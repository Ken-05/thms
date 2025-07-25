# training_pipeline/train_autoencoder.py
import os
import argparse
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, callbacks
from sklearn.model_selection import train_test_split
import keras_tuner as kt
from joblib import dump
from google.cloud import storage
from datetime import datetime
import io
import yaml
# from core_utilities.preprocess import preprocess # Import for internal module access


# --- Function to load cloud_settings.yaml ---
def load_cloud_settings():
    """
    Loads configuration settings from the cloud_settings.yaml file.
    Assumes cloud_settings.yaml is in the 'config' directory at the project root.
    """
    # Get the directory of the current script (train_autoencoder.py is in 'training_pipeline/')
    BASE_DIR = os.path.abspath(os.path.dirname(__file__))
    # Go up one level (..) from 'training_pipeline' to the project root, then into 'config'
    CONFIG_DIR = os.path.join(BASE_DIR, "..", "config")
    CLOUD_SETTINGS_FILE_PATH = os.path.join(CONFIG_DIR, "cloud_settings.yaml")

    try:
        with open(CLOUD_SETTINGS_FILE_PATH, 'r') as f:
            settings = yaml.safe_load(f)
        print(f"[AutoencoderTrain:Settings] Loaded cloud settings from {CLOUD_SETTINGS_FILE_PATH}")
        return settings
    except FileNotFoundError:
        print(f"[AutoencoderTrain:Settings] ERROR: cloud_settings.yaml not found at {CLOUD_SETTINGS_FILE_PATH}. Exiting.")
        exit(1) # Critical error, cannot proceed without cloud settings
    except yaml.YAMLError as e:
        print(f"[AutoencoderTrain:Settings] ERROR: Could not decode YAML from {CLOUD_SETTINGS_FILE_PATH}. Check file format. Error: {e}")
        exit(1) # Critical error

# Load cloud settings once when the module is imported
CLOUD_SETTINGS = load_cloud_settings()

# --- Configuration (now loaded from CLOUD_SETTINGS) ---
GCS_BUCKET_NAME = CLOUD_SETTINGS['gcs']['models_bucket']
GCS_VERSIONED_MODELS_PREFIX = CLOUD_SETTINGS['gcs']['models_versioned_prefix']
GCS_LATEST_MODELS_PREFIX = CLOUD_SETTINGS['gcs']['models_latest_prefix']

# Autoencoder training parameters
AE_EPOCHS = CLOUD_SETTINGS['model_training_params']['autoencoder']['epochs']
AE_BATCH_SIZE = CLOUD_SETTINGS['model_training_params']['autoencoder']['batch_size']
AE_LEARNING_RATE_CHOICES = CLOUD_SETTINGS['model_training_params']['autoencoder']['learning_rate_choices']
AE_ANOMALY_THRESHOLD_FACTOR = CLOUD_SETTINGS['model_training_params']['autoencoder']['anomaly_threshold_factor']

# Keras Tuner parameters for Autoencoder (Bayesian Optimization)
TUNER_MAX_TRIALS = CLOUD_SETTINGS['model_training_params']['autoencoder']['tuner_max_trials'] 
TUNER_EXEC_PER_TRIAL = CLOUD_SETTINGS['model_training_params']['autoencoder']['tuner_executions_per_trial']
AE_TUNER_LSTM_UNITS_CHOICES = CLOUD_SETTINGS['model_training_params']['autoencoder']['tuner_lstm_units_choices']
AE_TUNER_DROPOUT_MIN = CLOUD_SETTINGS['model_training_params']['autoencoder']['tuner_dropout_min']
AE_TUNER_DROPOUT_MAX = CLOUD_SETTINGS['model_training_params']['autoencoder']['tuner_dropout_max']
AE_TUNER_DROPOUT_STEP = CLOUD_SETTINGS['model_training_params']['autoencoder']['tuner_dropout_step']

# Sequence Length from data_preparation_params
SEQUENCE_LENGTH = CLOUD_SETTINGS['data_preparation_params']['sequence_length']

# Global variable for input shape (timesteps, num_features)
input_sequence_shape = None

# --- Setup Google Cloud Storage Client (for uploading models) ---
try:
    storage_client = storage.Client()
    bucket = storage_client.bucket(GCS_BUCKET_NAME)
    print(f"[AutoencoderTrain] GCS client initialized for bucket: {GCS_BUCKET_NAME}")
except Exception as e:
    print(f"[AutoencoderTrain] ERROR: Failed to initialize GCS client: {e}")
    # This is a critical error for model deployment, so exit.
    exit(1)


# -----------------------------
# Define HyperModel for Keras Tuner for the (LSTM Autoencoder)
# -----------------------------
def build_autoencoder(hp):
    """
    Builds a Keras LSTM Autoencoder model with tunable hyperparameters for Keras Tuner.
    Input shape is (timesteps, num_features).
    """
    timesteps = input_sequence_shape[0]
    num_features = input_sequence_shape[1]
    
    input_layer = keras.Input(shape=(timesteps, num_features))

    # Encoder
    # Using return_sequences=True for intermediate LSTM layers if stacking LSTMs
    encoded = layers.LSTM(
        units=hp.Choice('lstm_units_enc', AE_TUNER_LSTM_UNITS_CHOICES),
        activation='relu',
        return_sequences=True # Must return sequences if another LSTM follows, or if TimeDistributed Dense follows
    )(input_layer)
    encoded = layers.Dropout(rate=hp.Float('enc_dropout_1', AE_TUNER_DROPOUT_MIN, AE_TUNER_DROPOUT_MAX, step=AE_TUNER_DROPOUT_STEP))(encoded)
    
    # Another LSTM layer, but this one will output a single vector per sequence
    # This forms the latent space.
    encoded = layers.LSTM(
        units=hp.Int('latent_dim', 8, 32, step=8), # Tunable latent dimension
        activation='relu',
        return_sequences=False # Last encoder LSTM should not return sequences
    )(encoded)
    encoded = layers.Dropout(rate=hp.Float('enc_dropout_2', AE_TUNER_DROPOUT_MIN, AE_TUNER_DROPOUT_MAX, step=AE_TUNER_DROPOUT_STEP))(encoded)

    # Decoder
    # RepeatVector is used to replicate the latent vector for each timestep in the decoder sequence
    decoded = layers.RepeatVector(timesteps)(encoded)
    
    decoded = layers.LSTM(
        units=hp.Choice('lstm_units_dec', AE_TUNER_LSTM_UNITS_CHOICES),
        activation='relu',
        return_sequences=True # Must return sequences for subsequent TimeDistributed Dense layer
    )(decoded)
    decoded = layers.Dropout(rate=hp.Float('dec_dropout_1', AE_TUNER_DROPOUT_MIN, AE_TUNER_DROPOUT_MAX, step=AE_TUNER_DROPOUT_STEP))(decoded)

    # TimeDistributed Dense layer to output feature dimension for each timestep
    decoded = layers.TimeDistributed(layers.Dense(num_features, activation='linear'))(decoded)

   
    autoencoder = keras.Model(inputs=input_layer, outputs=decoded)
    autoencoder.compile(
        optimizer=keras.optimizers.Adam(learning_rate=hp.Choice("lr", AE_LEARNING_RATE_CHOICES)),
        loss="mse"
    )
    return autoencoder

# -----------------------------
# Run Tuner ( Bayesian Optimization)
# -----------------------------
def run_tuner(X_train: np.ndarray, X_val: np.ndarray):
    """
    Executes Keras Tuner's BayesianOptimization to find the best autoencoder hyperparameters.
    """
    # Sets global input_sequence_shape for build_autoencoder
    global input_sequence_shape
    input_sequence_shape = X_train.shape[1:] # (timesteps, num_features)

    # Creates a tuner instance, providing the build function and objective.
    tuner = kt.BayianOptimization(
        build_autoencoder,
        objective="val_loss",
        max_trials=TUNER_MAX_TRIALS,
        executions_per_trial=TUNER_EXEC_PER_TRIAL,
        overwrite=True,
        directory="tuner_logs_autoencoder", # Local directory within the container for ae tuner logs.
        project_name="autoencoder_tuning_bayesian"
    )

    tuner.search_space_summary() # Prints a summary of the hyperparameter search space.

    # Starts the hyperparameter search.
    tuner.search(
        X_train, X_train,
        epochs=AE_EPOCHS,
        validation_data=(X_val, X_val),
        callbacks=[callbacks.EarlyStopping(patience=5, restore_best_weights=True)],
        verbose=1
    )

    tuner.results_summary() # Prints a summary of the tuning results.

    best_model = tuner.get_best_models(num_models=1)[0] # Retrieves the best performing model.
    return best_model

# -----------------------------
# Train final model + evaluate
# -----------------------------
def train_and_evaluate(X_train: np.ndarray, X_val: np.ndarray, model: tf.keras.Model, h5_output_path: str, tflite_output_path: str, scaler):
    """
    Trains the autoencoder model on the full training data and saves the model.
    """
    es = callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1)

    # Model checkpoint callback to save the best model during training.
    checkpoint_cb = callbacks.ModelCheckpoint(
        filepath=h5_output_path, # Saves to the specified H5 output path.
        monitor="val_loss",
        save_best_only=True,
        verbose=1
    )

    # Trains the model.
    history = model.fit(
        X_train, X_train,
        epochs=AE_EPOCHS,
        batch_size=AE_BATCH_SIZE,
        validation_data=(X_val, X_val),
        callbacks=[es, checkpoint_cb],
        verbose=1
    )

    # Ensures the best model is loaded (if not already restored by EarlyStopping).
    model = keras.models.load_model(h5_output_path) # Loads the best saved model.
    print(f"Final model loaded from {h5_output_path}")

    # Converts the model to TFLite format and saves it.
    convert_and_save_tflite(model, tflite_output_path, scaler, X_val)


# -----------------------------
# Convert to TfLite
# -----------------------------
def convert_and_save_tflite(model: tf.keras.Model, tflite_output_path: str, scaler, X_val: np.ndarray):
    """
    Converts a Keras model to TFLite format and saves it, and uploads to GCS.
    Also calculates and uploads the anomaly detection threshold.
    """
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    # Post-training quantization for edge optimization
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    
    autoencoder_tflite_model = converter.convert()
    
    # Saves the TFLite model using tf.io.gfile for GCS/local path compatibility.
    with tf.io.gfile.GFile(tflite_output_path, "wb") as f:
        f.write(autoencoder_tflite_model)
    print(f"TFLite model saved locally to {tflite_output_path}")
    
    # Calculate reconstruction errors on validation set to set threshold
    # X_val is a 3D array (samples, timesteps, features)
    reconstructions = model.predict(X_val)
    # MSE needs to be calculated per sample across the entire sequence and features
    errors = np.mean(np.square(X_val - reconstructions), axis=(1,2))  # Mean over timesteps and features
    
    anomaly_threshold = np.mean(errors) + AE_ANOMALY_THRESHOLD_FACTOR * np.std(errors)
    print(f"[AutoencoderTrain] Calculated anomaly threshold: {anomaly_threshold:.6f}")
   
    # Get today's date for versioning
    version_date = datetime.now().strftime("%Y-%m-%d")
    
    # Path within GCS bucket for versioned models
    gcs_versioned_path = f"{GCS_VERSIONED_MODELS_PREFIX}{version_date}"
    # Path within GCS bucket for the 'latest' copy
    gcs_latest_path = GCS_LATEST_MODELS_PREFIX 
    
    
    # Upload TFLite model
    blob_ae = bucket.blob(f"{gcs_versioned_path}/autoencoder_tflite_model.tflite")
    blob_ae.upload_from_string(autoencoder_tflite_model, content_type="application/octet-stream")
    print(f"Uploaded autoencoder_model.tflite to gs://{GCS_BUCKET_NAME}/{gcs_versioned_path}/autoencoder_tflite_model.tflite")

    # Upload scaler - Scaler is saved here to align with model versioning.
    scaler_bytes = io.BytesIO()
    dump(scaler, scaler_bytes)
    scaler_bytes.seek(0) # Rewind the buffer to the beginning
    blob_scaler = bucket.blob(f"{gcs_versioned_path}/scaler.joblib") 
    blob_scaler.upload_from_file(scaler_bytes, content_type="application/octet-stream") 
    print(f"Uploaded scaler to gs://{GCS_BUCKET_NAME}/{gcs_versioned_path}/scaler.joblib")

    # Upload anomaly threshold
    threshold_blob = bucket.blob(f"{gcs_versioned_path}/anomaly_threshold.json")
    threshold_blob.upload_from_string(json.dumps({"anomaly_threshold": float(anomaly_threshold)}), content_type="application/json")
    print(f"Uploaded anomaly_threshold.json to gs://{GCS_BUCKET_NAME}/{gcs_versioned_path}/anomaly_threshold.json")

    # Update the 'latest' folder in GCS
    # Requires copying the files from the versioned path to the 'latest' path
    try:
        # Define the files that should be copied to 'latest'
        files_to_copy = [
            "autoencoder_tflite_model.tflite",
            "scaler.joblib",
            "anomaly_threshold.json"
        ]
        
        for filename in files_to_copy:
            source_blob_name = f"{gcs_versioned_path}/{filename}"
            destination_blob_name = os.path.join(gcs_latest_path, filename)
            source_blob = bucket.blob(source_blob_name)
            
            # Check if source blob exists before copying
            if source_blob.exists():
                bucket.copy_blob(source_blob, bucket, destination_blob_name)
                print(f"Copied {source_blob_name} to {destination_blob_name} (latest)")
            else:
                print(f"[AutoencoderTrain] Warning: Source blob {source_blob_name} not found for 'latest' update.")

        print(f"Updated 'latest' models in gs://{GCS_BUCKET_NAME}/{gcs_latest_path}")
                
    except Exception as e:
        print(f"[AutoencoderTrain] ERROR updating 'latest' models in GCS: {e}")

 
# -----------------------------
# Main function for script execution
# -----------------------------
def main():
    """
    Parses arguments, loads and preprocesses data, runs hyperparameter tuning,
    trains the final autoencoder model, and saves the model and scaler.
    """
    parser = argparse.ArgumentParser(description="Autoencoder Training Script")
    parser.add_argument("--train_path", type=str, required=True, help="Path to the preprocessed training features (NPY).")
    parser.add_argument("--train_labels_path", type=str, required=True, help="Path to the training labels (NPY).") # Added for consistency in pipeline, though AE doesn't use y
    parser.add_argument("--val_path", type=str, required=True, help="Path to the preprocessed validation features (NPY).")
    parser.add_argument("--val_labels_path", type=str, required=True, help="Path to the validation labels (NPY).") # Added for consistency in pipeline, though AE doesn't use y
    parser.add_argument("--h5_output_path", type=str, required=True, help="Path to save the trained Keras H5 model.")
    parser.add_argument("--tflite_output_path", type=str, required=True, help="Path to save the trained TFLite model.")
    parser.add_argument("--scaler_output_path", type=str, required=True, help="Path to save the MinMaxScaler object (joblib).")
    args = parser.parse_args()

    print("[Training] Loading preprocessed data for autoencoder...")
    # Load training and validation features from NPY files.
    X_train_ae = np.load(args.train_path)
    X_val_ae = np.load(args.val_path)
   
    # Load scaler (needed for TFLite conversion, as it's a model artifact now)
    # Ensure scaler is available (it's loaded by preprocess module itself)
    try:
        scaler = joblib.load(args.scaler_output_path)
        print(f"[Training] Scaler loaded from {args.scaler_output_path} for TFLite conversion.")
    except Exception as e:
        print(f"[Training] ERROR loading scaler from {args.scaler_output_path}: {e}")
        exit(1) # Critical error if scaler cannot be loaded
   
    # Ensure input_sequence_shape is set globally for build_autoencoder
    global input_sequence_shape
    input_sequence_shape = X_train_ae.shape[1:] # Should be (timesteps, num_features)

    print("[Training] Running hyperparameter tuning for autoencoder (Bayesian Optimization)...")
    best_model = run_tuner(X_train_ae, X_val_ae)

    print("[Training] Training final autoencoder model...")
    train_and_evaluate(X_train_ae, X_val_ae, best_model, args.h5_output_path, args.tflite_output_path, scaler)

    print("--- Autoencoder Training Script Finished ---")

if __name__ == "__main__":
    main()