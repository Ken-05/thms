# training_pipeline/train_classifier.py
import os
import argparse
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, callbacks
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import keras_tuner as kt
import shap # Used for model explainability
from google.cloud import storage
from datetime import datetime
import io
import yaml
#from core_utilities.preprocess import load_csv # Import for internal module access



# --- Function to load cloud_settings.yaml ---
def load_cloud_settings():
    """
    Loads configuration settings from the cloud_settings.yaml file.
    Assumes cloud_settings.yaml is in the 'config' directory at the project root.
    """
    # Get the directory of the current script (train_classifier.py is in 'training_pipeline/')
    BASE_DIR = os.path.abspath(os.path.dirname(__file__))
    # Go up one level (..) from 'training_pipeline' to the project root, then into 'config'
    CONFIG_DIR = os.path.join(BASE_DIR, "..", "config")
    CLOUD_SETTINGS_FILE_PATH = os.path.join(CONFIG_DIR, "cloud_settings.yaml")

    try:
        with open(CLOUD_SETTINGS_FILE_PATH, 'r') as f:
            settings = yaml.safe_load(f)
        print(f"[ClassifierTrain:Settings] Loaded cloud settings from {CLOUD_SETTINGS_FILE_PATH}")
        return settings
    except FileNotFoundError:
        print(f"[ClassifierTrain:Settings] ERROR: cloud_settings.yaml not found at {CLOUD_SETTINGS_FILE_PATH}. Exiting.")
        exit(1) # Critical error, cannot proceed without cloud settings
    except yaml.YAMLError as e:
        print(f"[ClassifierTrain:Settings] ERROR: Could not decode YAML from {CLOUD_SETTINGS_FILE_PATH}. Check file format. Error: {e}")
        exit(1) # Critical error

# Load cloud settings once when the module is imported
CLOUD_SETTINGS = load_cloud_settings()

# --- Configuration (now loaded from CLOUD_SETTINGS) ---
GCS_BUCKET_NAME = CLOUD_SETTINGS['gcs']['models_bucket']
GCS_VERSIONED_MODELS_PREFIX = CLOUD_SETTINGS['gcs']['models_versioned_prefix']
GCS_LATEST_MODELS_PREFIX = CLOUD_SETTINGS['gcs']['models_latest_prefix']

# Classifier training parameters
CLF_EPOCHS = CLOUD_SETTINGS['model_training_params']['classifier']['epochs']
CLF_BATCH_SIZE = CLOUD_SETTINGS['model_training_params']['classifier']['batch_size']
CLF_LEARNING_RATE_RANGE = (
    CLOUD_SETTINGS['model_training_params']['classifier']['learning_rate_min'],
    CLOUD_SETTINGS['model_training_params']['classifier']['learning_rate_max']
)
CLF_NUM_CLASSES = CLOUD_SETTINGS['model_training_params']['classifier']['num_classes']

# Keras Tuner parameters for Classifier (Bayesian Optimization)
CLF_TUNER_MAX_TRIALS = CLOUD_SETTINGS['model_training_params']['classifier']['tuner_max_trials']
CLF_TUNER_EXEC_PER_TRIAL = CLOUD_SETTINGS['model_training_params']['classifier']['tuner_executions_per_trial']
CLF_TUNER_NUM_CONV_LAYERS_MIN = CLOUD_SETTINGS['model_training_params']['classifier']['tuner_num_conv_layers_min']
CLF_TUNER_NUM_CONV_LAYERS_MAX = CLOUD_SETTINGS['model_training_params']['classifier']['tuner_num_conv_layers_max']
CLF_TUNER_FILTERS_CHOICES = CLOUD_SETTINGS['model_training_params']['classifier']['tuner_filters_choices']
CLF_TUNER_KERNEL_SIZE_CHOICES = CLOUD_SETTINGS['model_training_params']['classifier']['tuner_kernel_size_choices']
CLF_TUNER_NUM_LSTM_LAYERS_MIN = CLOUD_SETTINGS['model_training_params']['classifier']['tuner_num_lstm_layers_min']
CLF_TUNER_NUM_LSTM_LAYERS_MAX = CLOUD_SETTINGS['model_training_params']['classifier']['tuner_num_lstm_layers_max']
CLF_TUNER_LSTM_UNITS_CHOICES = CLOUD_SETTINGS['model_training_params']['classifier']['tuner_lstm_units_choices']
CLF_TUNER_DROPOUT_MIN = CLOUD_SETTINGS['model_training_params']['classifier']['tuner_dropout_min']
CLF_TUNER_DROPOUT_MAX = CLOUD_SETTINGS['model_training_params']['classifier']['tuner_dropout_max']
CLF_TUNER_DROPOUT_STEP = CLOUD_SETTINGS['model_training_params']['classifier']['tuner_dropout_step']

FEATURE_COLUMNS = CLOUD_SETTINGS['data_preparation_params']['feature_columns']

# Global variables for input shape (tuple: timesteps, num_features) and number of classes
input_sequence_shape = None # Will be (timesteps, num_features)
num_classes = None          # Will be set from CLF_NUM_CLASSES

# --- Setup Google Cloud Storage Client (for uploading models) ---
try:
    storage_client = storage.Client()
    bucket = storage_client.bucket(GCS_BUCKET_NAME)
    print(f"[ClassifierTrain] GCS client initialized for bucket: {GCS_BUCKET_NAME}")
except Exception as e:
    print(f"[ClassifierTrain] ERROR: Failed to initialize GCS client: {e}")
    exit(1)


# -----------------------------
# Build classifier model with tunable hyperparameters  (CNN + LSTM)
# -----------------------------
def build_classifier_model(hp):
    """
    Builds a Keras CNN+LSTM Classifier model with tunable hyperparameters for Keras Tuner.
    """
    model = models.Sequential()
    # Input layer expects (timesteps, num_features)
    model.add(layers.Input(shape=input_sequence_shape))
    
    # Tunable number of Conv1D layers
    for i in range(hp.Int('num_conv_layers', CLF_TUNER_NUM_CONV_LAYERS_MIN, CLF_TUNER_NUM_CONV_LAYERS_MAX)):
        model.add(layers.Conv1D(
            filters=hp.Choice(f'conv_filters_{i}', CLF_TUNER_FILTERS_CHOICES),
            kernel_size=hp.Choice(f'kernel_size_{i}', CLF_TUNER_KERNEL_SIZE_CHOICES),
            activation='relu',
            padding='same' # Keep output sequence length same as input
        ))
        model.add(layers.MaxPooling1D(pool_size=2, padding='same')) # Reduce sequence length
        model.add(layers.Dropout(rate=hp.Float(f'conv_dropout_{i}', CLF_TUNER_DROPOUT_MIN, CLF_TUNER_DROPOUT_MAX, step=CLF_TUNER_DROPOUT_STEP)))

        
        
    # Flatten before LSTM if only one LSTM layer, or use TimeDistributed Dense
    # For multiple LSTM layers, first LSTM should return_sequences=True
    
    # Tunable number of LSTM layers
    num_lstm_layers = hp.Int('num_lstm_layers', CLF_TUNER_NUM_LSTM_LAYERS_MIN, CLF_TUNER_NUM_LSTM_LAYERS_MAX)
    for i in range(num_lstm_layers):
        if i == num_lstm_layers - 1: # Last LSTM layer should not return sequences
            model.add(layers.LSTM(
                units=hp.Choice(f'lstm_units_{i}', CLF_TUNER_LSTM_UNITS_CHOICES),
                activation='relu'
            ))
        else: # Intermediate LSTM layers should return sequences
            model.add(layers.LSTM(
                units=hp.Choice(f'lstm_units_{i}', CLF_TUNER_LSTM_UNITS_CHOICES),
                activation='relu',
                return_sequences=True
            ))
        model.add(layers.Dropout(rate=hp.Float(f'lstm_dropout_{i}', CLF_TUNER_DROPOUT_MIN, CLF_TUNER_DROPOUT_MAX, step=CLF_TUNER_DROPOUT_STEP)))    
    
    # Output layer for multi-class classification
    model.add(layers.Dense(num_classes, activation='softmax')) # Uses the global num_classes.
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(hp.Float('learning_rate', CLF_LEARNING_RATE_RANGE[0], CLF_LEARNING_RATE_RANGE[1], sampling='log')),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

# -----------------------------
# Run hyperparameter tuner (Bayesian Optimization)
# -----------------------------
def run_tuner(X_train: np.ndarray, Y_train: np.ndarray):
    """
    Executes Keras Tuner's RandomSearch to find the best classifier hyperparameters.
    """
    tuner = kt.BayesianOptimization(
        build_classifier_model,
        objective='val_accuracy',
        max_trials=CLF_TUNER_MAX_TRIALS,
        executions_per_trial=CLF_TUNER_EXEC_PER_TRIAL,
        overwrite=True,
        directory='tuner_logs_classifier', # Local directory within the container for clf tuner logs.
        project_name='classifier_tuning_bayesian'
    )
    
    tuner.search_space_summary() # Prints a summary of the hyperparameter search space.
    
    # Starts the hyperparameter search.
    # Input data X_train and Y_train are now 3D and 1D NumPy arrays respectively.
    tuner.search(
        X_train, Y_train,
        epochs=CLF_EPOCHS,
        validation_split=0.2, # Using 20% of training data for validation during tuning.
        callbacks=[callbacks.EarlyStopping(monitor="val_loss", patience=5)],
        verbose=1
    )
    
    tuner.results_summary() # Prints a summary of the tuning results.
    best_model = tuner.get_best_models(num_models=1)[0] # Retrieves the best performing model.
    return best_model

# -----------------------------
# Explain predictions with SHAP
# -----------------------------
def explain_model(model: tf.keras.Model, X_sample: np.ndarray, feature_names: list):
    """
    Generates SHAP explanations for model predictions on a sample of data.
    For sequential models (CNN+LSTM), SHAP's KernelExplainer typically
    expects a 2D input. Flatten the sequences for SHAP explanation, which
    simplifies interpretation (but loses temporal context for SHAP itself).
    """
    
    if X_sample.size == 0: # Check if array is empty
        print("[ClassifierTrain:SHAP] No samples to explain.")
        return
    
    # Flatten 3D input to 2D for KernelExplainer: (samples, timesteps * features)
    original_timesteps = X_sample.shape[1]
    original_num_features = X_sample.shape[2]
    X_sample_flat = X_sample.reshape(X_sample.shape[0], -1) # Flatten the timesteps and features

    # Create flattened feature names for SHAP plot
    flattened_feature_names = []
    for t in range(original_timesteps):
        for feature_name in feature_names:
            flattened_feature_names.append(f"{feature_name}_t-{original_timesteps-1-t}") # e.g., temp_t-0, temp_t-1 (t-0 is most recent)

    # Use a sample of the flattened data for background
    background_data = shap.sample(X_sample_flat, min(100, X_sample_flat.shape[0]))

    # Using GradientExplainer for TensorFlow models as it's often more efficient than KernelExplainer
    try:
        explainer = shap.GradientExplainer(model, background_data)
    except Exception as e:
        print(f"[ClassifierTrain:SHAP] Warning: GradientExplainer failed ({e}). Falling back to KernelExplainer (slower).")
        explainer = shap.KernelExplainer(model.predict, background_data)

    num_explain_samples = min(200, X_sample_flat.shape[0])
    
    # Select samples from the flattened data for SHAP value calculation
    X_explain_samples = shap.sample(X_sample_flat, num_explain_samples)
    shap_values = explainer.shap_values(X_explain_samples)

    print(f"[ClassifierTrain:SHAP] Generated SHAP values for {num_explain_samples} samples.")
    # For local analysis 
    # mean_abs_shap = np.mean(np.abs(shap_values[0]), axis=0)
    # feature_importance_series = pd.Series(mean_abs_shap, index=flattened_feature_names).sort_values(ascending=False)
    # print("SHAP Feature Importance (Top 10):\n", feature_importance_series.head(10))
   
# -----------------------------
# Main training and evaluation function
# -----------------------------
def train_and_evaluate(X_Y_train: tuple, X_Y_val: tuple, model: tf.keras.Model, h5_output_path: str, tflite_output_path: str, feature_names: list):
    """
    Trains the classifier model on the full training data, evaluates it,
    explains predictions using SHAP, and saves the model.
    """
    X_train, y_train = X_Y_train
    X_val, y_val = X_Y_val
    
    es = callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1)

    # Model checkpoint callback to save the best model during training.
    checkpoint_cb = callbacks.ModelCheckpoint(
        filepath=h5_output_path, # Saves to the specified H5 output path.
        monitor="val_loss",
        save_best_only=True,
        verbose=1
    )
    
    # Trains the model.
    history = model.fit(
        X_train, y_train,
        epochs=CLF_EPOCHS,
        batch_size=CLF_BATCH_SIZE,
        validation_data=(X_val, y_val),
        callbacks=[es, checkpoint_cb],
        verbose=1
    )
    
        
    # Ensures the best model is loaded (if not already restored by EarlyStopping).
    model = models.load_model(h5_output_path) # Loads the best saved model.
    print(f"Final model loaded from {h5_output_path}")
    
    # Evaluates the model on the validation set.
    loss, acc = model.evaluate(X_val, y_val, verbose=0)
    print(f"[Evaluation] Test Loss: {loss:.4f}, Test Accuracy: {acc:.4f}")
    
    # Explains model predictions on the validation set.
    explain_model(model, X_val, feature_names)

    # Converts the model to TFLite format and saves it.
    convert_and_save_tflite(model, tflite_output_path)
 

# -----------------------------
# Convert to TfLite
# -----------------------------
def convert_and_save_tflite(model: tf.keras.Model, tflite_output_path: str):
    """
    Converts a Keras model to TFLite format and saves it and uploads to GCS.
    """
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    # Post-training quantization for edge optimization
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    
    classifier_tflite_model = converter.convert()

    # Saves the TFLite model using tf.io.gfile for GCS/local path compatibility.
    with tf.io.gfile.GFile(tflite_output_path, "wb") as f:
        f.write(classifier_tflite_model)
    print(f"TFLite model saved locally to {tflite_output_path}")

    # Get today's date for versioning
    version_date = datetime.now().strftime("%Y-%m-%d")
    # Path within GCS bucket for versioned models
    gcs_versioned_path = f"{GCS_VERSIONED_MODELS_PREFIX}{version_date}"
    # Path within GCS bucket for the 'latest' symlink/copy
    gcs_latest_path = GCS_LATEST_MODELS_PREFIX

    # Upload TFLite model
    blob_clf = bucket.blob(f"{gcs_versioned_path}/classifier_tflite_model.tflite")
    blob_clf.upload_from_string(classifier_tflite_model, content_type="application/octet-stream")
    print(f"Uploaded classifier_model.tflite to gs://{GCS_BUCKET_NAME}/{gcs_versioned_path}/classifier_tflite_model.tflite")

    # Update the 'latest' folder in GCS
    try:
        # Get the blobs from the newly uploaded versioned directory
        # Only the *specific* classifier model should be copied, not all blobs from the
        # versioned path as autoencoder training might have already uploaded its parts.
        source_blob_name = f"{gcs_versioned_path}/classifier_tflite_model.tflite"
        destination_blob_name = os.path.join(gcs_latest_path, "classifier_tflite_model.tflite") # Specific filename
        source_blob = bucket.blob(source_blob_name)
        bucket.copy_blob(source_blob, bucket, destination_blob_name)
        print(f"Copied {source_blob_name} to {destination_blob_name} (latest)")
    except Exception as e:
        print(f"[ClassifierTrain] ERROR updating 'latest' classifier model in GCS: {e}")

    
# -----------------------------
# Main function for script execution
# -----------------------------
def main():
    """
    Parses arguments, loads data, runs hyperparameter tuning,
    trains the final classifier model, and saves the model.
    """
    parser = argparse.ArgumentParser(description="Classifier Training Script")
    parser.add_argument("--train_path", type=str, required=True, help="Path to the preprocessed training features (NPY).")
    parser.add_argument("--train_labels_path", type=str, required=True, help="Path to the training labels (NPY).")
    parser.add_argument("--val_path", type=str, required=True, help="Path to the preprocessed validation features (NPY).")
    parser.add_argument("--val_labels_path", type=str, required=True, help="Path to the validation labels (NPY).")
    parser.add_argument("--test_path", type=str, required=True, help="Path to the preprocessed testing features (NPY).")
    parser.add_argument("--test_labels_path", type=str, required=True, help="Path to the testing labels (NPY).")
    parser.add_argument("--h5_output_path", type=str, required=True, help="Path to save the trained Keras H5 model.")
    parser.add_argument("--tflite_output_path", type=str, required=True, help="Path to save the trained TFLite model.")
    args = parser.parse_args()

    # Loads training, validation, and testing data from NPY files.
    X_train_3d = np.load(args.train_path)
    y_train_1d = np.load(args.train_labels_path)
    X_val_3d = np.load(args.val_path)
    y_val_1d = np.load(args.val_labels_path)
    X_test_3d = np.load(args.test_path)
    y_test_1d = np.load(args.test_labels_path)

    print(f"[Training] Loaded training data shape: {X_train_3d.shape}, labels: {y_train_1d.shape}")
    print(f"[Training] Loaded validation data shape: {X_val_3d.shape}, labels: {y_val_1d.shape}")
    print(f"[Training] Loaded testing data shape: {X_test_3d.shape}, labels: {y_test_1d.shape}")
    
    global input_sequence_shape, num_classes
    input_sequence_shape = X_train_3d.shape[1:] # (timesteps, num_features)
    num_classes = CLF_NUM_CLASSES
    
    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train_1d)
    y_val_encoded = label_encoder.transform(y_val_1d) 
    y_test_encoded = label_encoder.transform(y_test_1d)    
    

    print("[Training] Running hyperparameter tuning for classifier  (Bayesian Optimization)...")
    best_model = run_tuner(X_train_3d, y_train_encoded)

    print("[Training] Training final classifier model...")
    # Passes the combined X and y for training and validation to the evaluation function.
    train_and_evaluate(
        (X_train_3d, y_train_encoded),
        (X_val_3d, y_val_encoded), # Pass validation data to train_and_evaluate
        best_model,
        args.h5_output_path,
        args.tflite_output_path,
        FEATURE_COLUMNS # Pass original feature names for SHAP
    )

    print("--- Classifier Training Script Finished ---")
    
if __name__ == "__main__":
    main()