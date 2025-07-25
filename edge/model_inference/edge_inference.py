# edge_inference.py
import os
import json
import numpy as np
import pandas as pd 
import joblib
import tflite_runtime.interpreter as tflite


# --- Function to load settings from settings.json ---
def load_settings():
    """
    Loads configuration settings from the settings.json file.
    Assumes settings.json is in the 'config' directory one level up from this file.
    """
    # Get the directory of the current script (edge_inference.py is in 'model_inference/')
    BASE_DIR = os.path.abspath(os.path.dirname(__file__))
    CONFIG_DIR = os.path.join(BASE_DIR, "..", "config")
    SETTINGS_FILE_PATH = os.path.join(CONFIG_DIR, "settings.json")

    try:
        with open(SETTINGS_FILE_PATH, 'r') as f:
            settings = json.load(f)
        print(f"[EdgeInference:Settings] Loaded settings from {SETTINGS_FILE_PATH}")
        return settings
    except FileNotFoundError:
        print(f"[EdgeInference:Settings] ERROR: settings.json not found at {SETTINGS_FILE_PATH}")
        exit(1)
    except json.JSONDecodeError:
        print(f"[EdgeInference:Settings] ERROR: Could not decode JSON from {SETTINGS_FILE_PATH}. Check file format.")
        exit(1)

# Load settings once when the module is imported
SETTINGS = load_settings()


# ----------------------
# Configuration (loaded from SETTINGS)
# ----------------------
ANOMALY_THRESHOLD = SETTINGS['inference']['anomaly_threshold'] # Initially set based on validation (reminder to tweak as needed)
FAULT_CONFIDENCE_THRESHOLD = SETTINGS['inference']['fault_confidence_threshold']
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
MODELS_BASE_DIR = SETTINGS['inference']['model_dir']
AUTOENCODER_FILENAME = SETTINGS['inference']['autoencoder_filename']
CLASSIFIER_FILENAME = SETTINGS['inference']['classifier_filename']
STATUS_OUTPUT_FILE = SETTINGS['inference']['status_output_file'] 

# Global interpreters and details, initialized by a dedicated function
ae_interpreter = None
ae_input_details = None
ae_output_details = None
clf_interpreter = None
clf_input_details = None
clf_output_details = None

# ----------------------
# Model Loading Function
# ----------------------
def load_tflite_models(model_dir):
    """
    Loads the TFLite autoencoder and classifier models from the specified directory.
    This function is designed to be called when models need to be reloaded (after an update).
    """
    global ae_interpreter, ae_input_details, ae_output_details
    global clf_interpreter, clf_input_details, clf_output_details
    
    full_models_path = os.path.join(BASE_DIR, "..")
    autoencoder_model_path = os.path.join(full_models_path, model_dir, AUTOENCODER_FILENAME)
    classifier_model_path = os.path.join(full_models_path, model_dir, CLASSIFIER_FILENAME)

    # Load Autoencoder Interpreter
    try:
        ae_interpreter = tflite.Interpreter(model_path=autoencoder_model_path)
        ae_interpreter.allocate_tensors()
        ae_input_details = ae_interpreter.get_input_details()
        ae_output_details = ae_interpreter.get_output_details()
        print(f"[EdgeInference] Autoencoder TFLite model loaded from {autoencoder_model_path}")
    except Exception as e:
        print(f"[EdgeInference] ERROR loading Autoencoder model from {autoencoder_model_path}: {e}")
        ae_interpreter = None # Set to None on failure to prevent using a bad interpreter
        ae_input_details = None
        ae_output_details = None

    # Load Classifier Interpreter
    try:
        clf_interpreter = tflite.Interpreter(model_path=classifier_model_path)
        clf_interpreter.allocate_tensors()
        clf_input_details = clf_interpreter.get_input_details()
        clf_output_details = clf_interpreter.get_output_details()
        print(f"[EdgeInference] Classifier TFLite model loaded from {classifier_model_path}")
    except Exception as e:
        print(f"[EdgeInference] ERROR loading Classifier model from {classifier_model_path}: {e}")
        clf_interpreter = None # Set to None on failure
        clf_input_details = None
        clf_output_details = None

# Call the model loading function once when the module is first imported
# This ensures models are ready when perform_edge_inference is called.
# The `data_collect.py`'s restart mechanism will trigger this reload.
load_tflite_models(MODELS_BASE_DIR)


# ----------------------------
# Run TFLite Inference Helpers
# ----------------------------
def run_ae_inference(input_data):
    """
    Runs inference on the autoencoder TFLite model.
    """
    if ae_interpreter is None:
        print("[EdgeInference] Autoencoder interpreter not available. Cannot perform inference")
        return None
    
    # Ensure input data shape and type match the model's expectation (tflite expects float32)
    # Reshape if necessary (e.g., from (num_features,) to (1, num_features))
    if input_data.ndim == 1:
        input_data = np.expand_dims(input_data, axis=0) # Add batch dimension if missing
    input_data = input_data.astype(ae_input_details[0]['dtype'])
    ae_interpreter.set_tensor(ae_input_details[0]['index'], input_data)
    ae_interpreter.invoke()
    output = ae_interpreter.get_tensor(ae_output_details[0]['index'])
    return output

def run_clf_inference(input_data):
    """
    Runs inference on the classifier TFLite model.
    """
    if clf_interpreter is None:
        print("[EdgeInference] Classifier interpreter not available. Cannot perform inference.")
        return None

    # Ensure input data shape and type match the model's expectation
    if input_data.ndim == 1:
        input_data = np.expand_dims(input_data, axis=0) # Add batch dimension if missing
    input_data = input_data.astype(clf_input_details[0]['dtype'])
    clf_interpreter.set_tensor(clf_input_details[0]['index'], input_data)
    clf_interpreter.invoke()
    output = clf_interpreter.get_tensor(clf_output_details[0]['index'])
    return output

# ----------------------
# Reconstruction error calculation
# ----------------------
def calc_reconstruction_error(original, reconstructed):
    """
    Calculates the mean squared error between original and reconstructed data.
    """
    # Ensure both are 2D arrays for element-wise operation (e.g., (1, N))
    if original.ndim == 1:
        original = np.expand_dims(original, axis=0)
    if reconstructed.ndim == 1:
        reconstructed = np.expand_dims(reconstructed, axis=0)

    # Mean squared error per sample
    return np.mean((original - reconstructed) ** 2)

# ----------------------
# Main inference function to be called by InferenceRunner
# ----------------------
def perform_edge_inference(preprocessed_data):
    """
    Performs anomaly detection and fault classification on preprocessed data.
    
    Args:
        preprocessed_data (np.array): A 2D numpy array (1, num_features) of scaled features.
                                      Expected to be already scaled by edge_preprocess.

    Returns:
        dict: A dictionary containing inference results (anomaly status, fault class, error).
    """
    if preprocessed_data is None:
        print("[EdgeInference] No preprocessed data provided for inference.")
        return {"is_anomaly": False, "fault_class": -1, "reconstruction_error": -1.0,  "confidence": 0.0}

    # Ensure preprocessed_data has a batch dimension (1, num_features)
    if preprocessed_data.ndim == 1:
        preprocessed_data = np.expand_dims(preprocessed_data, axis=0)

    # Run anomaly detection (autoencoder)
    reconstructed = run_ae_inference(preprocessed_data)
    if reconstructed is None:
        return {"is_anomaly": False, "fault_class": -1, "reconstruction_error": -1.0, "confidence": 0.0}

    recon_error = calc_reconstruction_error(preprocessed_data, reconstructed)
    print(f"[EdgeInference] Reconstruction error: {recon_error}")

    is_anomaly = recon_error > ANOMALY_THRESHOLD
    predicted_class = -1 # Default to no fault
    confidence = 0.0     # Default confidence

    if is_anomaly:
        # Anomaly detected — run classifier
        clf_output = run_clf_inference(preprocessed_data)
        if clf_output is not None:
            # Softmax output gives probabilities, find the class with highest probability
            predicted_class = np.argmax(clf_output, axis=1)[0]
            confidence = np.max(clf_output, axis=1)[0] # Get the max probability as confidence

            # Only classify if confidence is above threshold
            if confidence >= FAULT_CONFIDENCE_THRESHOLD:
                print(f"[EdgeInference] Anomaly detected! Predicted fault class: {predicted_class} (Confidence: {confidence:.2f})")
            else:
                print(f"[EdgeInference] Anomaly detected, but classifier confidence too low ({confidence:.2f} < {SETTINGS['inference']['fault_confidence_threshold']}). No specific fault classified.")
                predicted_class = -1 # Revert to no specific fault if confidence is too low
                confidence = 0.0 # Reset confidence if not classified
        else:
            print("[EdgeInference] Classifier not available, cannot classify anomaly.")
            predicted_class = -1 # No specific fault if classifier fails
    else:
        print("[EdgeInference] No anomaly detected; skipping classification.")

    return {
        "is_anomaly": bool(is_anomaly), # Ensure boolean type for JSON
        "fault_class": int(predicted_class),
        "reconstruction_error": float(recon_error),
        "confidence": float(confidence)
    }

# This __main__ block is for local testing of this module's functions only.
if __name__ == "__main__":
    print("[EdgeInference] Running local test of perform_edge_inference.")
    
    # Simulate preprocessed data (should match model's expected input shape and type)    
    # Get expected input shape from the loaded autoencoder interpreter
    if ae_interpreter:
        dummy_num_features = ae_input_details[0]['shape'][1]
        dummy_preprocessed_data = np.random.rand(1, dummy_num_features).astype(np.float32)
        print(f"Simulating preprocessed data with shape: {dummy_preprocessed_data.shape}")

        inference_results = perform_edge_inference(dummy_preprocessed_data)
        print(f"\nTest Inference Results: {inference_results}")

        # Example of saving status for GUI for local test
        # Use the path from settings for consistency
        gui_status_file_path = os.path.join(
            os.path.abspath(os.path.dirname(__file__)), 
            "..",
            STATUS_OUTPUT_FILE # e.g., "gui/latest_status.json"
        )
        os.makedirs(os.path.dirname(gui_status_file_path), exist_ok=True) # Ensure directory exists
        with open(gui_status_file_path, "w") as f:
            json.dump(inference_results, f, indent=4)
        print(f"Test status payload saved to {gui_status_file_path}")
    else:
        print("[EdgeInference] Skipping local test: Models could not be loaded. Ensure dummy models exist in the configured path.")
