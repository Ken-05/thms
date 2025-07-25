# update_models.py

from google.cloud import storage
import os
import subprocess
import time
from datetime import datetime
import json

# This script is used to update the models used for 
# inference at the edge because retraining happens
# in the cloud on new data everyday, so the models
# are downloaded and updated locally to the same
# directory that contains the edge_inference.py 
# everyday. 

# This script is run periodically via a cron job 
# at 1:30 AM everyday(0.5 hours before another cron job 
# uploads a csv file of combined data gotten from 
# sensors and CANBus to the cloud and 1.5 hours before a
# new model is retrained, to allow 24 hours for time to 
# train new model before deploying to the edge)



# --- Function to load settings from settings.json ---
def load_settings():
    """
    Loads configuration settings from the settings.json file.
    Assumes settings.json is in the 'config' directory one level up from this file.
    """
    # Get the directory of the current script 
    BASE_DIR = os.path.abspath(os.path.dirname(__file__))
    # Go up one level and then into 'config'
    CONFIG_DIR = os.path.join(BASE_DIR, "..", "config")
    SETTINGS_FILE_PATH = os.path.join(CONFIG_DIR, "settings.json")

    try:
        with open(SETTINGS_FILE_PATH, 'r') as f:
            settings = json.load(f)
        print(f"[UpdateScript:Settings] Loaded settings from {SETTINGS_FILE_PATH}")
        return settings
    except FileNotFoundError:
        print(f"[UpdateScript:Settings] ERROR: settings.json not found at {SETTINGS_FILE_PATH}. Ensure it's deployed.")
        exit(1) # Critical error, cannot proceed without settings
    except json.JSONDecodeError:
        print(f"[UpdateScript:Settings] ERROR: Could not decode JSON from {SETTINGS_FILE_PATH}. Check file format.")
        exit(1) # Critical error

# Load settings at the very beginning of the script
SETTINGS = load_settings()


# --- Configuration (loaded from SETTINGS) ---
# Construct the absolute path for credentials based on the project structure
GCP_CREDENTIALS_PATH = os.path.join(
    os.path.abspath(os.path.dirname(__file__)), 
    "..", 
    SETTINGS['cloud_sync']['gcp_credentials_path_relative'] # "keys/gcp-service-account.json"
)

GCS_BUCKET_NAME = SETTINGS['cloud_sync']['gcs_models_bucket']
GCS_MODELS_PREFIX = SETTINGS['cloud_sync']['gcs_models_prefix'] 
MODELS_BASE_DIR = SETTINGS['inference']['model_dir']
LOCAL_MODELS_BASE_DIR = SETTINGS['inference']['temp_model_dir']


# Local directories for models
LOCAL_MODEL_DIR = os.path.normpath(os.path.join(os.path.abspath(os.path.dirname(__file__)), "..", MODELS_BASE_DIR))
LOCAL_TEMP_DIR = os.path.normpath(os.path.join(os.path.abspath(os.path.dirname(__file__)), "..", LOCAL_MODELS_BASE_DIR))

SERVICE_NAME = SETTINGS['data_collection']['systemd_service_name']

# --- Setup Google Cloud Storage Client ---
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = GCP_CREDENTIALS_PATH
try:
    storage_client = storage.Client()
    bucket = storage_client.bucket(GCS_BUCKET_NAME)
    print(f"[UpdateScript] GCS client initialized for bucket: {GCS_BUCKET_NAME}")
except Exception as e:
    print(f"[UpdateScript] ERROR: Failed to initialize GCS client with credentials {GCP_CREDENTIALS_PATH}: {e}")
    exit(1) # Exit if cannot connect to GCS



def get_latest_model_version():
    """
    Lists directories under GCS_MODELS_PREFIX and returns the latest YYYY-MM-DD version.
    Assumes directory names are in YYYY-MM-DD format and are sortable.
    """
    try:
        # List blobs, and filter for directories (common prefix) to find version folders
        # Using delimiter '/' to list 'directories'
        blobs = bucket.list_blobs(prefix=GCS_MODELS_PREFIX, delimiter='/')

        # The .prefixes attribute contains the "directories" found by the delimiter
        version_folders = [prefix.rstrip('/') for prefix in blobs.prefixes if prefix.rstrip('/')]

        # Filter for YYYY-MM-DD format and sort to get the latest
        valid_versions = []
        for vf in version_folders:
            # Extract just the date part, e.g., "tflite_models/2025-07-18" -> "2025-07-18"
            version_date_str = vf.split('/')[-1] 
            try:
                datetime.strptime(version_date_str, "%Y-%m-%d") # Validate format
                valid_versions.append(version_date_str)
            except ValueError:
                continue # Skip if not in YYYY-MM-DD format

        if not valid_versions:
            return None

        # Return the latest version (string comparison works for YYYY-MM-DD)
        return sorted(valid_versions)[-1]

    except Exception as e:
        print(f"[UpdateScript] ERROR listing GCS versions: {e}")
        return None

def download_model_files(gcs_version_path, local_target_dir):
    """
    Downloads model files from a given GCS version to a local directory.
    """
    gcs_source_prefix = f"{GCS_MODELS_PREFIX}{gcs_version_path}/"
    print(f"[UpdateScript] Downloading from GCS prefix: {gcs_source_prefix}")

    # Ensure local temporary directory exists and is empty
    os.makedirs(local_target_dir, exist_ok=True)
    for f in os.listdir(local_target_dir):
        os.remove(os.path.join(local_target_dir, f))

    try:
        blobs = bucket.list_blobs(prefix=gcs_source_prefix)
        downloaded_count = 0
        for blob in blobs:
            # Only download the actual files, not the directory itself
            if blob.name.endswith('/'): # Skip directory blobs
                continue
            file_name = os.path.basename(blob.name)
            local_file_path = os.path.join(local_target_dir, file_name)
            blob.download_to_filename(local_file_path)
            print(f"[UpdateScript] Downloaded {file_name} to {local_file_path}")
            downloaded_count += 1

        if downloaded_count == 0:
            print(f"[UpdateScript] Warning: No files found to download at {gcs_source_prefix}. This might indicate an issue with the GCS path or pipeline.")
            return False

        print(f"[UpdateScript] Successfully downloaded {downloaded_count} files.")
        return True
    except Exception as e:
        print(f"[UpdateScript] ERROR downloading model files: {e}")
        return False

def replace_models(source_dir, target_dir):
    """
    Atomically replaces old models with new ones.
    Removes existing files in target_dir and copies new ones.
    """
    try:
        # Clear existing models
        if os.path.exists(target_dir):
            for f in os.listdir(target_dir):
                os.remove(os.path.join(target_dir, f))
            print(f"[UpdateScript] Cleared existing models in {target_dir}")
        else:
            os.makedirs(target_dir, exist_ok=True)
            print(f"[UpdateScript] Created target directory: {target_dir}")

        # Copy new models
        for f in os.listdir(source_dir):
            src_file = os.path.join(source_dir, f)
            dst_file = os.path.join(target_dir, f)
            os.rename(src_file, dst_file) # os.rename is atomic on the same filesystem
            print(f"[UpdateScript] Replaced {f}")
        print("[UpdateScript] Models replaced successfully.")
        return True
    except Exception as e:
        print(f"[UpdateScript] ERROR replacing models: {e}")
        return False

def restart_service(service_name):
    """Restarts the systemd service."""
    try:
        print(f"[UpdateScript] Stopping {service_name}...")
        subprocess.run(["sudo", "systemctl", "stop", service_name], check=True, capture_output=True, text=True)
        print(f"[UpdateScript] Stop output: {result.stdout}")
        if result.stderr:
            print(f"[UpdateScript] Stop error: {result.stderr}")
            
        time.sleep(2) # Give a moment to stop
        
        print(f"[UpdateScript] Starting {service_name}...")
        result = subprocess.run(["sudo", "systemctl", "start", service_name], check=True, capture_output=True, text=True)
        print(f"[UpdateScript] Start output: {result.stdout}")
        if result.stderr:
            print(f"[UpdateScript] Start error: {result.stderr}")print(f"[UpdateScript] {service_name} restarted.")
        return True
    except subprocess.CalledProcessError as e:
        print(f"[UpdateScript] ERROR restarting service: {e.cmd} failed with exit code {e.returncode}. STDOUT: {e.stdout} STDERR: {e.stderr}")
        return False
    except Exception as e:
        print(f"[UpdateScript] An unexpected error occurred during service restart: {e}")
        return False


def get_current_model_version(local_model_dir):
    """
    Tries to determine the currently deployed model version.
    """
    version_file_path = os.path.join(local_model_dir, "VERSION")
    if os.path.exists(version_file_path):
        try:
            with open(version_file_path, 'r') as f:
                version = f.read().strip()
            print(f"[UpdateScript] Found local version marker: {version}")
            return version
        except Exception as e:
            print(f"[UpdateScript] ERROR reading local VERSION file: {e}")
            return None
    print(f"[UpdateScript] No local VERSION file found at {version_file_path}.")
    return None
    

def set_current_model_version(local_model_dir, version):
    """Writes the current model version to a file."""
    version_file_path = os.path.join(local_model_dir, "VERSION")
    try:
        with open(version_file_path, 'w') as f:
            f.write(version)
        print(f"[UpdateScript] Updated local model version to {version} in {version_file_path}")
    except Exception as e:
        print(f"[UpdateScript] ERROR writing local VERSION file: {e}")


if __name__ == "__main__":
    print(f"--- Model Update Script Started: {datetime.now()} ---")

    latest_gcs_version = get_latest_model_version()
    if not latest_gcs_version:
        print("[UpdateScript] No model versions found in GCS or error occurred. Exiting.")
        exit(0) # Not an error if no versions, just nothing to do

    current_local_version = get_current_model_version(LOCAL_MODEL_DIR)

    print(f"[UpdateScript] Latest GCS version: {latest_gcs_version}")
    print(f"[UpdateScript] Current local version: {current_local_version}")

    if latest_gcs_version == current_local_version:
        print("[UpdateScript] Local models are already up-to-date. No update needed.")
        exit(0)

    print(f"[UpdateScript] New version '{latest_gcs_version}' available. Initiating download...")

    if not download_model_files(latest_gcs_version, LOCAL_TEMP_DIR):
        print("[UpdateScript] Model download failed. Exiting.")
        exit(1)

    print("[UpdateScript] Download successful. Replacing models...")
    if not replace_models(LOCAL_TEMP_DIR, LOCAL_MODEL_DIR):
        print("[UpdateScript] Model replacement failed. Exiting.")
        exit(1)

    # Update the local version marker
    set_current_model_version(LOCAL_MODEL_DIR, latest_gcs_version)

    print("[UpdateScript] Attempting to restart data_collect.service...")
    if not restart_service(SERVICE_NAME):
        print("[UpdateScript] Service restart failed. Manual intervention might be needed.")
        exit(1)

    print("--- Model Update Script Finished ---")
    exit(0)