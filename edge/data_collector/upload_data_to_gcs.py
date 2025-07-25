from pathlib import Path
import os
from google.cloud import storage
from datetime import datetime
import json

# ---------------------------------------------------------------
# This is called by the cron job automatically every day at 2AM
# to upload the data gotten from streaming from Sensors and CAN
# to google cloud storage
# ---------------------------------------------------------------


# --- Function to load settings from settings.json ---
def load_settings():
    """
    Loads configuration settings from the settings.json file.
    Assumes settings.json is in the 'config' directory one level up from
    where this script is located relative to the project root.
    """
    # Get the directory of the current script 
    BASE_DIR = os.path.abspath(os.path.dirname(__file__))
    CONFIG_DIR = os.path.join(BASE_DIR, "..", "config")
    SETTINGS_FILE_PATH = os.path.join(CONFIG_DIR, "settings.json")
    
    try:
        with open(SETTINGS_FILE_PATH, 'r') as f:
            settings = json.load(f)
        print(f"[UploadScript:Settings] Loaded settings from {SETTINGS_FILE_PATH}")
        return settings
    except FileNotFoundError:
        print(f"[UploadScript:Settings] ERROR: settings.json not found at {SETTINGS_FILE_PATH}. Exiting.")
        exit(1) # Critical error, cannot proceed without settings
    except json.JSONDecodeError:
        print(f"[UploadScript:Settings] ERROR: Could not decode JSON from {SETTINGS_FILE_PATH}. Check file format.")
        exit(1) # Critical error

# Load settings at the very beginning of the script
SETTINGS = load_settings()

# --- Configuration (loaded from SETTINGS) ---
# Construct the absolute path for credentials based on the project structure
# Assuming gcp-service-account.json is at project_root/keys/gcp-service-account.json
KEYS_DIR = os.path.join(BASE_DIR, "..", "keys")

GCP_CREDENTIALS_PATH = os.path.join(
    KEYS_DIR, 
    SETTINGS['cloud_sync']['gcp_credentials_path_relative'] # "keys/gcp-service-account.json"
)

BUCKET_NAME = SETTINGS['cloud_sync']['gcs_raw_data_bucket']
GCS_DEST_FOLDER = SETTINGS['cloud_sync']['gcs_raw_data_prefix']

# Dynamically resolve the path to combined_log.csv using settings
# Example: "data/combined_log.csv" from settings.json
COMBINED_LOG_FILE_PATH = SETTINGS['data_collection']['combined_log_file_path']
CSV_PATH = os.path.join(BASE_DIR, "..", COMBINED_LOG_FILE_PATH)


def upload_to_gcs():
    print(f"[UploadScript] Attempting to upload data to GCS.")
    print(f"[UploadScript] Using credentials from: {GCP_CREDENTIALS_PATH}")
    print(f"[UploadScript] Target bucket: {BUCKET_NAME}, folder: {GCS_DEST_FOLDER}")
    print(f"[UploadScript] Local CSV path: {CSV_PATH}")

    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = GCP_CREDENTIALS_PATH
    try:
        client = storage.Client()
        bucket = client.bucket(BUCKET_NAME)
    except Exception as e:
        print(f"[UploadScript] ERROR: Failed to initialize GCS client: {e}")
        return # Exit if cannot connect to GCS

    # Name the uploaded file with timestamp for GCS
    timestamp = datetime.utcnow().strftime("%Y-%m-%dT%H-%M-%S")
    blob_name = f"{GCS_DEST_FOLDER}log_{timestamp}.csv"
    blob = bucket.blob(blob_name)

    # Check if the CSV file actually exists before attempting to upload/delete
    if not CSV_PATH.exists():
        print(f" Warning: {CSV_PATH} does not exist. No data to upload or delete.")
        return        
    
    # Upload to GCS
    try:    
        blob.upload_from_filename(str(CSV_PATH))
        print(f"Uploaded {CSV_PATH} to gs://{BUCKET_NAME}/{blob_name}")
        
        # Erase the local CSV file after successful upload
        os.remove(str(CSV_PATH))
        print(f"Erased local file: {CSV_PATH}")
        
    except Exception as e:
        print(f" Error uploading {CSV_PATH} to GCS: {e}")
        
        # Erase the local CSV file if upload unsuccessful
        os.remove(str(CSV_PATH))
        print(f" Erased local file: {CSV_PATH}")

if __name__ == "__main__":
    upload_to_gcs()
    print("--- Data Upload Script Finished ---")