# main.py for data_ingestion_component
import os
import argparse
from datetime import datetime
import pandas as pd
from google.cloud import storage, bigquery

def data_ingestion_component_main(
    gcs_bucket_name: str,
    gcs_data_folder: str,
    bigquery_table_id: str,
    output_data_path: str,
):
    """
    Ingests the latest daily raw sensor data from Google Cloud Storage
    and writes it to a BigQuery table.
    The ingested data is also saved as a local CSV artifact.
    """
    print(f"Starting data ingestion from gs://{gcs_bucket_name}/{gcs_data_folder}")
    print(f"Target BigQuery table: {bigquery_table_id}")

    storage_client = storage.Client()
    bucket = storage_client.bucket(gcs_bucket_name)

    blobs_iterator = storage_client.list_blobs(bucket, prefix=gcs_data_folder)

    latest_blob = None
    latest_timestamp = None
    
    # Iterates through blobs to find the most recent log file.
    for blob in blobs_iterator:
        # Checks if the blob name matches the expected log file pattern.
        if blob.name.startswith(f"{gcs_data_folder}log_") and blob.name.endswith(".csv"):
            try:
                # Extracts the timestamp from the filename for comparison.
                timestamp_str = blob.name.replace(f"{gcs_data_folder}log_", "").replace(".csv", "")
                current_blob_timestamp = datetime.strptime(timestamp_str, "%Y-%m-%dT%H-%M-%S")
                
                # Updates latest_blob if a newer file is found.
                if latest_timestamp is None or current_blob_timestamp > latest_timestamp:
                    latest_timestamp = current_blob_timestamp
                    latest_blob = blob
            except ValueError:
                print(f"Skipping malformed filename: {blob.name}")
                continue
    
    # Raises an error if no valid log files are found.
    if latest_blob is None:
        raise FileNotFoundError(f"No valid daily log files found in gs://{gcs_bucket_name}/{gcs_data_folder}")

    print(f"Identified latest daily log file: gs://{gcs_bucket_name}/{latest_blob.name} (timestamp: {latest_timestamp})")

    # Loads the latest CSV data into a pandas DataFrame.
    try:
        df = pd.read_csv(f"gs://{gcs_bucket_name}/{latest_blob.name}")
        print(f"Successfully loaded {len(df)} rows from {latest_blob.name}")
    except Exception as e:
        print(f"Error loading CSV from GCS: {e}")
        raise

    # Saves the ingested DataFrame to a BigQuery table.
    bigquery_client = bigquery.Client()
    job_config = bigquery.LoadJobConfig(
        write_disposition="WRITE_APPEND",  # Appends data to the table if it exists.
        autodetect=True,                   # Automatically detects schema.
    )
    job = bigquery_client.load_table_from_dataframe(
        df, bigquery_table_id, job_config=job_config
    )
    job.result() # Waits for the job to complete.
    print(f"Loaded {job.output_rows} rows into BigQuery table: {bigquery_table_id}")

    # Saves the DataFrame to the local output path provided by KFP.
    df.to_csv(output_data_path, index=False)
    print(f"Ingested data saved to local path: {output_data_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Data Ingestion Component")
    parser.add_argument("--gcs_bucket_name", type=str, required=True, help="Name of the GCS bucket for raw data.")
    parser.add_argument("--gcs_data_folder", type=str, required=True, help="Folder within GCS bucket containing sensor data.")
    parser.add_argument("--bigquery_table_id", type=str, required=True, help="Full ID of the BigQuery table for raw data.")
    parser.add_argument("--output_data_path", type=str, required=True, help="Local path for the output CSV artifact.")
    
    args = parser.parse_args()
    data_ingestion_component_main(
        gcs_bucket_name=args.gcs_bucket_name,
        gcs_data_folder=args.gcs_data_folder,
        bigquery_table_id=args.bigquery_table_id,
        output_data_path=args.output_data_path
    )