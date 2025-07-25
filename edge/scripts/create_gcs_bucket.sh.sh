#!/bin/bash

# First script run
# This script ensures that all Google Cloud Storage buckets required for the
# tractor health monitoring project's ML pipeline and data storage exist.
# It creates the buckets in the specified region if they don't already exist.

# --- CONFIGURATION ---
GCP_REGION="us-central1" 
RAW_DATA_BUCKET_NAME="tractor-health-monitoring-raw-data-bucket"
MODELS_BUCKET_NAME="tractor-health-monitoring-models-bucket"
PIPELINE_ARTIFACTS_BUCKET_NAME="tractor-health-monitoring-kfp-pipeline-artifacts"


# Function to check and create a bucket
check_and_create_bucket() {
    local bucket_name=$1
    local region=$2
    echo "Checking if bucket gs://${bucket_name} exists..."
    if gsutil ls -b "gs://${bucket_name}" > /dev/null 2>&1; then
        echo "Bucket gs://${bucket_name} already exists."
    else
        echo "Creating bucket: gs://${bucket_name} in ${region}..."
        if gsutil mb -l "${region}" "gs://${bucket_name}"; then
            echo "Bucket gs://${bucket_name} created successfully."
        else
            echo "Error: Failed to create bucket gs://${bucket_name}. Exiting."
            exit 1
        fi
    fi
}

echo "--- Starting GCS Bucket Setup ---"

echo "GCP Region: ${GCP_REGION}"

# Create Raw Data Bucket
check_and_create_bucket "$RAW_DATA_BUCKET_NAME" "$GCP_REGION"

# Create Models Bucket
check_and_create_bucket "$MODELS_BUCKET_NAME" "$GCP_REGION"

# Create Pipeline Artifacts Bucket
check_and_create_bucket "$PIPELINE_ARTIFACTS_BUCKET_NAME" "$GCP_REGION"

echo "--- GCS Bucket Setup Complete ---"