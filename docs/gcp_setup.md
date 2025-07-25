# Google Cloud Platform (GCP) Setup Guide
This document provides detailed, step-by-step instructions for setting up the necessary Google Cloud Platform (GCP) resources and configurations for the Tractor Health Monitoring MLOps pipeline.

## 1. Prerequisites
Before starting, ensure you have:
- A Google Cloud account with billing enabled.
- The gcloud CLI installed and authenticated.
  - gcloud init
  - gcloud auth login
  - gcloud config set project [YOUR_GCP_PROJECT_ID]
  - gcloud config set compute/region [YOUR_GCP_REGION] (e.g., us-central1)
- gsutil (comes with gcloud SDK)
- docker installed on your local machine where you'll build images.
- yq installed (for parsing YAML in shell scripts if you choose to use it in setup scripts).
  - sudo apt-get install yq (Debian/Ubuntu) or brew install yq (macOS)

## 2. Create a GCP Project
If you don't have an existing project, create a new one:
``` bash
gcloud projects create [YOUR_GCP_PROJECT_ID] --name="Tractor Health Monitoring"
gcloud config set project [YOUR_GCP_PROJECT_ID]
```
## 3. Enable Required GCP APIs
Enable all necessary APIs for the project. This can take a few minutes.
``` bash
gcloud services enable \
    compute.googleapis.com \
    storage.googleapis.com \
    bigquery.googleapis.com \
    artifactregistry.googleapis.com \
    aiplatform.googleapis.com \
    cloudbuild.googleapis.com \
    iam.googleapis.com
```
## 4. Create GCP Service Accounts
You need two primary service accounts: one for the edge device and one for the Vertex AI Pipelines.

### 4.1. Edge Device Service Account
This service account will be used by your edge device to upload data to GCS. It needs minimal permissions.

**1. Create Service Account:**
```bash 
gcloud iam service-accounts create edge-data-uploader \
    --display-name "Edge Data Uploader Service Account" \
    --project=[YOUR_GCP_PROJECT_ID]
```
**2. Grant Permissions:** Grant Storage Object Creator role to allow it to write new objects to your raw data bucket.
``` bash
gcloud projects add-iam-policy-binding [YOUR_GCP_PROJECT_ID] \
    --member="serviceAccount:edge-data-uploader@[YOUR_GCP_PROJECT_ID].iam.gserviceaccount.com" \
    --role="roles/storage.objectCreator"
```
**3. Create and Download JSON Key:** This key will be placed on your edge device.
``` bash
gcloud iam service-accounts keys create edge-gcp-key.json \
    --iam-account="edge-data-uploader@[YOUR_GCP_PROJECT_ID].iam.gserviceaccount.com" \
    --project=[YOUR_GCP_PROJECT_ID]
```
**Important:** Rename edge-gcp-key.json to gcp-service-account.json and place it in edge/keys/ within your project structure on the edge device. Keep this file secure.

### 4.2. Vertex AI Pipelines Service Account
This service account needs broad permissions to orchestrate the ML pipeline, including reading/writing to GCS, BigQuery, managing Vertex AI resources, and pulling Docker images from Artifact Registry.

**1. Create Service Account:**

gcloud iam service-accounts create vertex-ai-pipeline-runner \
    --display-name "Vertex AI Pipeline Runner Service Account" \
    --project=[YOUR_GCP_PROJECT_ID]

**2. Grant Permissions**:
``` bash
# Grant roles to the pipeline runner service account
gcloud projects add-iam-policy-binding [YOUR_GCP_PROJECT_ID] \
    --member="serviceAccount:vertex-ai-pipeline-runner@[YOUR_GCP_PROJECT_ID].iam.gserviceaccount.com" \
    --role="roles/aiplatform.user" # Vertex AI User
gcloud projects add-iam-policy-binding [YOUR_GCP_PROJECT_ID] \
    --member="serviceAccount:vertex-ai-pipeline-runner@[YOUR_GCP_PROJECT_ID].iam.gserviceaccount.com" \
    --role="roles/storage.admin" # Cloud Storage Admin (for all GCS operations)
gcloud projects add-iam-policy-binding [YOUR_GCP_PROJECT_ID] \
    --member="serviceAccount:vertex-ai-pipeline-runner@[YOUR_GCP_PROJECT_ID].iam.gserviceaccount.com" \
    --role="roles/bigquery.dataEditor" # BigQuery Data Editor (for writing to BQ)
gcloud projects add-iam-policy-binding [YOUR_GCP_PROJECT_ID] \
    --member="serviceAccount:vertex-ai-pipeline-runner@[YOUR_GCP_PROJECT_ID].iam.gserviceaccount.com" \
    --role="roles/bigquery.jobUser" # BigQuery Job User (for running BQ jobs)
gcloud projects add-iam-policy-binding [YOUR_GCP_PROJECT_ID] \
    --member="serviceAccount:vertex-ai-pipeline-runner@[YOUR_GCP_PROJECT_ID].iam.gserviceaccount.com" \
    --role="roles/artifactregistry.reader" # Artifact Registry Reader (for pulling images)
# If using Cloud Build for image builds, Cloud Build SA also needs permissions.
# The default Cloud Build service account usually has enough permissions, but verify.
```
## 5. Create GCS Buckets
Your project uses three dedicated GCS buckets. You can create them using the provided shell script.

**1. Navigate to the scripts directory:**
``` bash
cd edge/scripts/
```
**2. Run the bucket creation script:**
``` bash
./create_gcs_bucket.sh
```
This script will:

- Create tractor-health-monitoring-raw-data-bucket

- Create tractor-health-monitoring-models-bucket

- Create tractor-health-monitoring-kfp-pipeline-artifacts

- All buckets will be created in the us-central1 region (or the region you hardcoded in the script).

**3. Verify Buckets (Optional):**
``` bash
gsutil ls
```
You should see all three buckets listed.

## 6. Setup Artifact Registry Repository
Create a Docker repository in Artifact Registry to store your custom component images.
``` bash
gcloud artifacts repositories create tractor-mlops-containers \
    --repository-format=docker \
    --location=[YOUR_GCP_REGION] \
    --description="Docker repository for Tractor Health MLOps pipeline components" \
    --project=[YOUR_GCP_PROJECT_ID]
```
## 7. Build and Push Docker Images for Cloud Components
For each component (data_ingestion_container, data_preprocessing_container, autoencoder_training_container, classifier_training_container, model_deployment_container), you need to build its Docker image and push it to Artifact Registry.

**Important:** Ensure you are in your project root directory when running these docker build commands, as the Dockerfiles use paths relative to the project root.
``` bash
First, configure Docker to authenticate with Artifact Registry:
```
gcloud auth configure-docker [YOUR_GCP_REGION]-docker.pkg.dev

Then, build and push each image:

- Data Ingestion Component:
``` bash
docker build -t [YOUR_GCP_REGION]-docker.pkg.dev/[YOUR_GCP_PROJECT_ID]/tractor-mlops-containers/data_ingestion:latest -f cloud/end_to_end_automated_pipeline/components/data_ingestion_container/Dockerfile .
docker push [YOUR_GCP_REGION]-docker.pkg.dev/[YOUR_GCP_PROJECT_ID]/tractor-mlops-containers/data_ingestion:latest
```
- Data Preprocessing Component:
``` bash
docker build -t [YOUR_GCP_REGION]-docker.pkg.dev/[YOUR_GCP_PROJECT_ID]/tractor-mlops-containers/data_preprocessing:latest -f cloud/end_to_end_automated_pipeline/components/data_preprocessing_container/Dockerfile .
docker push [YOUR_GCP_REGION]-docker.pkg.dev/[YOUR_GCP_PROJECT_ID]/tractor-mlops-containers/data_preprocessing:latest
```
- Autoencoder Training Component:
``` bash
docker build -t [YOUR_GCP_REGION]-docker.pkg.dev/[YOUR_GCP_PROJECT_ID]/tractor-mlops-containers/autoencoder_training:latest -f cloud/end_to_end_automated_pipeline/components/autoencoder_training_container/Dockerfile .
docker push [YOUR_GCP_REGION]-docker.pkg.dev/[YOUR_GCP_PROJECT_ID]/tractor-mlops-containers/autoencoder_training:latest
```
- Classifier Training Component:
``` bash
docker build -t [YOUR_GCP_REGION]-docker.pkg.dev/[YOUR_GCP_PROJECT_ID]/tractor-mlops-containers/classifier_training:latest -f cloud/end_to_end_automated_pipeline/components/classifier_training_container/Dockerfile .
docker push [YOUR_GCP_REGION]-docker.pkg.dev/[YOUR_GCP_PROJECT_ID]/tractor-mlops-containers/classifier_training:latest
```
- Model Deployment Component:
``` bash
docker build -t [YOUR_GCP_REGION]-docker.pkg.dev/[YOUR_GCP_PROJECT_ID]/tractor-mlops-containers/model_deployment:latest -f cloud/end_to_end_automated_pipeline/components/model_deployment_container/Dockerfile .
docker push [YOUR_GCP_REGION]-docker.pkg.dev/[YOUR_GCP_PROJECT_ID]/tractor-mlops-containers/model_deployment:latest
```
## 8. Configure Cloud Settings (cloud_settings.yaml)
Update the cloud/end_to_end_automated_pipeline/config/cloud_settings.yaml file with your specific GCP project ID, region, and the exact names of the buckets and Artifact Registry repository you've created.
``` bash
# cloud/end_to_end_automated_pipeline/config/cloud_settings.yaml

gcp_project_id: "[YOUR_GCP_PROJECT_ID]"
gcp_region: "[YOUR_GCP_REGION]"

gcs:
  raw_data_bucket: "tractor-health-monitoring-raw-data-bucket"
  raw_data_prefix: "raw/"
  models_bucket: "tractor-health-monitoring-models-bucket"
  models_latest_prefix: "models/latest/"
  models_versioned_prefix: "models/versions/"
  pipeline_artifacts_bucket: "tractor-health-monitoring-kfp-pipeline-artifacts"

bigquery:
  dataset_id: "tractor_health_data"
  table_id: "sensor_can_logs"

vertex_ai_pipelines:
  pipeline_name: "tractor-health-mlops-pipeline"
  scheduler_cron: "0 0 * * 0" # Example: every Sunday at midnight UTC
  endpoint_display_name: "tractor-health-monitoring-endpoint" # Ensure this matches your deployment
  compiled_pipeline_output_file: "end_to_end_automated_pipeline.json" # Relative to end_to_end_pipeline/

artifact_registry:
  repository_name: "tractor-mlops-containers"

# ... (rest of your model_training_params, data_preparation_params)
```
## 9. Compile Kubeflow Pipeline
This step generates the pipeline's JSON blueprint.

**1. Navigate to the pipeline definition directory:**
``` bash
cd cloud/end_to_end_automated_pipeline/end_to_end_pipeline/
```
**2. Run the compilation script:**
``` bash
python end_to_end_automated_pipeline.py
```
This will create end_to_end_automated_pipeline.json in the same directory.

## 10. Run Vertex AI Pipeline
You can trigger the pipeline manually or set up a recurring schedule.

### 10.1. Manual Run
**1. Ensure you are in the pipeline definition directory:**
``` bash
cd cloud/end_to_end_automated_pipeline/end_to_end_pipeline/
```
**2. Run the pipeline execution script:**
``` bash
python run_pipeline.py
```
This script will initialize the Vertex AI SDK and submit the pipeline job. The sync=True argument will make your terminal wait until the pipeline completes.

### 10.2. Set Up Recurring Schedule (Recommended for Production)
For automated daily runs, it's best to set up a recurring schedule directly in the Vertex AI console:

**1. Upload the compiled pipeline JSON to GCS:**

- The run_pipeline.py script automatically references the local JSON. However, for a scheduled run, the JSON needs to be in GCS.

- You can manually upload it:
``` bash
gsutil cp cloud/end_to_end_automated_pipeline/end_to_end_pipeline/end_to_end_automated_pipeline.json gs://tractor-health-monitoring-kfp-pipeline-artifacts/pipelines/end_to_end_automated_pipeline.json
```
(Adjust GCS path as desired, but ensure it's accessible by the pipeline runner SA.)

**2.Navigate to Vertex AI Pipelines in GCP Console:**

- Go to Vertex AI -> Pipelines.

**3. Create a new Scheduled Run:**

- Click "Create Run" or "Create Scheduled Run".

- Select "Recurring run".

- Provide a name (e.g., "Tractor Health Daily Pipeline").

- Specify the GCS path to your uploaded end_to_end_automated_pipeline.json.

- Configure the recurrence (e.g., daily at 03:00 AM UTC, matching your scheduler_cron if defined).

- Select the vertex-ai-pipeline-runner service account.

- Set the pipeline root to gs://tractor-health-monitoring-kfp-pipeline-artifacts/.

- Create the schedule.

## 11. Verify Vertex AI Endpoint
After a successful pipeline run, verify that your models are deployed:

1. Go to Vertex AI -> Endpoints.

2. Look for the endpoint named tractor-health-monitoring-endpoint (or whatever you configured).

3. Ensure both your autoencoder and classifier models are listed as deployed to this endpoint.

