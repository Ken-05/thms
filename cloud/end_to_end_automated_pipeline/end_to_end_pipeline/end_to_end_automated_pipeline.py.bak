# end_to_end_automated_pipeline.py
import os
from kfp import dsl
from kfp.v2 import compiler
from kfp.v2.dsl import OutputPath, Model # Imports OutputPath for artifact handling and Model for model artifacts
import yaml

# This script defines the entire structure and logic of the machine 
# learning pipeline. It uses the Kubeflow Pipelines (KFP) SDK's 
# Domain Specific Language (DSL) to orchestrate the sequence of operations.

# This code doesnt run the pipeline directly on Google Cloud. 
# Instead, it compiles this Python definition into a machine-readable 
# format which is a JSON file in this case which is the static 
# blueprint of the pipeline


# --- Function to load cloud_settings.yaml ---
def load_cloud_settings():
    """
    Loads configuration settings from the cloud_settings.yaml file.
    Assumes cloud_settings.yaml is in the 'config' directory at the project root.
    """
    
    BASE_DIR = os.path.abspath(os.path.dirname(__file__))
    CONFIG_DIR = os.path.join(BASE_DIR, "..", "config") # Go up one level to project_root, then into 'config'
    CLOUD_SETTINGS_FILE_PATH = os.path.join(CONFIG_DIR, "cloud_settings.yaml")

    try:
        with open(CLOUD_SETTINGS_FILE_PATH, 'r') as f:
            settings = yaml.safe_load(f)
        print(f"[Pipeline:Settings] Loaded cloud settings from {CLOUD_SETTINGS_FILE_PATH}")
        return settings
    except FileNotFoundError:
        print(f"[Pipeline:Settings] ERROR: cloud_settings.yaml not found at {CLOUD_SETTINGS_FILE_PATH}. Exiting.")
        exit(1) # Critical error, cannot proceed without cloud settings
    except yaml.YAMLError as e:
        print(f"[Pipeline:Settings] ERROR: Could not decode YAML from {CLOUD_SETTINGS_FILE_PATH}. Check file format. Error: {e}")
        exit(1) # Critical error

# Load cloud settings once when the script is run (for compilation)
CLOUD_SETTINGS = load_cloud_settings()

# Define common constants for the pipeline, now sourced from CLOUD_SETTINGS.
GCP_REGION = CLOUD_SETTINGS['gcp_region']
PROJECT_ID = CLOUD_SETTINGS['gcp_project_id']
ARTIFACT_REGISTRY_REPO = CLOUD_SETTINGS['artifact_registry']['repository_name']

# BigQuery details
BIGQUERY_DATASET_ID = CLOUD_SETTINGS['bigquery']['dataset_id']
BIGQUERY_TABLE_ID = CLOUD_SETTINGS['bigquery']['table_id']
BIGQUERY_FULL_TABLE_ID = f"{PROJECT_ID}.{BIGQUERY_DATASET_ID}.{BIGQUERY_TABLE_ID}"

# GCS bucket names and prefixes
GCS_RAW_DATA_BUCKET = CLOUD_SETTINGS['gcs']['raw_data_bucket']
GCS_RAW_DATA_PREFIX = CLOUD_SETTINGS['gcs']['raw_data_prefix'] # The folder where edge devices upload
GCS_PIPELINE_ROOT_BUCKET = CLOUD_SETTINGS['gcs']['pipeline_artifacts_bucket']

# Endpoint for model deployment
ENDPOINT_DISPLAY_NAME = CLOUD_SETTINGS['vertex_ai_pipelines']['endpoint_display_name']

# Pipeline creation
PIPELINE_NAME = CLOUD_SETTINGS['vertex_ai_pipelines']['pipeline_name']
PIPELINE_DESCRIPTION = CLOUD_SETTINGS['vertex_ai_pipelines']['pipeline_description']
COMPILED_PIPELINE_OUTPUT_FILE = CLOUD_SETTINGS['vertex_ai_pipelines']['compiled_pipeline_output_file']

# Construct the full URI prefix for Docker images in Artifact Registry.
BASE_IMAGE_URI_PREFIX = f"{GCP_REGION}-docker.pkg.dev/{PROJECT_ID}/{ARTIFACT_REGISTRY_REPO}/"

# Define Kubeflow Pipeline components using dsl.ContainerSpec
# Each component specifies a custom Docker image and command-line arguments for inputs/outputs.

# Data Ingestion Component
data_ingestion_component = dsl.ContainerSpec(
    image=f"{BASE_IMAGE_URI_PREFIX}data_ingestion:latest",
    command=["python", "main.py"],
    args=[
        "--gcs_bucket_name", GCS_RAW_DATA_BUCKET,
        "--gcs_data_folder", GCS_RAW_DATA_PREFIX,
        "--bigquery_table_id", BIGQUERY_FULL_TABLE_ID,
        "--output_data_path", OutputPath("csv"), # KFP provides a local path for the output CSV.
    ]
)

# Data Preprocessing Component
data_preprocessing_component = dsl.ContainerSpec(
    image=f"{BASE_IMAGE_URI_PREFIX}data_preprocessing:latest",
    command=["python", "main.py"],
    args=[
        "--input", InputPath("csv"), 
        "--train_output", OutputPath("csv"), 
        "--val_output", OutputPath("csv"), 
        "--test_output", OutputPath("csv"), 
        "--scaler_output", OutputPath("joblib"),
    ]
)

# Autoencoder Model Training Component
autoencoder_model_training_component = dsl.ContainerSpec(
    image=f"{BASE_IMAGE_URI_PREFIX}autoencoder_training:latest",
    command=["python", "main.py"],
    args=[
        "--train_path", InputPath("csv"),
        "--val_path", InputPath("csv"),
        "--h5_output_path", OutputPath(Model), 
        "--tflite_output_path", OutputPath("tflite"),
        "--scaler_output_path", InputPath("joblib"),
    ]
)

# Classifier Model Training Component
classifier_model_training_component = dsl.ContainerSpec(
    image=f"{BASE_IMAGE_URI_PREFIX}classifier_training:latest",
    command=["python", "main.py"],
    args=[
        "--train_path", InputPath("csv"),
        "--val_path", InputPath("csv"), 
        "--test_path", InputPath("csv"), 
        "--h5_output_path", OutputPath(Model), 
        "--tflite_output_path", OutputPath("tflite"), 
    ]
)

# Model Deployment Component (for Vertex AI Endpoint)
model_deployment_component = dsl.ContainerSpec(
    image=f"{BASE_IMAGE_URI_PREFIX}model_deployment:latest",
    command=["python", "main.py"],
    args=[
        "--project", PROJECT_ID,
        "--region", GCP_REGION,
        "--autoencoder_model_uri", InputPath(Model), # KFP provides the GCS URI of the Autoencoder Model artifact.
        "--classifier_model_uri", InputPath(Model), # KFP provides the GCS URI of the Classifier Model artifact.
        "--endpoint_display_name", ENDPOINT_DISPLAY_NAME,
    ]
)

@dsl.pipeline(
    name=PIPELINE_NAME,
    description=PIPELINE_DESCRIPTION,
    pipeline_root=f"gs://{GCS_PIPELINE_ROOT_BUCKET}/" # Specifies the GCS bucket for pipeline artifacts.
)
def end_to_end_automated_pipeline():
    
    # 1. Data Ingestion: Ingests the latest daily data from GCS and loads it into BigQuery.
    # The component outputs a CSV artifact for the next step.
    ingest_data_task = dsl.ContainerOp(
        name="data-ingestion-container-op",
        image=data_ingestion_component.image,
        command=data_ingestion_component.command,
        arguments=[
            "--gcs_bucket_name", GCS_RAW_DATA_BUCKET,
            "--gcs_data_folder", GCS_RAW_DATA_PREFIX,
            "--bigquery_table_id", BIGQUERY_FULL_TABLE_ID,
            "--output_data_path", OutputPath("csv"), # This is the output artifact for the next step
        ]
    ).set_display_name("Data Ingestion (GCS to BQ)")

    

    # 2. Data Preprocessing + Feature Engineering: Processes the raw data and splits it.
    # It takes the ingested CSV as input and outputs processed training and testing CSVs.
    preprocess_data_task = dsl.ContainerOp(
        name="data-preprocessing",
        image=data_preprocessing_component.image,
        command=data_preprocessing_component.command,
        arguments=[
            "--input", ingest_data_task.outputs["output_data_path"], # Uses the output path from data ingestion.
            "--train_output", OutputPath("csv"), # Defines output path for training data.
            "--val_output", OutputPath("csv"),   # Defines output path for validation data.
            "--test_output", OutputPath("csv"),   # Defines output path for testing data.
            "--scaler_output", OutputPath("joblib"), # Defines output path for the scaler.
        ]
    ).set_display_name("Data Preprocessing")
    
    # 3. Train Autoencoder: Trains the autoencoder model for anomaly detection.
    # It takes the preprocessed training data and outputs a Keras H5 model and a TFLite model.
    train_autoencoder_task = dsl.ContainerOp(
        name="train-autoencoder",
        image=autoencoder_model_training_component.image,
        command=autoencoder_model_training_component.command,
        arguments=[
            "--train_path", preprocess_data_task.outputs["train_output"], # Uses training data from preprocessing.
            "--val_path", preprocess_data_task.outputs["val_output"], # Use validation data from preprocessing.
            "--h5_output_path", OutputPath(Model), # Defines output path for the H5 model artifact.
            "--tflite_output_path", OutputPath("tflite"), # Defines output path for the TFLite model artifact.
            "--scaler_output_path", preprocess_data_task.outputs["scaler_output"], # Pass the scaler output path
        ]
    ).set_display_name("Train Autoencoder")

    # 4. Train Classifier: Trains the classifier model for fault classification.
    # It takes preprocessed training and testing data and outputs a Keras H5 model and a TFLite model.
    train_classifier_task = dsl.ContainerOp(
        name="train-classifier",
        image=classifier_model_training_component.image,
        command=classifier_model_training_component.command,
        arguments=[
            "--train_path", preprocess_data_task.outputs["train_output"], # Uses training data from preprocessing.
            "--val_path", preprocess_data_task.outputs["val_output"], # Uses validation data from preprocessing.
            "--test_path", preprocess_data_task.outputs["test_output"], # Uses testing data from preprocessing.
            "--h5_output_path", OutputPath(Model), # Defines output path for the H5 model artifact.
            "--tflite_output_path", OutputPath("tflite"), # Defines output path for the TFLite model artifact.
        ]
    ).set_display_name("Train Classifier")

    # 5. Model Deployment: Deploys the trained models to a Vertex AI Endpoint.
    # It uses the H5 model artifacts from the training steps.
    deploy_models_task = dsl.ContainerOp(
        name="deploy-models",
        image=model_deployment_component.image,
        command=model_deployment_component.command,
        arguments=[
            "--project", PROJECT_ID,
            "--region", GCP_REGION,
            "--autoencoder_model_uri", train_autoencoder_task.outputs["h5_output_path"], # Passes the GCS URI of the Autoencoder H5 model.
            "--classifier_model_uri", train_classifier_task.outputs["h5_output_path"], # Passes the GCS URI of the Classifier H5 model.
            "--endpoint_display_name", ENDPOINT_DISPLAY_NAME,
        ]
    ).after(train_autoencoder_task, train_classifier_task).set_display_name("Deploy Models") # Ensures deployment runs after both models are trained.
    
# Compile the pipeline to a JSON file.
if __name__ == "__main__":
    print("Compiling pipeline definition to JSON...")
    BASE_DIR = os.path.abspath(os.path.dirname(__file__))
    OUTPUT_FILE_PATH = os.path.join(BASE_DIR, COMPILED_PIPELINE_OUTPUT_FILE)

    compiler.Compiler().compile(
        pipeline_func=end_to_end_automated_pipeline,
        package_path=OUTPUT_FILE_PATH
    )
    print(f"Pipeline compilation complete: {OUTPUT_FILE_PATH}")