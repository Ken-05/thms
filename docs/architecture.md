# Project Architecture: Tractor Health Monitoring

## 1. Introduction

This document provides a detailed overview of the architecture for the Tractor Health Monitoring MLOps pipeline. It describes the various components, their responsibilities, and how data and models flow through the system, from the edge device to the Google Cloud Platform (GCP) and back.

The architecture is designed for scalability, reliability, and continuous improvement, leveraging a hybrid cloud-edge approach to deliver real-time insights and enable predictive maintenance.

---

## 2. Overall System Diagram (Conceptual Flow)


```

+---------------------+
|   Tractor Sensors   |
| (Arduino, CAN Bus)  |
+---------------------+
           |
           v
+---------------------------------------------------------------------------------+
|                                 Edge Layer                                      |
|                                                                                 |
|  +-------------------+   +-------------------+   +-----------------------------+|
|  | Data Collection   |-->| Local Data Storage|-->| Data Upload Script          ||
|  | (data_collect.py) |   | (combined_log.csv)|   | (upload_data_to_gcs.py)     ||
|  +-------------------+   +-------------------+   +-----------------------------+|
|           |                                                                     |
|           v                                                                     |
|  +-----------------------+   +----------------------+   +----------------------+|
|  | Edge Preprocessing    |-->| Local Inference      |-->| Local GUI Display    ||
|  | (edge_preprocess.py)  |   | (edge_inference.py)  |   | (tractor_display_gui.py)||
|  +-----------------------+   +----------------------+   +----------------------+|
|           ^                                                                     |
|           | (Model Sync)                                                        |
|  +-----------------------+                                                      |
|  | Model Update Script   |                                                      |
|  | (update_models.py)    |                                                      |
|  +-----------------------+                                                      |
+---------------------------------------------------------------------------------+
           |
           | (Raw Data Upload: gs://<raw_data_bucket>/raw/)
           v
+---------------------------------------------------------------------------------+
|                                 Cloud Layer                                     |
|                                                                                 |
|  +-------------------+                                                          |
|  | GCS Raw Data      |                                                          |
|  | (raw_data_bucket) |                                                          |
|  +-------------------+                                                          |
           |
           | (Trigger/Input)
           v
+---------------------------------------------------------------------------------+
|                     Vertex AI Pipelines (Kubeflow Pipelines)                    |
|                                                                                 |
|  +-------------------+   +-------------------+   +---------------------------+  |
|  | 1. Data Ingestion |-->| 2. Data           |-->| 3. Autoencoder Training   |  |
|  | (GCS to BigQuery) |   | Preprocessing     |   |                           |  |
|  +-------------------+   +-------------------+   +---------------------------+  |
|                                     |                                           |
|                                     v                                           |
|  +-------------------+   +-----------------------------+                        |
|  | 4. Classifier     |-->| 5. Model Deployment         |                        |
|  | Training          |   | (Vertex AI Endpoint)        |                        |
|  +-------------------+   +-----------------------------+                        |
+---------------------------------------------------------------------------------+
           |
           | (Deployed Models)
           v
+-------------------+      +-------------------+
| Vertex AI Endpoint|----->| GCS Models Bucket |
| (Deployed Models) |      | (models_bucket)   |
+-------------------+      +-------------------+

```


---


## 3. Edge Layer Components  
The edge layer is responsible for direct interaction with the tractor, local data processing, real-time inference, and communication with the cloud.

### 3.1. Data Collection (edge/data_collector/data_collect.py)  
**Purpose**: Continuously collects sensor readings (via Arduino serial) and CAN bus messages.  
**Input**: Raw data streams from Arduino (e.g., via /dev/ttyACM0) and CAN bus (e.g., can0).  
**Output**: Appends combined raw data to a local CSV file (edge/data/combined_log.csv).  
**Technology**: Python, pyserial, python-can.  
**Operation**: Designed to run as a persistent service (e.g., systemd service).  

### 3.2. Local Data Storage (edge/data/combined_log.csv)  
**Purpose**: Temporary storage for raw sensor and CAN data collected on the edge device before cloud synchronization.  
**Format**: CSV.  
**Lifecycle**: Cleared after successful upload to GCS.  

### 3.3. Data Upload Script (edge/data_collector/upload_data_to_gcs.py)  
**Purpose**: Uploads the accumulated combined_log.csv from the edge device to a designated GCS bucket.  
**Input**: edge/data/combined_log.csv.  
**Output**: New CSV blob in gs://<raw_data_bucket>/raw/ with a timestamped filename. The local combined_log.csv is then deleted.  
**Technology**: Python, google-cloud-storage.  
**Operation**: Executed periodically via a cron job (e.g., daily).  

### 3.4. Edge Preprocessing (edge/model_inference/edge_preprocess.py)  
**Purpose**: Prepares raw sensor data for local inference by applying the same feature engineering and scaling transformations as the cloud preprocessing pipeline.  
**Input**: Raw sensor data (likely passed directly from data_collect.py or read from a temporary local file).  
**Output**: Transformed feature vector suitable for TFLite model input.  
**Technology**: Python, pandas, numpy, scikit-learn (for scaler).  
**Dependency**: Requires the scaler.joblib file downloaded from the cloud.  

### 3.5. Local Inference (edge/model_inference/edge_inference.py)  
**Purpose**: Performs real-time anomaly detection and fault classification on preprocessed data using lightweight TFLite models.  
**Input**: Preprocessed feature vector from edge_preprocess.py.  
**Output**: Anomaly scores, classification predictions, and overall health status, saved to edge/gui/latest_status.json.  
**Technology**: Python, TensorFlow Lite Interpreter.  
**Dependency**: Requires autoencoder_tflite_model.tflite and classifier_tflite_model.tflite downloaded from the cloud.  
**Operation**: Designed to run as a persistent service, processing data as it becomes available.  

### 3.6. Local GUI Display (edge/gui/tractor_display_gui.py)  
**Purpose**: Provides a local graphical interface on the edge device to visualize the tractor's real-time health status and sensor readings.  
**Input**: edge/gui/latest_status.json (updated by edge_inference.py).  
**Output**: Visual display.  
Technology: Python, Tkinter or similar GUI library.  

### 3.7. Model Update Script (edge/model_inference/update_models.py)  
**Purpose**: Periodically checks for and downloads the latest trained ML models (TFLite models and scaler) from GCS to the edge device.  
**Input**: GCS Models Bucket (gs://<models_bucket>/models/versions/ and gs://<models_bucket>/models/latest/).  
**Output**: Updates edge/models/current_models/ with the latest autoencoder_tflite_model.tflite, classifier_tflite_model.tflite, scaler.joblib, and a VERSION file.  
**Technology**: Python, google-cloud-storage.  
**Operation**: Executed periodically via a cron job (e.g., daily).  

## 4. Cloud Layer Components  
The cloud layer handles scalable data storage, robust ML pipeline orchestration, and model serving.  

### 4.1. Google Cloud Storage (GCS) Buckets  
- **Raw Data Bucket (gs://<raw_data_bucket>)**: Stores raw sensor and CAN data CSVs uploaded from edge devices. Prefix `raw/` is used for organization.  
- **Models Bucket (gs://<models_bucket>)**: Stores versioned TFLite models, H5 models, scalers, and anomaly thresholds from the training pipeline. Uses prefixes like `models/versions/<YYYY-MM-DD>/` for versioning and `models/latest/` for the current active models.  
- **Pipeline Artifacts Bucket (gs://<pipeline_artifacts_bucket>)**: Used by Kubeflow Pipelines on Vertex AI to store intermediate artifacts, logs, and temporary files generated during pipeline runs.  

### 4.2. BigQuery (<project_id>.tractor_health_data.sensor_can_logs)  
**Purpose**: A serverless, highly scalable data warehouse for storing and querying the ingested raw sensor and CAN bus data.  
**Schema**: Auto-detected during ingestion, but designed to capture all raw sensor fields and CAN log details.  

### 4.3. Vertex AI Pipelines (Kubeflow Pipelines)  
The core of the cloud ML workflow, orchestrated by Vertex AI Pipelines. The pipeline is defined in `cloud/end_to_end_automated_pipeline/end_to_end_pipeline/end_to_end_automated_pipeline.py` and compiled into `end_to_end_automated_pipeline.json`.  

#### 4.3.1. Data Ingestion Component (data_ingestion_container)  
**Input**: Raw CSVs from GCS Raw Data Bucket (`gs://<raw_data_bucket>/raw/`).  
**Output**:  
- Loads data into BigQuery table `<project_id>.tractor_health_data.sensor_can_logs`.  
- Outputs a local CSV artifact (passed to the next pipeline step).
  
**Technology**: Python, google-cloud-storage, google-cloud-bigquery.  

#### 4.3.2. Data Preprocessing Component (data_preprocessing_container)  
**Input**: Raw data CSV artifact from Data Ingestion.  
**Output**:  
- Processed training data CSV artifact.  
- Processed validation data CSV artifact.  
- Processed testing data CSV artifact.  
- Fitted MinMaxScaler object (serialized, e.g., using joblib).
   
**Technology**: Python, pandas, numpy, scikit-learn, tensorflow (for GCS file access).  
**Core Logic**: `cloud/end_to_end_automated_pipeline/core_utilities/preprocess.py`.  

#### 4.3.3. Autoencoder Training Component (autoencoder_training_container)  
**Input**: Processed training and validation data CSVs from Data Preprocessing, and the fitted scaler.  
**Output**:  
- Trained Keras H5 autoencoder model artifact.  
- Converted TFLite autoencoder model artifact.  
- Anomaly threshold value (derived from training).
  
**Technology**: Python, TensorFlow/Keras, keras-tuner, joblib, google-cloud-storage.  
**Core Logic**: `cloud/end_to_end_automated_pipeline/training_pipeline/train_autoencoder.py`.  

#### 4.3.4. Classifier Training Component (classifier_training_container)  
**Input**: Processed training, validation, and testing data CSVs from Data Preprocessing.  
**Output**:  
- Trained Keras H5 classifier model artifact.  
- Converted TFLite classifier model artifact.
  
**Technology**: Python, TensorFlow/Keras, keras-tuner, scikit-learn (for LabelEncoder), shap, google-cloud-storage.  
**Core Logic**: `cloud/end_to_end_automated_pipeline/training_pipeline/train_classifier.py`.  

#### 4.3.5. Model Deployment Component (model_deployment_container)  
**Input**: Keras H5 autoencoder model URI and Keras H5 classifier model URI (from their respective training tasks).  
**Output**: Deploys both models to a shared Vertex AI Endpoint.  
**Technology**: Python, google-cloud-aiplatform.  

### 4.4. Vertex AI Endpoints  
**Purpose**: Managed service for deploying and serving trained ML models for online predictions.  
**Models**: Hosts both the Autoencoder and Classifier models, allowing them to be queried via REST API.  

## 5. Data Flow  
- **Edge Data Collection**: `data_collect.py` gathers sensor and CAN data, writing it to `combined_log.csv`.  
- **Edge Data Upload**: `upload_data_to_gcs.py` (cron job) uploads `combined_log.csv` to `gs://<raw_data_bucket>/raw/`.  
- **Cloud Data Ingestion**: The Data Ingestion pipeline component reads the latest raw data from GCS and loads it into BigQuery. It also passes the data as a CSV artifact to the next step.  
- **Cloud Data Preprocessing**: The Preprocessing component performs feature engineering and scaling, producing training/validation/testing CSVs and the `scaler.joblib`.  
- **Cloud Model Training**:  
  - Autoencoder component trains using healthy data and produces H5/TFLite models and an anomaly threshold.  
  - Classifier component trains on labeled data and produces H5/TFLite models.  
- **Cloud Model Deployment**: Models are deployed to Vertex AI Endpoints.  
- **Edge Model Sync**: `update_models.py` downloads the latest models and scaler from GCS.  
- **Edge Inference**: `edge_inference.py` performs inference and updates `latest_status.json`.  
- **Edge GUI Display**: `tractor_display_gui.py` visualizes data from `latest_status.json`.  

## 6. Technology Stack  
**Edge:** Python, pyserial, python-can, pandas, numpy, scikit-learn, TensorFlow Lite, Tkinter, systemd, cron.  

**Cloud:**  
- **Data Storage:** Google Cloud Storage, BigQuery  
- **ML Platform:** Vertex AI (Pipelines, Model Registry, Endpoints)  
- **Containerization:** Docker, Artifact Registry  
- **Frameworks:** TensorFlow/Keras, scikit-learn, Keras Tuner, SHAP  
- **Orchestration:** Kubeflow Pipelines (KFP SDK)  
- **Language:** Python  








































### 3. Edge Layer Components
The edge layer is responsible for direct interaction with the tractor, local data processing, real-time inference, and communication with the cloud.

#### 3.1. Data Collection (`edge/data_collector/data_collect.py`)
- **Purpose**: Continuously collects sensor readings (via Arduino serial) and CAN bus messages.  
- **Input**: Raw data streams from Arduino (e.g., via `/dev/ttyACM0`) and CAN bus (e.g., `can0`).  
- **Output**: Appends combined raw data to a local CSV file (`edge/data/combined_log.csv`).  
- **Technology**: Python, `pyserial`, `python-can`.  
- **Operation**: Designed to run as a persistent service (e.g., `systemd` service).  

#### 3.2. Local Data Storage (`edge/data/combined_log.csv`)
- **Purpose**: Temporary storage for raw sensor and CAN data collected on the edge device before cloud synchronization.  
- **Format**: CSV.  
- **Lifecycle**: Cleared after successful upload to GCS.  

#### 3.3. Data Upload Script (`edge/data_collector/upload_data_to_gcs.py`)
- **Purpose**: Uploads the accumulated `combined_log.csv` from the edge device to a designated GCS bucket.  
- **Input**: `edge/data/combined_log.csv`  
- **Output**: New CSV blob in `gs://<raw_data_bucket>/raw/` with a timestamped filename. The local `combined_log.csv` is then deleted.  
- **Technology**: Python, `google-cloud-storage`.  
- **Operation**: Executed periodically via a cron job (e.g., daily).  

#### 3.4. Edge Preprocessing (`edge/model_inference/edge_preprocess.py`)
- **Purpose**: Prepares raw sensor data for local inference by applying the same feature engineering and scaling transformations as the cloud preprocessing pipeline.  
- **Input**: Raw sensor data (likely passed directly from `data_collect.py` or read from a temporary local file).  
- **Output**: Transformed feature vector suitable for TFLite model input.  
- **Technology**: Python, `pandas`, `numpy`, `scikit-learn` (for scaler).  
- **Dependency**: Requires the `scaler.joblib` file downloaded from the cloud.  

#### 3.5. Local Inference (`edge/model_inference/edge_inference.py`)
- **Purpose**: Performs real-time anomaly detection and fault classification on preprocessed data using lightweight TFLite models.  
- **Input**: Preprocessed feature vector from `edge_preprocess.py`.  
- **Output**: Anomaly scores, classification predictions, and overall health status, saved to `edge/gui/latest_status.json`.  
- **Technology**: Python, TensorFlow Lite Interpreter.  
- **Dependency**: Requires `autoencoder_tflite_model.tflite` and `classifier_tflite_model.tflite` downloaded from the cloud.  
- **Operation**: Designed to run as a persistent service, processing data as it becomes available.  

#### 3.6. Local GUI Display (`edge/gui/tractor_display_gui.py`)
- **Purpose**: Provides a local graphical interface on the edge device to visualize the tractor's real-time health status and sensor readings.  
- **Input**: `edge/gui/latest_status.json` (updated by `edge_inference.py`).  
- **Output**: Visual display.  
- **Technology**: Python, Tkinter or similar GUI library.  

#### 3.7. Model Update Script (`edge/model_inference/update_models.py`)
- **Purpose**: Periodically checks for and downloads the latest trained ML models (TFLite models and scaler) from GCS to the edge device.  
- **Input**: GCS Models Bucket (`gs://<models_bucket>/models/versions/` and `gs://<models_bucket>/models/latest/`).  
- **Output**: Updates `edge/models/current_models/` with the latest `autoencoder_tflite_model.tflite`, `classifier_tflite_model.tflite`, `scaler.joblib`, and a `VERSION` file.  
- **Technology**: Python, `google-cloud-storage`.  
- **Operation**: Executed periodically via a cron job (e.g., daily).  

---

### 4. Cloud Layer Components
The cloud layer handles scalable data storage, robust ML pipeline orchestration, and model serving.

#### 4.1. Google Cloud Storage (GCS) Buckets
- **Raw Data Bucket** (`gs://<raw_data_bucket>`): Stores raw sensor and CAN data CSVs uploaded from edge devices. Uses prefix `raw/` for organization.  
- **Models Bucket** (`gs://<models_bucket>`): Stores versioned TFLite models, H5 models, scalers, and anomaly thresholds from the training pipeline.  
- **Pipeline Artifacts Bucket** (`gs://<pipeline_artifacts_bucket>`): Used by Kubeflow Pipelines to store intermediate artifacts and logs.  

#### 4.2. BigQuery (`<project_id>.tractor_health_data.sensor_can_logs`)
- **Purpose**: A serverless data warehouse for storing and querying the ingested raw sensor and CAN bus data.  
- **Schema**: Auto-detected but captures all sensor fields and CAN log details.  

#### 4.3. Vertex AI Pipelines (Kubeflow Pipelines)
Pipeline defined in `cloud/end_to_end_automated_pipeline/end_to_end_pipeline/end_to_end_automated_pipeline.py` and compiled into `end_to_end_automated_pipeline.json`.

##### 4.3.1. Data Ingestion Component (`data_ingestion_container`)
- **Input**: Raw CSVs from GCS Raw Data Bucket.  
- **Output**: Loads into BigQuery and passes a local CSV artifact to the next step.  
- **Technology**: Python, `google-cloud-storage`, `google-cloud-bigquery`.  

##### 4.3.2. Data Preprocessing Component (`data_preprocessing_container`)
- **Input**: Raw data CSV artifact.  
- **Output**: Processed train/val/test CSVs and a serialized scaler.  
- **Technology**: Python, `pandas`, `numpy`, `scikit-learn`, `tensorflow`.  
- **Core Logic**: `core_utilities/preprocess.py`.  

##### 4.3.3. Autoencoder Training Component (`autoencoder_training_container`)
- **Input**: Processed data and fitted scaler.  
- **Output**: Keras H5 model, TFLite model, anomaly threshold.  
- **Technology**: Python, TensorFlow/Keras, keras-tuner, `joblib`, `google-cloud-storage`.  
- **Core Logic**: `training_pipeline/train_autoencoder.py`.  

##### 4.3.4. Classifier Training Component (`classifier_training_container`)
- **Input**: Processed train/val/test data.  
- **Output**: Keras H5 model and TFLite classifier.  
- **Technology**: Python, TensorFlow/Keras, keras-tuner, `scikit-learn`, `shap`, `google-cloud-storage`.  
- **Core Logic**: `training_pipeline/train_classifier.py`.  

##### 4.3.5. Model Deployment Component (`model_deployment_container`)
- **Input**: H5 autoencoder and classifier model URIs.  
- **Output**: Deployed models to Vertex AI Endpoint.  
- **Technology**: Python, `google-cloud-aiplatform`.  

#### 4.4. Vertex AI Endpoints
- **Purpose**: Hosts Autoencoder and Classifier models for online predictions.  
- **Interface**: Acce

ssible via REST API.  

---

### 5. Data Flow

1. **Edge Data Collection**: `data_collect.py` gathers sensor and CAN data → `combined_log.csv`.  
2. **Edge Upload**: `upload_data_to_gcs.py` uploads CSV to GCS → deletes local copy.  
3. **Cloud Ingestion**: Vertex pipeline reads from GCS and loads into BigQuery → forwards CSV.  
4. **Preprocessing**: CSV → feature engineering & scaling → `train_data.csv`, `scaler.joblib`, etc.  
5. **Model Training**:  
   - Autoencoder: uses healthy data → H5 + TFLite + threshold.  
   - Classifier: all labeled data → H5 + TFLite.  
6. **Model Deployment**: H5 models deployed to Vertex AI.  
7. **Edge Sync**: `update_models.py` fetches latest models and scaler.  
8. **Edge Inference**: `edge_inference.py` → outputs `latest_status.json`.  
9. **GUI Display**: `tractor_display_gui.py` reads status and visualizes results.  

---

### 6. Technology Stack

**Edge**:  
- Python, `pyserial`, `python-can`, `pandas`, `numpy`, `scikit-learn`, TensorFlow Lite  
- GUI: Tkinter  
- System Services: `systemd`, `cron`

**Cloud**:  
- **Storage**: Google Cloud Storage, BigQuery  
- **ML Platform**: Vertex AI (Pipelines, Endpoints)  
- **Containers**: Docker, Artifact Registry  
- **Frameworks**: TensorFlow/Keras, `scikit-learn`, Keras Tuner, SHAP  
- **Orchestration**: Kubeflow Pipelines (KFP SDK)  
- **Language**: Python