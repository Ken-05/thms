

# Tractor Health Monitoring: Edge-to-Cloud MLOps Pipeline

## Project Overview

This project implements a comprehensive MLOps pipeline for real-time tractor health monitoring, spanning from edge device data collection to cloud-based machine learning model training and deployment. The system is designed to detect anomalies and classify potential faults in tractor operations, enabling predictive maintenance and improving operational efficiency.

The solution leverages Google Cloud Platform (GCP) services for scalable data storage, robust machine learning workflows, and model serving, while utilizing edge devices (e.g., Raspberry Pi with Arduino/CAN bus interfaces) for local data acquisition and real-time inference.

## Key Features

- **Edge Data Collection**: Gathers sensor data (e.g., temperature, levels, vibrations) and CAN bus logs directly from tractors.
- **Edge Inference**: Runs lightweight TFLite models on the edge device for immediate anomaly detection and fault classification.
- **Edge to Cloud Raw Data Upload**: Periodically uploads raw data from edge devices to Google Cloud Storage (GCS).
- **Automated ML Pipeline (Kubeflow Pipelines on Vertex AI)**:
  - **Data Ingestion**: Ingest raw data daily from Google Cloud Storage and load it into BigQuery for centralized storage and preprocessing.
  - **Data Preprocessing**: Cleans, transforms, and engineers features from raw data.
  - **Autoencoder Training**: Trains an LSTM autoencoder model on healthy data for anomaly detection.
  - **Classifier Training**: Trains a multi-class CNN+LSTM classifier model to identify specific fault types.
  - **Model Deployment**: Deploys trained models to Vertex AI Endpoints for serving.
  - **Model Versioning & Sync**: New models trained in the cloud are automatically versioned and synced back to edge devices for continuous improvement.
- **Local GUI**: A simple graphical interface on the edge device to display real-time sensor data, inference results, and system status.

## High-Level Architecture

The system is divided into two primary layers: Edge and Cloud.

```
+---------------------+      +---------------------+      +---------------------+
|                     |      |                     |      |                     |
|    Tractor Sensors  |----->|   Edge Device       |----->|   Google Cloud      |
|    (Arduino/CAN)    |      | (Raspberry Pi)      |      |     Platform        |
|                     |      |                     |      |                     |
+---------------------+      +---------------------+      +---------------------+
                                     |       ^
                                     |       | (Model Sync)
                                     |       |
                                     v       |
+---------------------------------------------------------------------------------+
|                                 Edge Layer                                      |
|                                                                                 |
|  +-------------------+   +-------------------+   +----------------------------+ |
|  | Data Collection   |-->| Local Data Storage|-->| Data Upload Script         | |
|  | (data_collect.py) |   | (combined_log.csv)|   | (upload_data_to_gcs.py)    | |
|  +-------------------+   +-------------------+   +----------------------------+ |
|           |                                                                     |
|           v                                                                     |
|  +------------------------+   +-----------------------+   +-------------------+ |
|  | Edge Preprocessing     |-->| Local Inference       |-->| Local GUI Display | |
|  | (edge_preprocess.py)   |   | (edge_inference.py)   |   | (tractor_display_gui.py)| |
|  +------------------------+   +-----------------------+   +-------------------+ |
|           ^                                                                     |
|           | (Model Sync)                                                        |
|  +------------------------+                                                     |
|  | Model Update Script    |                                                     |
|  | (update_models.py)     |                                                     |
|  +------------------------+                                                     |
+---------------------------------------------------------------------------------+
                                     |
                                     | (Raw Data Upload)
                                     v
+---------------------------------------------------------------------------------+
|                                 Cloud Layer                                     |
|                                                                                 |
|  +-------------------+   +---------------------------------------------------+  |
|  | GCS Raw Data      |-->| BigQuery (tractor_health_data.sensor_can_logs)    |  |
|  | (raw_data_bucket) |   |                                                   |  |
|  +-------------------+   +---------------------------------------------------+  |
|           |                                                                     |
|           v                                                                     |
|  +---------------------------------------------------------------------------+  |
|  |                     Vertex AI Pipelines (Kubeflow Pipelines)              |  |
|  |  +-------------------+   +-------------------+   +-------------------+    |  |
|  |  | 1. Data Ingestion |-->| 2. Data           |-->| 3. Autoencoder    |    |  |
|  |  | (GCS to BigQuery) |   | Preprocessing     |   | Training          |    |  |
|  |  +-------------------+   +-------------------+   +-------------------+    |  |
|  |           |                       |                       |               |  |
|  |           v                       v                       v               |  |
|  |  +------------------------+   +-------------------+   +-------------------+  |
|  |  | (Intermediate Artifacts|   | (Train/Val/Test   |   | (H5/TFLite Models)|  |
|  |  | in Pipeline Root)      |   | CSVs, Scaler)     |   |                   |  |
|  |  +------------------------+   +-------------------+   +-------------------+  |
|  |                                   |                                       |  |
|  |                                   v                                       |  |
|  |  +-------------------+   +--------------------------+                     |  |
|  |  | 4. Classifier     |-->| 5. Model Deployment      |                     |  |
|  |  | Training          |   | (Vertex AI Endpoint)     |                     |  |
|  |  +-------------------+   +--------------------------+                     |  |
|  +---------------------------------------------------------------------------+  |
|                                     |                                           |
|                                     v                                           |
|  +-------------------+   +-------------------+                                  |
|  | GCS Models Bucket |<--| Vertex AI Endpoint|                                  |
|  | (models_bucket)   |   | (Deployed Models) |                                  |
|  +-------------------+   +-------------------+                                  |
+---------------------------------------------------------------------------------+
```



## Getting Started: Running the Entire System

This section outlines the general steps to get the entire edge-to-cloud MLOps pipeline up and running. Refer to the detailed `docs/` for specific instructions.

### Prerequisites

- **Google Cloud SDK** (`gcloud`, `gsutil`) — Authenticated to your GCP project.
- **Docker** — For building and pushing container images.
- **Python 3.8+ and pip**
- **jq** — A lightweight and flexible command-line JSON processor.
- **yq** — A portable YAML processor.
- **Edge Device** — A Raspberry Pi or similar Linux-based device with Python 3.8+, pip, and hardware interfaces.

### High-Level Order of Operations

**GCP Project Setup & APIs**:
- Create a GCP Project.
- Enable APIs: Compute Engine, Cloud Storage, BigQuery, Artifact Registry, Vertex AI, Cloud Build, IAM.
- Create Service Accounts for:
  - Edge Device (write access to raw data bucket).
  - Vertex AI Pipelines (access to GCS, BQ, Vertex AI, etc.).
- _Action_: Follow `docs/gcp_setup.md`.

**Create GCS Buckets**:
- Run: `cd edge/scripts && ./create_gcs_bucket.sh`

**Build and Push Cloud Docker Images**:
- Navigate to project root.
- Build/push each component under `cloud/end_to_end_automated_pipeline/components/`.
- _Action_: Follow `docs/gcp_setup.md`.

**Configure Cloud Settings**:
- Edit `cloud/end_to_end_automated_pipeline/config/cloud_settings.yaml`.

**Compile Kubeflow Pipeline**:
- `cd cloud/end_to_end_automated_pipeline/end_to_end_pipeline && python end_to_end_automated_pipeline.py`

**Run Cloud ML Pipeline**:
- Manual: `python run_pipeline.py`
- Scheduled: Use Vertex AI UI with the JSON blueprint.

**Edge Device Setup**:
- Clone repo, install dependencies.
- Edit `edge/config/settings.json`.
- Add `gcp-service-account.json` to `edge/keys/`.
- Configure Arduino or CAN interfaces.
- _Action_: See `docs/edge_setup.md`.

**Setup Edge Cron Jobs**:
- `cd edge/scripts && ./setup_cron_upload_data_to_gcs.sh`
- `cd edge/scripts && ./setup_cron_update_models.sh`

**Start Edge Apps**:
- `data_collect.py`, `edge_inference.py`, `tractor_display_gui.py`

---

## Project Structure
```
.
├── cloud/
│   ├── end_to_end_automated_pipeline/
│   │   ├── components/
│   │   │   ├── autoencoder_training_container/
│   │   │   │   ├── Dockerfile
│   │   │   │   ├── main.py
│   │   │   │   └── requirements
│   │   │   ├── classifier_training_container/
│   │   │   │   ├── Dockerfile
│   │   │   │   ├── main.py
│   │   │   │   └── requirements
│   │   │   ├── data_ingestion_container/
│   │   │   │   ├── Dockerfile
│   │   │   │   ├── main.py
│   │   │   │   └── requirements
│   │   │   ├── data_preprocessing_container/
│   │   │   │   ├── Dockerfile
│   │   │   │   ├── main.py
│   │   │   │   └── requirements
│   │   │   └── model_deployment_container/
│   │   │       ├── Dockerfile
│   │   │       ├── main.py
│   │   │       └── requirements
│   │   ├── config/
│   │   │   └── cloud_settings.yaml
│   │   ├── core_utilities/
│   │   │   ├── __init__.py
│   │   │   └── preprocess.py
│   │   ├── end_to_end_pipeline/
│   │   │   ├── end_to_end_automated_pipeline.json
│   │   │   ├── end_to_end_automated_pipeline.py
│   │   │   └── run_pipeline.py
│   │   └── training_pipeline/
│   │       ├── __init__.py
│   │       ├── train_autoencoder.py
│   │       └── train_classifier.py
├── docs/
│   ├── architecture.md
│   ├── data_schema.md
│   ├── edge_setup.md
│   └── gcp_setup.md
├── edge/
│   ├── config/
│   │   └── settings.json
│   ├── data/
│   │   ├── combined_log.csv
│   │   ├── combined_log_preprocessed.csv
│   │   ├── test_data.csv
│   │   └── train_data.csv
│   ├── data_collector/
│   │   ├── data_collect.py
│   │   ├── upload_data_to_gcs.py
│   │   └── upload_data_to_gcs_log.txt
│   ├── gui/
│   │   ├── latest_status.json
│   │   └── tractor_display_gui.py
│   ├── keys/
│   │   └── gcp-service-account.json
│   ├── model_inference/
│   │   ├── __init__.py
│   │   ├── edge_inference.py
│   │   ├── edge_preprocess.py
│   │   ├── update_models.log
│   │   └── update_models.py
│   ├── models/
│   │   ├── current_models/
│   │   │   ├── model_autoencoder.tflite
│   │   │   ├── model_classifier.tflite
│   │   │   ├── scaler.joblib
│   │   │   └── VERSION
│   │   └── temp_models_download/
│   └── scripts/
│       ├── create_gcs_bucket.sh
│       ├── setup_cron_update_models.sh
│       └── setup_cron_upload_data_to_gcs.sh
├── hardware/
│   └── arduino_code/
│       └── sensors/
│           └── sensors.ino
├── .gitignore
├── LICENSE
├── README.md
└── requirements.txt
```

### 📄 License

This project is not yet licensed under the MIT License - see the [LICENSE](./LICENSE) file for details.

