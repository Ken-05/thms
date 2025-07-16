# thms
An end-to-end machine learning and IoT system with cloud analytics for real-time thm using s and C b data.

# 🚜 Tractor Health Monitoring System

An end-to-end machine learning and IoT system with cloud analytics for real-time tractor health monitoring using sensor and CAN bus data.

---

## 📦 Features

- 📡 Real-time data collection from sensors and CAN bus
- 🤖 ML-based anomaly detection and fault classification
- 🔁 Multi-modal data fusion: Sensor + CAN data
- 📊 PyQt5 GUI for in-cabin tractor display
- ☁️ Google Cloud Platform integration for training and analytics
- 🔁 Model updates from cloud to edge (TensorFlow Lite)

---

## 🧱 Architecture Overview

- **Edge Layer**: Raspberry Pi + Arduino for data ingestion, ML inference, and GUI display
- **Cloud Layer (GCP)**: Model training, storage, fleet-wide analysis
- **Multi-Modal ML**: Separate encoders for sensor and CAN data, merged for prediction

---

## 🛠️ Tech Stack

| Layer        | Tools & Libraries |
|--------------|-------------------|
| Edge         | Arduino, Raspberry Pi, PyQt5, TensorFlow Lite |
| Data         | Python, pandas, scipy, python-can, cantools |
| ML Models    | TensorFlow, Keras, LSTM Autoencoder, CNN+LSTM Classifier |
| Cloud        | GCP Pub/Sub, Cloud Functions, Vertex AI, BigQuery, Looker Studio |

---

## 🚀 Getting Started

### Prerequisites

- Raspberry Pi running Linux (Raspbian or similar)
- Arduino board for sensor data ingestion
- Google Cloud Platform project with Storage bucket and Pub/Sub set up
- Python 3 installed on Raspberry Pi
- Required Python packages (see `requirements.txt`)

  
### 🧰 Requirements
Install dependencies:

```bash
pip install -r requirements.txt



### 🖥️ Run GUI (On Pi)
```bash
cd edge/gui
python main_window.py
```

### ☁️ Upload to GCP
```bash
# Trigger data upload
python edge/data_collector/data_collect.py
```

### 🤖 Train ML Models (Cloud)
```bash
python cloud/training_pipeline/train_classifier.py
python cloud/training_pipeline/train_autoencoder.py
```

## Running the Project

### 1. Setting Up the Project Locally

Create the project directory structure and clone this repo.

---

### 2. Data Collection and Upload

The main data collection script is:

```bash
edge/data_collector/data_collect.py
```
This script reads sensor and CAN bus data, logs to CSV, performs ML inference, and uploads data to Google Cloud Pub/Sub and Storage.

### 3. Schedule Daily Upload Cron Job

To automate daily uploads of the combined log CSV to Google Cloud Storage:

1. SSH into your Raspberry Pi or open a terminal.

2. Navigate to the project root folder:

```bash
cd /home/pi/tractor-health-monitoring
```

3. Make the setup script executable:

```bash
chmod +x scripts/setup_cron.sh
```

4. Run the setup script to add the cron job:

```bash
./scripts/setup_cron.sh
```
This will add a cron job that runs every day at 2:00 AM to upload the latest combined log to GCS.

### 4. Manual Cron Job Setup (Alternative)

If you prefer to add the cron job manually:

1. Open the crontab editor:

```bash
crontab -e
```

2. Add the following line at the bottom:

```bash
0 2 * * * /usr/bin/python3 /home/pi/tractor-health-monitoring/cloud/gcp_functions/upload_to_gcs.py >> /home/pi/upload_log.txt 2>&1
```

3. Save and exit the editor.

### 📁 Repo Structure

| Path         | Description                               |
|--------------|-------------------------------------------|
| `edge/`      | Raspberry Pi-side code: data collection, inference, GUI     |
| `cloud/`     | GCP scripts: model training, upload, dashboards |
| `hardware/`  | Arduino code and wiring diagrams          |
| `models/`    | Saved ML models (H5 and TFLite)              |
| `data/`      | Sample or testing data logs               |
| `scripts/`   | Utility scripts (e.g., cron setup)               |
| `docs/`      | System diagrams and documentation         |

---

### 📄 License

This project is not yet licensed under the MIT License - see the [LICENSE](./LICENSE) file for details.
