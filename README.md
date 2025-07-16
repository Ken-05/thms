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
python edge/data_collector/collect.py
```

### 🤖 Train ML Models (Cloud)
```bash
python cloud/training_pipeline/train_classifier.py
python cloud/training_pipeline/train_autoencoder.py
```

### 📁 Repo Structure

| Path         | Description                               |
|--------------|-------------------------------------------|
| `edge/`      | Pi-side code: sensors, inference, GUI     |
| `cloud/`     | GCP scripts: model training, upload, dashboards |
| `hardware/`  | Arduino code and wiring diagrams          |
| `models/`    | Saved models (H5 and TFLite)              |
| `data/`      | Sample or testing data logs               |
| `docs/`      | System diagrams and documentation         |

---

### 📄 License

This project is licensed under the MIT License - see the [LICENSE](./LICENSE) file for details.
