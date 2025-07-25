# Edge Device Setup Guide
This document provides detailed instructions for setting up your edge device (e.g., Raspberry Pi) to collect tractor data, perform local inference, and synchronize with the Google Cloud Platform (GCP).

## 1. Prerequisites
- A Raspberry Pi (or similar Linux-based SBC) with an internet connection.

- Raspberry Pi OS (or other Debian-based Linux distribution) installed.

- Python 3.8+ and pip installed.

- git installed.

- cron daemon running (usually pre-installed).

- **Hardware:**

  - Arduino (e.g., Arduino Uno/Nano) connected via USB for sensor data.

  - CAN bus interface (e.g., Waveshare SN65HVD230 CAN board) connected to the Raspberry Pi's GPIO pins, configured for CAN communication.

## 2. Initial Device Setup
**1. Update and Upgrade System:**
``` bash
sudo apt update && sudo apt upgrade -y
```
**2. Install Python Development Tools:**
``` bash
sudo apt install python3-dev python3-pip -y
```
**3. Install jq (for shell scripts):**
``` bash
sudo apt install jq -y
```
## 3. Clone the Repository
Clone your project repository onto the edge device. It's recommended to clone it into a user's home directory, e.g., /home/pi/tractor_monitor/.
``` bash
cd /home/pi/
git clone [YOUR_REPOSITORY_URL] tractor_monitor
cd tractor_monitor
```
## 4. Install Python Dependencies
Install all Python libraries required for the edge components.
``` bash
pip install -r requirements.txt
```
(Note: Ensure requirements.txt at the project root contains all edge-specific dependencies like pyserial, python-can, pandas, numpy, scikit-learn, tensorflow-lite, google-cloud-storage.)

## 5. Configure Edge Settings (edge/config/settings.json)
This file contains all local and cloud synchronization settings for the edge device. Update it with your specific hardware details and cloud resource names.

Navigate to the config directory:
``` bash
cd edge/config/
nano settings.json # or your preferred editor
```
Adjust the following sections:

- data_collection:

  - arduino_serial_port: E.g., "/dev/ttyACM0" (verify with ls /dev/ttyA* or ls /dev/ttyU* after plugging in Arduino).

  - arduino_baud_rate: Must match your Arduino sketch's baud rate.

  - can_interface: E.g., "can0" (after CAN setup).

  - can_bitrate: Must match your CAN bus network's bitrate.

  - combined_log_file_path: This is relative to the project root (edge/data/combined_log.csv). Ensure this path is correct.

- inference:

  - model_dir: "models/current_models/" (relative to edge/).

  - temp_model_dir: "models/temp_models_download/" (relative to edge/).

  - autoencoder_filename, classifier_filename, scaler_params_filename: These should match the names used by your cloud training pipeline when uploading to GCS.

  - anomaly_threshold, fault_confidence_threshold: Adjust as needed based on model performance.

  - rolling_window: Must match the value used in cloud/end_to_end_automated_pipeline/core_utilities/preprocess.py (which is read from cloud_settings.yaml during cloud training).

- cloud_sync:

  - gcs_raw_data_bucket: Must exactly match the name of your raw data GCS bucket (e.g., tractor-health-monitoring-raw-data-bucket).

  - gcs_raw_data_prefix: Must match the prefix used in the cloud (e.g., "raw/").

  - gcs_models_bucket: Must exactly match the name of your models GCS bucket (e.g., tractor-health-monitoring-models-bucket).

  - gcs_models_prefix: Must match the prefix used in the cloud (e.g., "models/versions/").

  - gcp_credentials_path_relative: Set to "keys/gcp-service-account.json".

- logging: Adjust log file paths and levels as desired.

## 6. Place GCP Service Account Key
Place the gcp-service-account.json key file (downloaded during GCP setup) into the edge/keys/ directory of your project on the Raspberry Pi.

``` bash
# Example: If you downloaded it to your Downloads folder on your computer
# and are using scp to transfer:
# scp ~/Downloads/edge-gcp-key.json pi@<RASPBERRY_PI_IP>:/home/pi/tractor_monitor/edge/keys/gcp-service-account.json
# On the Raspberry Pi:
mv /path/to/downloaded/gcp-service-account.json /home/pi/tractor_monitor/edge/keys/
```
Ensure the file is named gcp-service-account.json.

## 7. Hardware Setup (Arduino & CAN Bus)
### 7.1. Arduino Serial Communication
**1. Upload Arduino Sketch:** Upload your hardware/arduino_code/sensors/sensors.ino sketch to your Arduino board. Ensure the baud rate in the sketch matches arduino_baud_rate in settings.json.

**2. Connect Arduino:** Connect the Arduino to the Raspberry Pi via a USB cable.

**3. Verify Serial Port:**
``` bash
ls /dev/ttyA* /dev/ttyU*
```
Identify the correct serial port (e.g., /dev/ttyACM0, /dev/ttyUSB0). Update arduino_serial_port in settings.json if necessary.

**4. Grant Permissions (if needed):**
``` bash
sudo usermod -a -G dialout pi # Add your user to dialout group
sudo chmod a+rw /dev/ttyACM0 # Or your specific serial port
# Reboot for group changes to take effect: sudo reboot
```
### 7.2. CAN Bus Configuration (Example for Raspberry Pi)
This assumes you are using a CAN HAT or similar interface.

**1. Enable SPI (if using SPI-based CAN HAT):**
``` bash
sudo raspi-config
# Go to Interface Options -> SPI -> Yes
# Reboot if prompted
```
**2. Add CAN Bus Overlay: Edit /boot/config.txt**
``` bash
sudo nano /boot/config.txt
```
Add or uncomment lines similar to these (adjusting for your specific HAT/chipset, e.g., MCP2515, MCP2517FD):
``` bash
dtparam=spi=on
dtoverlay=mcp2515-can0,oscillator=16000000,interrupt=25
dtparam=spi=on
dtoverlay=spi-bcm2835
```
- oscillator: Your crystal frequency (e.g., 8MHz, 16MHz).

- interrupt: The GPIO pin connected to the INT pin of your CAN controller.

**3. Reboot:**
``` bash
sudo reboot
```
**4. Bring up CAN Interface:** After reboot, configure the CAN interface.
``` bash
sudo ip link set can0 up type can bitrate 250000 # Use bitrate from settings.json
```
- Verify with ip -s link show can0.

- You can test with candump can0.

**5. Make CAN Config Persistent (Optional, Recommended):**
Edit /etc/network/interfaces.d/can0 (create if it doesn't exist)
``` bash
sudo nano /etc/network/interfaces.d/can0
```
Add:
``` bash
auto can0
iface can0 inet manual
    pre-up ip link set can0 type can bitrate 250000
    up ip link set can0 up
    down ip link set can0 down
``` 
Or use rc.local or a systemd service for persistent setup.

## 8. Setup Cron Jobs
Cron jobs automate the daily data upload and model update processes.

### 8.1. Data Upload Cron Job (upload_data_to_gcs.py)
This script uploads combined_log.csv daily.

**1. Navigate to scripts directory:**
``` bash
cd /home/pi/tractor_monitor/edge/scripts/
```
**2. Make script executable:**
``` bash
chmod +x setup_cron_upload_data_to_gcs.sh
```
**3. Run setup script:**
``` bash
./setup_cron_upload_data_to_gcs.sh
```
This script will add a cron entry to run upload_data_to_gcs.py at 2:00 AM daily (or as configured in the script).

### 8.2. Model Update Cron Job (update_models.py)
This script checks for and downloads new models daily.

**1. Navigate to scripts directory:**
``` bash
cd /home/pi/tractor_monitor/edge/scripts/
```
**2. Make script executable:**
``` bash
chmod +x setup_cron_update_models.sh
```
**3. Run setup script:**
``` bash
./setup_cron_update_models.sh
```
This script will add a cron entry to run update_models.py at 1:30 AM daily (or as configured in the script).

## 9. Running Edge Applications
For robust, continuous operation, it's highly recommended to run data_collect.py and edge_inference.py as systemd services.

### 9.1. Data Collection Service (data_collect.py)
**1. Create a systemd service file:**
``` bash
sudo nano /etc/systemd/system/data_collect.service
```
Add the following content (adjust /home/pi/tractor_monitor if your path differs):
``` bash
[Unit]
Description=Tractor Data Collection Service
After=network.target

[Service]
ExecStart=/usr/bin/python3 /home/pi/tractor_monitor/edge/data_collector/data_collect.py
WorkingDirectory=/home/pi/tractor_monitor/
StandardOutput=inherit
StandardError=inherit
Restart=always
User=pi # Or your username
Group=pi # Or your groupname

[Install]
WantedBy=multi-user.target
```
**2. Reload systemd, enable, and start the service:**
``` bash
sudo systemctl daemon-reload
sudo systemctl enable data_collect.service
sudo systemctl start data_collect.service
```
**3. Check status:**
``` bash
sudo systemctl status data_collect.service
```
### 9.2. Edge Inference Service (edge_inference.py)
**1. Create a systemd service file:**
``` bash
sudo nano /etc/systemd/system/edge_inference.service
```
Add the following content:
``` bash
[Unit]
Description=Tractor Edge Inference Service
After=data_collect.service # Ensure data collection starts first

[Service]
ExecStart=/usr/bin/python3 /home/pi/tractor_monitor/edge/model_inference/edge_inference.py
WorkingDirectory=/home/pi/tractor_monitor/
StandardOutput=inherit
StandardError=inherit
Restart=always
User=pi # Or your username
Group=pi # Or your groupname

[Install]
WantedBy=multi-user.target
```
**2. Reload systemd, enable, and start the service:**
``` bash
sudo systemctl daemon-reload
sudo systemctl enable edge_inference.service
sudo systemctl start edge_inference.service
```
**3. Check status:**
``` bash
sudo systemctl status edge_inference.service
```
### 9.3. Local GUI (tractor_display_gui.py)
The GUI application can be run manually or as part of your desktop environment's startup.

**1. Navigate to GUI directory:**
``` bash
cd /home/pi/tractor_monitor/edge/gui/
```
**2. Run the GUI:**
``` bash
python tractor_display_gui.py
```
- If running headless, you might need to configure X server or VNC.

- For autostart on boot with a desktop environment, add an entry to .config/autostart/ or similar.

## 10. Verification
- **Check logs:** Monitor the log files defined in settings.json (logs/app.log, logs/inference.log).

- **Verify GCS uploads:** Check your raw_data_bucket in the GCP console for new CSV files.

- **Verify model downloads:** Check edge/models/current_models/ for updated model files and the VERSION file.

- **Check latest_status.json:** Verify that edge/gui/latest_status.json is being updated by edge_inference.py.