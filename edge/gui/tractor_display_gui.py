import sys
import json
import os
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QLabel, QGridLayout, QFrame, QTimer
)
from PyQt5.QtGui import QFont
from PyQt5.QtCore import Qt


# --- Function to load settings from settings.json ---
def load_settings():
    """
    Loads configuration settings from the settings.json file.
    Assumes settings.json is in the 'config' directory one level up from this file.
    """
    # Get the directory of the current script)
    BASE_DIR = os.path.abspath(os.path.dirname(__file__))
    # Go up one level and then into 'config'
    CONFIG_DIR = os.path.join(BASE_DIR, "..", "config")
    SETTINGS_FILE_PATH = os.path.join(CONFIG_DIR, "settings.json")

    try:
        with open(SETTINGS_FILE_PATH, 'r') as f:
            settings = json.load(f)
        print(f"[GUI:Settings] Loaded settings from {SETTINGS_FILE_PATH}")
        return settings
    except FileNotFoundError:
        print(f"[GUI:Settings] ERROR: settings.json not found at {SETTINGS_FILE_PATH}. GUI may not function correctly.")
        # In a GUI, you might show a message box, but for now, print and return empty dict
        return {}
    except json.JSONDecodeError:
        print(f"[GUI:Settings] ERROR: Could not decode JSON from {SETTINGS_FILE_PATH}. Check file format.")
        return {}

# Load settings once when the module is imported
SETTINGS = load_settings()

# Config (load from settings)
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
STATUS_FILE = os.path.join(BASE_DIR, SETTINGS['inference']['status_output_file'])
FAULT_CONFIDENCE_THRESHOLD = SETTINGS['inference']['fault_confidence_threshold']


FAULT_CLASS_MAP = {
    0: "Healthy",
    1: "Overheated Front Brakes",
    2: "Overheated Rear Brakes",
    3: "Low Hydraulic Oil level",
    4: "Too high Hydraulic Oil level",
    5: "Low Transmission Oil level",
    6: "Too high Transmission Oil level",
    7: "Vibration Issue",
    8: "Clutch Failure",
    9: "Transmission Overheat",
    10: "Engine Overheat",
    11: "Alternator Overheat",
    12: "Clutch Overheat",
    13: "Engine Coolant Overheat",
    14: "Engine RPM Failure",
    15: "Fuel Overheat"
}

PART_MAP = {
    "Overheated Front Brakes": "Front Brakes",
    "Overheated Rear Brakes": "Rear Brakes",
    "Low Hydraulic Oil level": "Hydraulic Tank",
    "Too high Hydraulic Oil level": "Hydraulic Tank",
    "Low Transmission Oil level": "Transmission Oil",
    "Too high Transmission Oil level": "Transmission Oil",
    "Vibration Issue": "Cabin Vibration",
    "Clutch Failure": "Engine",
    "Transmission Overheat": "Transmission",
    "Engine Overheat": "Engine Compartment",
    "Alternator Overheat": "Alternator",
    "Clutch Overheat": "Clutch",
    "Engine Coolant Overheat": "Engine Coolant",
    "Engine Overheat": "Engine", 
    "Fuel Overheat": "Fuel"
    
}


class StatusCard(QFrame):
    def __init__(self, title):
        super().__init__()
        self.setFrameShape(QFrame.Box)
        self.setLineWidth(2)
        self.setStyleSheet("background-color: #f0f0f0; border-radius: 10px;")
        layout = QVBoxLayout()

        self.title_label = QLabel(title)
        self.title_label.setFont(QFont("Arial", 14, QFont.Bold))
        self.title_label.setAlignment(Qt.AlignCenter)

        self.status_label = QLabel("Waiting...")
        self.status_label.setFont(QFont("Arial", 12))
        self.status_label.setAlignment(Qt.AlignCenter)

        layout.addWidget(self.title_label)
        layout.addWidget(self.status_label)
        self.setLayout(layout)

    def update_status(self, status_text, is_fault=False):
        self.status_label.setText(status_text)
        if is_fault:
            self.setStyleSheet("background-color: #ffcccc; border-radius: 10px;")
        elif status_text == "Normal":
            self.setStyleSheet("background-color: #ccffcc; border-radius: 10px;")
        else:
            self.setStyleSheet("background-color: #f0f0f0; border-radius: 10px;")


class TractorHealthGUI(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Tractor Health Dashboard")
        self.setGeometry(100, 100, 800, 400)
        self.setStyleSheet("background-color: #ffffff;")

        layout = QVBoxLayout()
        title = QLabel("Real-Time Tractor Health Monitoring")
        title.setFont(QFont("Arial", 20, QFont.Bold))
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)

        self.grid = QGridLayout()

        # Define the main components to display on the GUI
        self.components = {
            "Engine": StatusCard("Engine"),
            "Brakes": StatusCard("Brakes"),
            "Hydraulic Oil": StatusCard("Hydraulic Oil"),
            "Cabin Vibration": StatusCard("Cabin Vibration"),
            "Transmission": StatusCard("Transmission")
            # Future Note: add more components being monitored
        }

        positions = [(i, j) for i in range(2) for j in range(3)]
        for pos, key in zip(positions, self.components):
            self.grid.addWidget(self.components[key], *pos)

        layout.addLayout(self.grid)
        self.setLayout(layout)

        # Timer to update display every 2 seconds
        self.timer = QTimer()
        self.timer.timeout.connect(self.refresh_status)
        self.timer.start(2000)

    def refresh_status(self):
        if not os.path.exists(STATUS_FILE):
            for key in self.components:
                self.components[key].update_status("No Data")
            return

        try:
            with open(STATUS_FILE, "r") as f:
                data = json.load(f)

            is_anomaly = data.get("is_anomaly", False)
            fault_class = data.get("fault_class", -1)
            confidence = data.get("confidence", 0.0)
            fault_label = FAULT_CLASS_MAP.get(fault_class, "Unknown Fault")


            # Reset all parts
            for key in self.components:
                self.components[key].update_status("Normal")

            # Update based on overall system anomaly status
            if is_anomaly:
                if fault_class != -1 and confidence >= FAULT_CONFIDENCE_THRESHOLD:
                    part_affected = PART_MAP.get(fault_label, None)
                    if part_affected and part_affected in self.components:
                        self.components[part_affected].update_status(f"FAULT: {fault_label} ({confidence:.0%})", is_fault=True)
                    else:
                        print(f"[GUI] Warning: Fault '{fault_label}' mapped to unknown part '{part_affected}' or not in display components.")
                else:
                    print(f"[GUI] Anomaly detected, but no specific fault classified or low confidence (Class: {fault_class}, Conf: {confidence:.2f}).")
            else:
                pass # Already reset to "Normal" above

        except json.JSONDecodeError:
            print(f"[GUI] Error decoding JSON from {STATUS_FILE}. File might be corrupted or incomplete.")
            for key in self.components:
                self.components[key].update_status("Error Reading Data")
        except Exception as e:
            print(f"[GUI] Failed to update GUI: {e}")
            for key in self.components:
                self.components[key].update_status("Update Error")
                

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = TractorHealthGUI()
    window.show()
    sys.exit(app.exec_())
