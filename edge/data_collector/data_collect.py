# data_collect.py

import threading
import queue
import random
import time
import serial
import can
import csv
import os
import json
import struct
import numpy as np
from datetime import datetime
from google.cloud import pubsub_v1

# This data_collect.py should run as a systemd service for 
# reliable restarts. This script is restarted periodically 
# everyday to load the new models from the cloud which are  
# gotten because of the continuous retraining done at the cloud 

# When data_collect.py restarts, it re-imports edge_inference.py 
# and edge_preprocess.py, so these scripts will load the latest
# version of tflite models and scaler in the directory

# Import the edge inference and preprocessing modules
import edge_preprocess # Contains process_single_sample and scaler loading for single data preprocessing for inference
import edge_inference  # Contains perform_edge_inference


# --- Function to load settings from settings.json ---
def load_settings():
    """
    Loads configuration settings from the settings.json file.
    Assumes settings.json is in the 'config' directory one level up from this file.
    """
    # Get the directory of the current script 
    BASE_DIR = os.path.abspath(os.path.dirname(__file__))
    CONFIG_DIR = os.path.join(BASE_DIR, "..", "config")
    SETTINGS_FILE_PATH = os.path.join(CONFIG_DIR, "settings.json")

    try:
        with open(SETTINGS_FILE_PATH, 'r') as f:
            settings = json.load(f)
        print(f"[Settings] Loaded settings from {SETTINGS_FILE_PATH}")
        return settings
    except FileNotFoundError:
        print(f"[Settings] ERROR: settings.json not found at {SETTINGS_FILE_PATH}")
        # Exit if critical configuration is missing
        exit(1)
    except json.JSONDecodeError:
        print(f"[Settings] ERROR: Could not decode JSON from {SETTINGS_FILE_PATH}. Check file format.")
        exit(1)

# Load settings at the very beginning of the script
SETTINGS = load_settings()

# ------------------ Config (loaded from SETTINGS) ------------------ #
# Access settings using the SETTINGS dictionary
SERIAL_PORT = SETTINGS['data_collection']['arduino_serial_port']
BAUD_RATE = SETTINGS['data_collection']['arduino_baud_rate']
CAN_INTERFACE = SETTINGS['data_collection']['can_interface'] 
CAN_BITRATE = SETTINGS['data_collection']['can_bitrate'] 
PUBSUB_TOPIC = SETTINGS['cloud_sync']['pubsub_topic'] 
# Construct the absolute path for credentials based on the project structure
GCP_CREDENTIALS_PATH = os.path.join(
    os.path.abspath(os.path.dirname(__file__)), 
    "..",
    SETTINGS['cloud_sync']['gcp_credentials_path_relative'] # "keys/gcp-service-account.json"
)


# Shared queues for data passing between threads
raw_sensor_data_queue = queue.Queue() # Readers put raw data here
csv_writer_queue = queue.Queue() # CSVWriter consumes from here
cloud_uploader_queue = queue.Queue() # CloudUploader consumes from here
inference_input_queue = queue.Queue()# InferenceRunner consumes from here
inference_output_queue = queue.Queue() # InferenceRunner puts results here (e.g., for GUI updates, logging)


# ========== J1939 Decoder ========== #
class J1939Decoder:
    """
    Decodes J1939 CAN bus messages into human-readable sensor values.
    """
    @staticmethod
    def decode_pgn(arbitration_id, data_bytes):
        """
        Decodes specific PGNs (Parameter Group Numbers) from CAN messages.
        """
        pgn = J1939Decoder.extract_pgn(arbitration_id)
        decoded = {"pgn": pgn}

        if pgn == 65262: # Engine Temperature 1
            if len(data_bytes) >= 2:
                # SPN 110: Engine Coolant Temperature
                raw_temp = data_bytes[0]
                decoded["engine_coolant_temp_C"] = (raw_temp * 1.0) - 40
        elif pgn == 61444: # Engine Speed
            if len(data_bytes) >= 4:
                # SPN 190: Engine Speed (bytes 3-4)
                rpm_raw = struct.unpack("<H", bytes(data_bytes[2:4]))[0]
                decoded["engine_rpm"] = rpm_raw * 0.125
        elif pgn == 65263: # Fuel Economy
            if len(data_bytes) >= 2:
                # SPN 174: Fuel Temperature
                raw_temp = data_bytes[1]
                decoded["fuel_temp_C"] = (raw_temp * 1.0) - 40
        # Future: Add more PGN decoders for other CANBus Data
        return decoded

    @staticmethod
    def extract_pgn(arbitration_id):
        """
        Extracts the PGN from a 29-bit CAN ID (J1939).
        """
        pgn = (arbitration_id >> 8) & 0xFFFF
        if pgn >= 0xF000:
            pgn &= 0xFF00 # For PDU1 format, clear destination address
        return pgn

# ========== Serial Reader ========== #
class SerialReader(threading.Thread):
    """
    Reads data from a serial port (e.g., Arduino) and puts it into a queue.
    """
    def __init__(self, port, baudrate, output_queue):
        super().__init__()
        self.port = port
        self.baudrate = baudrate
        self.output_queue = output_queue
        self.stop_flag = threading.Event()

    def run(self):
        try:
            ser = serial.Serial(self.port, self.baudrate, timeout=1)
            print("[SerialReader] Connected to Arduino")
        except Exception as e:
            print(f"[SerialReader] ERROR: {e}")
            return

        while not self.stop_flag.is_set():
            try:
                if ser.in_waiting:
                    line = ser.readline().decode('utf-8', errors='ignore').strip()
                    if line:
                        # Put raw serial data into the queue
                        self.output_queue.put({"source": "serial", "data": line})
            except Exception as e:
                print(f"[SerialReader] ERROR while reading: {e}")
            time.sleep(0.01)

    def stop(self):
        """Signals the thread to stop."""
        self.stop_flag.set()

# ========== CAN Reader ========== #
class CANReader(threading.Thread):
    """
    Reads data from the CAN bus and puts decoded messages into a queue.
    """
    def __init__(self, interface, output_queue):
        super().__init__()
        self.interface = interface
        self.output_queue = output_queue
        self.stop_flag = threading.Event()

    def run(self):
        try:
            bus = can.interface.Bus(channel=self.interface, bustype='socketcan')
            print("[CANReader] CAN bus initialized")
        except Exception as e:
            print(f"[CANReader] ERROR: {e}")
            return

        while not self.stop_flag.is_set():
            try:
                msg = bus.recv(timeout=0.5)
                if msg:
                    raw_data_bytes = list(msg.data)
                    pgn_data = J1939Decoder.decode_pgn(msg.arbitration_id, raw_data_bytes)
                    # Put decoded CAN data into the queue
                    self.output_queue.put({
                        "source": "can",
                        "id": hex(msg.arbitration_id),
                        "pgn": pgn_data.get("pgn"),
                        "decoded": pgn_data,
                        "data": msg.data.hex()
                    })
            except Exception as e:
                print(f"[CANReader] ERROR while reading, watch for potential issues or faulty ECU: {e}")

    def stop(self):
        """Signals the thread to stop."""
        self.stop_flag.set()

# ========== CSV Writer ========== #
class CSVWriter(threading.Thread):
    """
    Appends combined sensor data to a CSV file.
    """
    def __init__(self, input_queue, filename):
        super().__init__()
        self.input_queue = input_queue
        self.filename = filename
        self.stop_flag = threading.Event()
        self.fields = [
            "timestamp", "source", "id", "pgn",
            "engine_coolant_temp_C", "engine_rpm", "fuel_temp_C",
            "data", "raw_line" # "data" for CAN hex, "raw_line" for serial raw string
        ]
        # Create file with header if it doesn't exist
        try:
            with open(self.filename, "x", newline='') as f:
                writer = csv.DictWriter(f, fieldnames=self.fields)
                writer.writeheader()
        except FileExistsError:
            pass # File already exists, continue appending
        except Exception as e:
            print(f"[CSVWriter] ERROR creating/opening CSV file {self.filename}: {e}")
        print("[CSVWriter] Initialized.")

    def run(self):
        while not self.stop_flag.is_set() or not self.input_queue.empty():
            try:
                item = self.input_queue.get(timeout=0.1) # Use timeout to allow stop_flag check
                with open(self.filename, "a", newline='') as f:
                    writer = csv.DictWriter(f, fieldnames=self.fields)
                    row = {
                        "timestamp": datetime.utcnow().isoformat(),
                        "source": item.get("source"),
                        "id": item.get("id", ""),
                        "pgn": item.get("pgn", ""),
                        "engine_coolant_temp_C": item.get("decoded", {}).get("engine_coolant_temp_C", ""),
                        "engine_rpm": item.get("decoded", {}).get("engine_rpm", ""),
                        "fuel_temp_C": item.get("decoded", {}).get("fuel_temp_C", ""),
                        "data": item.get("data", ""),
                        "raw_line": item.get("data", "") if item["source"] == "serial" else ""
                    }
                    writer.writerow(row)
                # print(f"[CSVWriter] Wrote data to CSV: {row.get('source')}")
            except queue.Empty:
                continue
            except Exception as e:
                print(f"[CSVWriter] ERROR writing to CSV: {e}")
        print("[CSVWriter] Stopped.")

    def stop(self):
        """Signals the thread to stop."""
        self.stop_flag.set()

# ========== Cloud Uploader ========== #
class CloudUploader(threading.Thread):
    """
    Uploads raw sensor data to a GCP Pub/Sub topic for cloud processing.
    """
    def __init__(self, input_queue, topic_id=PUBSUB_TOPIC, credentials_path=GCP_CREDENTIALS_PATH):
        super().__init__()
        self.input_queue = input_queue
        self.topic_id = topic_id
        self.stop_flag = threading.Event()
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = credentials_path
        self.project_id = self._extract_project_id(credentials_path)
        self.publisher = pubsub_v1.PublisherClient()
        self.topic_path = self.publisher.topic_path(self.project_id, self.topic_id)
        print("[CloudUploader] Initialized.")

    def _extract_project_id(self, cred_path):
        """Extracts the GCP project ID from the service account credentials file."""
        with open(cred_path) as f:
            key = json.load(f)
            return key["project_id"]

    def run(self):
        while not self.stop_flag.is_set() or not self.input_queue.empty():
            try:
                item = self.input_queue.get(timeout=0.1)
                # Construct the JSON message payload
                message = {
                    "timestamp": datetime.utcnow().isoformat(),
                    "source": item.get("source"),
                    "id": item.get("id", ""),
                    "pgn": item.get("pgn", ""),
                    "engine_coolant_temp_C": item.get("decoded", {}).get("engine_coolant_temp_C", ""),
                    "engine_rpm": item.get("decoded", {}).get("engine_rpm", ""),
                    "fuel_temp_C": item.get("decoded", {}).get("fuel_temp_C", ""),
                    "data": item.get("data", "")
                }
                # Convert to bytes and publish to Pub/Sub
                future = self.publisher.publish(self.topic_path, json.dumps(message).encode("utf-8"))
                # future.add_done_callback(lambda f: print(f"[CloudUploader] Published: {f.result()}"))
            except queue.Empty:
                continue
            except Exception as e:
                print(f"[CloudUploader] ERROR publishing: {e}")
        print("[CloudUploader] Stopped.")

    def stop(self):
        """Signals the thread to stop."""
        self.stop_flag.set()

# ========== Inference Runner ========== #
class InferenceRunner(threading.Thread):
    """
    Runs real-time TFLite model inference on incoming sensor data.
    Manages a rolling buffer for feature engineering and outputs results.
    """
    def __init__(self, input_queue, output_queue, rolling_window, model_dir, status_output_file):
        super().__init__()
        self.input_queue = input_queue
        self.output_queue = output_queue
        self.stop_flag = threading.Event()
        self.rolling_window = rolling_window
        self.model_dir = model_dir
        self.status_output_file = status_output_file
        
        # Initialize an empty DataFrame for the rolling buffer with the same
        # columns as the raw data before feature engineering for consistency.
        # These are the columns that will be directly populated from incoming sensor data.
        self.rolling_buffer_df = pd.DataFrame(columns=[
            "timestamp_ms", "engine_temp_C", "front_brake_temp_C", "rear_brake_temp_C", "alt_temp_C",
            "clutch_temp_C", "hydraulic_level_pct", "transmission_level", "cabin_acX", "cabin_acY",
            "cabin_acZ", "body_frame_abX", "body_frame_abY", "body_frame_abZ",
            "engine_coolant_temp_C", "engine_rpm", "fuel_temp_C"
        ])
        print("[InferenceRunner] Initialized.")

    def run(self):
        print("[InferenceRunner] Ready to run inference...")
        while not self.stop_flag.is_set() or not self.input_queue.empty():
            try:
                item = self.input_queue.get(timeout=0.1) # Get raw data item
                
                # Preprocess the single item using edge_preprocess module               
                
                parsed_values = {} # Parse raw data into a dictionary suitable for DataFrame row
                if item.get("source") == "serial":
                    # For serial, 'data' holds the raw line to be parsed
                    parsed_values = edge_preprocess.parse_raw_serial_data(item.get("data", ""))
                elif item.get("source") == "can":
                    # For CAN, 'decoded' holds the already parsed values
                    parsed_values = edge_preprocess.parse_raw_can_data(item.get("decoded", {}))
                else:
                    print(f"[InferenceRunner] Skipping unknown source for inference: {item.get('source')}")
                    continue

                # Ensure parsed_values has a timestamp_ms for the buffer
                parsed_values["timestamp_ms"] = int(time.time() * 1000)
                
                # Convert parsed_values to a DataFrame row and append to buffer
                current_row_df = pd.DataFrame([parsed_values])
                self.rolling_buffer_df = pd.concat([self.rolling_buffer_df, current_row_df], ignore_index=True)
                
                # Keep only the last 'rolling_window' samples
                if len(self.rolling_buffer_df) > self.rolling_window:
                    self.rolling_buffer_df = self.rolling_buffer_df.iloc[-self.rolling_window:].reset_index(drop=True)

                # Perform preprocessing and inference if the buffer has enough data for rolling features
                if len(self.rolling_buffer_df) >= self.rolling_window:                    
                    preprocessed_data = edge_preprocess.process_single_sample(item, self.rolling_buffer_df.copy())
                    
                    if preprocessed_data is not None:
                        # Run inference using edge_inference module
                        inference_results = edge_inference.perform_edge_inference(
                            preprocessed_data,
                            model_dir=self.model_dir # Pass model_dir for model loading
                        )
                        
                        # Add original timestamp and source to results for context
                        inference_results["original_timestamp"] = datetime.utcnow().isoformat()
                        inference_results["original_source"] = item.get("source")

                        self.output_queue.put(inference_results) # Put results into output queue
                        print(f"[InferenceRunner] Inference complete: {inference_results.get('is_anomaly')}, Class: {inference_results.get('fault_class')}")
                        
                        # Write inference results to local status file for GUI
                        try:
                            os.makedirs(os.path.dirname(self.status_output_file), exist_ok=True)
                            with open(self.status_output_file, "w") as f:
                                json.dump(inference_results, f, indent=4)
                        except Exception as write_e:
                            print(f"[InferenceRunner] ERROR writing GUI status file {self.status_output_file}: {write_e}")


                    else:
                        print("[InferenceRunner] Preprocessing returned None. Skipping inference.")
                else:
                    print(f"[InferenceRunner] Rolling buffer not full ({len(self.rolling_buffer_df)}/{self.rolling_window}). Skipping inference.")

            except queue.Empty:
                continue
            except Exception as e:
                print(f"[InferenceRunner] ERROR: {e}")

    def stop(self):
        """Signals the thread to stop."""
        self.stop_flag.set()

# ========== Main Execution ========== #
def main():
    """
    Main function to start and manage all data collection, processing, and upload threads.
    """
    print("[Main] Starting threads...")
    
    # Construct absolute path for combined_log.csv
    COMBINED_LOG_FILE_PATH = os.path.join(
        os.path.abspath(os.path.dirname(__file__)),
        "..", 
        SETTINGS['data_collection']['combined_log_file_path_relative'] # "data/combined_log.csv"
    )

    # Construct absolute path for GUI status file
    GUI_STATUS_FILE_PATH = os.path.join(
        os.path.abspath(os.path.dirname(__file__)),
        "..",
        SETTINGS['inference']['status_output_file_relative'] # "gui/latest_status.json"
    )
    
    
    threads = [
        SerialReader(SERIAL_PORT, BAUD_RATE, raw_sensor_data_queue),
        CANReader(CAN_INTERFACE, raw_sensor_data_queue),
        CSVWriter(csv_writer_queue, filename=COMBINED_LOG_FILE_PATH),
        CloudUploader(cloud_uploader_queue, PUBSUB_TOPIC, GCP_CREDENTIALS_PATH),
        InferenceRunner(
            inference_input_queue,
            inference_output_queue,
            rolling_window=SETTINGS['inference']['rolling_window'], # Assuming you add this to settings.json
            model_dir=SETTINGS['inference']['model_dir'],
            status_output_file=GUI_STATUS_FILE_PATH
        )
    ]

    for t in threads:
        t.start()

    print("[Main] All threads started. Dispatching data...")
    try:
        while True:
            # Main dispatcher loop: gets raw data and puts copies into consumer queues
            try:
                raw_item = raw_sensor_data_queue.get(timeout=1)
                
                # Put copies into respective queues
                csv_writer_queue.put(raw_item.copy())
                cloud_uploader_queue.put(raw_item.copy())
                inference_input_queue.put(raw_item.copy()) # For inference

                # Update GUI inference results
                if not inference_output_queue.empty():
                    inference_result = inference_output_queue.get()
                    print(f"[Main Dispatcher] Received inference result: {inference_result}")
                    # Example: write to a GUI status file
                    gui_status_path = os.path.join(os.path.dirname(__file__), "..", "gui", "latest_status.json")
                    os.makedirs(os.path.dirname(gui_status_path), exist_ok=True)
                    with open(gui_status_path, "w") as f:
                        json.dump(inference_result, f, indent=4)


            except queue.Empty:
                # No raw data yet, continue to check for stop signal
                pass
            
            time.sleep(0.01) # Small sleep to prevent busy-waiting

    except KeyboardInterrupt:
        print("\n[Main] Shutting down all threads...")
        for t in threads:
            t.stop() # Signal threads to stop

        for t in threads:
            t.join() # Wait for threads to complete
        print("[Main] All threads stopped gracefully.")


# Testing and Simulation
def simulate_data_loop_for_main_test():
    """
    Simulates sensor data and puts it into the raw_sensor_data_queue.
    Mimics the real data structure used by SerialReader and CANReader.
    """
    print("[simulate_data_loop_for_main_test] Starting simulated data generation...")
    try:
        while True:
            # Simulate serial data
            simulated_serial_item = {
                "source": "serial",
                "data": f"1000,{random.uniform(70, 100):.1f},{random.uniform(20, 40):.1f},{random.uniform(20, 40):.1f},{random.uniform(30, 50):.1f},{random.uniform(40, 60):.1f},{random.uniform(60, 90):.1f},{random.uniform(5, 15):.1f},{random.uniform(-0.5, 0.5):.1f},{random.uniform(-0.5, 0.5):.1f},{random.uniform(-0.5, 0.5):.1f},{random.uniform(0.5, 1.5):.1f},{random.uniform(1.5, 2.5):.1f},{random.uniform(2.5, 3.5):.1f}"
            }
            raw_sensor_data_queue.put(simulated_serial_item)
            # print(f"[simulate_data_loop_for_main_test] Simulated serial data put: {simulated_serial_item['data']}")

            # Simulate CAN data
            simulated_can_item = {
                "source": "can",
                "id": "sim-001",
                "pgn": 65262,
                "decoded": {
                    "engine_coolant_temp_C": random.uniform(70, 100),
                    "engine_rpm": random.uniform(500, 1500),
                    "fuel_temp_C": random.uniform(30, 50)
                },
                "data": "simulated_hex_data"
            }
            raw_sensor_data_queue.put(simulated_can_item)
            # print(f"[simulate_data_loop_for_main_test] Simulated CAN data put: {simulated_can_item['decoded']}")

            time.sleep(0.1) # Simulate faster data rate for testing
    except KeyboardInterrupt:
        print("[simulate_data_loop_for_main_test] Stopped by user")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Data collection script")
    parser.add_argument("--debug", action="store_true", help="Enable debug/test mode (simulate data)")
    args = parser.parse_args()

    if args.debug:
        # If debug is enabled, run the simulation loop instead of main
        # This will fill raw_sensor_data_queue, which main() will then consume and dispatch.
        # So, we call main() first, and then run the simulation in a separate thread.
        simulation_thread = threading.Thread(target=simulate_data_loop_for_main_test)
        simulation_thread.daemon = True # Allow main thread to exit even if simulation is running
        simulation_thread.start()
        main() # Start the main data collection and dispatch loop
    else:
        main()