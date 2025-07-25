# Data Schema Definition
This document defines the structure and content of the data used throughout the Tractor Health Monitoring MLOps pipeline, from raw sensor readings at the edge to processed features in the cloud. Consistent data schema is critical for seamless data flow and model compatibility.

## 1. Raw Data Schema (Edge Device & GCS)
Raw data is collected from Arduino-connected sensors and the CAN bus, then combined and stored locally before being uploaded to GCS.

- **Source:** edge/data_collector/data_collect.py

- **Local Storage:** edge/data/combined_log.csv

- **Cloud Storage:** gs://<raw_data_bucket>/raw/log_<timestamp>.csv

The combined_log.csv file captures data from two primary sources: serial (Arduino) and CAN bus. It attempts to normalize these into a common set of columns, with NaN or default values where data is not applicable to a specific source.

***Expected Columns:***

| Column Name              | Type    | Description                                                                 | Source (Typical) | Notes                                                                   |
| :----------------------- | :------ | :-------------------------------------------------------------------------- | :--------------- | :---------------------------------------------------------------------- |
| `timestamp`              | String  | UTC timestamp of data collection (e.g., `YYYY-MM-DDTHH-MM-SS`).             | Combined         | Used for ordering and partitioning.                                     |
| `source`                 | String  | Origin of the data: "serial" (Arduino) or "can" (CAN Bus).                  | Combined         | Categorical identifier.                                                 |
| `id`                     | String  | Identifier for the specific message/sensor (e.g., CAN ID, Arduino sensor ID). | Combined         | Can be `NaN` or a placeholder if not applicable.                        |
| `raw_line`               | String  | The original raw string received from Arduino serial or CAN bus.            | Combined         | Useful for debugging, dropped during preprocessing.                     |
| `pgn`                    | Integer | Parameter Group Number for CAN bus messages.                                | CAN              | `NaN` for serial data, dropped during preprocessing.                    |
| `data`                   | String  | Raw data payload string (e.g., comma-separated values from Arduino).        | Serial           | `NaN` for CAN data, dropped during preprocessing.                       |
| `timestamp_ms`           | Integer | Millisecond timestamp from Arduino.                                         | Serial           | `NaN` for CAN data.                                                     |
| `engine_temp_C`          | Float   | Engine temperature in Celsius.                                              | Serial           |                                                                         |
| `front_brake_temp_C`     | Float   | Front brake temperature in Celsius.                                         | Serial           |                                                                         |
| `rear_brake_temp_C`      | Float   | Rear brake temperature in Celsius.                                          | Serial           |                                                                         |
| `alternator_temp_C`      | Float   | Alternator temperature in Celsius.                                          | Serial           | (Note: Original code had `alt_temp_C`, adjust if necessary)            |
| `clutch_temp_C`          | Float   | Clutch temperature in Celsius.                                              | Serial           |                                                                         |
| `hydraulic_level_pct`    | Float   | Hydraulic oil level percentage (0-100).                                     | Serial           | (Note: Original code had `hydraulic_level_pct`, adjust if necessary)    |
| `transmission_level_pct` | Float   | Transmission oil level percentage (0-100).                                  | Serial           | (Note: Original code had `transmission_level`, adjust if necessary)     |
| `cabin_vibration_x`      | Float   | Cabin vibration (X-axis).                                                   | Serial           | (Note: Original code had `cabin_acX`, adjust if necessary)              |
| `cabin_vibration_y`      | Float   | Cabin vibration (Y-axis).                                                   | Serial           | (Note: Original code had `cabin_acY`, adjust if necessary)              |
| `cabin_vibration_z`      | Float   | Cabin vibration (Z-axis).                                                   | Serial           | (Note: Original code had `cabin_acZ`, adjust if necessary)              |
| `frame_vibration_x`      | Float   | Frame vibration (X-axis).                                                   | Serial           | (Note: Original code had `body_frame_abX`, adjust if necessary)         |
| `frame_vibration_y`      | Float   | Frame vibration (Y-axis).                                                   | Serial           | (Note: Original code had `body_frame_abY`, adjust if necessary)         |
| `frame_vibration_z`      | Float   | Frame vibration (Z-axis).                                                   | Serial           | (Note: Original code had `body_frame_abZ`, adjust if necessary)         |
| `engine_coolant_temp_C`  | Float   | Engine coolant temperature in Celsius.                                      | CAN              | `NaN` for serial data.                                                  |
| `engine_rpm`             | Float   | Engine Revolutions Per Minute.                                              | CAN              | `NaN` for serial data.                                                  |
| `fuel_temp_C`            | Float   | Fuel temperature in Celsius.                                                | CAN              | `NaN` for serial data.                                                  |
| `clutch_slip_ratio`      | Float   | Clutch slip ratio (0-1, or percentage).                                     | CAN/Derived      | Used for specific fault labeling, might be derived or direct CAN.       |




**Example combined_log.csv Row (Conceptual):**
``` csv
timestamp,source,id,raw_line,pgn,data,timestamp_ms,engine_temp_C,front_brake_temp_C,rear_brake_temp_C,alternator_temp_C,clutch_temp_C,hydraulic_level_pct,transmission_level_pct,cabin_vibration_x,cabin_vibration_y,cabin_vibration_z,frame_vibration_x,frame_vibration_y,frame_vibration_z,engine_coolant_temp_C,engine_rpm,fuel_temp_C,clutch_slip_ratio
2025-07-21T10-30-00,serial,TRACTOR-001,12345,25.5,80.1,75.2,60.0,90.5,50.3,1.2,0.8,0.5,0.1,0.2,0.3,,,,,,
2025-07-21T10-30-01,can,0x18F00400,CAN_MSG_DATA_HEX,65284,,,,,,,,,,,,,,85.0,1800.0,45.0,0.01
```
## 2. Processed Data Schema (Cloud & Edge Inference)
After the data_preprocessing component, raw data is transformed into a feature-rich dataset. This schema is critical for both cloud model training and edge inference.

- **Source:** cloud/end_to_end_automated_pipeline/core_utilities/preprocess.py

- **Cloud Artifacts:** gs://<pipeline_artifacts_bucket>/<run_id>/train_data.csv, -val_data.csv, test_data.csv

- **Edge Usage:** Input to edge_inference.py (after edge_preprocess.py generates these features).

The processed data includes original sensor readings (scaled), and newly engineered features (rolling means, standard deviations, and deltas). The fault_label is also added during this step.

**Columns (Feature Columns + fault_label):**

The feature_columns are explicitly defined in cloud_settings.yaml (and edge/config/settings.json for edge consistency). The list includes:

- Original Sensor Columns (scaled):

  - engine_temp_C

  - front_brake_temp_C

  - rear_brake_temp_C

  - alternator_temp_C

  - clutch_temp_C

  - hydraulic_level_pct

  - transmission_level_pct

  - cabin_vibration_x

  - cabin_vibration_y

  - cabin_vibration_z

  - frame_vibration_x

  - frame_vibration_y

  - frame_vibration_z

  - engine_coolant_temp_C

  - engine_rpm

  - fuel_temp_C


- Engineered Features (derived from above, scaled):

  - <original_column>_roll_mean_5 (e.g., engine_temp_C_roll_mean_5)

  - <original_column>_roll_std_5 (e.g., engine_temp_C_roll_std_5)

  - <original_column>_delta (e.g., engine_temp_C_delta)

  - (This pattern applies to all relevant original sensor columns.)

- Label Column:

  - fault_label (Integer): The classification label for the tractor's health status.

**fault_label Mapping:**

| Label | Description                   |
| :---- | :---------------------------- |
| 0     | Healthy                       |
| 1     | Overheated Front Brakes       |
| 2     | Overheated Rear Brakes        |
| 3     | Low Hydraulic Oil level       |
| 4     | Too high Hydraulic Oil level  |
| 5     | Low Transmission Oil level    |
| 6     | Too high Transmission Oil level |
| 7     | Vibration Issue               |
| 8     | Clutch Failure                |
| 9     | Transmission Overheat         |
| 10    | Engine Overheat               |
| 11    | Alternator Overheat           |
| 12    | Clutch Overheat               |
| 13    | Engine Coolant Overheat       |
| 14    | Engine RPM Failure            |
| 15    | Fuel Overheat                 |

**Example Processed Data Row (Conceptual, after scaling):**
``` csv
engine_temp_C,front_brake_temp_C,...,engine_temp_C_roll_mean_5,...,fault_label
0.52,0.78,...,0.55,...,0
0.61,0.85,...,0.60,...,1
```
## 3. BigQuery Table Schema (<project_id>.tractor_health_data.sensor_can_logs)
The data_ingestion component uses BigQuery's autodetect=True feature, so the schema is inferred from the incoming CSVs. However, it will generally align with the Raw Data Schema described above.

**Key Considerations for BigQuery:**

- **Data Types:** BigQuery will infer types like STRING, INTEGER, FLOAT, TIMESTAMP.

- **Partitioning/Clustering:** For large datasets, consider partitioning by timestamp column and clustering by id or source for optimized query performance. This is typically configured during table creation or managed by the ingestion process.

- **Nullability:** Inferred based on data, but can be explicitly defined.