# main.py for data_preprocessing_component
import os
import argparse
import subprocess

def data_preprocessing_component_main(
    raw_data_path: str,
    train_data_path: str,
    val_data_path: str, 
    test_data_path: str,
    scaler_output_path: str,
):
    """
    Executes the preprocessing and feature engineering script.
    The script processes raw data and splits it into training, validation, and testing sets,
    and saves the fitted scaler.
    """
    print(f"Starting preprocessing for data from {raw_data_path}")
    print(f"Train output: {train_data_path}")
    print(f"Validation output: {val_data_path}")
    print(f"Test output: {test_data_path}")
    print(f"Scaler output: {scaler_output_path}")

    
    # Executes the external 'preprocess.py' script.
    # The script is expected to be located at /end_to_end_automated_pipeline/core_utilities/preprocess.py
    # within the Docker container.
    subprocess.run([
        "python", "/end_to_end_automated_pipeline/core_utilities/preprocess.py",
        "--input", raw_data_path,
        "--train_output", train_data_path,
        "--val_output", val_data_path,
        "--test_output", test_data_path,
        "--scaler_output", scaler_output_path
    ], check=True) # Ensures the subprocess completes successfully.

    print(f"Preprocessing complete. Data splits and scaler saved.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Data Preprocessing Component")
    parser.add_argument("--raw_data_path", type=str, required=True, help="Local path to the raw data CSV.")
    parser.add_argument("--train_data_path", type=str, required=True, help="Local path for the preprocessed training data CSV.")
    parser.add_argument("--val_data_path", type=str, required=True, help="Local path for the preprocessed validation data CSV.")
    parser.add_argument("--test_data_path", type=str, required=True, help="Local path for the preprocessed testing data CSV.")
    parser.add_argument("--scaler_output_path", type=str, required=True, help="Local path for the fitted MinMaxScaler.")
    
    args = parser.parse_args()
    data_preprocessing_component_main(
        raw_data_path=args.raw_data_path,
        train_data_path=args.train_data_path,
        val_data_path=args.val_data_path,
        test_data_path=args.test_data_path,
        scaler_output_path=args.scaler_output_path
    )