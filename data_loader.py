import pandas as pd
import os
from config import RAW_DATA_FILE # Assuming churn.csv is in DATA_DIR specified in config

def load_data(file_path=RAW_DATA_FILE):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Data file not found at {file_path}. Please ensure 'churn.csv' is in the '{os.path.dirname(file_path)}' directory.")
    try:
        df = pd.read_csv(file_path)
        return df
    except Exception as e:
        raise Exception(f"Error loading data from {file_path}: {e}")

if __name__ == '__main__':
    # Example usage:
    try:
        data = load_data()
        print("Data loaded successfully:")
        print(data.head())
        print(f"Shape: {data.shape}")
    except Exception as e:
        print(e)