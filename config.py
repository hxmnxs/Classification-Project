import os

# --- Paths ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
OUTPUT_DIR = os.path.join(BASE_DIR, 'output')
MODEL_DIR = os.path.join(OUTPUT_DIR, 'models')
METRICS_DIR = os.path.join(OUTPUT_DIR, 'metrics')
PLOTS_DIR = os.path.join(OUTPUT_DIR, 'plots')

RAW_DATA_FILE = os.path.join(DATA_DIR, 'churn.csv') # Ensure 'churn.csv' is in a 'data' subdirectory

# --- Data ---
TARGET_COLUMN = 'Churn'
CUSTOMER_ID_COLUMN = 'customerID'

# Based on notebook EDA and common sense for this dataset
# Note: SeniorCitizen is initially 0/1, will be mapped if needed, or treated as numeric/categorical
# For simplicity in this setup, features that are 'Yes'/'No' or specific categories
# will be listed and handled.

NUMERICAL_FEATURES = ['tenure', 'MonthlyCharges', 'TotalCharges']

# Features to be one-hot encoded
CATEGORICAL_FEATURES_ONEHOT = [
    'gender', # Initially Male/Female, will be mapped to 0/1 then can be one-hot if preferred, or left as binary
    'MultipleLines',
    'InternetService',
    'OnlineSecurity',
    'OnlineBackup',
    'DeviceProtection',
    'TechSupport',
    'StreamingTV',
    'StreamingMovies',
    'Contract',
    'PaymentMethod'
]

# Features to be mapped directly to 0/1
BINARY_FEATURES_MAP_YES_NO = [
    'Partner',
    'Dependents',
    'PhoneService',
    'PaperlessBilling'
    # 'SeniorCitizen' is already 0/1 in raw data, but EDA converted to Yes/No.
    # If it's 0/1, it can be directly used or mapped for consistency if other binary features are string.
    # For this example, we'll assume it's 0/1. If it's string 'Yes'/'No', add to this list.
]
# SeniorCitizen is special: it's 0/1 but sometimes EDA treats it as categorical "No"/"Yes"
# We will handle it by ensuring it's int 0/1.
SENIOR_CITIZEN_COLUMN = 'SeniorCitizen'


# --- Preprocessing ---
# For 'TotalCharges' imputation strategy
IMPUTATION_STRATEGY = 'median' # or 'mean'

# --- Model Training ---
TEST_SPLIT_SIZE = 0.2
VALIDATION_SPLIT_SIZE = 0.1 # Percentage of training data to use for validation
RANDOM_SEED = 42

# ANN Parameters
EPOCHS = 100
BATCH_SIZE = 32
LEARNING_RATE = 0.001
EARLY_STOPPING_PATIENCE = 10

# Define ANN layer configuration
# Example: [{'units': 64, 'activation': 'relu', 'dropout': 0.3}, {'units': 32, 'activation': 'relu', 'dropout': 0.3}]
ANN_LAYER_CONFIG = [
    {'units': 128, 'activation': 'relu', 'dropout_rate': 0.3}, # Added dropout_rate key
    {'units': 64, 'activation': 'relu', 'dropout_rate': 0.3},
    {'units': 32, 'activation': 'relu', 'dropout_rate': 0.2}
]
OUTPUT_LAYER_ACTIVATION = 'sigmoid'
LOSS_FUNCTION = 'binary_crossentropy'
METRICS = ['accuracy'] # Keras metrics

# --- Output Files ---
TRAINED_MODEL_NAME = 'ann_churn_model.keras' # Use .keras for modern Keras
# For plots and reports
CONFUSION_MATRIX_FILE = os.path.join(PLOTS_DIR, 'confusion_matrix.png')
ROC_CURVE_FILE = os.path.join(PLOTS_DIR, 'roc_curve.png')
PR_CURVE_FILE = os.path.join(PLOTS_DIR, 'pr_curve.png')
TRAINING_HISTORY_FILE = os.path.join(PLOTS_DIR, 'training_history.png')
CLASSIFICATION_REPORT_FILE = os.path.join(METRICS_DIR, 'classification_report.txt')
MODEL_SUMMARY_FILE = os.path.join(METRICS_DIR, 'model_summary.txt')

# Create directories if they don't exist
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(METRICS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)