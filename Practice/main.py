import os
import pandas as pd
import numpy as np
import tensorflow as tf
import joblib # For saving preprocessor artifacts

import Practice.config as config
from Practice.data_loader import load_data
from Practice.data_validator import validate_data_schema, check_missing_values
from Practice.preprocessor import preprocess_data, split_data
from model_builder import build_ann_model
from Practice.trainer import train_model
from Practice.evaluator import generate_evaluation_dashboard


def set_seeds(seed_value):
    np.random.seed(seed_value)
    tf.random.set_seed(seed_value)
    os.environ['PYTHONHASHSEED'] = str(seed_value)
    # Potentially: random.seed(seed_value) if using Python's random module directly

def run_pipeline():
    print("Starting Churn Prediction Pipeline...")
    set_seeds(config.RANDOM_SEED)

    # 1. Load Data
    print("\n--- 1. Loading Data ---")
    raw_df = load_data(config.RAW_DATA_FILE)
    print(f"Raw data loaded. Shape: {raw_df.shape}")

    # 2. Validate Data
    print("\n--- 2. Validating Data ---")
    all_expected_features = ([config.CUSTOMER_ID_COLUMN] + config.NUMERICAL_FEATURES + 
                             config.CATEGORICAL_FEATURES_ONEHOT + config.BINARY_FEATURES_MAP_YES_NO + 
                             [config.SENIOR_CITIZEN_COLUMN] + [config.TARGET_COLUMN])
    expected_cols_list = sorted(list(set(all_expected_features)))
    
    is_valid, schema_issues = validate_data_schema(
        raw_df, 
        expected_columns=expected_cols_list, 
        numerical_features=config.NUMERICAL_FEATURES,
        customer_id_col=config.CUSTOMER_ID_COLUMN,
        target_col=config.TARGET_COLUMN
    )
    if not is_valid:
        print(f"Schema validation failed: {schema_issues}. Exiting pipeline.")
        return
    check_missing_values(raw_df) # Informational

    # 3. Preprocess Data (Training mode to get artifacts)
    print("\n--- 3. Preprocessing Data ---")
    X_processed, y_processed, fit_artifacts = preprocess_data(raw_df, is_train=True)
    
    # Save the fit_artifacts (scaler, imputation values, feature columns)
    preprocessor_artifact_path = os.path.join(config.MODEL_DIR, 'preprocessor_artifacts.joblib')
    joblib.dump(fit_artifacts, preprocessor_artifact_path)
    print(f"Preprocessor artifacts saved to {preprocessor_artifact_path}")
    
    # Combine X and y for consistent splitting
    processed_df_for_split = pd.concat([X_processed, y_processed.rename(config.TARGET_COLUMN)], axis=1)


    # 4. Split Data
    print("\n--- 4. Splitting Data ---")
    X_train, y_train, X_val, y_val, X_test, y_test = split_data(processed_df_for_split)
    
    if X_train.empty:
        print("Training data is empty. Cannot proceed with model training. Check data and split ratios.")
        return

    # 5. Build Model
    print("\n--- 5. Building ANN Model ---")
    input_shape = X_train.shape[1]
    model = build_ann_model(
        input_shape=input_shape,
        learning_rate=config.LEARNING_RATE,
        layer_config=config.ANN_LAYER_CONFIG,
        output_activation=config.OUTPUT_LAYER_ACTIVATION,
        loss_function=config.LOSS_FUNCTION,
        metrics_list=config.METRICS
    )
    model.summary()

    # 6. Train Model
    print("\n--- 6. Training Model ---")
    trained_model, history = train_model(
        model, X_train, y_train, X_val, y_val,
        epochs=config.EPOCHS,
        batch_size=config.BATCH_SIZE,
        early_stopping_patience=config.EARLY_STOPPING_PATIENCE,
        use_class_weights=True # Enable class weights for imbalanced data
    )
    
    # Save the trained model
    model_save_path = os.path.join(config.MODEL_DIR, config.TRAINED_MODEL_NAME)
    trained_model.save(model_save_path)
    print(f"Trained model saved to {model_save_path}")

    # 7. Evaluate Model & Generate Dashboard
    print("\n--- 7. Evaluating Model & Generating Dashboard ---")
    if not X_test.empty and not y_test.empty:
        generate_evaluation_dashboard(trained_model, history, X_test, y_test)
    else:
        print("Test data is empty. Skipping evaluation dashboard generation.")

    print("\nChurn Prediction Pipeline finished successfully.")

if __name__ == '__main__':
    # Before running, ensure:
    # 1. 'churn.csv' is inside a 'data' subdirectory next to your Python files.
    #    (e.g., project_root/data/churn.csv, project_root/main.py, project_root/config.py etc.)
    # 2. All required libraries (pandas, numpy, tensorflow, scikit-learn, joblib, matplotlib, seaborn) are installed.
    run_pipeline()