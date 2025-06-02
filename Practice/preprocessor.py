import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

from Practice.config import (TARGET_COLUMN, CUSTOMER_ID_COLUMN,
                    NUMERICAL_FEATURES, CATEGORICAL_FEATURES_ONEHOT,
                    BINARY_FEATURES_MAP_YES_NO, SENIOR_CITIZEN_COLUMN,
                    RANDOM_SEED, TEST_SPLIT_SIZE, VALIDATION_SPLIT_SIZE)

def clean_total_charges(df, column_name='TotalCharges'):
    df[column_name] = pd.to_numeric(df[column_name], errors='coerce')
    return df

def impute_missing(df, column_name, strategy='median', fill_value=None):
    if strategy == 'median':
        fill_value = df[column_name].median() if fill_value is None else fill_value
    elif strategy == 'mean':
        fill_value = df[column_name].mean() if fill_value is None else fill_value
    elif strategy == 'constant':
        if fill_value is None:
            raise ValueError("Fill value must be provided for 'constant' strategy.")
    else: # Specific value or other
        if fill_value is None:
            raise ValueError("Fill value must be provided if no strategy like median/mean.")
            
    df[column_name] = df[column_name].fillna(fill_value)
    return df, fill_value

def map_binary_features(df, feature_list):
    for feature in feature_list:
        if feature in df.columns:
            df[feature] = df[feature].map({'Yes': 1, 'No': 0}).astype(int)
    return df

def map_gender_feature(df, feature_name='gender'):
    if feature_name in df.columns:
        df[feature_name] = df[feature_name].map({'Male': 1, 'Female': 0}).astype(int)
    return df

def map_target_column(df, target_column_name):
    if target_column_name in df.columns:
        df[target_column_name] = df[target_column_name].map({'Yes': 1, 'No': 0}).astype(int)
    return df
    
def encode_categorical_features(df, feature_list):
    df = pd.get_dummies(df, columns=feature_list, drop_first=True, dummy_na=False) # Ensure no NaN columns created by get_dummies
    return df

def scale_numerical_features(df, feature_list, scaler_obj=None):
    if not scaler_obj:
        scaler_obj = MinMaxScaler()
        df[feature_list] = scaler_obj.fit_transform(df[feature_list])
    else:
        df[feature_list] = scaler_obj.transform(df[feature_list])
    return df, scaler_obj

def preprocess_data(df_raw, is_train=True, fit_artifacts=None):
    df = df_raw.copy()

    if CUSTOMER_ID_COLUMN in df.columns:
        df = df.drop(columns=[CUSTOMER_ID_COLUMN])

    df = clean_total_charges(df, 'TotalCharges')
    
    # Target column mapping
    df = map_target_column(df, TARGET_COLUMN)

    # Handle SeniorCitizen: ensure it's int (0 or 1)
    if SENIOR_CITIZEN_COLUMN in df.columns:
        # If it was read as float or object due to NaNs elsewhere, convert
        df[SENIOR_CITIZEN_COLUMN] = pd.to_numeric(df[SENIOR_CITIZEN_COLUMN], errors='coerce').fillna(0).astype(int)


    # Imputation, scaling, and encoding artifacts
    imputation_values = {}
    scaler = None

    if is_train:
        # Impute 'TotalCharges'
        df, tc_fill_value = impute_missing(df, 'TotalCharges', strategy='median')
        imputation_values['TotalCharges'] = tc_fill_value
        
        # Binary mappings
        df = map_binary_features(df, BINARY_FEATURES_MAP_YES_NO)
        df = map_gender_feature(df) # Specific for gender

        # One-hot encoding (after binary mapping for consistency if some were strings)
        # Ensure all categorical features intended for one-hot are strings before pd.get_dummies
        for col in CATEGORICAL_FEATURES_ONEHOT:
            if col in df.columns:
                df[col] = df[col].astype(str)
        df = encode_categorical_features(df, CATEGORICAL_FEATURES_ONEHOT)
        
        # Scaling
        # Ensure numerical features are float for scaler
        for col in NUMERICAL_FEATURES:
            if col in df.columns:
                df[col] = df[col].astype(float)
        df, scaler = scale_numerical_features(df, NUMERICAL_FEATURES)
        
        fit_artifacts = {'imputation_values': imputation_values, 'scaler': scaler, 'feature_columns': df.drop(TARGET_COLUMN, axis=1).columns.tolist()}
    else:
        if not fit_artifacts:
            raise ValueError("fit_artifacts (scaler, imputation_values, feature_columns) must be provided for non-training data.")
        
        imputation_values = fit_artifacts['imputation_values']
        scaler = fit_artifacts['scaler']
        
        # Impute 'TotalCharges'
        df, _ = impute_missing(df, 'TotalCharges', fill_value=imputation_values.get('TotalCharges'))
        
        # Binary mappings
        df = map_binary_features(df, BINARY_FEATURES_MAP_YES_NO)
        df = map_gender_feature(df)

        # One-hot encoding
        for col in CATEGORICAL_FEATURES_ONEHOT:
             if col in df.columns:
                df[col] = df[col].astype(str)
        df = encode_categorical_features(df, CATEGORICAL_FEATURES_ONEHOT)
        
        # Scaling
        for col in NUMERICAL_FEATURES:
            if col in df.columns:
                df[col] = df[col].astype(float)
        df, _ = scale_numerical_features(df, NUMERICAL_FEATURES, scaler_obj=scaler)

        # Align columns with training data (add missing, remove extra)
        train_cols = fit_artifacts['feature_columns']
        current_cols_X = df.drop(TARGET_COLUMN, axis=1).columns
        
        # Add missing columns (that were in train, not in current) - fill with 0
        for col in train_cols:
            if col not in current_cols_X:
                df[col] = 0
        
        # Ensure order is the same and drop columns not in train_cols from X features
        # Preserve target column if it exists
        y_temp = None
        if TARGET_COLUMN in df.columns:
            y_temp = df[TARGET_COLUMN]
            df_X = df[train_cols]
            df = pd.concat([df_X, y_temp], axis=1)
        else: # If target is not there (e.g. pure inference data)
            df = df[train_cols]


    X = df.drop(TARGET_COLUMN, axis=1)
    y = df[TARGET_COLUMN]
    
    return X, y, fit_artifacts


def split_data(df_processed):
    X = df_processed.drop(columns=[TARGET_COLUMN])
    y = df_processed[TARGET_COLUMN]

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=TEST_SPLIT_SIZE, random_state=RANDOM_SEED, stratify=y
    )
    
    # Calculate validation size relative to the temporary set to achieve overall desired val size from original data
    # val_size_abs = VALIDATION_SPLIT_SIZE (original)
    # test_size_abs = TEST_SPLIT_SIZE (original)
    # temp_size_abs = TEST_SPLIT_SIZE (original)
    # desired_val_in_temp = VALIDATION_SPLIT_SIZE / TEST_SPLIT_SIZE (if VALIDATION_SPLIT_SIZE < TEST_SPLIT_SIZE)
    # This logic is tricky. A simpler way: split train into train and validation
    
    if X_temp.shape[0] < 2 or len(np.unique(y_temp)) < 2: # If temp set is too small for stratified split
        # Handle case with very small X_temp or single class in y_temp for validation split
        # This might happen if initial dataset is tiny. For now, we'll proceed, but production code might log warning or adjust.
        # If VALIDATION_SPLIT_SIZE is 0, then X_val, y_val will be empty.
        if VALIDATION_SPLIT_SIZE > 0 and X_temp.shape[0] >=2 and len(np.unique(y_temp)) >=2 :
             X_val, X_test, y_val, y_test = train_test_split(
                X_temp, y_temp, test_size= (TEST_SPLIT_SIZE-VALIDATION_SPLIT_SIZE)/TEST_SPLIT_SIZE if TEST_SPLIT_SIZE > VALIDATION_SPLIT_SIZE else 0.5, # Adjust this ratio carefully
                random_state=RANDOM_SEED, stratify=y_temp
            )
        elif VALIDATION_SPLIT_SIZE > 0 and X_temp.shape[0] > 0: # Not enough for stratify, but can still split
             X_val, X_test, y_val, y_test = train_test_split(
                X_temp, y_temp, test_size= (TEST_SPLIT_SIZE-VALIDATION_SPLIT_SIZE)/TEST_SPLIT_SIZE if TEST_SPLIT_SIZE > VALIDATION_SPLIT_SIZE else 0.5,
                random_state=RANDOM_SEED # No stratify
            )
        else: # No validation set or X_temp is empty
            X_val, y_val = pd.DataFrame(), pd.Series(dtype='int64')
            X_test, y_test = X_temp, y_temp

    else: # Standard case
         X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size= (TEST_SPLIT_SIZE-VALIDATION_SPLIT_SIZE)/TEST_SPLIT_SIZE if TEST_SPLIT_SIZE > VALIDATION_SPLIT_SIZE else 0.5, # test_size here is proportion of X_temp
            random_state=RANDOM_SEED, stratify=y_temp
        )
        
    # A more common way is to split train further:
    # X_train_full, X_test, y_train_full, y_test = train_test_split(X,y, test_size=TEST_SPLIT_SIZE, stratify=y, random_state=RANDOM_SEED)
    # X_train, X_val, y_train, y_val = train_test_split(X_train_full,y_train_full, test_size=VALIDATION_SPLIT_SIZE / (1-TEST_SPLIT_SIZE), stratify=y_train_full, random_state=RANDOM_SEED)
    # This split logic needs to be very careful.
    # For simplicity and robustness:
    # 1. Split into Train + Test
    # 2. Split Train into Train_final + Validation

    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X, y, test_size=TEST_SPLIT_SIZE, random_state=RANDOM_SEED, stratify=y
    )

    if VALIDATION_SPLIT_SIZE > 0 and X_train_full.shape[0] > 0:
        # Calculate actual fraction for validation from the training set
        val_frac = VALIDATION_SPLIT_SIZE / (1.0 - TEST_SPLIT_SIZE)
        if val_frac >= 1.0 or val_frac <= 0.0: # Safety check
            if X_train_full.shape[0] > 1 and len(np.unique(y_train_full)) > 1: # Can make a tiny val set
                 val_frac = 0.1 # Default to a small validation set
            else:
                 val_frac = 0 # No validation set possible

        if val_frac > 0 and X_train_full.shape[0] > 1 and len(np.unique(y_train_full)) > 1 :
            X_train, X_val, y_train, y_val = train_test_split(
                X_train_full, y_train_full, test_size=val_frac, random_state=RANDOM_SEED, stratify=y_train_full
            )
        else: # Not enough data for validation split or val_frac is 0
            X_train, y_train = X_train_full, y_train_full
            X_val, y_val = pd.DataFrame(columns=X_train.columns), pd.Series(dtype='int64')
    else: # No validation split needed or possible
        X_train, y_train = X_train_full, y_train_full
        X_val, y_val = pd.DataFrame(columns=X_train.columns), pd.Series(dtype='int64')


    print(f"Train set: X_train shape {X_train.shape}, y_train shape {y_train.shape}")
    print(f"Validation set: X_val shape {X_val.shape}, y_val shape {y_val.shape}")
    print(f"Test set: X_test shape {X_test.shape}, y_test shape {y_test.shape}")
    
    return X_train, y_train, X_val, y_val, X_test, y_test


if __name__ == '__main__':
    from Practice.data_loader import load_data
    raw_df = load_data() # Assumes churn.csv is in ./data/
    
    # Test preprocessing in training mode
    X_processed, y_processed, artifacts = preprocess_data(raw_df, is_train=True)
    print("Processed X (train mode):")
    print(X_processed.head())
    print(f"X_processed shape: {X_processed.shape}")
    print("Processed y (train mode):")
    print(y_processed.head())
    print(f"Scaler: {artifacts['scaler']}")
    print(f"Imputation Values: {artifacts['imputation_values']}")
    print(f"Feature columns: {artifacts['feature_columns']}")

    # Create a combined DataFrame for splitting
    processed_df_for_split = pd.concat([X_processed, y_processed], axis=1)

    # Test data splitting
    X_train, y_train, X_val, y_val, X_test, y_test = split_data(processed_df_for_split)

    # Test preprocessing in non-training (inference/test) mode
    # Simulate a new batch of raw data (e.g., the first 10 rows of original)
    sample_raw_test_df = raw_df.head(10).copy()
    X_new_processed, y_new_processed, _ = preprocess_data(sample_raw_test_df, is_train=False, fit_artifacts=artifacts)
    print("\nProcessed X (test mode using sample_raw_test_df):")
    print(X_new_processed.head())
    print(f"X_new_processed shape: {X_new_processed.shape}")
    
    # Verify column alignment
    if not X_new_processed.columns.equals(X_train.columns):
        print("Error: Columns in new processed data do not match training data columns.")
        print(f"New data columns: {X_new_processed.columns}")
        print(f"Train data columns: {X_train.columns}")