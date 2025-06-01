import pandas as pd
import numpy as np

def validate_data_schema(df, expected_columns, numerical_features, customer_id_col, target_col):
    issues = []
    
    # Check for presence of all expected columns
    missing_expected_cols = [col for col in expected_columns if col not in df.columns]
    if missing_expected_cols:
        issues.append(f"Missing expected columns: {', '.join(missing_expected_cols)}")

    # Check for ID and Target column
    if customer_id_col not in df.columns:
        issues.append(f"Customer ID column '{customer_id_col}' not found.")
    if target_col not in df.columns:
        issues.append(f"Target column '{target_col}' not found.")

    # Check numerical features (can they be made numeric?)
    for col in numerical_features:
        if col in df.columns:
            try:
                # Attempt to convert to numeric, coercing errors. If all are NaN, it's an issue.
                # A more robust check might be needed depending on acceptable NaN levels.
                pd.to_numeric(df[col], errors='coerce') 
            except Exception as e:
                issues.append(f"Column '{col}' (expected numerical) cannot be converted to numeric: {e}")
        else:
            if col not in missing_expected_cols: # Avoid double reporting if already missing
                 issues.append(f"Numerical column '{col}' not found in DataFrame.")


    if issues:
        print("Data Schema Validation Issues Found:")
        for issue in issues:
            print(f"- {issue}")
        return False, issues
    
    print("Data schema validation successful.")
    return True, []

def check_missing_values(df):
    missing_summary = df.isnull().sum()
    missing_summary = missing_summary[missing_summary > 0]
    if not missing_summary.empty:
        print("Missing values found:")
        print(missing_summary)
        return missing_summary
    print("No missing values found (or will be handled by preprocessor).")
    return pd.Series(dtype='int64')


if __name__ == '__main__':
    # This is a mock example, replace with actual data loading for testing
    data = {
        'customerID': ['1', '2', '3'],
        'gender': ['Male', 'Female', 'Male'],
        'tenure': [10, 20, ' '], # Intentionally problematic
        'MonthlyCharges': [100, 200, 300],
        'TotalCharges': [1000, 4000, ' '],
        'Churn': ['No', 'Yes', 'No'],
        'Partner': ['Yes', 'No', 'Yes'],
        'Dependents': ['No', 'No', 'No'],
        'PhoneService': ['Yes', 'Yes', 'No'],
        'PaperlessBilling': ['Yes', 'No', 'Yes'],
        'SeniorCitizen': [0, 1, 0],
        'MultipleLines': ['No', 'Yes', 'No phone service'],
        'InternetService': ['DSL', 'Fiber optic', 'No'],
        'OnlineSecurity': ['Yes', 'No', 'No internet service'],
        'OnlineBackup': ['No', 'Yes', 'No internet service'],
        'DeviceProtection': ['No', 'No', 'Yes'],
        'TechSupport': ['No', 'No', 'Yes'],
        'StreamingTV': ['No', 'Yes', 'No'],
        'StreamingMovies': ['No', 'Yes', 'No'],
        'Contract': ['Month-to-month', 'One year', 'Two year'],
        'PaymentMethod': ['Electronic check', 'Mailed check', 'Bank transfer (automatic)']
    }
    sample_df = pd.DataFrame(data)

    from config import (TARGET_COLUMN, CUSTOMER_ID_COLUMN, 
                        NUMERICAL_FEATURES, CATEGORICAL_FEATURES_ONEHOT, 
                        BINARY_FEATURES_MAP_YES_NO, SENIOR_CITIZEN_COLUMN)
    
    all_features = ([CUSTOMER_ID_COLUMN] + NUMERICAL_FEATURES + CATEGORICAL_FEATURES_ONEHOT + 
                    BINARY_FEATURES_MAP_YES_NO + [SENIOR_CITIZEN_COLUMN] + [TARGET_COLUMN])
    # Remove duplicates if any feature listed in multiple lists (e.g. gender)
    expected_cols_list = sorted(list(set(all_features)))


    is_valid, issues_found = validate_data_schema(sample_df, expected_cols_list, NUMERICAL_FEATURES, CUSTOMER_ID_COLUMN, TARGET_COLUMN)
    if is_valid:
        print("Sample data schema is valid.")
    else:
        print(f"Sample data schema invalid: {issues_found}")
    
    check_missing_values(sample_df)