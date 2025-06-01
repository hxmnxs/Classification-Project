from sklearn.preprocessing import MinMaxScaler

def preprocess_data(df):
    """Preprocess the churn data (encoding, scaling, etc.)."""
    # Example: scale numeric columns
    scaler = MinMaxScaler()
    numeric_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
    # Add encoding and other preprocessing as needed
    return df
