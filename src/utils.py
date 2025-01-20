import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from typing import Tuple, Dict

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean the data by handling missing values and converting datatypes."""
    df = df.copy()
    
    # Drop customerID if exists
    if 'customerID' in df.columns:
        df = df.drop(['customerID'], axis=1)
    
    # Convert TotalCharges to numeric
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    
    # Remove rows where tenure is 0
    df = df[df['tenure'] != 0]
    
    # Fill missing TotalCharges with MonthlyCharges
    df['TotalCharges'].fillna(df['MonthlyCharges'], inplace=True)
    
    return df

def preprocess_features(df: pd.DataFrame) -> pd.DataFrame:
    """Preprocess features for model training."""
    df = df.copy()
    
    # Binary encoding (Yes/No columns)
    binary_mapping = {'Yes': 1, 'No': 0}
    binary_cols = ['Partner', 'Dependents', 'PhoneService', 'PaperlessBilling']
    
    for col in binary_cols:
        df[col] = df[col].map(binary_mapping)
    
    # Gender encoding
    df['gender'] = df['gender'].map({'Male': 1, 'Female': 0})
    
    # SeniorCitizen is already encoded as 0/1
    
    # Service columns with multiple values
    service_mapping = {
        'Yes': 1,
        'No': 0,
        'No internet service': 0,
        'No phone service': 0
    }
    
    service_cols = [
        'MultipleLines', 'OnlineSecurity', 'OnlineBackup',
        'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies'
    ]
    
    for col in service_cols:
        df[col] = df[col].map(service_mapping)
    
    # One-hot encoding for categorical columns
    categorical_cols = ['InternetService', 'Contract', 'PaymentMethod']
    df = pd.get_dummies(df, columns=categorical_cols)
    
    return df

def prepare_data(df: pd.DataFrame, scaler: StandardScaler = None) -> Tuple[np.ndarray, StandardScaler]:
    """Prepare data for model training/prediction."""
    # Clean data
    df = clean_data(df)
    
    # Scale numerical features
    numerical_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
    if scaler is None:
        scaler = StandardScaler()
        df[numerical_cols] = scaler.fit_transform(df[numerical_cols])
    else:
        df[numerical_cols] = scaler.transform(df[numerical_cols])
    
    # Preprocess features
    df_processed = preprocess_features(df)
    
    # Remove target if present
    if 'Churn' in df_processed.columns:
        df_processed = df_processed.drop('Churn', axis=1)
    
    return df_processed.values, scaler

def format_input_data(data: Dict) -> pd.DataFrame:
    """Format input data for prediction."""
    df = pd.DataFrame([data])
    
    required_columns = [
        'gender', 'SeniorCitizen', 'Partner', 'Dependents', 'tenure',
        'PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity',
        'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV',
        'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod',
        'MonthlyCharges', 'TotalCharges'
    ]
    
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"Missing required feature: {col}")
    
    return df