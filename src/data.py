import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor

# Load data with error handling
def load_data():
    try:
        df = pd.read_excel('data/raw/data of fyp.xlsx')  # Use forward slashes for consistency
        print("Data loaded successfully.")
        return df
    except FileNotFoundError:
        print("Error: 'data/raw/data of fyp.xlsx' not found. Please upload the file to the data/raw folder.")
        return None
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

# Clean data and convert to numeric
def data_cleaning(df):
    if df is None:
        return None
    df = df.copy()
    
    # Remove trailing spaces and replace internal spaces with underscores in column names
    df.columns = [col.strip().replace(' ', '_') for col in df.columns]
    
    # Identify object columns
    object_columns = df.select_dtypes(include=['object']).columns
    for col in object_columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Handle missing values
    df = df.dropna(subset=['Maximum_Temperature', 'Minimun_Temperature'])  # Fixed typo 'Minimun'
    df['Rainfall'] = df['Rainfall'].fillna(0.05)
    print("Data cleaned successfully.")
    return df

# Fill missing values with KNN
def fill_missing_with_knn(df, feature_cols, target_col, n_neighbors=5):
    if df is None:
        return None
    # Split known and unknown target values
    known = df[df[target_col].notna()].copy()
    unknown = df[df[target_col].isna()].copy()

    if unknown.empty:
        print("No missing values found in target column.")
        return df

    # Prepare features and target
    X_known = known[feature_cols]
    y_known = known[target_col].values.ravel()  # Ensure 1D array
    X_unknown = unknown[feature_cols]

    # Scale features
    scaler = StandardScaler()
    X_known_scaled = scaler.fit_transform(X_known)
    X_unknown_scaled = scaler.transform(X_unknown)

    # Train KNN Regressor
    knn = KNeighborsRegressor(n_neighbors=n_neighbors)
    knn.fit(X_known_scaled, y_known)

    # Predict missing target values
    y_pred = knn.predict(X_unknown_scaled)

    # Fill in the missing values
    df.loc[df[target_col].isna(), target_col] = y_pred
    print(f"Missing values in {target_col} filled with KNN predictions.")
    return df

# Main execution
df = load_data()
if df is not None:
    df = data_cleaning(df)
    if df is not None:
        print("Null values after cleaning:")
        print(df.isnull().sum())
        feature_cols = ['Maximum_Temperature', 'Minimun_Temperature', 'Rainfall', 'Average_Temperature']
        target_col = 'water_Discharge'
        df = fill_missing_with_knn(df, feature_cols, target_col)
        print("\nFinal DataFrame:")
        print(df.head())
        print(df.isnull().sum())
        # to save the data to the data/intermidiate/cleaned.csv for further processing
        df.to_csv('data/intermidiate/cleaned.csv',index=False)
        