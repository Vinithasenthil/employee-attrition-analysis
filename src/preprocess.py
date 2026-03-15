import sqlite3
import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import LabelEncoder, StandardScaler

# -------------------------------
# FILE PATHS
# -------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(BASE_DIR, "employee_attrition.db")

# -------------------------------
# LOAD DATA FROM SQLITE
# -------------------------------
def load_data():
    """
    Loads employee data from SQLite database.
    Returns raw DataFrame.
    """
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql("SELECT * FROM employees", conn)
    conn.close()
    print(f"Data loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    return df

# -------------------------------
# DROP USELESS COLUMNS
# -------------------------------
def drop_useless_columns(df):
    """
    Drops constant columns that add no value to model.
    EmployeeCount → always 1
    Over18        → always Y
    StandardHours → always 80
    EmployeeNumber→ just an ID
    """
    cols_to_drop = [
        'EmployeeCount',
        'Over18',
        'StandardHours',
        'EmployeeNumber'
    ]
    df = df.drop(columns=cols_to_drop)
    print(f"Dropped useless columns. Remaining: {df.shape[1]} columns")
    return df

# -------------------------------
# ENCODE TARGET COLUMN
# -------------------------------
def encode_target(df):
    """
    Encodes Attrition column:
    'Yes' → 1 (will leave)
    'No'  → 0 (will stay)
    """
    df['Attrition'] = df['Attrition'].map({'Yes': 1, 'No': 0})
    print(f"Target encoded → Yes:1, No:0")
    print(f"Attrition value counts:\n{df['Attrition'].value_counts()}")
    return df

# -------------------------------
# ENCODE CATEGORICAL COLUMNS
# -------------------------------
def encode_categorical(df):
    """
    Encodes all text columns to numbers using LabelEncoder.

    Categorical columns:
    → BusinessTravel, Department, EducationField
    → Gender, JobRole, MaritalStatus, OverTime
    """
    le = LabelEncoder()
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()

    for col in categorical_cols:
        df[col] = le.fit_transform(df[col])
        print(f"Encoded: {col}")

    return df

# -------------------------------
# SCALE NUMERIC FEATURES
# -------------------------------
def scale_features(X_train, X_test):
    """
    Scales numeric features using StandardScaler.
    Mean = 0, Standard Deviation = 1

    IMPORTANT RULE:
    → fit_transform on X_train only (learns mean & std)
    → transform on X_test only (applies same mean & std)
    → Never fit on test data = prevents data leakage!
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    print("Features scaled successfully")
    return X_train_scaled, X_test_scaled, scaler

# -------------------------------
# MASTER FUNCTION
# -------------------------------
# MASTER FUNCTION
def get_clean_data():
    df = load_data()
    df = drop_useless_columns(df)
    df = encode_target(df)
    df = encode_categorical(df)
    return df

# PIPELINE FUNCTION  ← ADD THIS
def preprocess_pipeline():
    df = get_clean_data()
    X = df.drop("Attrition", axis=1)
    y = df["Attrition"]
    return X, y

# MAIN
def main():
    df = get_clean_data()
    print("\nSample data:")
    print(df.head(3))

if __name__ == "__main__":
    main()