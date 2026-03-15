import sqlite3
import pandas as pd
import os
# -------------------------------
# FILE PATHS
# -------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "Employee-Attrition.xlsx")
DB_PATH = os.path.join(BASE_DIR, "employee_attrition.db")
# -------------------------------
# CREATE DATABASE CONNECTION
# -------------------------------
def create_connection():
    """
    Creates connection to SQLite database.
    If database does not exist, SQLite creates it.
    """
    conn = sqlite3.connect(DB_PATH)
    return conn
# -------------------------------
# CREATE REQUIRED TABLES
# -------------------------------
def create_tables():
    """
    Creates tables used in the project.
    """
    conn = create_connection()
    cursor = conn.cursor()
    # Table to store model predictions
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS predictions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        employee_number INTEGER,
        attrition_prediction INTEGER,
        probability REAL,
        risk_level TEXT,
        prediction_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """)
    # Table to store model evaluation metrics
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS model_metrics (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    model_name TEXT,
    target TEXT,
    accuracy REAL,
    precision_score REAL,
    recall_score REAL,
    f1_score REAL,
    auc_roc REAL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP

   )
    """)
    conn.commit()
    conn.close()
    print("Tables created successfully")
# -------------------------------
# LOAD DATASET INTO DATABASE
# -------------------------------
def load_dataset():
    """
    Loads the Excel dataset into SQLite.
    """
    conn = create_connection()
    # Read dataset
    df = pd.read_excel(DATA_PATH)
    # Store dataset into database
    df.to_sql("employees", conn, if_exists="replace", index=False)
    conn.close()
    print("Dataset loaded into employees table")
# -------------------------------
# MAIN FUNCTION
# -------------------------------
def main():
    create_tables()
    load_dataset()
# -------------------------------
# PROGRAM ENTRY POINT
# -------------------------------
if __name__ == "__main__":
    main()