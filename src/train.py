import os
import sys
import sqlite3
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# -------------------------------
# FILE PATHS
# -------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(BASE_DIR, "employee_attrition.db")
MODELS_DIR = os.path.join(BASE_DIR, "models")
sys.path.append(BASE_DIR)

from src.feature_engineering import feature_pipeline
from src.preprocess import get_clean_data


# -------------------------------
# SAVE METRICS TO SQLITE
# -------------------------------
def save_metrics(model_name, target, accuracy, precision, recall, f1):

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute("""
        INSERT INTO model_metrics
        (model_name, target, accuracy, precision_score, recall_score, f1_score)
        VALUES (?, ?, ?, ?, ?, ?)
    """, (model_name, target, accuracy, precision, recall, f1))

    conn.commit()
    conn.close()

    print("  Metrics saved to SQLite ✓")


# -------------------------------
# TRAIN ONE MODEL
# -------------------------------
def train_model(model, model_name, X_train, X_test, y_train, y_test, target):

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    accuracy  = round(accuracy_score(y_test, y_pred), 4)
    precision = round(precision_score(y_test, y_pred, average="weighted", zero_division=0), 4)
    recall    = round(recall_score(y_test, y_pred, average="weighted", zero_division=0), 4)
    f1        = round(f1_score(y_test, y_pred, average="weighted", zero_division=0), 4)

    print(f"\n  {model_name}")
    print(f"  Accuracy  : {accuracy}")
    print(f"  Precision : {precision}")
    print(f"  Recall    : {recall}")
    print(f"  F1 Score  : {f1}")

    save_metrics(model_name, target, accuracy, precision, recall, f1)

    return model, accuracy


# -------------------------------
# TRAIN ATTRITION MODELS
# -------------------------------
def train_attrition(X_train, X_test, y_train, y_test):

    print("\n" + "="*50)
    print("TRAINING — ATTRITION PREDICTION")
    print("="*50)

    models = {
        "LogisticRegression_Attrition": LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=42
        ),

        "DecisionTree_Attrition": DecisionTreeClassifier(
            max_depth=5,
            class_weight="balanced",
            random_state=42
        ),

        "RandomForest_Attrition": RandomForestClassifier(
            n_estimators=100,
            class_weight="balanced",
            random_state=42
        )
    }

    best_model = None
    best_accuracy = 0

    for model_name, model in models.items():

        trained_model, accuracy = train_model(
            model,
            model_name,
            X_train,
            X_test,
            y_train,
            y_test,
            target="Attrition"
        )

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model = trained_model

    model_path = os.path.join(MODELS_DIR, "attrition_model.pkl")
    joblib.dump(best_model, model_path)

    print("\nBest Attrition model saved → attrition_model.pkl")


# -------------------------------
# TRAIN PERFORMANCE MODELS
# -------------------------------
def train_performance(X_train, X_test, y_train, y_test):

    print("\n" + "="*50)
    print("TRAINING — PERFORMANCE RATING PREDICTION")
    print("="*50)

    models = {
        "LogisticRegression_Performance": LogisticRegression(max_iter=1000),

        "DecisionTree_Performance": DecisionTreeClassifier(max_depth=5),

        "RandomForest_Performance": RandomForestClassifier(n_estimators=100)
    }

    best_model = None
    best_accuracy = 0

    for model_name, model in models.items():

        trained_model, accuracy = train_model(
            model,
            model_name,
            X_train,
            X_test,
            y_train,
            y_test,
            target="PerformanceRating"
        )

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model = trained_model

    model_path = os.path.join(MODELS_DIR, "performance_model.pkl")
    joblib.dump(best_model, model_path)

    print("\nBest Performance model saved → performance_model.pkl")


# -------------------------------
# MAIN TRAINING PIPELINE
# -------------------------------
def main():

    os.makedirs(MODELS_DIR, exist_ok=True)

    print("Loading data...")

    X, y_attrition = feature_pipeline()

    # IMPORTANT: save feature order
    feature_columns_path = os.path.join(MODELS_DIR, "feature_columns.pkl")
    joblib.dump(X.columns.tolist(), feature_columns_path)

    print("Feature columns saved → feature_columns.pkl")

    df_full = get_clean_data()
    y_performance = df_full["PerformanceRating"]

    X_train, X_test, y_att_train, y_att_test = train_test_split(
        X,
        y_attrition,
        test_size=0.2,
        random_state=42,
        stratify=y_attrition
    )

    print(f"\nTrain size : {X_train.shape[0]}")
    print(f"Test size  : {X_test.shape[0]}")

    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)

    scaler_path = os.path.join(MODELS_DIR, "scaler.pkl")
    joblib.dump(scaler, scaler_path)

    print("Scaler saved → scaler.pkl")

    y_perf_train = y_performance.iloc[y_att_train.index]
    y_perf_test  = y_performance.iloc[y_att_test.index]

    train_attrition(X_train_scaled, X_test_scaled, y_att_train, y_att_test)

    train_performance(X_train_scaled, X_test_scaled, y_perf_train, y_perf_test)

    print("\nALL MODELS TRAINED AND SAVED!")


# -------------------------------
# ENTRY POINT
# -------------------------------
if __name__ == "__main__":
    main()