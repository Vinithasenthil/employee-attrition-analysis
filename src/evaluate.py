import os
import sys
import sqlite3
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # non-interactive backend for saving plots
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report,
    roc_auc_score, roc_curve
)
from sklearn.model_selection import train_test_split

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
# LOAD MODELS AND SCALER
# -------------------------------
def load_artifacts():
    """
    Loads saved model files and scaler from models/ folder.
    Returns attrition model, performance model, scaler.
    """
    attrition_model  = joblib.load(os.path.join(MODELS_DIR, "attrition_model.pkl"))
    performance_model = joblib.load(os.path.join(MODELS_DIR, "performance_model.pkl"))
    scaler           = joblib.load(os.path.join(MODELS_DIR, "scaler.pkl"))
    print("Models and scaler loaded successfully ✓")
    return attrition_model, performance_model, scaler

# -------------------------------
# PREPARE TEST DATA
# -------------------------------
def prepare_test_data():
    """
    Runs full pipeline and returns scaled test data.
    Uses same random_state=42 as train.py to get same split.
    """
    # Load features
    X, y_attrition = feature_pipeline()

    # Get PerformanceRating
    df_full = get_clean_data()
    y_performance = df_full["PerformanceRating"]

    # Same split as train.py
    X_train, X_test, y_att_train, y_att_test = train_test_split(
        X, y_attrition,
        test_size=0.2,
        random_state=42,
        stratify=y_attrition
    )

    # Load saved scaler
    scaler = joblib.load(os.path.join(MODELS_DIR, "scaler.pkl"))
    X_test_scaled = scaler.transform(X_test)

    # Performance split
    y_perf_test = y_performance.iloc[y_att_test.index]

    print(f"Test data prepared: {X_test_scaled.shape[0]} rows ✓")
    return X_test_scaled, y_att_test, y_perf_test, X_test

# -------------------------------
# PLOT CONFUSION MATRIX
# -------------------------------
def plot_confusion_matrix(y_test, y_pred, title, filename):
    """
    Saves confusion matrix as PNG in models/ folder.
    """
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(
        cm, annot=True, fmt="d",
        cmap="Blues",
        xticklabels=["Predicted No", "Predicted Yes"],
        yticklabels=["Actual No", "Actual Yes"]
    )
    plt.title(title)
    plt.tight_layout()
    save_path = os.path.join(MODELS_DIR, filename)
    plt.savefig(save_path)
    plt.close()
    print(f"  Confusion matrix saved → {filename}")

# -------------------------------
# PLOT ROC CURVE
# -------------------------------
def plot_roc_curve(model, X_test, y_test, title, filename):
    """
    Saves ROC curve as PNG in models/ folder.
    Only works for binary classification (Attrition).
    """
    try:
        y_prob = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        auc = roc_auc_score(y_test, y_prob)

        plt.figure(figsize=(6, 4))
        plt.plot(fpr, tpr, color="blue", label=f"AUC = {auc:.4f}")
        plt.plot([0, 1], [0, 1], color="gray", linestyle="--")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(title)
        plt.legend()
        plt.tight_layout()
        save_path = os.path.join(MODELS_DIR, filename)
        plt.savefig(save_path)
        plt.close()
        print(f"  ROC curve saved → {filename} (AUC: {auc:.4f})")
        return auc
    except Exception as e:
        print(f"  ROC curve skipped: {e}")
        return None

# -------------------------------
# PLOT FEATURE IMPORTANCE
# -------------------------------
def plot_feature_importance(model, feature_names, filename):
    """
    Saves top 15 feature importance chart as PNG.
    Works for RandomForest and DecisionTree models.
    """
    try:
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1][:15]

        plt.figure(figsize=(10, 6))
        plt.bar(
            range(15),
            importances[indices],
            color="steelblue"
        )
        plt.xticks(
            range(15),
            [feature_names[i] for i in indices],
            rotation=45, ha="right"
        )
        plt.title("Top 15 Feature Importances")
        plt.tight_layout()
        save_path = os.path.join(MODELS_DIR, filename)
        plt.savefig(save_path)
        plt.close()
        print(f"  Feature importance saved → {filename}")
    except Exception as e:
        print(f"  Feature importance skipped: {e}")

# -------------------------------
# EVALUATE ATTRITION MODEL
# -------------------------------
def evaluate_attrition(model, X_test, y_test, feature_names):
    """
    Full evaluation of attrition prediction model.
    Prints metrics, saves confusion matrix, ROC curve,
    and feature importance plots.
    """
    print("\n" + "="*50)
    print("EVALUATION — ATTRITION PREDICTION")
    print("="*50)

    y_pred = model.predict(X_test)

    # Metrics
    accuracy  = round(accuracy_score(y_test, y_pred), 4)
    precision = round(precision_score(y_test, y_pred, average="weighted", zero_division=0), 4)
    recall    = round(recall_score(y_test, y_pred, average="weighted", zero_division=0), 4)
    f1        = round(f1_score(y_test, y_pred, average="weighted", zero_division=0), 4)

    print(f"\n  Accuracy  : {accuracy}")
    print(f"  Precision : {precision}")
    print(f"  Recall    : {recall}")
    print(f"  F1 Score  : {f1}")

    # Classification report
    print(f"\n  Classification Report:")
    print(classification_report(y_test, y_pred,
          target_names=["Stayed (0)", "Left (1)"],
          zero_division=0))

    # Plots
    plot_confusion_matrix(
        y_test, y_pred,
        "Confusion Matrix — Attrition",
        "confusion_matrix_attrition.png"
    )
    plot_roc_curve(
        model, X_test, y_test,
        "ROC Curve — Attrition",
        "roc_curve_attrition.png"
    )
    plot_feature_importance(
        model, feature_names,
        "feature_importance_attrition.png"
    )

# -------------------------------
# EVALUATE PERFORMANCE MODEL
# -------------------------------
def evaluate_performance(model, X_test, y_test, feature_names):
    """
    Full evaluation of performance rating prediction model.
    Prints metrics and saves confusion matrix plot.
    """
    print("\n" + "="*50)
    print("EVALUATION — PERFORMANCE RATING PREDICTION")
    print("="*50)

    y_pred = model.predict(X_test)

    # Metrics
    accuracy  = round(accuracy_score(y_test, y_pred), 4)
    precision = round(precision_score(y_test, y_pred, average="weighted", zero_division=0), 4)
    recall    = round(recall_score(y_test, y_pred, average="weighted", zero_division=0), 4)
    f1        = round(f1_score(y_test, y_pred, average="weighted", zero_division=0), 4)

    print(f"\n  Accuracy  : {accuracy}")
    print(f"  Precision : {precision}")
    print(f"  Recall    : {recall}")
    print(f"  F1 Score  : {f1}")

    # Classification report
    print(f"\n  Classification Report:")
    print(classification_report(y_test, y_pred,
          target_names=["Rating 3", "Rating 4"],
          zero_division=0))

    # Confusion matrix only (no ROC for multiclass)
    plot_confusion_matrix(
        y_test, y_pred,
        "Confusion Matrix — Performance Rating",
        "confusion_matrix_performance.png"
    )
    plot_feature_importance(
        model, feature_names,
        "feature_importance_performance.png"
    )

# -------------------------------
# MAIN
# -------------------------------
def main():
    # Load models
    attrition_model, performance_model, scaler = load_artifacts()

    # Prepare test data
    X_test_scaled, y_att_test, y_perf_test, X_test_raw = prepare_test_data()

    # Get feature names
    feature_names = X_test_raw.columns.tolist()

    # Evaluate both models
    evaluate_attrition(attrition_model, X_test_scaled, y_att_test, feature_names)
    evaluate_performance(performance_model, X_test_scaled, y_perf_test, feature_names)

    print("\n" + "="*50)
    print("EVALUATION COMPLETE!")
    print("="*50)
    print("Plots saved in models/:")
    print("  → confusion_matrix_attrition.png")
    print("  → roc_curve_attrition.png")
    print("  → feature_importance_attrition.png")
    print("  → confusion_matrix_performance.png")
    print("  → feature_importance_performance.png")

# -------------------------------
# ENTRY POINT
# -------------------------------
if __name__ == "__main__":
    main()