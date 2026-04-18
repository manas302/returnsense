"""
Day 6 Task — Train XGBoost behavioral fraud scorer.
Run: python models/xgboost/train_xgb.py
"""

import os
import sys
import json
import joblib
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    classification_report, roc_auc_score,
    confusion_matrix, ConfusionMatrixDisplay,
)
from sklearn.preprocessing import LabelEncoder
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from utils.config import DATA_PROCESSED, MODEL_XGB, XGB_PARAMS, RANDOM_SEED

FEATURE_COLS = [
    "total_orders", "total_returns", "return_rate",
    "account_age_days", "order_value", "days_to_return",
    "is_suspicious_pin", "high_value_flag",
    "new_account_flag", "high_return_cat",
]
TARGET_COL = "is_fraud"


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add derived features on top of raw behavioral data."""
    df = df.copy()

    # Return rate buckets
    df["return_rate_bucket"] = pd.cut(
        df["return_rate"],
        bins=[-0.01, 0.10, 0.30, 0.60, 1.01],
        labels=[0, 1, 2, 3]
    ).astype(int)

    # Velocity: returns per account age
    df["return_velocity"] = df["total_returns"] / (df["account_age_days"] + 1)

    # High value + quick return = red flag
    df["quick_high_value_return"] = (
        (df["order_value"] > 5000) & (df["days_to_return"] <= 3)
    ).astype(int)

    return df


def main():
    os.makedirs(MODEL_XGB, exist_ok=True)

    # Load data
    data_path = os.path.join(DATA_PROCESSED, "behavioral_data.csv")
    if not os.path.exists(data_path):
        print(f"Behavioral data not found at {data_path}")
        print("Run: python utils/generate_synthetic_data.py first")
        sys.exit(1)

    df = pd.read_csv(data_path)
    df = engineer_features(df)

    # Final feature list (includes engineered)
    all_features = FEATURE_COLS + [
        "return_rate_bucket", "return_velocity", "quick_high_value_return"
    ]

    X = df[all_features]
    y = df[TARGET_COL]

    print(f"Dataset: {X.shape} | Fraud rate: {y.mean():.2%}")

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=RANDOM_SEED
    )

    # Train XGBoost
    print("\nTraining XGBoost...")
    model = XGBClassifier(**XGB_PARAMS)
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=50,
    )

    # Evaluate
    y_pred      = model.predict(X_test)
    y_prob      = model.predict_proba(X_test)[:, 1]
    auc         = roc_auc_score(y_test, y_prob)
    cv_scores   = cross_val_score(model, X, y, cv=5, scoring="roc_auc")

    print(f"\nTest AUC-ROC  : {auc:.4f}")
    print(f"CV AUC-ROC    : {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=["Genuine", "Fraud"]))

    # Feature importance plot
    fig, ax = plt.subplots(figsize=(8, 6))
    feat_imp = pd.Series(model.feature_importances_, index=all_features).sort_values()
    feat_imp.plot(kind="barh", ax=ax, color="#378ADD")
    ax.set_title("XGBoost Feature Importances")
    ax.set_xlabel("Importance Score")
    plt.tight_layout()
    plt.savefig(os.path.join(MODEL_XGB, "feature_importance.png"), dpi=120)
    plt.close()

    # Save model + metadata
    joblib.dump(model, os.path.join(MODEL_XGB, "xgb_model.pkl"))
    joblib.dump(all_features, os.path.join(MODEL_XGB, "feature_cols.pkl"))

    metrics = {
        "test_auc":    round(auc, 4),
        "cv_auc_mean": round(float(cv_scores.mean()), 4),
        "cv_auc_std":  round(float(cv_scores.std()), 4),
    }
    with open(os.path.join(MODEL_XGB, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\nModel saved to: {MODEL_XGB}")
    print(f"Feature importance plot saved.")


if __name__ == "__main__":
    main()
