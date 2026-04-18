"""
Day 7 Task — The heart of ReturnSense.
DecisionEngine combines NLP + XGBoost scores into a final business verdict.
"""

import os
import sys
import joblib
import numpy as np
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.config import (
    MODEL_XGB, FRAUD_PROB_THRESHOLD, NLP_CONFIDENCE_THRESHOLD,
    HIGH_FRAUD_CATEGORIES, VERDICT_APPROVE, VERDICT_REJECT, VERDICT_MANUAL,
)

# Lazy import NLP classifier (heavy — loads only when needed)
_nlp_classifier = None

def get_nlp_classifier():
    global _nlp_classifier
    if _nlp_classifier is None:
        from models.nlp.nlp_inference import NLPClassifier
        _nlp_classifier = NLPClassifier()
    return _nlp_classifier


class DecisionEngine:
    """
    Combines NLP return reason classifier + XGBoost behavioral fraud scorer
    to produce a final business verdict on a return request.

    Usage:
        engine = DecisionEngine()
        result = engine.analyze(
            return_text="Received empty box, product missing",
            behavioral_features={
                "total_orders": 8, "total_returns": 6,
                "return_rate": 0.75, "account_age_days": 30,
                "order_value": 12000, "days_to_return": 1,
                "is_suspicious_pin": 1, "high_value_flag": 1,
                "new_account_flag": 1, "high_return_cat": 1,
            }
        )
    """

    def __init__(self):
        model_path   = os.path.join(MODEL_XGB, "xgb_model.pkl")
        features_path = os.path.join(MODEL_XGB, "feature_cols.pkl")

        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"XGBoost model not found at {model_path}.\n"
                "Run models/xgboost/train_xgb.py first."
            )

        self.xgb_model    = joblib.load(model_path)
        self.feature_cols = joblib.load(features_path)
        self.nlp          = get_nlp_classifier()
        print("DecisionEngine ready.")

    def _engineer_features(self, bf: dict) -> pd.DataFrame:
        """Add derived features to raw behavioral dict."""
        bf = bf.copy()
        return_rate = bf.get("return_rate", 0)

        # Bucketize return rate
        if return_rate <= 0.10:   bf["return_rate_bucket"] = 0
        elif return_rate <= 0.30: bf["return_rate_bucket"] = 1
        elif return_rate <= 0.60: bf["return_rate_bucket"] = 2
        else:                     bf["return_rate_bucket"] = 3

        # Return velocity
        bf["return_velocity"] = bf.get("total_returns", 0) / (bf.get("account_age_days", 1) + 1)

        # Quick high-value return flag
        bf["quick_high_value_return"] = int(
            bf.get("order_value", 0) > 5000 and bf.get("days_to_return", 99) <= 3
        )

        return pd.DataFrame([{col: bf.get(col, 0) for col in self.feature_cols}])

    def analyze(self, return_text: str, behavioral_features: dict) -> dict:
        """
        Full analysis pipeline.

        Args:
            return_text           : Customer's return reason (string)
            behavioral_features   : Dict of customer behavioral signals

        Returns:
            dict with keys:
                verdict            : AUTO_APPROVE / AUTO_REJECT / MANUAL_REVIEW
                fraud_probability  : 0.0–1.0 (XGBoost score)
                nlp_category       : detected return reason category
                nlp_confidence     : NLP model confidence
                risk_level         : LOW / MEDIUM / HIGH
                explanation        : Human-readable reason for verdict
                all_scores         : All NLP class probabilities
        """
        # Step 1 — NLP classification
        nlp_result = self.nlp.predict(return_text)
        nlp_cat    = nlp_result["category"]
        nlp_conf   = nlp_result["confidence"]

        # Step 2 — XGBoost fraud probability
        X            = self._engineer_features(behavioral_features)
        fraud_prob   = float(self.xgb_model.predict_proba(X)[0][1])

        # Step 3 — Risk level
        if fraud_prob >= 0.75 or (nlp_cat in HIGH_FRAUD_CATEGORIES and nlp_conf >= 0.80):
            risk_level = "HIGH"
        elif fraud_prob >= 0.45 or nlp_cat in HIGH_FRAUD_CATEGORIES:
            risk_level = "MEDIUM"
        else:
            risk_level = "LOW"

        # Step 4 — Final verdict logic
        if fraud_prob >= FRAUD_PROB_THRESHOLD and nlp_cat in HIGH_FRAUD_CATEGORIES:
            verdict     = VERDICT_REJECT
            explanation = (
                f"High fraud probability ({fraud_prob:.0%}) combined with "
                f"suspicious return reason '{nlp_cat}' ({nlp_conf:.0%} confidence). "
                f"Return request automatically rejected."
            )
        elif fraud_prob >= FRAUD_PROB_THRESHOLD:
            verdict     = VERDICT_MANUAL
            explanation = (
                f"High behavioral fraud score ({fraud_prob:.0%}) detected. "
                f"Return reason classified as '{nlp_cat}'. "
                f"Flagged for manual review."
            )
        elif nlp_cat in HIGH_FRAUD_CATEGORIES and nlp_conf >= NLP_CONFIDENCE_THRESHOLD:
            verdict     = VERDICT_MANUAL
            explanation = (
                f"Return text strongly suggests '{nlp_cat}' ({nlp_conf:.0%} confidence). "
                f"Behavioral risk is moderate ({fraud_prob:.0%}). "
                f"Flagged for manual review."
            )
        else:
            verdict     = VERDICT_APPROVE
            explanation = (
                f"Low fraud risk ({fraud_prob:.0%}). "
                f"Return reason: '{nlp_cat}' ({nlp_conf:.0%} confidence). "
                f"Return automatically approved."
            )

        return {
            "verdict":           verdict,
            "fraud_probability": round(fraud_prob, 4),
            "nlp_category":      nlp_cat,
            "nlp_confidence":    round(nlp_conf, 4),
            "risk_level":        risk_level,
            "explanation":       explanation,
            "all_scores":        nlp_result["all_scores"],
        }


# ── Quick test ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    engine = DecisionEngine()

    test_cases = [
        {
            "text": "Empty box delivered, product is completely missing",
            "features": {
                "total_orders": 10, "total_returns": 8, "return_rate": 0.80,
                "account_age_days": 25, "order_value": 15000, "days_to_return": 1,
                "is_suspicious_pin": 1, "high_value_flag": 1,
                "new_account_flag": 1, "high_return_cat": 1,
            },
            "expected": VERDICT_REJECT,
        },
        {
            "text": "The product arrived broken, screen is cracked",
            "features": {
                "total_orders": 45, "total_returns": 3, "return_rate": 0.07,
                "account_age_days": 800, "order_value": 4500, "days_to_return": 10,
                "is_suspicious_pin": 0, "high_value_flag": 0,
                "new_account_flag": 0, "high_return_cat": 1,
            },
            "expected": VERDICT_APPROVE,
        },
        {
            "text": "I changed my mind and no longer need this",
            "features": {
                "total_orders": 20, "total_returns": 7, "return_rate": 0.35,
                "account_age_days": 180, "order_value": 2000, "days_to_return": 8,
                "is_suspicious_pin": 0, "high_value_flag": 0,
                "new_account_flag": 0, "high_return_cat": 0,
            },
            "expected": VERDICT_MANUAL,
        },
    ]

    print("\n── DecisionEngine Test ──────────────────────────")
    for i, case in enumerate(test_cases, 1):
        result = engine.analyze(case["text"], case["features"])
        status = "PASS" if result["verdict"] == case["expected"] else "FAIL"
        print(f"\nCase {i} [{status}]")
        print(f"  Text       : {case['text'][:55]}...")
        print(f"  Verdict    : {result['verdict']} (expected: {case['expected']})")
        print(f"  Risk       : {result['risk_level']}")
        print(f"  Fraud Prob : {result['fraud_probability']:.2%}")
        print(f"  NLP Cat    : {result['nlp_category']} ({result['nlp_confidence']:.2%})")
        print(f"  Explanation: {result['explanation'][:80]}...")
