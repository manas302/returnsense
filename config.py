import os

# ── Paths ──────────────────────────────────────────────
BASE_DIR        = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_RAW        = os.path.join(BASE_DIR, "data", "raw")
DATA_PROCESSED  = os.path.join(BASE_DIR, "data", "processed")
MODEL_NLP       = os.path.join(BASE_DIR, "models", "nlp")
MODEL_XGB       = os.path.join(BASE_DIR, "models", "xgboost")
DB_PATH         = os.path.join(BASE_DIR, "returnsense.db")

# ── NLP Model ──────────────────────────────────────────
NLP_MODEL_NAME  = "distilbert-base-uncased"
MAX_LENGTH      = 128
BATCH_SIZE      = 32
EPOCHS          = 5
LEARNING_RATE   = 2e-5
WARMUP_STEPS    = 100

# ── Return Categories ──────────────────────────────────
RETURN_CATEGORIES = {
    0: "genuine_damage",
    1: "buyers_remorse",
    2: "fraud",
    3: "wardrobing",
    4: "wrong_item"
}

# Fraud risk categories (used in decision engine)
HIGH_FRAUD_CATEGORIES = {"fraud", "wardrobing"}

# ── XGBoost ────────────────────────────────────────────
XGB_PARAMS = {
    "n_estimators":   200,
    "max_depth":      6,
    "learning_rate":  0.05,
    "subsample":      0.8,
    "colsample_bytree": 0.8,
    "use_label_encoder": False,
    "eval_metric":    "logloss",
    "random_state":   42,
}

# ── Decision Engine Thresholds ─────────────────────────
FRAUD_PROB_THRESHOLD    = 0.65   # XGBoost fraud probability
NLP_CONFIDENCE_THRESHOLD = 0.70  # DistilBERT confidence

# Verdict labels
VERDICT_APPROVE = "AUTO_APPROVE"
VERDICT_REJECT  = "AUTO_REJECT"
VERDICT_MANUAL  = "MANUAL_REVIEW"

# ── Synthetic Data ─────────────────────────────────────
SYNTHETIC_RECORDS = 15000
FRAUD_RATIO       = 0.20         # 20% fraud in synthetic data
RANDOM_SEED       = 42
