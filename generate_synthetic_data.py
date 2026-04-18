"""
Day 2 Task — Generate synthetic behavioral data for XGBoost fraud scorer.
Run: python utils/generate_synthetic_data.py
"""

import pandas as pd
import numpy as np
from faker import Faker
import random
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.config import SYNTHETIC_RECORDS, FRAUD_RATIO, RANDOM_SEED, DATA_PROCESSED

fake = Faker("en_IN")
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

# ── Suspicious pin code clusters (fake high-fraud zones) ───────────────
SUSPICIOUS_PINCODES = {"110001", "400001", "600001", "700001", "560001"}

PRODUCT_CATEGORIES = [
    "Electronics", "Fashion", "HomeAppliances",
    "Books", "Beauty", "Sports", "Grocery"
]

# High-return categories (used in fraud logic)
HIGH_RETURN_CATEGORIES = {"Electronics", "Fashion"}


def generate_record(is_fraud: bool) -> dict:
    """Generate one synthetic customer return record."""

    if is_fraud:
        # Fraud pattern: high return rate, new account, high value orders
        total_orders   = random.randint(5, 30)
        total_returns  = random.randint(int(total_orders * 0.5), total_orders)
        account_age    = random.randint(1, 90)       # days — new account
        order_value    = random.uniform(3000, 25000) # high value
        days_to_return = random.randint(1, 5)        # returned very quickly
        category       = random.choice(list(HIGH_RETURN_CATEGORIES))
        pincode        = random.choice(list(SUSPICIOUS_PINCODES)) \
                         if random.random() < 0.4 else fake.postcode()
    else:
        # Genuine pattern
        total_orders   = random.randint(1, 50)
        total_returns  = random.randint(0, max(1, int(total_orders * 0.15)))
        account_age    = random.randint(30, 1500)    # established account
        order_value    = random.uniform(200, 8000)
        days_to_return = random.randint(3, 25)
        category       = random.choice(PRODUCT_CATEGORIES)
        pincode        = fake.postcode()

    return_rate = round(total_returns / total_orders, 4) if total_orders > 0 else 0

    return {
        "customer_id":        fake.uuid4(),
        "total_orders":       total_orders,
        "total_returns":      total_returns,
        "return_rate":        return_rate,
        "account_age_days":   account_age,
        "order_value":        round(order_value, 2),
        "days_to_return":     days_to_return,
        "product_category":   category,
        "pincode":            pincode,
        "is_suspicious_pin":  int(pincode in SUSPICIOUS_PINCODES),
        "high_value_flag":    int(order_value > 5000),
        "new_account_flag":   int(account_age < 90),
        "high_return_cat":    int(category in HIGH_RETURN_CATEGORIES),
        "is_fraud":           int(is_fraud),
    }


def generate_dataset(n_records: int, fraud_ratio: float) -> pd.DataFrame:
    n_fraud   = int(n_records * fraud_ratio)
    n_genuine = n_records - n_fraud

    print(f"Generating {n_genuine} genuine + {n_fraud} fraud records...")

    records = (
        [generate_record(is_fraud=True)  for _ in range(n_fraud)] +
        [generate_record(is_fraud=False) for _ in range(n_genuine)]
    )

    df = pd.DataFrame(records).sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)
    return df


if __name__ == "__main__":
    os.makedirs(DATA_PROCESSED, exist_ok=True)

    df = generate_dataset(SYNTHETIC_RECORDS, FRAUD_RATIO)

    out_path = os.path.join(DATA_PROCESSED, "behavioral_data.csv")
    df.to_csv(out_path, index=False)

    print(f"\nDataset saved to: {out_path}")
    print(f"Shape          : {df.shape}")
    print(f"Fraud rate     : {df['is_fraud'].mean():.2%}")
    print(f"\nSample records:")
    print(df.head(3).to_string())
