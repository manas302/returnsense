"""
Day 2–3 Task — Prepare NLP dataset for DistilBERT fine-tuning.

Two modes:
  1. Use Amazon reviews from Kaggle (recommended — put CSV in data/raw/)
  2. Generate fully synthetic return reasons (fallback)

Run: python utils/prepare_nlp_data.py
"""

import pandas as pd
import numpy as np
import os
import sys
import json
import random

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.config import DATA_RAW, DATA_PROCESSED, RETURN_CATEGORIES, RANDOM_SEED

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# ── Synthetic return reason templates per category ─────────────────────
TEMPLATES = {
    "genuine_damage": [
        "The product arrived completely broken",
        "Item was damaged during delivery, packaging was torn",
        "Screen cracked when I opened the box",
        "Product stopped working after one day of use",
        "Received a defective item, doesn't turn on",
        "The product has a manufacturing defect",
        "Item was shattered, clearly damaged in transit",
        "Product has scratches and dents, not new condition",
        "Battery not working, device is dead on arrival",
        "The item was crushed, completely unusable",
    ],
    "buyers_remorse": [
        "I changed my mind after purchasing",
        "The product doesn't match what I expected",
        "Found a better deal elsewhere after ordering",
        "I accidentally ordered the wrong size",
        "No longer need this item",
        "The color looks different from the website photos",
        "Product is good but not suitable for my needs",
        "I ordered by mistake, please accept the return",
        "My requirement changed so I don't need this anymore",
        "Decided to go with a different brand",
    ],
    "fraud": [
        "Item not received but marked as delivered",
        "Empty box was delivered, product is missing",
        "Received a completely different product",
        "This is a fake or counterfeit product",
        "The seal was broken when I received it",
        "Product looks used and refurbished, not new",
        "Wrong item was sent, this is not what I ordered",
        "Box contains stones instead of the product",
        "Received someone else's order entirely",
        "Product is clearly a cheap duplicate",
    ],
    "wardrobing": [
        "I used it for an event and now want to return",
        "Wore this once for a wedding, returning now",
        "Used it for a single occasion, still in good condition",
        "Tried it out and it works but I want a refund",
        "Only used it once for a photoshoot",
        "Returning after using it for one party",
        "Used the product for a day, want my money back",
        "Tested it for a week, no longer needed",
        "I needed it temporarily and am done with it",
        "Borrowed for a trip and now returning",
    ],
    "wrong_item": [
        "I ordered red but received blue color",
        "Wrong size was delivered, I ordered medium",
        "Received a completely different model",
        "The variant I ordered was not sent",
        "I ordered 500ml but received 250ml",
        "Wrong product entirely, nothing like what I ordered",
        "Size XL was ordered but size S was delivered",
        "Different brand was delivered instead",
        "I ordered the 2024 model but got the 2022 version",
        "Package contains wrong items from another order",
    ],
}


def generate_augmented_text(template: str) -> str:
    """Light augmentation — add filler phrases to increase variety."""
    prefixes = [
        "", "Hi, ", "Hello, ", "Please note that ",
        "I want to report that ", "I am very disappointed, ",
        "This is urgent, ", "Kindly help me, ",
    ]
    suffixes = [
        "", " Please process my return ASAP.",
        " I want a full refund.", " Very unhappy with this.",
        " Need resolution immediately.", " This is unacceptable.",
        "", " Please look into this.",
    ]
    return random.choice(prefixes) + template + random.choice(suffixes)


def build_synthetic_nlp_dataset(n_per_class: int = 800) -> pd.DataFrame:
    """Build a balanced synthetic NLP dataset."""
    rows = []
    label_map = {v: k for k, v in RETURN_CATEGORIES.items()}

    for category, templates in TEMPLATES.items():
        label = label_map[category]
        for _ in range(n_per_class):
            text = generate_augmented_text(random.choice(templates))
            rows.append({"text": text, "label": label, "category": category})

    df = pd.DataFrame(rows).sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)
    return df


def load_amazon_reviews(csv_path: str, n_per_class: int = 800) -> pd.DataFrame:
    """
    Map Amazon reviews to return categories.
    Expects columns: reviewText, overall (star rating)
    """
    print(f"Loading Amazon reviews from {csv_path}...")
    df = pd.read_csv(csv_path, usecols=["reviewText", "overall"]).dropna()
    df = df.rename(columns={"reviewText": "text"})

    # Simple heuristic mapping based on star rating
    def map_label(rating):
        if rating == 1:   return 0  # genuine_damage
        elif rating == 2: return 2  # fraud (low rating complaints)
        elif rating == 3: return 1  # buyers_remorse
        else:             return 4  # wrong_item / neutral
    df["label"] = df["overall"].apply(map_label)
    df["category"] = df["label"].map(RETURN_CATEGORIES)
    return df[["text", "label", "category"]].groupby("label") \
             .apply(lambda x: x.sample(min(len(x), n_per_class), random_state=RANDOM_SEED)) \
             .reset_index(drop=True)


def split_and_save(df: pd.DataFrame):
    """Split into train/val/test and save."""
    from sklearn.model_selection import train_test_split

    train, temp = train_test_split(df, test_size=0.30, stratify=df["label"], random_state=RANDOM_SEED)
    val,   test = train_test_split(temp, test_size=0.50, stratify=temp["label"], random_state=RANDOM_SEED)

    os.makedirs(DATA_PROCESSED, exist_ok=True)
    train.to_csv(os.path.join(DATA_PROCESSED, "nlp_train.csv"), index=False)
    val.to_csv(os.path.join(DATA_PROCESSED,   "nlp_val.csv"),   index=False)
    test.to_csv(os.path.join(DATA_PROCESSED,  "nlp_test.csv"),  index=False)

    print(f"\nTrain : {len(train)} | Val : {len(val)} | Test : {len(test)}")
    print(f"Label distribution (train):\n{train['category'].value_counts()}")

    # Save label map
    with open(os.path.join(DATA_PROCESSED, "label_map.json"), "w") as f:
        json.dump(RETURN_CATEGORIES, f, indent=2)
    print("\nLabel map saved.")


if __name__ == "__main__":
    amazon_path = os.path.join(DATA_RAW, "amazon_reviews.csv")

    if os.path.exists(amazon_path):
        print("Amazon reviews CSV found — using real data.")
        df = load_amazon_reviews(amazon_path)
    else:
        print("Amazon reviews CSV not found — generating synthetic data.")
        print("(Download from Kaggle: 'Amazon Customer Reviews Dataset')")
        df = build_synthetic_nlp_dataset(n_per_class=800)

    print(f"\nTotal records : {len(df)}")
    split_and_save(df)
    print("\nNLP data preparation complete!")
