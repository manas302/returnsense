"""
Day 5 Task — NLP inference pipeline.
Clean predict() function that takes raw text → returns category + confidence.
"""

import os
import sys
import torch
import torch.nn.functional as F
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from utils.config import MODEL_NLP, MAX_LENGTH, RETURN_CATEGORIES

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class NLPClassifier:
    """
    Wrapper around fine-tuned DistilBERT for return reason classification.

    Usage:
        clf = NLPClassifier()
        result = clf.predict("The product arrived completely broken")
        print(result)
        # {'category': 'genuine_damage', 'label': 0, 'confidence': 0.94,
        #   'all_scores': {'genuine_damage': 0.94, 'fraud': 0.02, ...}}
    """

    def __init__(self, model_path: str = None):
        model_path = model_path or os.path.join(MODEL_NLP, "best_model")

        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"No trained model found at {model_path}.\n"
                f"Run models/nlp/train_nlp.py first."
            )

        print(f"Loading NLP model from {model_path}...")
        self.tokenizer = DistilBertTokenizerFast.from_pretrained(model_path)
        self.model     = DistilBertForSequenceClassification.from_pretrained(model_path)
        self.model.to(DEVICE)
        self.model.eval()
        print("NLP model loaded.")

    def predict(self, text: str) -> dict:
        """
        Predict return reason category from raw text.

        Returns:
            dict with keys: category, label, confidence, all_scores
        """
        enc = self.tokenizer(
            text,
            max_length=MAX_LENGTH,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        input_ids      = enc["input_ids"].to(DEVICE)
        attention_mask = enc["attention_mask"].to(DEVICE)

        with torch.no_grad():
            logits = self.model(input_ids=input_ids, attention_mask=attention_mask).logits

        probs      = F.softmax(logits, dim=-1).squeeze().cpu().numpy()
        label      = int(probs.argmax())
        confidence = float(probs.max())
        category   = RETURN_CATEGORIES[label]

        all_scores = {
            RETURN_CATEGORIES[i]: round(float(probs[i]), 4)
            for i in range(len(RETURN_CATEGORIES))
        }

        return {
            "category":   category,
            "label":      label,
            "confidence": round(confidence, 4),
            "all_scores": all_scores,
        }

    def predict_batch(self, texts: list) -> list:
        """Predict for a list of texts."""
        return [self.predict(t) for t in texts]


# ── Quick test ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    clf = NLPClassifier()

    test_cases = [
        "The product arrived completely broken, packaging was torn",
        "I changed my mind, no longer need this item",
        "Empty box was delivered, product is missing",
        "I wore this to a wedding once and now want to return",
        "I ordered size M but received size XL",
    ]

    print("\n── Inference Test ──────────────────────────────")
    for text in test_cases:
        result = clf.predict(text)
        print(f"\nText      : {text[:60]}...")
        print(f"Category  : {result['category']}")
        print(f"Confidence: {result['confidence']:.2%}")
