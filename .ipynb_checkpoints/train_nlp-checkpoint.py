"""
Day 3–4 Task — Fine-tune DistilBERT for return reason classification.

Run on Google Colab (free T4 GPU) if no local GPU:
  1. Upload this file + data/processed/ to Colab
  2. !pip install transformers torch datasets
  3. !python models/nlp/train_nlp.py

Run locally (CPU — slow but works for testing):
  python models/nlp/train_nlp.py
"""

import os
import sys
import json
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import (
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification,
    get_linear_schedule_with_warmup,
)
from sklearn.metrics import classification_report, confusion_matrix
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from utils.config import (
    DATA_PROCESSED, MODEL_NLP, NLP_MODEL_NAME,
    MAX_LENGTH, BATCH_SIZE, EPOCHS, LEARNING_RATE,
    WARMUP_STEPS, RETURN_CATEGORIES,
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")


# ── Dataset ────────────────────────────────────────────────────────────
class ReturnDataset(Dataset):
    def __init__(self, df: pd.DataFrame, tokenizer, max_length: int):
        self.texts  = df["text"].tolist()
        self.labels = df["label"].tolist()
        self.tokenizer  = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.texts[idx],
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids":      enc["input_ids"].squeeze(),
            "attention_mask": enc["attention_mask"].squeeze(),
            "label":          torch.tensor(self.labels[idx], dtype=torch.long),
        }


# ── Training loop ──────────────────────────────────────────────────────
def train_epoch(model, loader, optimizer, scheduler):
    model.train()
    total_loss, correct, total = 0, 0, 0

    for batch in tqdm(loader, desc="Training"):
        input_ids      = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)
        labels         = batch["label"].to(DEVICE)

        optimizer.zero_grad()
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss    = outputs.loss
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()
        preds       = outputs.logits.argmax(dim=-1)
        correct    += (preds == labels).sum().item()
        total      += len(labels)

    return total_loss / len(loader), correct / total


def eval_epoch(model, loader):
    model.eval()
    total_loss, correct, total = 0, 0, 0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating"):
            input_ids      = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            labels         = batch["label"].to(DEVICE)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            total_loss += outputs.loss.item()

            preds       = outputs.logits.argmax(dim=-1)
            correct    += (preds == labels).sum().item()
            total      += len(labels)
            all_preds  += preds.cpu().tolist()
            all_labels += labels.cpu().tolist()

    return total_loss / len(loader), correct / total, all_preds, all_labels


# ── Main ───────────────────────────────────────────────────────────────
def main():
    os.makedirs(MODEL_NLP, exist_ok=True)

    # Load data
    train_df = pd.read_csv(os.path.join(DATA_PROCESSED, "nlp_train.csv"))
    val_df   = pd.read_csv(os.path.join(DATA_PROCESSED, "nlp_val.csv"))
    test_df  = pd.read_csv(os.path.join(DATA_PROCESSED, "nlp_test.csv"))

    # Tokenizer + model
    tokenizer = DistilBertTokenizerFast.from_pretrained(NLP_MODEL_NAME)
    model     = DistilBertForSequenceClassification.from_pretrained(
        NLP_MODEL_NAME, num_labels=len(RETURN_CATEGORIES)
    ).to(DEVICE)

    # DataLoaders
    train_loader = DataLoader(ReturnDataset(train_df, tokenizer, MAX_LENGTH), batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(ReturnDataset(val_df,   tokenizer, MAX_LENGTH), batch_size=BATCH_SIZE)
    test_loader  = DataLoader(ReturnDataset(test_df,  tokenizer, MAX_LENGTH), batch_size=BATCH_SIZE)

    # Optimizer + scheduler
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
    total_steps = len(train_loader) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(optimizer, WARMUP_STEPS, total_steps)

    # Training loop
    best_val_acc = 0.0
    history = []

    for epoch in range(1, EPOCHS + 1):
        print(f"\n{'='*50}")
        print(f"Epoch {epoch}/{EPOCHS}")

        train_loss, train_acc = train_epoch(model, train_loader, optimizer, scheduler)
        val_loss, val_acc, _, _ = eval_epoch(model, val_loader)

        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"Val   Loss: {val_loss:.4f} | Val   Acc: {val_acc:.4f}")

        history.append({
            "epoch": epoch,
            "train_loss": round(train_loss, 4), "train_acc": round(train_acc, 4),
            "val_loss":   round(val_loss,   4), "val_acc":   round(val_acc,   4),
        })

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            model.save_pretrained(os.path.join(MODEL_NLP, "best_model"))
            tokenizer.save_pretrained(os.path.join(MODEL_NLP, "best_model"))
            print(f"  Best model saved (val_acc={val_acc:.4f})")

    # Final evaluation on test set
    print(f"\n{'='*50}")
    print("Final Test Evaluation")
    _, test_acc, test_preds, test_labels = eval_epoch(model, test_loader)
    category_names = [RETURN_CATEGORIES[i] for i in range(len(RETURN_CATEGORIES))]

    print(f"\nTest Accuracy: {test_acc:.4f}")
    print("\nClassification Report:")
    print(classification_report(test_labels, test_preds, target_names=category_names))

    # Save training history + metrics
    with open(os.path.join(MODEL_NLP, "training_history.json"), "w") as f:
        json.dump({
            "history":      history,
            "best_val_acc": round(best_val_acc, 4),
            "test_acc":     round(test_acc, 4),
        }, f, indent=2)

    print(f"\nTraining complete! Best val accuracy: {best_val_acc:.4f}")
    print(f"Model saved to: {os.path.join(MODEL_NLP, 'best_model')}")


if __name__ == "__main__":
    main()
