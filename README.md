# ReturnSense — AI-Powered Return Fraud Detection

![Python](https://img.shields.io/badge/Python-3.10-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.1-orange)
![XGBoost](https://img.shields.io/badge/XGBoost-2.0-green)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28-red)
![FastAPI](https://img.shields.io/badge/FastAPI-0.104-teal)

> An end-to-end return fraud detection system combining **DistilBERT NLP classifier** + **XGBoost behavioral scorer** to automatically approve, reject, or flag return requests — solving a ₹4000 crore annual problem for e-commerce platforms.

---

## Live Demo
**[ReturnSense Dashboard →](YOUR_STREAMLIT_LINK_HERE)**

---

## The Problem
E-commerce platforms lose thousands of crores annually to return fraud:
- Customers claiming "damaged product" for items they intentionally broke
- Wardrobing — buying, using, and returning items
- Empty box scams
- Serial returners exploiting refund policies

Existing rule-based systems miss complex patterns. ReturnSense uses AI to detect fraud intelligently.

---

## Architecture

```
Customer Return Request
        │
        ├── Return Text (NLP)
        │       └── DistilBERT fine-tuned
        │               └── Category + Confidence
        │
        └── Behavioral Signals (Tabular)
                └── XGBoost Classifier
                        └── Fraud Probability
                                │
                        DecisionEngine
                                │
              ┌─────────────────┼─────────────────┐
         AUTO_APPROVE     MANUAL_REVIEW      AUTO_REJECT
```

---

## Results

| Model | Metric | Score |
|---|---|---|
| DistilBERT NLP | Test Accuracy | 89% |
| DistilBERT NLP | F1-Score (macro) | 0.87 |
| XGBoost Fraud Scorer | AUC-ROC | 0.92 |
| XGBoost Fraud Scorer | CV AUC (5-fold) | 0.91 ± 0.02 |
| DecisionEngine | Precision (fraud) | 0.88 |
| DecisionEngine | Recall (fraud) | 0.85 |

Trained on 15,000 synthetic behavioral records + Amazon Customer Reviews dataset.

---

## Tech Stack

| Component | Technology |
|---|---|
| NLP Classifier | DistilBERT (HuggingFace Transformers) |
| Fraud Scorer | XGBoost + SHAP |
| Decision Engine | Custom Python logic |
| REST API | FastAPI + SQLite |
| Dashboard | Streamlit + Plotly |
| Deployment | Streamlit Cloud |

---

## Project Structure

```
ReturnSense/
├── data/
│   ├── raw/              # Raw datasets (Kaggle downloads)
│   └── processed/        # Cleaned + split datasets
├── models/
│   ├── nlp/
│   │   ├── train_nlp.py       # DistilBERT fine-tuning
│   │   └── nlp_inference.py   # Inference pipeline
│   └── xgboost/
│       └── train_xgb.py       # XGBoost training
├── api/
│   └── main.py           # FastAPI backend
├── dashboard/
│   └── app.py            # Streamlit dashboard
├── utils/
│   ├── config.py                  # All constants
│   ├── generate_synthetic_data.py # Synthetic data generator
│   ├── prepare_nlp_data.py        # NLP dataset preparation
│   └── decision_engine.py         # Core logic
├── tests/
├── requirements.txt
└── README.md
```

---

## Quickstart

```bash
# 1. Clone repo
git clone https://github.com/YOUR_USERNAME/ReturnSense.git
cd ReturnSense

# 2. Install dependencies
pip install -r requirements.txt

# 3. Generate data
python utils/generate_synthetic_data.py
python utils/prepare_nlp_data.py

# 4. Train models (use Google Colab for NLP training)
python models/nlp/train_nlp.py
python models/xgboost/train_xgb.py

# 5. Start API
uvicorn api.main:app --reload --port 8000

# 6. Start dashboard
streamlit run dashboard/app.py
```

---

## Return Categories Detected

| Category | Description |
|---|---|
| `genuine_damage` | Product arrived broken or defective |
| `buyers_remorse` | Customer changed mind |
| `fraud` | Empty box, missing product, counterfeit |
| `wardrobing` | Used and returning |
| `wrong_item` | Wrong size/color/model delivered |

---

## Inspiration
This project was inspired by real e-commerce challenges explored in Flipkart GRiD engineering hackathons, where return fraud detection is a critical business problem at scale.
