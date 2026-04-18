"""
Day 8 Task — FastAPI backend.
Run: uvicorn api.main:app --reload --port 8000
"""

import os
import sys
import sqlite3
import json
from datetime import datetime
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.config import DB_PATH, VERDICT_APPROVE, VERDICT_REJECT, VERDICT_MANUAL
from utils.decision_engine import DecisionEngine

app    = FastAPI(title="ReturnSense API", version="1.0.0")
engine = None  # Lazy loaded on first request

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)

# ── DB Setup ───────────────────────────────────────────────────────────
def init_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS return_requests (
            id               INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp        TEXT,
            return_text      TEXT,
            verdict          TEXT,
            fraud_probability REAL,
            nlp_category     TEXT,
            nlp_confidence   REAL,
            risk_level       TEXT,
            order_value      REAL,
            product_category TEXT,
            explanation      TEXT
        )
    """)
    conn.commit()
    conn.close()

init_db()


def get_engine() -> DecisionEngine:
    global engine
    if engine is None:
        engine = DecisionEngine()
    return engine


# ── Request / Response schemas ─────────────────────────────────────────
class BehavioralFeatures(BaseModel):
    total_orders:     int   = Field(..., ge=1)
    total_returns:    int   = Field(..., ge=0)
    return_rate:      float = Field(..., ge=0.0, le=1.0)
    account_age_days: int   = Field(..., ge=0)
    order_value:      float = Field(..., gt=0)
    days_to_return:   int   = Field(..., ge=0)
    is_suspicious_pin: int  = Field(0, ge=0, le=1)
    high_value_flag:  int   = Field(0, ge=0, le=1)
    new_account_flag: int   = Field(0, ge=0, le=1)
    high_return_cat:  int   = Field(0, ge=0, le=1)
    product_category: Optional[str] = "Unknown"


class AnalyzeRequest(BaseModel):
    return_text:          str
    behavioral_features:  BehavioralFeatures


class AnalyzeResponse(BaseModel):
    verdict:           str
    fraud_probability: float
    nlp_category:      str
    nlp_confidence:    float
    risk_level:        str
    explanation:       str
    all_scores:        dict


# ── Endpoints ──────────────────────────────────────────────────────────
@app.get("/")
def root():
    return {"message": "ReturnSense API is running!", "version": "1.0.0"}


@app.get("/health")
def health():
    return {"status": "ok", "timestamp": datetime.now().isoformat()}


@app.post("/analyze", response_model=AnalyzeResponse)
def analyze_return(request: AnalyzeRequest):
    """
    Analyze a return request and return a fraud verdict.
    """
    try:
        eng    = get_engine()
        bf     = request.behavioral_features.model_dump()
        result = eng.analyze(request.return_text, bf)

        # Store in DB
        conn = sqlite3.connect(DB_PATH)
        conn.execute("""
            INSERT INTO return_requests
            (timestamp, return_text, verdict, fraud_probability,
             nlp_category, nlp_confidence, risk_level,
             order_value, product_category, explanation)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            datetime.now().isoformat(),
            request.return_text[:500],
            result["verdict"],
            result["fraud_probability"],
            result["nlp_category"],
            result["nlp_confidence"],
            result["risk_level"],
            bf.get("order_value", 0),
            bf.get("product_category", "Unknown"),
            result["explanation"],
        ))
        conn.commit()
        conn.close()

        return result

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/stats")
def get_stats():
    """Return aggregate fraud statistics for dashboard."""
    conn   = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # Overall counts
    cursor.execute("SELECT verdict, COUNT(*) FROM return_requests GROUP BY verdict")
    verdict_counts = dict(cursor.fetchall())

    # Fraud by category
    cursor.execute("""
        SELECT product_category, COUNT(*) as total,
               SUM(CASE WHEN verdict = ? THEN 1 ELSE 0 END) as fraud_count
        FROM return_requests
        GROUP BY product_category
    """, (VERDICT_REJECT,))
    category_stats = [
        {"category": r[0], "total": r[1], "fraud_count": r[2]}
        for r in cursor.fetchall()
    ]

    # Fraud by NLP category
    cursor.execute("""
        SELECT nlp_category, COUNT(*) as count
        FROM return_requests
        WHERE verdict IN (?, ?)
        GROUP BY nlp_category ORDER BY count DESC
    """, (VERDICT_REJECT, VERDICT_MANUAL))
    fraud_by_nlp = dict(cursor.fetchall())

    # Recent 20 requests
    cursor.execute("""
        SELECT timestamp, verdict, fraud_probability, nlp_category,
               risk_level, order_value, product_category
        FROM return_requests
        ORDER BY id DESC LIMIT 20
    """)
    cols   = ["timestamp", "verdict", "fraud_prob", "nlp_category",
              "risk_level", "order_value", "category"]
    recent = [dict(zip(cols, row)) for row in cursor.fetchall()]

    conn.close()

    total = sum(verdict_counts.values()) or 1
    return {
        "total_requests":   total,
        "verdict_counts":   verdict_counts,
        "fraud_rate":       round(verdict_counts.get(VERDICT_REJECT, 0) / total, 4),
        "category_stats":   category_stats,
        "fraud_by_nlp":     fraud_by_nlp,
        "recent_requests":  recent,
    }
