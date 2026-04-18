"""
Day 9 Task — Streamlit dashboard.
Run: streamlit run dashboard/app.py
"""

import os
import sys
import requests
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.config import VERDICT_APPROVE, VERDICT_REJECT, VERDICT_MANUAL

API_URL = os.getenv("API_URL", "http://localhost:8000")

st.set_page_config(
    page_title="ReturnSense — Fraud Detection",
    page_icon="🔍",
    layout="wide",
)

# ── Helpers ────────────────────────────────────────────────────────────
VERDICT_COLOR = {
    VERDICT_APPROVE: "#1D9E75",
    VERDICT_REJECT:  "#E24B4A",
    VERDICT_MANUAL:  "#EF9F27",
}

RISK_COLOR = {"LOW": "#1D9E75", "MEDIUM": "#EF9F27", "HIGH": "#E24B4A"}


def verdict_badge(verdict: str) -> str:
    color = VERDICT_COLOR.get(verdict, "#888")
    return f'<span style="background:{color};color:white;padding:3px 10px;border-radius:5px;font-size:13px;">{verdict}</span>'


def fetch_stats():
    try:
        r = requests.get(f"{API_URL}/stats", timeout=5)
        return r.json() if r.status_code == 200 else {}
    except Exception:
        return {}


# ── Sidebar ────────────────────────────────────────────────────────────
st.sidebar.title("ReturnSense")
st.sidebar.caption("AI-powered return fraud detection")
page = st.sidebar.radio("Navigate", ["Live Analyzer", "Analytics Dashboard", "Model Metrics"])
st.sidebar.markdown("---")
st.sidebar.info("Built for Flipkart GRiD 8.0\n\nStack: DistilBERT + XGBoost + FastAPI")


# ══════════════════════════════════════════════════════════════════════
# PAGE 1 — Live Analyzer
# ══════════════════════════════════════════════════════════════════════
if page == "Live Analyzer":
    st.title("Live Return Fraud Analyzer")
    st.caption("Enter a return request to get an instant fraud verdict.")

    col1, col2 = st.columns([1.2, 1])

    with col1:
        st.subheader("Return Details")
        return_text = st.text_area(
            "Customer's return reason",
            placeholder="e.g. The product arrived completely broken, screen is cracked...",
            height=120,
        )

        st.subheader("Customer Behavioral Signals")
        c1, c2, c3 = st.columns(3)
        total_orders   = c1.number_input("Total Orders",   min_value=1,   value=20)
        total_returns  = c2.number_input("Total Returns",  min_value=0,   value=3)
        account_age    = c3.number_input("Account Age (days)", min_value=0, value=365)

        c4, c5, c6 = st.columns(3)
        order_value    = c4.number_input("Order Value (₹)", min_value=1.0, value=2500.0)
        days_to_return = c5.number_input("Days to Return", min_value=0,   value=7)
        category       = c6.selectbox("Product Category",
            ["Electronics", "Fashion", "HomeAppliances", "Books", "Beauty", "Sports", "Grocery"])

        pincode = st.text_input("Customer Pincode", value="400001")

        return_rate = round(total_returns / total_orders, 4) if total_orders > 0 else 0

        if st.button("Analyze Return Request", type="primary", use_container_width=True):
            if not return_text.strip():
                st.warning("Please enter the customer's return reason.")
            else:
                payload = {
                    "return_text": return_text,
                    "behavioral_features": {
                        "total_orders":     total_orders,
                        "total_returns":    total_returns,
                        "return_rate":      return_rate,
                        "account_age_days": account_age,
                        "order_value":      order_value,
                        "days_to_return":   days_to_return,
                        "is_suspicious_pin": 0,
                        "high_value_flag":  int(order_value > 5000),
                        "new_account_flag": int(account_age < 90),
                        "high_return_cat":  int(category in ["Electronics", "Fashion"]),
                        "product_category": category,
                    },
                }
                with st.spinner("Analyzing..."):
                    try:
                        r = requests.post(f"{API_URL}/analyze", json=payload, timeout=30)
                        if r.status_code == 200:
                            st.session_state["result"] = r.json()
                        else:
                            st.error(f"API Error: {r.text}")
                    except Exception as e:
                        st.error(f"Could not connect to API: {e}")

    with col2:
        st.subheader("Analysis Result")
        result = st.session_state.get("result")

        if result:
            verdict = result["verdict"]
            color   = VERDICT_COLOR.get(verdict, "#888")

            st.markdown(f"### {verdict_badge(verdict)}", unsafe_allow_html=True)
            st.markdown(f"**Risk Level:** <span style='color:{RISK_COLOR[result['risk_level']]}'>{result['risk_level']}</span>", unsafe_allow_html=True)

            st.markdown("---")

            # Fraud probability gauge
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=result["fraud_probability"] * 100,
                number={"suffix": "%"},
                title={"text": "Fraud Probability"},
                gauge={
                    "axis": {"range": [0, 100]},
                    "bar": {"color": color},
                    "steps": [
                        {"range": [0, 40],  "color": "#E1F5EE"},
                        {"range": [40, 65], "color": "#FAEEDA"},
                        {"range": [65, 100],"color": "#FCEBEB"},
                    ],
                },
            ))
            fig.update_layout(height=220, margin=dict(t=40, b=10, l=20, r=20))
            st.plotly_chart(fig, use_container_width=True)

            # NLP scores
            st.markdown("**Return Reason Scores**")
            scores_df = pd.DataFrame(
                list(result["all_scores"].items()),
                columns=["Category", "Score"]
            ).sort_values("Score", ascending=True)
            fig2 = px.bar(scores_df, x="Score", y="Category", orientation="h",
                          color="Score", color_continuous_scale="Blues")
            fig2.update_layout(height=200, margin=dict(t=10, b=10, l=10, r=10),
                               showlegend=False, coloraxis_showscale=False)
            st.plotly_chart(fig2, use_container_width=True)

            st.info(result["explanation"])
        else:
            st.info("Submit a return request to see the analysis here.")


# ══════════════════════════════════════════════════════════════════════
# PAGE 2 — Analytics Dashboard
# ══════════════════════════════════════════════════════════════════════
elif page == "Analytics Dashboard":
    st.title("Fraud Analytics Dashboard")

    stats = fetch_stats()
    if not stats:
        st.warning("No data yet. Analyze some return requests first!")
    else:
        total    = stats.get("total_requests", 0)
        verdicts = stats.get("verdict_counts", {})
        fraud_r  = stats.get("fraud_rate", 0)

        # KPI metrics
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Requests",  total)
        c2.metric("Auto Approved",   verdicts.get(VERDICT_APPROVE, 0))
        c3.metric("Auto Rejected",   verdicts.get(VERDICT_REJECT, 0))
        c4.metric("Fraud Rate",      f"{fraud_r:.1%}")

        st.markdown("---")

        col1, col2 = st.columns(2)

        # Verdict pie
        with col1:
            st.subheader("Verdict Distribution")
            if verdicts:
                fig = px.pie(
                    names=list(verdicts.keys()),
                    values=list(verdicts.values()),
                    color=list(verdicts.keys()),
                    color_discrete_map=VERDICT_COLOR,
                )
                fig.update_layout(height=300)
                st.plotly_chart(fig, use_container_width=True)

        # Fraud by NLP category
        with col2:
            st.subheader("Fraud by Return Reason")
            fraud_nlp = stats.get("fraud_by_nlp", {})
            if fraud_nlp:
                fig = px.bar(
                    x=list(fraud_nlp.values()),
                    y=list(fraud_nlp.keys()),
                    orientation="h",
                    color_discrete_sequence=["#E24B4A"],
                )
                fig.update_layout(height=300)
                st.plotly_chart(fig, use_container_width=True)

        # Category stats table
        st.subheader("Fraud by Product Category")
        cat_stats = stats.get("category_stats", [])
        if cat_stats:
            df = pd.DataFrame(cat_stats)
            df["fraud_rate"] = (df["fraud_count"] / df["total"]).map("{:.1%}".format)
            st.dataframe(df, use_container_width=True, hide_index=True)

        # Recent requests
        st.subheader("Recent Return Requests")
        recent = stats.get("recent_requests", [])
        if recent:
            df_recent = pd.DataFrame(recent)
            st.dataframe(df_recent, use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════════════════
# PAGE 3 — Model Metrics
# ══════════════════════════════════════════════════════════════════════
elif page == "Model Metrics":
    st.title("Model Performance Metrics")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("NLP Classifier — DistilBERT")
        import json, os
        from utils.config import MODEL_NLP
        history_path = os.path.join(MODEL_NLP, "training_history.json")
        if os.path.exists(history_path):
            with open(history_path) as f:
                h = json.load(f)
            st.metric("Best Val Accuracy", f"{h['best_val_acc']:.2%}")
            st.metric("Test Accuracy",     f"{h['test_acc']:.2%}")
            history_df = pd.DataFrame(h["history"])
            fig = px.line(history_df, x="epoch",
                          y=["train_acc", "val_acc"],
                          title="Training Accuracy")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Train the NLP model first (models/nlp/train_nlp.py)")

    with col2:
        st.subheader("XGBoost Fraud Scorer")
        from utils.config import MODEL_XGB
        metrics_path = os.path.join(MODEL_XGB, "metrics.json")
        feat_img     = os.path.join(MODEL_XGB, "feature_importance.png")
        if os.path.exists(metrics_path):
            with open(metrics_path) as f:
                m = json.load(f)
            st.metric("Test AUC-ROC",  f"{m['test_auc']:.4f}")
            st.metric("CV AUC (mean)", f"{m['cv_auc_mean']:.4f} ± {m['cv_auc_std']:.4f}")
            if os.path.exists(feat_img):
                st.image(feat_img, caption="Feature Importances", use_column_width=True)
        else:
            st.info("Train the XGBoost model first (models/xgboost/train_xgb.py)")
