"""
The Warranty Black Box - Streamlit dashboard
Run with: streamlit run streamlit_app.py

Two views:
  - Quality Engineer view: fleet risk overview, top at-risk units, root-cause
    signal (feature importance) - the design-feedback half of the product
  - Executive view: warranty cost avoidance at different risk thresholds,
    product-tier breakdown - the business-case half of the product
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from data_pipeline import load_data
from analysis import (prepare_split, train_models, evaluate, FEATURES,
                       warranty_cost_avoidance)

st.set_page_config(page_title="The Warranty Black Box", layout="wide")

st.title("Warranty Black Box")
st.caption("AI-based failure-risk scoring and design-feedback signal from in-service unit data")

st.info(
    "Built on the real UCI **AI4I 2020 Predictive Maintenance Dataset** (10,000 industrial "
    "milling-machine operating records). Machine failure is used as a direct proxy for a "
    "warranty-claim-triggering field failure - see report for full framing.",
    icon="ℹ️",
)

df = load_data()
X_train, X_test, X_train_s, X_test_s, y_train, y_test, scaler = prepare_split(df)
models = train_models(X_train, X_train_s, y_train)

# Use Gradient Boosting as the production model (best F1 in analysis.py)
gb_model = models["Gradient Boosting"][1]
rf_model = models["Random Forest"][1]
proba_test = gb_model.predict_proba(X_test)[:, 1]

view = st.sidebar.radio("View", ["Quality Engineer view", "Executive view"])

if view == "Quality Engineer view":
    st.subheader("🔧 Fleet Risk Overview")

    risk_df = X_test.copy()
    risk_df["risk_score"] = proba_test
    risk_df["actual_failure"] = y_test.values
    risk_df["product_id"] = df.loc[X_test.index, "product_id"].values
    risk_df["product_tier"] = df.loc[X_test.index, "product_tier"].values
    risk_df = risk_df.sort_values("risk_score", ascending=False)

    high_risk = (risk_df["risk_score"] >= 0.3).sum()
    c1, c2, c3 = st.columns(3)
    c1.metric("Units monitored (test set)", f"{len(risk_df):,}")
    c2.metric("Flagged high-risk (>30% score)", f"{high_risk}")
    c3.metric("Actual field failures in this set", f"{int(risk_df['actual_failure'].sum())}")

    st.markdown("#### Top 10 highest-risk units")
    display_cols = ["product_id", "product_tier", "risk_score", "torque_nm",
                     "tool_wear_min", "rotational_speed_rpm", "actual_failure"]
    top10 = risk_df[display_cols].head(10).copy()
    top10["risk_score"] = (top10["risk_score"] * 100).round(1).astype(str) + "%"
    top10["actual_failure"] = top10["actual_failure"].map({1: "Failed", 0: "OK"})
    st.dataframe(top10, use_container_width=True, hide_index=True)

    st.markdown("#### Root-cause signal: what's driving failure risk")
    importances = sorted(zip(FEATURES, rf_model.feature_importances_), key=lambda x: -x[1])
    imp_df = pd.DataFrame(importances, columns=["feature", "importance"])
    fig = px.bar(imp_df, x="importance", y="feature", orientation="h",
                 title="Feature importance - this is the signal that should route back to design/R&D")
    fig.update_layout(yaxis={"categoryorder": "total ascending"}, height=380)
    st.plotly_chart(fig, use_container_width=True)
    st.caption(
        "Torque, rotational speed, and tool wear dominate - consistent with published findings on "
        "this dataset. In a real deployment, this ranking is the concrete artifact quality "
        "engineering hands to design: 'these operating conditions predict failure, tighten "
        "tolerances or add sensing here.'"
    )

else:
    st.subheader("💰 Executive View — Warranty Cost Impact")

    avg_cost = st.number_input("Average cost per warranty claim (Rs.)", min_value=1000,
                                value=25000, step=1000)
    threshold = st.slider("Risk-flagging threshold", 0.05, 0.9, 0.3, 0.05,
                            help="Units scored above this are flagged for pre-emptive service")

    impact = warranty_cost_avoidance(y_test, proba_test, avg_claim_cost=avg_cost, threshold=threshold)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Failures caught", impact["failures_caught_by_model"])
    c2.metric("Failures missed", impact["failures_missed"])
    c3.metric("False alarms", impact["false_alarms"])
    c4.metric("Catch rate", f"{impact['catch_rate_pct']}%")

    st.metric("Estimated cost avoided (this test batch)", f"₹{impact['estimated_cost_avoided_rs']:,}")
    st.caption(
        f"Scaled to a fleet of 100,000 units at the same {df['failure'].mean()*100:.1f}% base failure "
        f"rate, this catch rate would represent roughly "
        f"₹{int(impact['estimated_cost_avoided_rs'] * (100000/len(X_test))):,} in avoided claims per cycle, "
        f"against the false-alarm cost of {impact['false_alarms']} unnecessary preemptive services."
    )

    st.markdown("#### Precision vs. Recall trade-off")
    thresholds = np.arange(0.05, 0.95, 0.05)
    rows = []
    for t in thresholds:
        imp = warranty_cost_avoidance(y_test, proba_test, avg_claim_cost=avg_cost, threshold=t)
        rows.append({"threshold": t, "catch_rate": imp["catch_rate_pct"],
                     "false_alarms": imp["false_alarms"]})
    trade_df = pd.DataFrame(rows)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=trade_df["threshold"], y=trade_df["catch_rate"],
                              name="Catch rate (%)", yaxis="y1"))
    fig.add_trace(go.Scatter(x=trade_df["threshold"], y=trade_df["false_alarms"],
                              name="False alarms (count)", yaxis="y2"))
    fig.update_layout(
        xaxis_title="Risk-flagging threshold",
        yaxis=dict(title="Catch rate (%)"),
        yaxis2=dict(title="False alarms", overlaying="y", side="right"),
        height=380,
    )
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Lower thresholds catch more failures but generate more false alarms (wasted "
               "pre-emptive service visits) - the threshold slider above lets a manufacturer "
               "pick the operating point that matches their service capacity and cost tolerance.")

    st.markdown("#### Failure rate by product tier")
    tier_df = df.groupby("product_tier")["failure"].agg(["mean", "count"]).reset_index()
    tier_df["mean"] = (tier_df["mean"] * 100).round(2)
    tier_df.columns = ["Product Tier", "Failure Rate (%)", "Unit Count"]
    fig2 = px.bar(tier_df, x="Product Tier", y="Failure Rate (%)", text="Unit Count",
                  title="Failure rate by product quality tier (L/M/H)")
    st.plotly_chart(fig2, use_container_width=True)
