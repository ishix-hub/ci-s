"""
Analysis for "The Warranty Black Box".

This is a binary classification problem (severe class imbalance: 3.39%
failure rate) rather than a forecasting problem, unlike the mandi-pricing
project. Three models compared:
  - Logistic Regression (interpretable baseline)
  - Random Forest (matches Matzka 2020's original benchmark model)
  - Gradient Boosting (stronger ensemble, common in follow-up literature)

Evaluated on Precision/Recall/F1/ROC-AUC - accuracy alone is meaningless
here since predicting "no failure" for every unit already scores 96.6%.

Final output: converts model precision/recall at a chosen threshold into
a WARRANTY COST AVOIDANCE estimate - the actual managerial deliverable.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (precision_score, recall_score, f1_score,
                              roc_auc_score, confusion_matrix, precision_recall_curve)
import warnings
warnings.filterwarnings("ignore")

from data_pipeline import load_data

FEATURES = ["ambient_temp_k", "process_temp_k", "rotational_speed_rpm",
            "torque_nm", "tool_wear_min", "temp_differential_k",
            "power_proxy_w", "tier_L", "tier_M", "tier_H"]
TARGET = "failure"


def prepare_split(df, test_size=0.25, seed=42):
    X = df[FEATURES]
    y = df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=seed
    )
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    return X_train, X_test, X_train_s, X_test_s, y_train, y_test, scaler


def train_models(X_train, X_train_s, y_train):
    models = {}

    lr = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42)
    lr.fit(X_train_s, y_train)
    models["Logistic Regression"] = ("scaled", lr)

    rf = RandomForestClassifier(n_estimators=300, max_depth=10, class_weight="balanced",
                                 random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    models["Random Forest"] = ("raw", rf)

    gb = GradientBoostingClassifier(n_estimators=300, max_depth=3, learning_rate=0.05,
                                     random_state=42)
    gb.fit(X_train, y_train)
    models["Gradient Boosting"] = ("raw", gb)

    return models


def evaluate(name, kind, model, X_test, X_test_s, y_test):
    X_eval = X_test_s if kind == "scaled" else X_test
    proba = model.predict_proba(X_eval)[:, 1]
    pred = (proba >= 0.5).astype(int)

    prec = precision_score(y_test, pred, zero_division=0)
    rec = recall_score(y_test, pred, zero_division=0)
    f1 = f1_score(y_test, pred, zero_division=0)
    auc = roc_auc_score(y_test, proba)

    print(f"{name:20s}  Precision={prec:.3f}  Recall={rec:.3f}  F1={f1:.3f}  AUC={auc:.3f}")
    return {"model": name, "precision": prec, "recall": rec, "f1": f1, "auc": auc}, proba


def warranty_cost_avoidance(y_test, proba, avg_claim_cost=25000, threshold=0.3):
    """
    Translate model performance into a business number: at a given risk
    threshold, how many field failures does the model catch (true
    positives) vs miss (false negatives), and what does that mean in
    avoided claim cost if a flagged unit gets pre-emptive service instead
    of failing in the field?

    avg_claim_cost: representative average cost of a single warranty
    claim (service + parts + logistics + goodwill), in Rs. This is an
    illustrative planning figure, not a real company's cost data.
    """
    pred = (proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, pred).ravel()

    total_failures = tp + fn
    caught = tp
    missed = fn
    false_alarms = fp

    avoided_cost = caught * avg_claim_cost
    catch_rate = caught / total_failures if total_failures else 0

    return {
        "threshold": threshold,
        "total_field_failures_in_test": int(total_failures),
        "failures_caught_by_model": int(caught),
        "failures_missed": int(missed),
        "false_alarms": int(false_alarms),
        "catch_rate_pct": round(catch_rate * 100, 1),
        "estimated_cost_avoided_rs": int(avoided_cost),
    }


if __name__ == "__main__":
    df = load_data()
    X_train, X_test, X_train_s, X_test_s, y_train, y_test, scaler = prepare_split(df)

    print(f"Train: {len(X_train)} units ({y_train.sum()} failures) | "
          f"Test: {len(X_test)} units ({y_test.sum()} failures)\n")

    models = train_models(X_train, X_train_s, y_train)

    results = []
    probas = {}
    for name, (kind, model) in models.items():
        r, proba = evaluate(name, kind, model, X_test, X_test_s, y_test)
        results.append(r)
        probas[name] = proba

    # Best model = highest F1 (balances precision/recall under imbalance)
    best_name = max(results, key=lambda r: r["f1"])["model"]
    best_proba = probas[best_name]
    print(f"\nBest model by F1: {best_name}")

    print("\n=== Feature importance (Random Forest) ===")
    rf_model = models["Random Forest"][1]
    importances = sorted(zip(FEATURES, rf_model.feature_importances_), key=lambda x: -x[1])
    for f, imp in importances:
        print(f"  {f:22s} {imp:.3f}")

    print(f"\n=== Warranty Cost Avoidance ({best_name}, threshold=0.3) ===")
    impact = warranty_cost_avoidance(y_test, best_proba, threshold=0.3)
    for k, v in impact.items():
        print(f"  {k}: {v}")

    pd.DataFrame(results).to_csv("model_comparison.csv", index=False)
    pd.DataFrame(importances, columns=["feature", "importance"]).to_csv("feature_importance.csv", index=False)
    print("\nSaved model_comparison.csv and feature_importance.csv")
