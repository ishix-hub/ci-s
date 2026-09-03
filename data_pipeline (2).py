"""
Data pipeline for "The Warranty Black Box" project.

Source: AI4I 2020 Predictive Maintenance Dataset (Matzka, 2020), UCI Machine
Learning Repository, dataset ID 601. This is the REAL dataset (not synthetic
by us) - 10,000 operating records from industrial milling machines, with a
binary failure flag and 5 sub-failure-mode flags (TWF, HDF, PWF, OSF, RNF).
Citation: S. Matzka, "Explainable Artificial Intelligence for Predictive
Maintenance Applications," 2020 IEEE Third International Conference on
Artificial Intelligence for Industries (AI4I), 2020.

FRAMING FOR THIS PROJECT: each row represents one manufactured unit's
in-service operating parameters at a point in time. "Machine failure" is
used as a direct proxy for a warranty-claim-triggering field failure - i.e.
this is real sensor/operating data standing in for the sensor telemetry a
modern connected product (vehicle, appliance, industrial machine) would
report back to the manufacturer before or at the point of failure. This
substitution is disclosed explicitly, as recommended when a proxy dataset
is used in place of a company's proprietary warranty database.
"""

import pandas as pd
import numpy as np

RAW_PATH = "ai4i2020.csv"


def load_raw():
    df = pd.read_csv(RAW_PATH)
    df.columns = [c.strip() for c in df.columns]
    return df


def clean_and_rename(df):
    """Rename into warranty/manufacturing-friendly business names and do
    basic validation cleaning."""
    df = df.rename(columns={
        "UDI": "unit_id",
        "Product ID": "product_id",
        "Type": "product_tier",          # L / M / H quality variant
        "Air temperature [K]": "ambient_temp_k",
        "Process temperature [K]": "process_temp_k",
        "Rotational speed [rpm]": "rotational_speed_rpm",
        "Torque [Nm]": "torque_nm",
        "Tool wear [min]": "tool_wear_min",
        "Machine failure": "failure",          # proxy for "warranty claim filed"
        "TWF": "mode_tool_wear",
        "HDF": "mode_heat_dissipation",
        "PWF": "mode_power",
        "OSF": "mode_overstrain",
        "RNF": "mode_random",
    })

    # basic validation
    df = df.drop_duplicates()
    df = df.dropna()
    numeric_cols = ["ambient_temp_k", "process_temp_k", "rotational_speed_rpm",
                     "torque_nm", "tool_wear_min"]
    for c in numeric_cols:
        df = df[df[c] >= 0]  # drop any physically impossible negative readings

    return df.reset_index(drop=True)


def engineer_features(df):
    """Add derived features used in the published literature as strong
    predictors (temperature differential, power proxy)."""
    d = df.copy()
    d["temp_differential_k"] = d["process_temp_k"] - d["ambient_temp_k"]
    d["power_proxy_w"] = d["torque_nm"] * d["rotational_speed_rpm"] * (2 * np.pi / 60)
    d["tier_L"] = (d["product_tier"] == "L").astype(int)
    d["tier_M"] = (d["product_tier"] == "M").astype(int)
    d["tier_H"] = (d["product_tier"] == "H").astype(int)
    return d


def load_data():
    df = load_raw()
    df = clean_and_rename(df)
    df = engineer_features(df)
    return df


if __name__ == "__main__":
    df = load_data()
    print(f"Loaded {len(df)} units after cleaning.")
    print(f"Failure rate: {df['failure'].mean()*100:.2f}%")
    print(df.head())
    df.to_csv("warranty_units_clean.csv", index=False)
    print("Saved warranty_units_clean.csv")
