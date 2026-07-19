"""Reproduce the leakage audit for the invalidated legacy dataset."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import GroupShuffleSplit, train_test_split

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "legacy_invalidated" / "training_data_all_inputs.csv"
RESULT_DIR = ROOT / "results"
RESULT_DIR.mkdir(exist_ok=True)

FEATURES = [
    "volume_m3",
    "head_m",
    "pipe_diameter_m",
    "pump_power_kw",
    "turbine_power_kw",
    "pv_kwp",
    "daily_energy_kwh",
    "evaporation_rate_mm_month",
    "reservoir_type_code",
]


def evaluate_split(X_train, X_test, y_train, y_test):
    model = xgb.XGBRegressor(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=4,
        objective="reg:squarederror",
    )
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    return {
        "r2": float(r2_score(y_test, predictions)),
        "mae": float(mean_absolute_error(y_test, predictions)),
    }


def main():
    df = pd.read_csv(DATA_PATH)
    X = df[FEATURES]
    y = df["efficiency"]

    random_train, random_test = train_test_split(
        np.arange(len(df)), test_size=0.2, random_state=42
    )
    random_metrics = evaluate_split(
        X.iloc[random_train], X.iloc[random_test], y.iloc[random_train], y.iloc[random_test]
    )

    # Legacy generation used the same 400 LHS rows in each of eight location blocks.
    design_group = np.arange(len(df)) % 400
    grouped_split = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    grouped_train, grouped_test = next(
        grouped_split.split(X, y, groups=design_group)
    )
    grouped_metrics = evaluate_split(
        X.iloc[grouped_train], X.iloc[grouped_test], y.iloc[grouped_train], y.iloc[grouped_test]
    )

    result = {
        "rows": int(len(df)),
        "unique_feature_vectors": int(len(df.drop_duplicates(subset=FEATURES))),
        "rows_belonging_to_duplicated_feature_vectors": int(
            df.duplicated(subset=FEATURES, keep=False).sum()
        ),
        "random_row_split": random_metrics,
        "grouped_unseen_design_split": grouped_metrics,
    }
    output = RESULT_DIR / "legacy_dataset_leakage_audit.json"
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    print(f"Saved: {output}")


if __name__ == "__main__":
    main()
