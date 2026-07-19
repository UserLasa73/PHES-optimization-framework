"""Train and evaluate XGBoost surrogate models without duplicate-row leakage."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, RandomizedSearchCV

from src.model_features import SURROGATE_FEATURES

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "training_data_all_inputs.csv"
MODEL_DIR = ROOT / "models"
RESULT_DIR = ROOT / "results"
MODEL_DIR.mkdir(exist_ok=True)
RESULT_DIR.mkdir(exist_ok=True)

HOLDOUT_LOCATION = "Anuradhapura"
RANDOM_SEED = 42


def _metrics(actual, predicted):
    return {
        "r2": float(r2_score(actual, predicted)),
        "mae": float(mean_absolute_error(actual, predicted)),
        "rmse": float(np.sqrt(mean_squared_error(actual, predicted))),
    }


def _fit_model(X_train, y_train, groups):
    estimator = xgb.XGBRegressor(
        objective="reg:squarederror",
        random_state=RANDOM_SEED,
        n_jobs=-1,
    )
    distributions = {
        "n_estimators": [100, 200, 300, 500],
        "learning_rate": [0.03, 0.05, 0.10, 0.15],
        "max_depth": [3, 4, 5, 6, 7],
        "min_child_weight": [1, 3, 5],
        "subsample": [0.7, 0.8, 0.9, 1.0],
        "colsample_bytree": [0.7, 0.8, 0.9, 1.0],
        "reg_alpha": [0.0, 0.01, 0.1],
        "reg_lambda": [1.0, 2.0, 5.0],
    }
    # Groups are locations. This prevents the same site from appearing in both
    # training and validation folds during hyperparameter selection.
    cv = GroupKFold(n_splits=min(5, groups.nunique()))
    search = RandomizedSearchCV(
        estimator,
        param_distributions=distributions,
        n_iter=30,
        scoring="r2",
        cv=cv,
        random_state=RANDOM_SEED,
        n_jobs=-1,
        verbose=1,
    )
    search.fit(X_train, y_train, groups=groups)
    return search.best_estimator_, search.best_params_, float(search.best_score_)


def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Training dataset not found at {DATA_PATH}. Run "
            "`python -m scripts.generate_dataset` first."
        )

    df = pd.read_csv(DATA_PATH)
    missing = [column for column in SURROGATE_FEATURES if column not in df.columns]
    if missing:
        raise ValueError(
            "Dataset uses the old feature schema. Regenerate it before training. "
            f"Missing columns: {missing}"
        )
    if "location" not in df.columns:
        raise ValueError("Dataset must retain location for grouped validation.")
    if df.duplicated(subset=SURROGATE_FEATURES).any():
        raise ValueError(
            "Duplicate feature vectors detected. This would leak nearly identical "
            "designs across train/test splits. Regenerate the dataset."
        )

    train_df = df[df["location"] != HOLDOUT_LOCATION].copy()
    test_df = df[df["location"] == HOLDOUT_LOCATION].copy()
    if train_df.empty or test_df.empty:
        raise ValueError(f"Could not create holdout set for {HOLDOUT_LOCATION}.")

    X_train = train_df[SURROGATE_FEATURES]
    X_test = test_df[SURROGATE_FEATURES]
    groups = train_df["location"]

    all_results = {
        "holdout_location": HOLDOUT_LOCATION,
        "training_rows": int(len(train_df)),
        "test_rows": int(len(test_df)),
        "features": SURROGATE_FEATURES,
        "models": {},
    }

    model_specs = {
        "efficiency": ("efficiency", MODEL_DIR / "xgboost_efficiency.pkl"),
        "autonomy": ("autonomy", MODEL_DIR / "xgboost_autonomy.pkl"),
    }

    prediction_output = test_df[["location", "design_id"]].copy()
    for model_name, (target, model_path) in model_specs.items():
        print(f"\nTraining {model_name} model...")
        model, best_params, best_cv_r2 = _fit_model(
            X_train, train_df[target], groups
        )
        predictions = model.predict(X_test)
        test_metrics = _metrics(test_df[target], predictions)
        joblib.dump(model, model_path)

        prediction_output[f"actual_{target}"] = test_df[target].to_numpy()
        prediction_output[f"predicted_{target}"] = predictions
        prediction_output[f"residual_{target}"] = (
            test_df[target].to_numpy() - predictions
        )

        all_results["models"][model_name] = {
            "target": target,
            "best_group_cv_r2": best_cv_r2,
            "best_parameters": best_params,
            "holdout_metrics": test_metrics,
        }
        print(f"Holdout metrics: {test_metrics}")

    joblib.dump(SURROGATE_FEATURES, MODEL_DIR / "feature_names.pkl")
    prediction_output.to_csv(RESULT_DIR / "surrogate_holdout_predictions.csv", index=False)
    with open(RESULT_DIR / "surrogate_metrics.json", "w", encoding="utf-8") as handle:
        json.dump(all_results, handle, indent=2)

    efficiency_model = joblib.load(MODEL_DIR / "xgboost_efficiency.pkl")
    importance = pd.DataFrame(
        {
            "feature": SURROGATE_FEATURES,
            "importance": efficiency_model.feature_importances_,
        }
    ).sort_values("importance", ascending=False)
    importance.to_csv(RESULT_DIR / "efficiency_feature_importance.csv", index=False)

    print(f"\nModels saved to {MODEL_DIR}")
    print(f"Evaluation artifacts saved to {RESULT_DIR}")


if __name__ == "__main__":
    main()
