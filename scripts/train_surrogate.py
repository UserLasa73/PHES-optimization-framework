"""Train and evaluate XGBoost surrogate models for the three-location dataset.

Validation design:
1. Vavuniya is held out as an unseen location for final evaluation.
2. Hyperparameters are selected using grouped cross-validation on Colombo and Jaffna.
3. After reporting holdout metrics, a final deployment model is retrained on all
   three locations using the selected hyperparameters.
"""

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

HOLDOUT_LOCATION = "Vavuniya"
RANDOM_SEED = 42


def metrics(actual, predicted):
    return {
        "r2": float(r2_score(actual, predicted)),
        "mae": float(mean_absolute_error(actual, predicted)),
        "rmse": float(np.sqrt(mean_squared_error(actual, predicted))),
    }


def tune_model(X_train, y_train, groups):
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

    unique_groups = int(groups.nunique())
    if unique_groups < 2:
        raise ValueError("At least two training locations are required for grouped CV.")

    cv = GroupKFold(n_splits=unique_groups)
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


def final_model(best_params):
    return xgb.XGBRegressor(
        objective="reg:squarederror",
        random_state=RANDOM_SEED,
        n_jobs=-1,
        **best_params,
    )


def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Training dataset not found at {DATA_PATH}. "
            "Run `python -m scripts.generate_dataset` first."
        )

    df = pd.read_csv(DATA_PATH)

    required = SURROGATE_FEATURES + [
        "location",
        "design_id",
        "efficiency",
        "autonomy",
    ]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")

    if df[required].isna().any().any():
        raise ValueError("Missing values detected in required training columns.")

    if df.duplicated(subset=SURROGATE_FEATURES).any():
        raise ValueError(
            "Duplicate feature vectors detected. Regenerate the dataset before training."
        )

    locations = sorted(df["location"].unique().tolist())
    if HOLDOUT_LOCATION not in locations:
        raise ValueError(
            f"Holdout location {HOLDOUT_LOCATION!r} is unavailable. "
            f"Dataset locations: {locations}"
        )

    train_df = df[df["location"] != HOLDOUT_LOCATION].copy()
    test_df = df[df["location"] == HOLDOUT_LOCATION].copy()

    print("=" * 72)
    print("XGBOOST SURROGATE TRAINING — THREE-LOCATION DATASET")
    print("=" * 72)
    print(f"Dataset rows: {len(df)}")
    print(f"Locations: {locations}")
    print(f"Training locations: {sorted(train_df['location'].unique().tolist())}")
    print(f"Unseen holdout location: {HOLDOUT_LOCATION}")
    print(f"Training rows: {len(train_df)} | Holdout rows: {len(test_df)}")

    X_train = train_df[SURROGATE_FEATURES]
    X_test = test_df[SURROGATE_FEATURES]
    X_all = df[SURROGATE_FEATURES]
    groups = train_df["location"]

    results = {
        "validation_design": (
            "Hyperparameter tuning on Colombo/Jaffna with grouped CV; "
            "Vavuniya held out for unseen-location evaluation; final models "
            "retrained on all three locations."
        ),
        "holdout_location": HOLDOUT_LOCATION,
        "locations": locations,
        "training_rows": int(len(train_df)),
        "holdout_rows": int(len(test_df)),
        "features": SURROGATE_FEATURES,
        "models": {},
    }

    prediction_output = test_df[["location", "design_id"]].reset_index(drop=True)

    model_specs = {
        "efficiency": ("efficiency", MODEL_DIR / "xgboost_efficiency.pkl"),
        "autonomy": ("autonomy", MODEL_DIR / "xgboost_autonomy.pkl"),
    }

    final_models = {}

    for model_name, (target, model_path) in model_specs.items():
        print(f"\nTraining and validating {model_name} model...")

        tuned_model, best_params, best_group_cv_r2 = tune_model(
            X_train,
            train_df[target],
            groups,
        )

        holdout_predictions = tuned_model.predict(X_test)
        holdout_metrics = metrics(test_df[target], holdout_predictions)

        prediction_output[f"actual_{target}"] = test_df[target].to_numpy()
        prediction_output[f"predicted_{target}"] = holdout_predictions
        prediction_output[f"residual_{target}"] = (
            test_df[target].to_numpy() - holdout_predictions
        )

        # Retrain the deployment model on all three locations after evaluation.
        deployment_model = final_model(best_params)
        deployment_model.fit(X_all, df[target])
        joblib.dump(deployment_model, model_path)
        final_models[model_name] = deployment_model

        results["models"][model_name] = {
            "target": target,
            "best_group_cv_r2_on_training_locations": best_group_cv_r2,
            "best_parameters": best_params,
            "unseen_location_holdout_metrics": holdout_metrics,
            "deployment_training_rows": int(len(df)),
        }

        print(f"  Group-CV R2: {best_group_cv_r2:.4f}")
        print(
            "  Vavuniya holdout — "
            f"R2: {holdout_metrics['r2']:.4f}, "
            f"MAE: {holdout_metrics['mae']:.4f}, "
            f"RMSE: {holdout_metrics['rmse']:.4f}"
        )
        print(f"  Final model saved: {model_path}")

    joblib.dump(SURROGATE_FEATURES, MODEL_DIR / "feature_names.pkl")

    prediction_output.to_csv(
        RESULT_DIR / "surrogate_holdout_predictions.csv",
        index=False,
    )

    with (RESULT_DIR / "surrogate_metrics.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(results, handle, indent=2)

    importance = pd.DataFrame(
        {
            "feature": SURROGATE_FEATURES,
            "importance": final_models["efficiency"].feature_importances_,
        }
    ).sort_values("importance", ascending=False)
    importance.to_csv(
        RESULT_DIR / "efficiency_feature_importance.csv",
        index=False,
    )

    print("\nTraining completed.")
    print(f"Models saved to: {MODEL_DIR}")
    print(f"Evaluation files saved to: {RESULT_DIR}")


if __name__ == "__main__":
    main()
