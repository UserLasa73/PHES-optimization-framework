"""Leave-one-location-out validation for PHES XGBoost surrogates.

For each location:
1. Train/tune on the other two locations using grouped cross-validation.
2. Evaluate on the held-out location.
3. Discard the temporary model.

This script does NOT overwrite the deployment models in models/.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, RandomizedSearchCV

from src.model_features import SURROGATE_FEATURES


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "training_data_all_inputs.csv"
RESULT_DIR = ROOT / "results"
RESULT_DIR.mkdir(exist_ok=True)

RANDOM_SEED = 42
TARGETS = ("efficiency", "autonomy")


def calculate_metrics(actual: pd.Series | np.ndarray,
                      predicted: np.ndarray) -> dict[str, float]:
    """Return R2, MAE and RMSE."""
    return {
        "r2": float(r2_score(actual, predicted)),
        "mae": float(mean_absolute_error(actual, predicted)),
        "rmse": float(np.sqrt(mean_squared_error(actual, predicted))),
    }


def tune_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    groups: pd.Series,
) -> tuple[xgb.XGBRegressor, dict, float]:
    """Tune one temporary XGBoost model using grouped CV."""
    estimator = xgb.XGBRegressor(
        objective="reg:squarederror",
        random_state=RANDOM_SEED,
        n_jobs=-1,
    )

    parameter_distributions = {
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
        raise ValueError("At least two training locations are required.")

    grouped_cv = GroupKFold(n_splits=unique_groups)
    search = RandomizedSearchCV(
        estimator=estimator,
        param_distributions=parameter_distributions,
        n_iter=30,
        scoring="r2",
        cv=grouped_cv,
        random_state=RANDOM_SEED,
        n_jobs=-1,
        verbose=0,
    )
    search.fit(X_train, y_train, groups=groups)

    return (
        search.best_estimator_,
        search.best_params_,
        float(search.best_score_),
    )


def main() -> None:
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found at {DATA_PATH}. Run dataset generation first."
        )

    data = pd.read_csv(DATA_PATH)

    required_columns = (
        SURROGATE_FEATURES
        + ["location", "design_id", "efficiency", "autonomy"]
    )
    missing_columns = [
        column for column in required_columns if column not in data.columns
    ]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    if data[required_columns].isna().any().any():
        raise ValueError("Missing values found in required columns.")

    if data.duplicated(subset=SURROGATE_FEATURES).any():
        raise ValueError(
            "Duplicate feature vectors found. Regenerate the dataset before validation."
        )

    locations = sorted(data["location"].unique().tolist())
    if len(locations) < 3:
        raise ValueError(
            "Leave-one-location-out validation requires at least three locations."
        )

    print("=" * 76)
    print("LEAVE-ONE-LOCATION-OUT SURROGATE VALIDATION")
    print("=" * 76)
    print(f"Dataset rows: {len(data)}")
    print(f"Locations: {locations}")
    print("Temporary validation models will not overwrite deployment models.\n")

    location_rows: list[dict] = []
    prediction_frames: list[pd.DataFrame] = []
    detailed_results: dict = {
        "validation_design": (
            "Each location is held out once. Hyperparameters are tuned only on "
            "the other locations using grouped cross-validation."
        ),
        "features": SURROGATE_FEATURES,
        "folds": {},
    }

    for holdout_location in locations:
        train_data = data[data["location"] != holdout_location].copy()
        test_data = data[data["location"] == holdout_location].copy()
        training_locations = sorted(train_data["location"].unique().tolist())

        print("-" * 76)
        print(f"Held out: {holdout_location}")
        print(f"Training on: {training_locations}")
        print(f"Train rows: {len(train_data)} | Test rows: {len(test_data)}")

        X_train = train_data[SURROGATE_FEATURES]
        X_test = test_data[SURROGATE_FEATURES]
        groups = train_data["location"]

        fold_predictions = test_data[["location", "design_id"]].reset_index(
            drop=True
        )
        fold_record: dict = {
            "held_out_location": holdout_location,
            "training_locations": " + ".join(training_locations),
            "training_rows": int(len(train_data)),
            "test_rows": int(len(test_data)),
        }
        detailed_results["folds"][holdout_location] = {
            "training_locations": training_locations,
            "training_rows": int(len(train_data)),
            "test_rows": int(len(test_data)),
            "targets": {},
        }

        for target in TARGETS:
            temporary_model, best_parameters, group_cv_r2 = tune_model(
                X_train,
                train_data[target],
                groups,
            )

            predictions = temporary_model.predict(X_test)
            holdout_metrics = calculate_metrics(test_data[target], predictions)

            fold_predictions[f"actual_{target}"] = test_data[target].to_numpy()
            fold_predictions[f"predicted_{target}"] = predictions
            fold_predictions[f"residual_{target}"] = (
                test_data[target].to_numpy() - predictions
            )

            fold_record[f"{target}_group_cv_r2"] = group_cv_r2
            fold_record[f"{target}_r2"] = holdout_metrics["r2"]
            fold_record[f"{target}_mae"] = holdout_metrics["mae"]
            fold_record[f"{target}_rmse"] = holdout_metrics["rmse"]

            detailed_results["folds"][holdout_location]["targets"][target] = {
                "group_cv_r2": group_cv_r2,
                "holdout_metrics": holdout_metrics,
                "best_parameters": best_parameters,
            }

            print(
                f"  {target.capitalize():10s} — "
                f"R2: {holdout_metrics['r2']:.4f}, "
                f"MAE: {holdout_metrics['mae']:.4f}, "
                f"RMSE: {holdout_metrics['rmse']:.4f}"
            )

        location_rows.append(fold_record)
        prediction_frames.append(fold_predictions)

    metrics_table = pd.DataFrame(location_rows)
    all_predictions = pd.concat(prediction_frames, ignore_index=True)

    numeric_metric_columns = [
        column
        for column in metrics_table.columns
        if column.endswith(("_r2", "_mae", "_rmse"))
    ]

    macro_mean = {
        "held_out_location": "MACRO_MEAN",
        "training_locations": "",
        "training_rows": int(metrics_table["training_rows"].mean()),
        "test_rows": int(metrics_table["test_rows"].mean()),
    }
    macro_std = {
        "held_out_location": "MACRO_STD",
        "training_locations": "",
        "training_rows": "",
        "test_rows": "",
    }

    for column in numeric_metric_columns:
        macro_mean[column] = float(metrics_table[column].mean())
        macro_std[column] = float(metrics_table[column].std(ddof=1))

    pooled_row = {
        "held_out_location": "POOLED",
        "training_locations": "",
        "training_rows": "",
        "test_rows": int(len(all_predictions)),
    }
    for target in TARGETS:
        pooled_metrics = calculate_metrics(
            all_predictions[f"actual_{target}"],
            all_predictions[f"predicted_{target}"],
        )
        pooled_row[f"{target}_group_cv_r2"] = np.nan
        pooled_row[f"{target}_r2"] = pooled_metrics["r2"]
        pooled_row[f"{target}_mae"] = pooled_metrics["mae"]
        pooled_row[f"{target}_rmse"] = pooled_metrics["rmse"]

    final_metrics_table = pd.concat(
        [
            metrics_table,
            pd.DataFrame([macro_mean, macro_std, pooled_row]),
        ],
        ignore_index=True,
    )

    metrics_path = RESULT_DIR / "location_holdout_metrics.csv"
    predictions_path = RESULT_DIR / "location_holdout_predictions.csv"
    details_path = RESULT_DIR / "location_holdout_details.json"

    final_metrics_table.to_csv(metrics_path, index=False)
    all_predictions.to_csv(predictions_path, index=False)

    detailed_results["summary"] = {
        "macro_mean": macro_mean,
        "macro_std": macro_std,
        "pooled": pooled_row,
    }
    with details_path.open("w", encoding="utf-8") as output_file:
        json.dump(detailed_results, output_file, indent=2)

    print("\n" + "=" * 76)
    print("VALIDATION COMPLETED")
    print("=" * 76)
    print(final_metrics_table.to_string(index=False))
    print(f"\nMetrics saved to: {metrics_path}")
    print(f"Predictions saved to: {predictions_path}")
    print(f"Details saved to: {details_path}")
    print("Deployment models in models/ were not changed.")


if __name__ == "__main__":
    main()
