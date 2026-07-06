"""
train_surrogate.py
Train XGBoost surrogate models for efficiency and cost predictions.
Matches research proposal specifications:
- 80/20 train-test split
- 5-fold cross-validation
- Grid search for hyperparameters
- R2 > 0.95, MAE < 5%
- Feature importance analysis
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

print("=" * 70)
print("XGBOOST SURROGATE TRAINING")
print("=" * 70)

# ============================================================================
# LOAD DATA
# ============================================================================

df = pd.read_csv('training_data_2000_samples.csv')
print(f"Loaded {len(df)} samples")

# ============================================================================
# FEATURES AND TARGETS
# ============================================================================

features = ['volume_m3', 'head_m', 'pipe_diameter_m', 'pump_power_kw', 'turbine_power_kw']
X = df[features].values
y_eff = df['efficiency'].values
y_cost = df['cost'].values

print(f"\nFeatures: {features}")
print(f"Targets: efficiency (%), cost (LKR)")

# ============================================================================
# 80/20 TRAIN-TEST SPLIT
# ============================================================================

X_train, X_test, y_eff_train, y_eff_test = train_test_split(
    X, y_eff, test_size=0.2, random_state=42
)

_, _, y_cost_train, y_cost_test = train_test_split(
    X, y_cost, test_size=0.2, random_state=42
)

print(f"\nTraining set: {len(X_train)} samples (80%)")
print(f"Test set: {len(X_test)} samples (20%)")

# ============================================================================
# EFFICIENCY MODEL WITH GRID SEARCH + 5-FOLD CV
# ============================================================================

print("\n" + "-" * 70)
print("MODEL 1: EFFICIENCY")
print("-" * 70)

print("Performing grid search with 5-fold cross-validation...")

param_grid = {
    'n_estimators': [100, 200, 300],
    'learning_rate': [0.05, 0.1, 0.2],
    'max_depth': [3, 5, 7],
    'subsample': [0.7, 0.8, 0.9],
    'colsample_bytree': [0.7, 0.8, 0.9]
}

grid_eff = GridSearchCV(
    xgb.XGBRegressor(random_state=42),
    param_grid,
    cv=5,
    scoring='r2',
    n_jobs=-1,
    verbose=1
)

grid_eff.fit(X_train, y_eff_train)
model_eff = grid_eff.best_estimator_

print(f"\nBest parameters: {grid_eff.best_params_}")

# Cross-validation scores
cv_scores_eff = cross_val_score(model_eff, X_train, y_eff_train, cv=5, scoring='r2')
print(f"5-fold CV R2: {cv_scores_eff.mean():.4f} (+/- {cv_scores_eff.std():.4f})")

# Test set evaluation
y_eff_pred = model_eff.predict(X_test)
r2_test_eff = r2_score(y_eff_test, y_eff_pred)
mae_test_eff = mean_absolute_error(y_eff_test, y_eff_pred)
rmse_test_eff = np.sqrt(mean_squared_error(y_eff_test, y_eff_pred))

print(f"\nTest set performance:")
print(f"  R2:  {r2_test_eff:.4f}")
print(f"  MAE: {mae_test_eff:.2f}%  (target: < 5%)")
print(f"  RMSE: {rmse_test_eff:.2f}%")

if r2_test_eff > 0.95 and mae_test_eff < 5.0:
    print("  PASS: R2 > 0.95 and MAE < 5%")
else:
    print(f"  WARNING: R2={r2_test_eff:.4f}, MAE={mae_test_eff:.2f}%")

# ============================================================================
# COST MODEL WITH GRID SEARCH + 5-FOLD CV
# ============================================================================

print("\n" + "-" * 70)
print("MODEL 2: COST")
print("-" * 70)

print("Performing grid search with 5-fold cross-validation...")

grid_cost = GridSearchCV(
    xgb.XGBRegressor(random_state=42),
    param_grid,
    cv=5,
    scoring='r2',
    n_jobs=-1,
    verbose=1
)

grid_cost.fit(X_train, y_cost_train)
model_cost = grid_cost.best_estimator_

print(f"\nBest parameters: {grid_cost.best_params_}")

cv_scores_cost = cross_val_score(model_cost, X_train, y_cost_train, cv=5, scoring='r2')
print(f"5-fold CV R2: {cv_scores_cost.mean():.4f} (+/- {cv_scores_cost.std():.4f})")

y_cost_pred = model_cost.predict(X_test)
r2_test_cost = r2_score(y_cost_test, y_cost_pred)
mae_test_cost = mean_absolute_error(y_cost_test, y_cost_pred)
rmse_test_cost = np.sqrt(mean_squared_error(y_cost_test, y_cost_pred))

print(f"\nTest set performance:")
print(f"  R2:  {r2_test_cost:.4f}")
print(f"  MAE: {mae_test_cost:,.0f} LKR")
print(f"  RMSE: {rmse_test_cost:,.0f} LKR")

# ============================================================================
# SAVE MODELS
# ============================================================================

os.makedirs('models', exist_ok=True)

joblib.dump(model_eff, 'models/xgboost_efficiency.pkl')
joblib.dump(model_cost, 'models/xgboost_cost.pkl')
joblib.dump(grid_eff.best_params_, 'models/efficiency_best_params.pkl')
joblib.dump(grid_cost.best_params_, 'models/cost_best_params.pkl')

print("\n" + "=" * 70)
print("MODELS SAVED")
print("=" * 70)
print("  models/xgboost_efficiency.pkl")
print("  models/xgboost_cost.pkl")
print("  models/efficiency_best_params.pkl")
print("  models/cost_best_params.pkl")

# ============================================================================
# FEATURE IMPORTANCE
# ============================================================================

print("\n" + "=" * 70)
print("FEATURE IMPORTANCE")
print("=" * 70)

importance_eff = pd.DataFrame({
    'feature': features,
    'importance': model_eff.feature_importances_
}).sort_values('importance', ascending=False)

importance_cost = pd.DataFrame({
    'feature': features,
    'importance': model_cost.feature_importances_
}).sort_values('importance', ascending=False)

print("\nEfficiency Model (Key variables like head height):")
print(importance_eff.to_string(index=False))

print("\nCost Model:")
print(importance_cost.to_string(index=False))

# ============================================================================
# SUMMARY
# ============================================================================

print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)

print(f"""
Efficiency Model:
  R2: {r2_test_eff:.4f} (target: > 0.95)
  MAE: {mae_test_eff:.2f}% (target: < 5%)
  Status: {'PASS' if r2_test_eff > 0.95 and mae_test_eff < 5.0 else 'NEED MORE DATA'}

Cost Model:
  R2: {r2_test_cost:.4f} (target: > 0.95)
  Status: {'PASS' if r2_test_cost > 0.95 else 'NEED MORE DATA'}

Key Features:
  Efficiency: {importance_eff.iloc[0]['feature']} (importance: {importance_eff.iloc[0]['importance']:.3f})
  Cost: {importance_cost.iloc[0]['feature']} (importance: {importance_cost.iloc[0]['importance']:.3f})
""")

print("Training complete.")