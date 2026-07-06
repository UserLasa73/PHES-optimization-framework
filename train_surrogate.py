"""
train_surrogate.py
Train XGBoost models on ALL user inputs (9 features).
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import r2_score, mean_absolute_error
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

print("=" * 70)
print("XGBOOST SURROGATE TRAINING (9 Features)")
print("=" * 70)

# ============================================================================
# LOAD DATA
# ============================================================================

df = pd.read_csv('training_data_all_inputs.csv')
print(f"Loaded {len(df)} samples")

# ============================================================================
# FEATURES (ALL 9)
# ============================================================================

features = [
    'volume_m3', 'head_m', 'pipe_diameter_m', 
    'pump_power_kw', 'turbine_power_kw',
    'pv_kwp', 'daily_energy_kwh', 
    'evaporation_rate_mm_month', 'reservoir_type_code'
]

X = df[features].values
y_eff = df['efficiency'].values
y_cost = df['cost'].values
y_auto = df['autonomy'].values

print(f"\nFeatures ({len(features)}):")
for i, f in enumerate(features):
    print(f"  {i+1}. {f}")

# ============================================================================
# TRAIN-TEST SPLIT
# ============================================================================

X_train, X_test, y_eff_train, y_eff_test = train_test_split(
    X, y_eff, test_size=0.2, random_state=42
)
_, _, y_cost_train, y_cost_test = train_test_split(
    X, y_cost, test_size=0.2, random_state=42
)
_, _, y_auto_train, y_auto_test = train_test_split(
    X, y_auto, test_size=0.2, random_state=42
)

print(f"\nTraining: {len(X_train)} | Test: {len(X_test)}")

# ============================================================================
# GRID SEARCH
# ============================================================================

param_grid = {
    'n_estimators': [100, 200, 300],
    'learning_rate': [0.05, 0.1, 0.2],
    'max_depth': [3, 5, 7],
    'subsample': [0.7, 0.8, 0.9],
    'colsample_bytree': [0.7, 0.8, 0.9]
}

print("\n" + "-" * 70)
print("GRID SEARCH (this will take a few minutes)")
print("-" * 70)

# ============================================================================
# TRAIN EFFICIENCY
# ============================================================================

print("\nTraining Efficiency Model...")
grid_eff = GridSearchCV(
    xgb.XGBRegressor(random_state=42),
    param_grid,
    cv=5,
    scoring='r2',
    n_jobs=-1,
    verbose=0
)
grid_eff.fit(X_train, y_eff_train)
model_eff = grid_eff.best_estimator_

y_pred = model_eff.predict(X_test)
r2_eff = r2_score(y_eff_test, y_pred)
mae_eff = mean_absolute_error(y_eff_test, y_pred)

print(f"  R2: {r2_eff:.4f} | MAE: {mae_eff:.2f}%")
print(f"  Best params: {grid_eff.best_params_}")

# ============================================================================
# TRAIN COST
# ============================================================================

print("\nTraining Cost Model...")
grid_cost = GridSearchCV(
    xgb.XGBRegressor(random_state=42),
    param_grid,
    cv=5,
    scoring='r2',
    n_jobs=-1,
    verbose=0
)
grid_cost.fit(X_train, y_cost_train)
model_cost = grid_cost.best_estimator_

y_pred = model_cost.predict(X_test)
r2_cost = r2_score(y_cost_test, y_pred)
mae_cost = mean_absolute_error(y_cost_test, y_pred)

print(f"  R2: {r2_cost:.4f} | MAE: {mae_cost:,.0f} LKR")
print(f"  Best params: {grid_cost.best_params_}")

# ============================================================================
# TRAIN AUTONOMY
# ============================================================================

print("\nTraining Autonomy Model...")
grid_auto = GridSearchCV(
    xgb.XGBRegressor(random_state=42),
    param_grid,
    cv=5,
    scoring='r2',
    n_jobs=-1,
    verbose=0
)
grid_auto.fit(X_train, y_auto_train)
model_auto = grid_auto.best_estimator_

y_pred = model_auto.predict(X_test)
r2_auto = r2_score(y_auto_test, y_pred)
mae_auto = mean_absolute_error(y_auto_test, y_pred)

print(f"  R2: {r2_auto:.4f} | MAE: {mae_auto:.2f} days")
print(f"  Best params: {grid_auto.best_params_}")

# ============================================================================
# SAVE MODELS
# ============================================================================

os.makedirs('models', exist_ok=True)

joblib.dump(model_eff, 'models/xgboost_efficiency.pkl')
joblib.dump(model_cost, 'models/xgboost_cost.pkl')
joblib.dump(model_auto, 'models/xgboost_autonomy.pkl')
joblib.dump(features, 'models/feature_names.pkl')

print("\n" + "=" * 70)
print("MODELS SAVED")
print("=" * 70)
print("  models/xgboost_efficiency.pkl")
print("  models/xgboost_cost.pkl")
print("  models/xgboost_autonomy.pkl")
print("  models/feature_names.pkl")

# ============================================================================
# FEATURE IMPORTANCE
# ============================================================================

print("\n" + "=" * 70)
print("FEATURE IMPORTANCE (Efficiency)")
print("=" * 70)

importance = pd.DataFrame({
    'feature': features,
    'importance': model_eff.feature_importances_
}).sort_values('importance', ascending=False)

print(importance.to_string(index=False))

# ============================================================================
# SUMMARY
# ============================================================================

print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
print(f"""
Efficiency:  R2 = {r2_eff:.4f}  | MAE = {mae_eff:.2f}%
Cost:        R2 = {r2_cost:.4f}  | MAE = {mae_cost:,.0f} LKR
Autonomy:    R2 = {r2_auto:.4f}  | MAE = {mae_auto:.2f} days

Top features for efficiency:
  1. {importance.iloc[0]['feature']} ({importance.iloc[0]['importance']:.3f})
  2. {importance.iloc[1]['feature']} ({importance.iloc[1]['importance']:.3f})
  3. {importance.iloc[2]['feature']} ({importance.iloc[2]['importance']:.3f})
""")

print("Training complete.")