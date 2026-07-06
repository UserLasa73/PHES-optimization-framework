"""
test_model.py
Test the trained XGBoost models.
"""

import joblib
import numpy as np

# Load models
model_eff = joblib.load('models/xgboost_efficiency.pkl')
model_cost = joblib.load('models/xgboost_cost.pkl')

# Test design: 200 m³, 20m head, 0.25m pipe, 20kW pump, 15kW turbine
design = np.array([[200, 20, 0.25, 20, 15]])

efficiency = model_eff.predict(design)[0]
cost = model_cost.predict(design)[0]

print("=" * 60)
print("MODEL TEST")
print("=" * 60)
print(f"Design:")
print(f"  Volume: 200 m³")
print(f"  Head: 20 m")
print(f"  Pipe: 0.25 m")
print(f"  Pump: 20 kW")
print(f"  Turbine: 15 kW")
print(f"\nPredictions:")
print(f"  Efficiency: {efficiency:.1f}%")
print(f"  Cost: {cost:,.0f} LKR")