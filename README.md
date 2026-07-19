# Solar–PHES Optimization Research Prototype — Corrected v1

This repository implements a preliminary computational framework for small-scale
solar-integrated Pumped Hydro Energy Storage design. It combines:

1. an hourly physics-based simulator;
2. simulation-generated XGBoost surrogate models;
3. NSGA-II multi-objective optimization; and
4. a Streamlit decision-support interface.

The Streamlit application is the delivery interface, not the main research contribution.
Outputs are preliminary design alternatives and are not construction-ready specifications.

## Important migration notice

The dataset and XGBoost models included in the original repository were invalidated after a
code audit identified simulator errors and train-test leakage. They have been moved to
`data/legacy_invalidated/` and `models/legacy_invalidated/`.

Read [IMPLEMENTATION_AUDIT_AND_FIXES.md](IMPLEMENTATION_AUDIT_AND_FIXES.md)
before running new experiments.

## Setup

```bash
python -m venv venv
```

Windows:

```bash
venv\Scripts\activate
```

Linux/macOS:

```bash
source venv/bin/activate
```

Install dependencies:

```bash
pip install -r Requirements.txt
```

## 1. Run tests

```bash
python -m unittest discover -s tests -v
```

## 2. Generate corrected training data

```bash
python -m scripts.generate_dataset
```

Outputs:

- `data/training_data_all_inputs.csv`
- `data/training_data_diagnostics.csv`

## 3. Train corrected surrogate models

```bash
python -m scripts.train_surrogate
```

Outputs:

- `models/xgboost_efficiency.pkl`
- `models/xgboost_autonomy.pkl`
- `models/feature_names.pkl`
- `results/surrogate_metrics.json`
- `results/surrogate_holdout_predictions.csv`
- `results/efficiency_feature_importance.csv`

The training procedure uses grouped cross-validation by location and reserves Anuradhapura
as a complete external holdout location.

## 4. Start the interface

```bash
streamlit run app.py
```

ML mode will refuse to run until corrected models have been trained. Physics mode uses the
corrected simulator directly.

## Current model scope

The first corrected surrogate is intentionally restricted to its training ranges:

- combined reservoir capacity: 20–800 m³;
- head: 5–45 m;
- pipe diameter: 0.05–0.35 m;
- pump power: 2–30 kW;
- turbine power: 2–25 kW;
- PV capacity: 5–30 kWp;
- daily demand: 10–50 kWh/day;
- evaporation: 30–80 mm/month.

Factory or island scenarios outside these ranges require an expanded simulation dataset and
new model training. Do not extrapolate the current XGBoost model.

## Scientific limitations

The current PV profile is a reproducible PVlib clear-sky baseline, not measured weather.
The cost model is for relative comparison and requires calibration with local quotations.
External simulator/field validation remains required for the final thesis.

## 5. Physics revalidation of an ML Pareto front

After exporting the Pareto alternatives from Streamlit, re-evaluate the exact same designs
with the corrected hourly simulator:

```bash
python -m scripts.validate_pareto phes_pareto_designs.csv
```

This creates a row-by-row comparison and a JSON summary under `results/`. This comparison,
not a comparison between two separately optimized "best" designs, is the defensible way to
validate surrogate-assisted optimization.
