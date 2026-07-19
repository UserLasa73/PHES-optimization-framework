# Implementation Audit and Corrective Changes

## Status

The original repository contained a working end-to-end prototype, but its numerical results
must not be treated as final research evidence. The audit confirmed several implementation
errors that affect the simulator, dataset, surrogate accuracy, cost objective, and Pareto-front
claims. The corrected source code in this package addresses the main structural errors.

The legacy dataset and model files have been moved to:

- `data/legacy_invalidated/`
- `models/legacy_invalidated/`

They are retained only for traceability. Do not use them for final experiments.

## Confirmed issues in the uploaded repository

### 1. NSGA-II maximized cost instead of minimizing it

The original code returned `[-efficiency, cost]` while using fitness weights
`(-1.0, 1.0)`. In DEAP, this combination rewards both higher efficiency and higher cost.
The infeasible penalty also included an extremely large cost, so infeasible candidates could
remain attractive during non-dominated selection.

**Correction:** both optimizers now return `(efficiency, cost)` and use weights
`(1.0, -1.0)`. Infeasible candidates receive a dominated penalty.

### 2. The displayed population was not a Pareto front

The original extraction function labelled every feasible final-population member as
Pareto-optimal and merely sorted by efficiency.

**Correction:** the first non-dominated front is now extracted using
`tools.sortNondominated(..., first_front_only=True)`.

### 3. Pumping friction had the wrong sign

The original pumping branch used `gross_head - head_loss`. A pump must overcome
`gross_head + head_loss`.

**Correction:** pump flow is solved under total dynamic head using a bisection calculation.

### 4. Pumped energy did not match actual pump power

The original code calculated `power_used` but recorded
`min(excess, pump_power)` as pumped energy.

**Correction:** the simulator records actual electrical pump input multiplied by the
one-hour timestep.

### 5. Turbine generation could exceed the electrical deficit

The original simulator used water based on turbine rated power even when the load deficit
was smaller, then counted all generated energy. This inflated annual generated energy and
could empty the reservoir unnecessarily.

**Correction:** turbine flow targets the actual deficit and generated energy is never allowed
to exceed that deficit.

### 6. Initial upper-reservoir energy biased annual efficiency

Both reservoirs originally started at 70% of their maxima. The upper reservoir could therefore
generate energy before any simulated pumping occurred.

**Correction:** the upper reservoir starts at minimum operating level and the lower reservoir
starts at maximum operating level. The simulator reports both realized efficiency and an
endpoint-storage-adjusted efficiency.

### 7. Reservoir-volume meaning was inconsistent

The cost model split `volume_m3` between two reservoirs, while the simulator assigned almost
the full value to each reservoir.

**Correction:** `volume_m3` now consistently means combined installed capacity. Each
reservoir receives half.

### 8. Autonomy included dead storage

The original autonomy calculation used the maximum upper-reservoir volume rather than the
usable volume above minimum operating level.

**Correction:** autonomy is calculated from usable upper storage
`max_upper - min_upper`.

### 9. Dataset targets did not correspond to sampled PV and load features

In the original dataset generator, solar and load profiles were created before sampled
`pv_kwp` and `daily_energy_kwh` were assigned. The feature columns varied, but simulations
continued to use the default PV and load profiles.

**Correction:** sampled user inputs are assigned before simulation. A one-kWp solar profile
is precomputed per location and scaled correctly for each sample.

### 10. Severe train-test leakage in the 3,200-row dataset

The same 400 Latin Hypercube feature vectors were reused for all eight locations. The stored
training file therefore contained 3,200 rows but only 400 unique feature vectors. A random row
split placed repeated feature vectors in both training and test sets.

Audit measurements on the legacy dataset:

- Random row split: approximately `R² = 0.996`, `MAE = 1.01 percentage points`
- Grouped split with unseen feature vectors: approximately `R² = 0.767`,
  `MAE = 7.92 percentage points`

This confirms that the thesis-level surrogate accuracy was strongly inflated by leakage.

**Correction:** each location now receives an independently seeded LHS sample. Training
uses grouped cross-validation by location and one complete held-out location for testing.

### 11. ML mode ignored the selected location

Location was removed from the training CSV and was absent from the model input vector.
Changing location in Streamlit therefore could not directly change ML predictions.

**Correction:** latitude, longitude, and annual clear-sky solar yield per kWp are included in
the new feature schema.

### 12. Streamlit allowed out-of-distribution ML inputs

The UI allowed PV up to 100 kWp and load up to 200 kWh/day, while the surrogate training
ranges were only 5–30 kWp and 10–50 kWh/day.

**Correction:** the current ML interface is restricted to the trained domain. Wider factory
and island case studies require an expanded dataset and retraining.

### 13. Physics optimization repeatedly rebuilt identical profiles

The original physics fitness function fetched solar and load data for every candidate.

**Correction:** profiles are generated once per optimization run and reused. Candidate
fitness values are also cached.

### 14. Pipe diameter did not affect pipe cost

The optimizer could select the largest pipe to reduce friction without paying additional cost.

**Correction:** pipe cost now increases monotonically with diameter. The coefficient remains
provisional and requires local quotation calibration.

### 15. Reservoir cost decreased at volume thresholds

The old stepwise unit-rate method made some larger reservoirs cheaper than slightly smaller
ones because the lower unit rate was retroactively applied to the whole volume.

**Correction:** a marginal tiered-cost function is used, so total cost is monotonic.

### 16. Generated load did not equal the requested daily energy

The old profile started with `daily_kwh / 24` and then multiplied morning and evening hours,
causing daily energy to exceed the user input.

**Correction:** a normalized 24-hour shape is scaled so each normal day sums exactly to the
requested daily demand.

### 17. Demand spikes could be applied twice

Spikes were added in the load generator and again inside the simulator.

**Correction:** all demand-profile construction occurs in the loader. The simulator consumes
the supplied profile without introducing new randomness.

### 18. Solar modelling was overstated

The repository used a PVlib clear-sky profile, a hard-coded year, and north-facing azimuth
under pvlib's convention. It was not measured or real-time weather.

**Correction:** the selected year is respected, leap-day handling is explicit, and the default
azimuth is south-facing (`180°`). The interface clearly labels solar input as a clear-sky
baseline. Historical/TMY validation is still required.

### 19. Training paths and test imports were broken

The training CSV was stored under `data/`, but the scripts read it from the repository root.
The optimizer test imported from an empty package `__init__.py`.

**Correction:** all paths are resolved using `pathlib`, and smoke tests use correct imports.

## New validation safeguards

The corrected simulator now reports:

- realized and storage-adjusted efficiency;
- initial and final reservoir volumes;
- water-balance residual;
- evaporation and seepage loss totals;
- load-served ratio;
- a physical-validity flag.

The package includes unit tests for:

- monotonic cost behaviour;
- diameter-sensitive pipe cost;
- exact daily load energy;
- friction behaviour;
- prevention of unearned initial generation;
- prevention of over-generation;
- reservoir-volume consistency;
- water and energy diagnostics;
- independent LHS designs.

Run them with:

```bash
python -m unittest discover -s tests -v
```

## Required experiment order

1. Install dependencies.
2. Run the unit tests.
3. Generate a new dataset with the corrected simulator.
4. Train the new surrogate models.
5. Review `results/surrogate_metrics.json` and holdout predictions.
6. Run ML optimization.
7. Re-evaluate every ML Pareto solution using the physics simulator.
8. Run case studies, sensitivity analysis, and runtime benchmarks.
9. Rewrite thesis results only from saved reproducible artifacts.

## Important remaining research limitations

The fixed code removes major programming errors, but it does not by itself prove external
engineering accuracy. The following still require evidence:

- historical or TMY weather inputs;
- real load profiles or justified archetypes;
- local component quotations for cost calibration;
- comparison against published calculations or another trusted model;
- sensitivity and uncertainty analysis;
- case-study validation;
- equipment efficiency curves if the thesis continues to claim variable efficiencies.
