"""Optimizer smoke test. Corrected imports and terminology."""

from optimization.optimization import extract_pareto_front, run_optimization
from src.user_inputs import UserInputs

user = UserInputs()
user.pv_kwp = 20.0
user.daily_energy_kwh = 20.0
user.autonomy_days = 0.5
user.max_volume_m3 = 800.0
user.budget_lkr = 5_000_000

population = run_optimization(user)
front = extract_pareto_front(population)
print(f"Non-dominated feasible solutions: {len(front)}")
for design in front[:10]:
    print(design)
