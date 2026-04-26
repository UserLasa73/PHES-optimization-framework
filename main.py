# main.py
from src.physics import simulate_one_hour
from src.config import DEFAULT_DESIGN

# Run 1 hour
v_upper, v_lower, status = simulate_one_hour(DEFAULT_DESIGN)

print(f"--- 1-Hour Simulation Result ---")
print(f"Status: {status}, Upper: {v_upper:.2f}, Lower: {v_lower:.2f}")