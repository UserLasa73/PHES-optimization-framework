"""Streamlit interface for the corrected PHES decision-support prototype."""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from optimization.common import select_compromise_design
from optimization.optimization import run_optimization, extract_pareto_front
from optimization.optimization_physics import (
    extract_pareto_front_physics,
    run_optimization_physics,
)
from src.cost_model import calculate_capital_cost
from src.model_features import TRAINING_BOUNDS
from src.user_inputs import UserInputs

st.set_page_config(page_title="PHES Design Explorer", layout="wide")
st.title("Solar–Pumped Hydro Design Explorer")
st.info(
    "Research prototype for preliminary design comparison. Results are not "
    "construction-ready engineering specifications or supplier quotations."
)

LOCATIONS = [
    {"name": "Vavuniya", "lat": 8.9, "lon": 79.9},
    {"name": "Colombo", "lat": 6.9, "lon": 79.9},
    {"name": "Jaffna", "lat": 9.7, "lon": 80.0},
    {"name": "Kandy", "lat": 7.3, "lon": 80.6},
    {"name": "Galle", "lat": 6.0, "lon": 80.2},
    {"name": "Trincomalee", "lat": 8.6, "lon": 81.2},
    {"name": "Batticaloa", "lat": 7.7, "lon": 81.7},
    {"name": "Anuradhapura", "lat": 8.3, "lon": 80.4},
]

location_name = st.sidebar.selectbox(
    "Location", [item["name"] for item in LOCATIONS]
)
location = next(item for item in LOCATIONS if item["name"] == location_name)
st.sidebar.caption(f"{location['lat']}° N, {location['lon']}° E")

st.sidebar.header("System requirements")
pv_kwp = st.sidebar.number_input(
    "Existing PV capacity (kWp)",
    min_value=float(TRAINING_BOUNDS["pv_kwp"][0]),
    max_value=float(TRAINING_BOUNDS["pv_kwp"][1]),
    value=20.0,
    step=1.0,
    help="ML mode is restricted to the current surrogate training range.",
)
daily_load = st.sidebar.number_input(
    "Daily energy demand (kWh/day)",
    min_value=float(TRAINING_BOUNDS["daily_energy_kwh"][0]),
    max_value=float(TRAINING_BOUNDS["daily_energy_kwh"][1]),
    value=20.0,
    step=1.0,
)
autonomy_days = st.sidebar.number_input(
    "Required storage autonomy (days)", min_value=0.0, max_value=3.0, value=0.5, step=0.1
)
reservoir_type = st.sidebar.selectbox(
    "Reservoir type", ["new_tank", "excavated", "pond", "river"]
)
max_volume_m3 = st.sidebar.number_input(
    "Maximum combined reservoir capacity (m³)",
    min_value=int(TRAINING_BOUNDS["volume_m3"][0]),
    max_value=int(TRAINING_BOUNDS["volume_m3"][1]),
    value=800,
    step=10,
    help="This is upper gross capacity plus lower gross capacity.",
)

use_budget = st.sidebar.checkbox("Set budget limit")
budget_lkr = None
if use_budget:
    budget_lkr = st.sidebar.number_input(
        "Maximum preliminary PHES cost (LKR)",
        min_value=500_000,
        max_value=20_000_000,
        value=5_000_000,
        step=100_000,
    )

st.sidebar.header("Model assumptions")
evaporation = st.sidebar.number_input(
    "Evaporation (mm/month)",
    min_value=float(TRAINING_BOUNDS["evaporation_rate_mm_month"][0]),
    max_value=float(TRAINING_BOUNDS["evaporation_rate_mm_month"][1]),
    value=50.0,
    step=5.0,
)
pipe_roughness = st.sidebar.number_input(
    "Pipe roughness (m)", value=0.00015, format="%.7f"
)
mode = st.sidebar.radio(
    "Evaluation mode",
    ["ML surrogate", "Physics simulator"],
    help=(
        "ML mode is faster but requires corrected retrained model files. "
        "Physics mode runs the hourly model directly."
    ),
)


def make_user():
    user = UserInputs()
    user.location = location_name
    user.latitude = location["lat"]
    user.longitude = location["lon"]
    user.pv_kwp = pv_kwp
    user.daily_energy_kwh = daily_load
    user.autonomy_days = autonomy_days
    user.upper_reservoir_type = reservoir_type
    user.lower_reservoir_type = reservoir_type
    user.max_volume_m3 = max_volume_m3
    user.budget_lkr = budget_lkr
    user.evaporation_rate_mm_month = evaporation
    user.pipe_roughness_m = pipe_roughness
    user.tilt_angle = 10.0
    user.azimuth_angle = 180.0
    user.year = 2021
    user.demand_spike_factor = 1.0
    user.has_grid_backup = False
    return user


def show_cost_breakdown(design, user):
    result = calculate_capital_cost(
        design["volume_m3"],
        design["head_m"],
        design["pipe_diameter_m"],
        design["pump_power_kw"],
        design["turbine_power_kw"],
        user.pv_kwp,
        user.upper_reservoir_type,
        user.lower_reservoir_type,
    )
    st.subheader("Preliminary cost breakdown")
    table = pd.DataFrame(
        [
            {"Component": key.replace("_lkr", "").replace("_", " ").title(), "Cost (LKR)": value}
            for key, value in result["breakdown"].items()
        ]
    )
    st.dataframe(table.style.format({"Cost (LKR)": "{:,.0f}"}), hide_index=True)
    st.write(f"**Estimated total:** LKR {result['total_lkr']:,.0f}")
    st.caption(result["cost_model_note"])


if st.sidebar.button("Run design search", type="primary"):
    user = make_user()
    try:
        with st.spinner("Evaluating candidate designs..."):
            if mode == "ML surrogate":
                population = run_optimization(user)
                records = extract_pareto_front(population)
            else:
                population = run_optimization_physics(user)
                records = extract_pareto_front_physics(population)
    except Exception as error:
        st.error(str(error))
        st.stop()

    if not records:
        st.warning(
            "No feasible non-dominated designs were found. Relax the autonomy or "
            "budget requirement, or increase the permitted volume."
        )
        st.stop()

    frame = pd.DataFrame(records)
    compromise = select_compromise_design(records)
    st.success(f"Found {len(frame)} non-dominated design alternatives.")

    chart = px.scatter(
        frame,
        x="cost",
        y="efficiency",
        hover_data=[
            "volume_m3",
            "head_m",
            "pipe_diameter_m",
            "pump_power_kw",
            "turbine_power_kw",
        ],
        labels={"cost": "Estimated PHES cost (LKR)", "efficiency": "Efficiency (%)"},
        title="Non-dominated cost–efficiency trade-off",
    )
    st.plotly_chart(chart, use_container_width=True)

    st.subheader("Balanced compromise design")
    columns = st.columns(4)
    columns[0].metric("Efficiency", f"{compromise['efficiency']:.2f}%")
    columns[1].metric("Estimated cost", f"LKR {compromise['cost']:,.0f}")
    columns[2].metric("Combined volume", f"{compromise['volume_m3']:.0f} m³")
    columns[3].metric("Head", f"{compromise['head_m']:.1f} m")

    st.write(
        {
            "Pipe diameter (m)": round(compromise["pipe_diameter_m"], 3),
            "Pump power (kW)": round(compromise["pump_power_kw"], 2),
            "Turbine power (kW)": round(compromise["turbine_power_kw"], 2),
        }
    )

    st.subheader("True first Pareto front")
    st.dataframe(
        frame.style.format(
            {
                "volume_m3": "{:.1f}",
                "head_m": "{:.2f}",
                "pipe_diameter_m": "{:.3f}",
                "pump_power_kw": "{:.2f}",
                "turbine_power_kw": "{:.2f}",
                "efficiency": "{:.3f}",
                "cost": "{:,.0f}",
            }
        ),
        use_container_width=True,
    )

    show_cost_breakdown(compromise, user)
    export = frame.copy()
    export.insert(0, "location", location_name)
    export["latitude"] = location["lat"]
    export["longitude"] = location["lon"]
    export["pv_kwp"] = pv_kwp
    export["daily_energy_kwh"] = daily_load
    export["required_autonomy_days"] = autonomy_days
    export["reservoir_type"] = reservoir_type
    export["evaporation_rate_mm_month"] = evaporation
    export["pipe_roughness_m"] = pipe_roughness
    export["max_volume_m3"] = max_volume_m3
    export["budget_lkr"] = budget_lkr if budget_lkr is not None else ""
    export["evaluation_mode"] = mode
    st.download_button(
        "Download alternatives (CSV)",
        export.to_csv(index=False),
        file_name="phes_pareto_designs.csv",
        mime="text/csv",
    )

st.caption(
    "Solar input currently uses a reproducible PVlib clear-sky profile. Historical/TMY "
    "weather and external engineering validation remain required for the final thesis."
)
