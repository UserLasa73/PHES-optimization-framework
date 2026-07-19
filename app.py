"""Streamlit interface for the corrected PHES decision-support prototype."""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from optimization.common import select_compromise_design
from optimization.optimization import (
    extract_pareto_front,
    run_optimization,
)
from optimization.optimization_physics import (
    extract_pareto_front_physics,
    run_optimization_physics,
)
from src.cost_model import calculate_capital_cost
from src.model_features import TRAINING_BOUNDS
from src.user_inputs import UserInputs


# ============================================================================
# PAGE CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="PHES Design Explorer",
    layout="wide",
)

st.title("Solar–Pumped Hydro Design Explorer")

st.info(
    "Research prototype for preliminary design comparison. Results are not "
    "construction-ready engineering specifications or supplier quotations."
)


# ============================================================================
# AVAILABLE LOCATIONS
# ============================================================================

LOCATIONS = [
    {
        "name": "Vavuniya",
        "lat": 8.7542,
        "lon": 80.4982,
    },
    {
        "name": "Colombo",
        "lat": 6.9271,
        "lon": 79.8612,
    },
    {
        "name": "Jaffna",
        "lat": 9.6615,
        "lon": 80.0255,
    },
]


# ============================================================================
# SIDEBAR INPUTS
# ============================================================================

location_name = st.sidebar.selectbox(
    "Location",
    [item["name"] for item in LOCATIONS],
)

location = next(
    item for item in LOCATIONS if item["name"] == location_name
)

st.sidebar.caption(
    f"{location['lat']}° N, {location['lon']}° E"
)

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
    "Required storage autonomy (days)",
    min_value=0.0,
    max_value=3.0,
    value=0.5,
    step=0.1,
)

reservoir_type = st.sidebar.selectbox(
    "Reservoir type",
    [
        "new_tank",
        "excavated",
        "pond",
        "river",
    ],
)

max_volume_m3 = st.sidebar.number_input(
    "Maximum combined reservoir capacity (m³)",
    min_value=int(TRAINING_BOUNDS["volume_m3"][0]),
    max_value=int(TRAINING_BOUNDS["volume_m3"][1]),
    value=800,
    step=10,
    help="This is the combined gross capacity of the upper and lower reservoirs.",
)

use_budget = st.sidebar.checkbox(
    "Set budget limit",
    value=False,
)

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
    min_value=float(
        TRAINING_BOUNDS["evaporation_rate_mm_month"][0]
    ),
    max_value=float(
        TRAINING_BOUNDS["evaporation_rate_mm_month"][1]
    ),
    value=50.0,
    step=5.0,
)

pipe_roughness = st.sidebar.number_input(
    "Pipe roughness (m)",
    value=0.00015,
    format="%.7f",
)

mode = st.sidebar.radio(
    "Evaluation mode",
    [
        "ML surrogate",
        "Physics simulator",
    ],
    help=(
        "ML mode uses the trained XGBoost surrogate. "
        "Physics mode evaluates every candidate with the annual hourly simulator."
    ),
)


# ============================================================================
# USER OBJECT
# ============================================================================

def make_user() -> UserInputs:
    """Create a UserInputs object from the sidebar selections."""

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
    user.year = 2023

    user.demand_spike_factor = 1.0
    user.has_grid_backup = False

    return user


# ============================================================================
# DESIGN SPECIFICATION AND COST BREAKDOWN
# ============================================================================

def show_cost_breakdown(
    design: dict,
    user: UserInputs,
) -> None:
    """Display component specifications with preliminary costs."""

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

    combined_volume = float(design["volume_m3"])
    volume_per_reservoir = combined_volume / 2.0

    st.subheader("Preliminary cost breakdown")

    descriptions = {
        "reservoir_lkr": (
            f"Two {user.upper_reservoir_type.replace('_', ' ')} reservoirs; "
            f"{combined_volume:.2f} m³ combined "
            f"(approximately {volume_per_reservoir:.2f} m³ each)"
        ),
        "pump_lkr": (
            f"{design['pump_power_kw']:.2f} kW water pump"
        ),
        "turbine_lkr": (
            f"{design['turbine_power_kw']:.2f} kW turbine-generator"
        ),
        "pipe_lkr": (
            f"{design['pipe_diameter_m']:.4f} m internal diameter penstock; "
            f"{design['head_m']:.2f} m gross head"
        ),
        "bos_lkr": (
            "Controllers, electrical equipment, sensors and wiring"
        ),
        "installation_lkr": (
            "Installation, transport and preliminary civil works"
        ),
    }

    cost_rows = []

    for key, value in result["breakdown"].items():
        cost_rows.append(
            {
                "Component": (
                    key.replace("_lkr", "")
                    .replace("_", " ")
                    .title()
                ),
                "Selected specification": descriptions.get(key, ""),
                "Estimated cost (LKR)": value,
            }
        )

    cost_table = pd.DataFrame(cost_rows)

    st.dataframe(
        cost_table.style.format(
            {
                "Estimated cost (LKR)": "{:,.0f}",
            }
        ),
        hide_index=True,
        use_container_width=True,
    )

    st.metric(
        "Total preliminary PHES cost",
        f"LKR {result['total_lkr']:,.0f}",
    )

    st.caption(
        f"{result['cost_model_note']} "
        f"The existing {user.pv_kwp:.2f} kWp PV system is not included in this PHES cost."
    )


# ============================================================================
# TERMINAL OUTPUT
# ============================================================================

def print_terminal_results(
    frame: pd.DataFrame,
    compromise: dict,
) -> None:
    """Print inputs, compromise design and top ten Pareto entries."""

    print("\n" + "=" * 90)
    print("PHES DESIGN SEARCH RESULT")
    print("=" * 90)

    print("\nINPUTS")
    print("-" * 90)

    print(f"Evaluation mode:               {mode}")
    print(f"Location:                      {location_name}")
    print(f"Latitude:                      {location['lat']}")
    print(f"Longitude:                     {location['lon']}")
    print("Weather source:                PVGIS-ERA5 historical hourly data")
    print("Weather year:                  2023")
    print(f"PV capacity:                   {pv_kwp:.2f} kWp")
    print(f"Daily energy demand:           {daily_load:.2f} kWh/day")
    print(f"Required autonomy:             {autonomy_days:.2f} days")
    print(f"Reservoir type:                {reservoir_type}")
    print(f"Maximum combined volume:       {max_volume_m3:.2f} m³")
    print(f"Evaporation rate:              {evaporation:.2f} mm/month")
    print(f"Pipe roughness:                {pipe_roughness:.7f} m")

    if budget_lkr is None:
        print("Budget limit:                  None")
    else:
        print(
            f"Budget limit:                  LKR {budget_lkr:,.0f}"
        )

    print("\nBALANCED COMPROMISE DESIGN")
    print("-" * 90)

    print(
        f"Combined reservoir volume:     "
        f"{compromise['volume_m3']:.2f} m³"
    )
    print(
        f"Approximate volume each:       "
        f"{compromise['volume_m3'] / 2.0:.2f} m³"
    )
    print(
        f"Head height:                   "
        f"{compromise['head_m']:.2f} m"
    )
    print(
        f"Pipe diameter:                 "
        f"{compromise['pipe_diameter_m']:.4f} m"
    )
    print(
        f"Pump power:                    "
        f"{compromise['pump_power_kw']:.2f} kW"
    )
    print(
        f"Turbine power:                 "
        f"{compromise['turbine_power_kw']:.2f} kW"
    )
    print(
        f"Round-trip efficiency:         "
        f"{compromise['efficiency']:.4f}%"
    )
    print(
        f"Estimated capital cost:        "
        f"LKR {compromise['cost']:,.2f}"
    )
    print(
        f"Pareto-front alternatives:     "
        f"{len(frame)}"
    )

    print("\nTOP 10 PARETO-FRONT ALTERNATIVES")
    print("-" * 90)

    print(
        frame.head(10).to_string(
            index=False,
            formatters={
                "volume_m3": lambda value: f"{value:.2f}",
                "head_m": lambda value: f"{value:.2f}",
                "pipe_diameter_m": lambda value: f"{value:.4f}",
                "pump_power_kw": lambda value: f"{value:.2f}",
                "turbine_power_kw": lambda value: f"{value:.2f}",
                "efficiency": lambda value: f"{value:.4f}",
                "cost": lambda value: f"{value:,.2f}",
            },
        )
    )

    print("=" * 90 + "\n")


# ============================================================================
# OPTIMIZATION
# ============================================================================

if st.sidebar.button(
    "Run design search",
    type="primary",
):
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
        st.error(f"Design search failed: {error}")
        st.stop()

    if not records:
        st.warning(
            "No feasible non-dominated designs were found. Relax the autonomy "
            "or budget requirement, or increase the permitted volume."
        )
        st.stop()

    frame = pd.DataFrame(records)
    compromise = select_compromise_design(records)

    print_terminal_results(
        frame,
        compromise,
    )

    st.success(
        f"Found {len(frame)} non-dominated design alternatives."
    )

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
        labels={
            "cost": "Estimated PHES cost (LKR)",
            "efficiency": "Round-trip efficiency (%)",
        },
        title="Non-dominated cost–efficiency trade-off",
    )

    st.plotly_chart(
        chart,
        use_container_width=True,
    )

    st.subheader("Balanced compromise design")

    columns = st.columns(4)

    columns[0].metric(
        "Efficiency",
        f"{compromise['efficiency']:.2f}%",
    )

    columns[1].metric(
        "Estimated cost",
        f"LKR {compromise['cost']:,.0f}",
    )

    columns[2].metric(
        "Combined volume",
        f"{compromise['volume_m3']:.0f} m³",
    )

    columns[3].metric(
        "Head",
        f"{compromise['head_m']:.1f} m",
    )

    show_cost_breakdown(
        compromise,
        user,
    )

    st.subheader("True first Pareto front")

    st.dataframe(
        frame.style.format(
            {
                "volume_m3": "{:.2f}",
                "head_m": "{:.2f}",
                "pipe_diameter_m": "{:.4f}",
                "pump_power_kw": "{:.2f}",
                "turbine_power_kw": "{:.2f}",
                "efficiency": "{:.4f}",
                "cost": "{:,.0f}",
            }
        ),
        hide_index=True,
        use_container_width=True,
    )

    export = frame.copy()

    export.insert(
        0,
        "location",
        location_name,
    )

    export["latitude"] = location["lat"]
    export["longitude"] = location["lon"]
    export["weather_source"] = "PVGIS-ERA5 historical hourly data"
    export["weather_year"] = 2023
    export["pv_kwp"] = pv_kwp
    export["daily_energy_kwh"] = daily_load
    export["required_autonomy_days"] = autonomy_days
    export["reservoir_type"] = reservoir_type
    export["evaporation_rate_mm_month"] = evaporation
    export["pipe_roughness_m"] = pipe_roughness
    export["max_combined_volume_m3"] = max_volume_m3
    export["budget_lkr"] = (
        budget_lkr if budget_lkr is not None else ""
    )
    export["evaluation_mode"] = mode

    st.download_button(
        "Download Pareto alternatives (CSV)",
        export.to_csv(index=False),
        file_name="phes_pareto_designs.csv",
        mime="text/csv",
    )


# ============================================================================
# FOOTER
# ============================================================================

st.caption(
    "Solar input uses cached PVGIS-ERA5 historical hourly weather data for "
    "2023, aligned to Sri Lankan local time, for Vavuniya, Colombo and Jaffna. "
    "Results are preliminary decision-support estimates and require engineering "
    "validation before implementation."
)