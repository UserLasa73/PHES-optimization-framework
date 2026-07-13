"""
app.py
Streamlit web interface for PHES optimization.
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px

from user_inputs import UserInputs
from cost_model import calculate_capital_cost

st.set_page_config(page_title="PHES Optimizer", layout="wide")

st.title(" Solar-Pumped Hydro Energy Storage Optimizer")
st.markdown("Design your own small-scale PHES system in seconds.")

# ============================================================================
# LOCATIONS DATABASE (Hardcoded)
# ============================================================================

LOCATIONS = [
    {'name': 'Vavuniya', 'lat': 8.9, 'lon': 79.9},
    {'name': 'Colombo', 'lat': 6.9, 'lon': 79.9},
    {'name': 'Jaffna', 'lat': 9.7, 'lon': 80.0},
    {'name': 'Kandy', 'lat': 7.3, 'lon': 80.6},
    {'name': 'Galle', 'lat': 6.0, 'lon': 80.2},
    {'name': 'Trincomalee', 'lat': 8.6, 'lon': 81.2},
    {'name': 'Batticaloa', 'lat': 7.7, 'lon': 81.7},
    {'name': 'Anuradhapura', 'lat': 8.3, 'lon': 80.4},
]

LOCATION_NAMES = [loc['name'] for loc in LOCATIONS]

# ============================================================================
# SIDEBAR - USER INPUTS
# ============================================================================

st.sidebar.header("Site Parameters")

# Location dropdown (instead of lat/lon)
selected_location = st.sidebar.selectbox("Location", LOCATION_NAMES, index=0)

# Get coordinates from selected location
location_data = next(loc for loc in LOCATIONS if loc['name'] == selected_location)
latitude = location_data['lat']
longitude = location_data['lon']

# Show coordinates (read-only feedback)
st.sidebar.caption(f"Latitude: {latitude}°N, Longitude: {longitude}°E")

st.sidebar.divider()

st.sidebar.header(" System Parameters")

pv_kwp = st.sidebar.number_input("PV Capacity (kWp)", value=10.0, min_value=5.0, max_value=100.0, step=1.0)
daily_load = st.sidebar.number_input("Daily Load (kWh/day)", value=20.0, min_value=10.0, max_value=200.0, step=5.0)
autonomy_days = st.sidebar.number_input("Autonomy (days)", value=2.0, min_value=0.0, max_value=5.0, step=0.5)
reservoir_type = st.sidebar.selectbox("Reservoir Type", ["new_tank", "excavated", "pond", "river"], index=0)
# Reservoir Volume Constraint
st.sidebar.subheader("Reservoir Volume Constraint")
max_volume_m3 = st.sidebar.number_input(
    "Maximum Total Volume (m³)",
    min_value=20,
    value=800,
    step=10,
    help="Designs with volume exceeding this will be penalized"
)

st.sidebar.divider()

st.sidebar.header(" Advanced")

evap_rate = st.sidebar.number_input("Evaporation Rate (mm/month)", value=50.0, min_value=20.0, max_value=100.0, step=5.0)
pipe_roughness = st.sidebar.number_input("Pipe Roughness (m)", value=0.00015, format="%.5f", help="0.00015 = steel pipe, 0.0000015 = PVC")


st.sidebar.divider()

st.sidebar.header("Optimization Mode")

optimization_mode = st.sidebar.radio(
    "Select Optimizer",
    ["ML Surrogate (Fast)", "Physics Simulator (Slow)"],
    index=0,
    help="ML Surrogate is fast but approximate. Physics Simulator is slow but accurate."
)



# ============================================================================
# DISPLAY SHOPPING LIST
# ============================================================================

def display_shopping_list(design, user):
    """Display shopping list using the cost model."""
    
    st.subheader("Shopping List / Bill of Materials")
    
    cost_dict = calculate_capital_cost(
        design['volume_m3'],
        design['head_m'],
        design['pipe_diameter_m'],
        design['pump_power_kw'],
        design['turbine_power_kw'],
        user.pv_kwp,
        user.upper_reservoir_type,
        user.lower_reservoir_type
    )
    
    total = cost_dict['total_lkr']
    breakdown = cost_dict['breakdown']
    
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        st.markdown("**1. Two Reservoirs**")
        st.caption(f"Upper: {cost_dict['upper_volume_m3']:.0f} m³")
        st.caption(f"Lower: {cost_dict['lower_volume_m3']:.0f} m³")
        st.caption(f"Total: {cost_dict['total_volume_m3']:.0f} m³")
        st.caption(f"Type: {user.upper_reservoir_type} / {user.lower_reservoir_type}")
    with col2:
        st.markdown(" ")
        st.caption("Cost:")
    with col3:
        st.markdown(" ")
        st.markdown(f"**LKR {breakdown['reservoir_lkr']:,.0f}**")
    
    st.divider()
    
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        st.markdown("**2. Water Pump**")
        st.caption(f"Power: {design['pump_power_kw']:.1f} kW")
    with col2:
        st.markdown(" ")
        st.caption("Cost:")
    with col3:
        st.markdown(" ")
        st.markdown(f"**LKR {breakdown['pump_lkr']:,.0f}**")
    
    st.divider()
    
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        st.markdown("**3. Turbine**")
        st.caption(f"Power: {design['turbine_power_kw']:.1f} kW")
    with col2:
        st.markdown(" ")
        st.caption("Cost:")
    with col3:
        st.markdown(" ")
        st.markdown(f"**LKR {breakdown['turbine_lkr']:,.0f}**")
    
    st.divider()
    
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        st.markdown("**4. Penstock Pipes**")
        st.caption(f"Diameter: {design['pipe_diameter_m']:.2f} m")
        st.caption(f"Length: {design['head_m'] * 2.5:.0f} m (supply + return)")
    with col2:
        st.markdown(" ")
        st.caption("Cost:")
    with col3:
        st.markdown(" ")
        st.markdown(f"**LKR {breakdown['pipe_lkr']:,.0f}**")
    
    st.divider()
    
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        st.markdown("**5. Control System**")
        st.caption("Controllers, sensors, wiring, switches")
    with col2:
        st.markdown(" ")
        st.caption("Cost:")
    with col3:
        st.markdown(" ")
        st.markdown(f"**LKR {breakdown['bos_lkr']:,.0f}**")
    
    st.divider()
    
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        st.markdown("**6. Installation & Civil**")
        st.caption("Excavation, foundations, labor, transport")
    with col2:
        st.markdown(" ")
        st.caption("Cost:")
    with col3:
        st.markdown(" ")
        st.markdown(f"**LKR {breakdown['installation_lkr']:,.0f}**")
    
    st.divider()
    
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        st.markdown("### TOTAL ESTIMATED COST")
        st.caption("(PV panels EXCLUDED - user already has them)")
        st.caption(f"≈ USD {total/300:,.0f} (at 1 USD = 300 LKR)")
    with col2:
        st.markdown(" ")
    with col3:
        st.markdown(" ")
        st.markdown(f"### **LKR {total:,.0f}**")
    
    st.caption("*Estimated costs for design comparison. Actual prices may vary.*")

# ============================================================================
# RUN OPTIMIZATION
# ============================================================================

if st.sidebar.button(" Optimize Design", type="primary"):
    
    with st.spinner("Running optimization... This may take a minute."):
        
        # Create user inputs
        user = UserInputs()
        user.location = selected_location
        user.latitude = latitude
        user.longitude = longitude
        user.pv_kwp = pv_kwp
        user.tilt_angle = 10.0
        user.azimuth_angle = 0.0
        user.daily_energy_kwh = daily_load
        user.autonomy_days = autonomy_days
        user.upper_reservoir_type = reservoir_type
        user.lower_reservoir_type = reservoir_type
        user.evaporation_rate_mm_month = evap_rate
        user.pipe_roughness_m = pipe_roughness
        user.demand_spike_factor = 1.0
        user.has_grid_backup = False
        user.max_volume_m3= max_volume_m3 
        
        # Run optimization
        # ===== RUN OPTIMIZATION (Choose mode) =====
        if optimization_mode == "ML Surrogate (Fast)":
            from optimization import run_optimization, extract_pareto_front
            population = run_optimization(user)
            pareto_front = extract_pareto_front(population)
        else:
            from optimization_physics import run_optimization_physics, extract_pareto_front_physics
            population = run_optimization_physics(user)
            pareto_front = extract_pareto_front_physics(population)
        
        if pareto_front:
            df = pd.DataFrame(pareto_front)
            
            # ===== PRINT TO TERMINAL =====
            print("\n" + "=" * 70)
            print("OPTIMAL DESIGN (from Streamlit)")
            print("=" * 70)
            print(f"Mode: {optimization_mode}")
            print(f"Location: {selected_location}")
            print(f"PV Capacity: {pv_kwp} kWp")
            print(f"Daily Load: {daily_load} kWh/day")
            print(f"Autonomy Required: {autonomy_days} days")
            print(f"Reservoir Type: {reservoir_type}")
            print("-" * 70)
            
            best = df.iloc[0]
            print(f"BEST DESIGN:")
            print(f"  Reservoir Volume: {best['volume_m3']:.0f} m3")
            print(f"  Head Height:      {best['head_m']:.1f} m")
            print(f"  Pipe Diameter:    {best['pipe_diameter_m']:.3f} m")
            print(f"  Pump Power:       {best['pump_power_kw']:.1f} kW")
            print(f"  Turbine Power:    {best['turbine_power_kw']:.1f} kW")
            print(f"  Efficiency:       {best['efficiency']:.1f}%")
            print(f"  Cost:             LKR {best['cost']:,.0f}")
            print("=" * 70)
            
            # ===== STREAMLIT DISPLAY =====


            st.success(f"Found {len(df)} optimal designs!")
            
            st.subheader("Optimal Designs (Pareto Front)")
            st.dataframe(df.style.format({
                'volume_m3': '{:.0f}',
                'head_m': '{:.1f}',
                'pipe_diameter_m': '{:.2f}',
                'pump_power_kw': '{:.1f}',
                'turbine_power_kw': '{:.1f}',
                'efficiency': '{:.1f}%',
                'cost': 'LKR {:.0f}'
            }))
            
            best = df.iloc[0]
            st.subheader(" Best Design")
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Efficiency", f"{best['efficiency']:.1f}%")
            col2.metric("Cost", f"LKR {best['cost']:,.0f}")
            col3.metric("Volume", f"{best['volume_m3']:.0f} m³")
            col4.metric("Head", f"{best['head_m']:.1f} m")
            
            # Pareto front plot
            fig = px.scatter(df, x='cost', y='efficiency', 
                             title='Pareto Front: Efficiency vs Cost',
                             labels={'cost': 'Cost (LKR)', 'efficiency': 'Efficiency (%)'},
                             hover_data=['volume_m3', 'head_m', 'pipe_diameter_m'])
            st.plotly_chart(fig, use_container_width=True)
            
            # Shopping list
            display_shopping_list(best, user)
        
            # Download
            csv = df.to_csv(index=False)
            st.download_button("📥 Download Results (CSV)", csv, "phos_designs.csv")
            
        else:
            st.error(" No valid designs found. Try relaxing constraints or increasing bounds.")