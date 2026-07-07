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
from optimization import run_optimization, extract_pareto_front
from cost_model import calculate_capital_cost

st.set_page_config(page_title="PHES Optimizer", layout="wide")

st.title("⚡ Solar-Pumped Hydro Energy Storage Optimizer")
st.markdown("Design your own small-scale PHES system in seconds.")

# ============================================================================
# SIDEBAR - USER INPUTS
# ============================================================================

st.sidebar.header("Site Parameters")

latitude = st.sidebar.number_input("Latitude", value=8.9, format="%.2f")
longitude = st.sidebar.number_input("Longitude", value=79.9, format="%.2f")
pv_kwp = st.sidebar.number_input("PV Capacity (kWp)", value=30.0, min_value=5.0, max_value=100.0)
daily_load = st.sidebar.number_input("Daily Load (kWh/day)", value=50.0, min_value=10.0, max_value=200.0)
autonomy_days = st.sidebar.number_input("Autonomy (days)", value=2.0, min_value=1.0, max_value=5.0)
reservoir_type = st.sidebar.selectbox("Reservoir Type", ["new_tank", "excavated", "pond", "river"])

st.sidebar.header("Advanced")
evap_rate = st.sidebar.number_input("Evaporation Rate (mm/month)", value=50.0)

# ============================================================================
# Display Prices
# ============================================================================
def display_shopping_list(design, user):
    """Display shopping list using the cost model."""
    
    st.subheader("Shopping List / Bill of Materials")
    
    # ===== USE THE COST MODEL =====
    cost_dict = calculate_capital_cost(
        design['volume_m3'],
        design['head_m'],
        design['pipe_diameter_m'],
        design['pump_power_kw'],
        design['turbine_power_kw'],
        user.pv_kwp,  # Not used for cost, but needed for function signature
        user.upper_reservoir_type,
        user.lower_reservoir_type
    )
    
    total = cost_dict['total_lkr']
    breakdown = cost_dict['breakdown']
    
    # ===== DISPLAY =====
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        st.markdown("**1. Two Reservoirs**")
        st.caption(f"Upper: {cost_dict['upper_volume_m3']:.0f} m³")
        st.caption(f"Lower: {cost_dict['lower_volume_m3']:.0f} m³")
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
        st.caption("Controllers, sensors, wiring")
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
        st.caption("Excavation, foundations, labor")
    with col2:
        st.markdown(" ")
        st.caption("Cost:")
    with col3:
        st.markdown(" ")
        st.markdown(f"**LKR {breakdown['installation_lkr']:,.0f}**")
    
    st.divider()
    
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        st.markdown("### TOTAL")
        st.caption("(PV panels EXCLUDED)")
        st.caption(f"≈ USD {total/300:,.0f}")
    with col2:
        st.markdown(" ")
    with col3:
        st.markdown(" ")
        st.markdown(f"### **LKR {total:,.0f}**")
    
    st.caption(" *Estimated costs for design comparison.*")

# ============================================================================
# RUN OPTIMIZATION
# ============================================================================

if st.sidebar.button(" Optimize Design", type="primary"):
    
    with st.spinner("Running optimization... This may take a minute."):
        
        # Create user inputs
        user = UserInputs()
        user.latitude = latitude
        user.longitude = longitude
        user.pv_kwp = pv_kwp
        user.daily_energy_kwh = daily_load
        user.autonomy_days = autonomy_days
        user.upper_reservoir_type = reservoir_type
        user.lower_reservoir_type = reservoir_type
        user.evaporation_rate_mm_month = evap_rate
        
        # Run optimization (simplified)
        population = run_optimization()
        pareto_front = extract_pareto_front(population)
        
        if pareto_front:
            df = pd.DataFrame(pareto_front)
            
            # Display results
            st.success(f" Found {len(df)} optimal designs!")
            
            # Table
            st.subheader(" Optimal Designs (Pareto Front)")
            st.dataframe(df.style.format({
                'volume_m3': '{:.0f}',
                'head_m': '{:.1f}',
                'pipe_diameter_m': '{:.2f}',
                'pump_power_kw': '{:.1f}',
                'turbine_power_kw': '{:.1f}',
                'efficiency': '{:.1f}%',
                'cost': 'LKR {:.0f}'
            }))
            
            # Best design
            best = df.iloc[0]
            st.subheader(" Best Design")
            col1, col2, col3 = st.columns(3)
            col1.metric("Efficiency", f"{best['efficiency']:.1f}%")
            col2.metric("Cost", f"LKR {best['cost']:,.0f}")
            col3.metric("Volume", f"{best['volume_m3']:.0f} m³")
            
            # Pareto front plot
            fig = px.scatter(df, x='cost', y='efficiency', 
                             title='Pareto Front: Efficiency vs Cost',
                             labels={'cost': 'Cost (LKR)', 'efficiency': 'Efficiency (%)'})
            st.plotly_chart(fig, use_container_width=True)
            
            # ===== SHOW SHOPPING LIST =====
            best = df.iloc[0]  # Get the best design
            display_shopping_list(best, user)
        
            # Download
            csv = df.to_csv(index=False)
            st.download_button("📥 Download Results (CSV)", csv, "phos_designs.csv")
            
        else:
            st.error(" No valid designs found. Try relaxing constraints or increasing bounds.")

