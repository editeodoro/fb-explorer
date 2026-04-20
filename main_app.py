####################################################################################################
# Interactive FB modelling with StreamLit.
# Run it in browser with:
# > streamlit run main_app.py
####################################################################################################

import streamlit as st
import numpy as np
import pandas as pd
from LOS_explorer import LOS_explorer
from wind_simulator import wind_simulator

st.set_page_config(layout="wide", page_title="Fermi Bubble Explorer")

# --- INITIALIZE DEFAULT PARAMETERS IN SESSION STATE ---
default_params = {
    'a': 6.0, 'b': 4.0, 'c': 4.0, 'z0': 5.0, 'polar_angle': 0.0, 'az_angle': 0.0,
    'sun_x': -8.275, 'sun_y': 0.0, 'sun_z': 0.0, 'N': 5000,
    'distribution_mode': "Volume Filling",
    'density_profile': "Constant per Volume", 
    'kinematic_model': "Radial Outflow",
    'wind_profile': "Constant Velocity Wind", 'v_r_const': 500.0,
    'm_slope': 125.0, 'v_r_max': 500.0, 'v_c': 240.0, 'min_lat': 0.0, 'max_lat': 90.0
}

for k, v in default_params.items():
    if k not in st.session_state:
        st.session_state[k] = v

# --- SIDEBAR: Simulation mode selector ---
st.sidebar.title("Simulation Mode")
mode = st.sidebar.selectbox("Select Mode:", ["Wind Simulator", "LOS Explorer"], index=0)

# --- SIDEBAR: Common Parameters for bubbles and Sun ---
st.sidebar.divider()
with st.sidebar.expander("Bubble Geometry", expanded=False):
    a = st.number_input("Vertical Semi-axis (a)", key='a', step=1.0)
    b = st.number_input("Lateral Semi-axis (b)", key='b', step=1.0)
    c = st.number_input("Lateral Semi-axis (c)", key='c', step=1.0)
    z0 = st.number_input("Center Offset (z0)", key='z0', step=1.0)
    polar_angle = st.number_input("Polar Tilt [deg]", key='polar_angle', step=10.0)
    az_angle = st.number_input("Azimuthal Rotation [deg]", key='az_angle', step=10.0)

with st.sidebar.expander("Observer (Sun)", expanded=False):
    sun_x = st.number_input("Sun X [kpc]", key='sun_x', format="%.3f", step=1.0)
    sun_y = st.number_input("Sun Y [kpc]", key='sun_y', format="%.3f", step=1.0)
    sun_z = st.number_input("Sun Z [kpc]", key='sun_z', format="%.3f", step=1.0)
    sun_v_c = st.number_input("Sun Circular Velocity (km/s)", key='v_c', step=10)
    sun_pos = np.array([sun_x, sun_y, sun_z])

live_params = {'a': a, 'b': b, 'c': c, 'z0': z0, 'polar_angle': polar_angle, 'az_angle': az_angle, \
               'sun_x': sun_x, 'sun_y': sun_y, 'sun_z': sun_z, 'v_c': sun_v_c}

# ==========================================
# MODE 1: Wind Simulator
# ==========================================
if mode == "Wind Simulator":
    wind_simulator(live_params, default_params)
    
# ==========================================
# MODE 2: LOS Explorer
# ==========================================
elif mode == "LOS Explorer":
    LOS_explorer(live_params)

