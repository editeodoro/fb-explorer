####################################################################################################
# Interactive FB modelling with StreamLit.
# Run it in browser with:
# > streamlit run main_app.py
####################################################################################################

import streamlit as st
import numpy as np
import pandas as pd
from LOS_explorer import LOS_explorer
from model import generate_wind_particles, get_selected_particles
from plotting import create_3d_wind_plot, create_2d_scatter_plot, create_2d_histogram


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

if 'calc_state' not in st.session_state:
    st.session_state['calc_state'] = {
        'data': None, 'sample_data': None, 'N': 0,
        'a': 6.0, 'b': 4.0, 'c': 4.0, 'z0': 5.0, 'polar_angle': 0.0, 'az_angle': 0.0,
        'sun_pos': np.array([-8.275, 0.0, 0.0])
    }

def process_uploaded_config():
    uploaded_file = st.session_state.get('config_uploader')
    if uploaded_file is not None:
        try:
            lines = uploaded_file.getvalue().decode("utf-8").splitlines()
            for line in lines:
                line = line.strip()
                if not line or line.startswith("#"): continue
                if "=" in line:
                    k, v = [part.strip() for part in line.split("=", 1)]
                    if k in default_params:
                        default_val = default_params[k]
                        if isinstance(default_val, bool): st.session_state[k] = v.lower() in ['true', '1', 't', 'y', 'yes']
                        elif isinstance(default_val, int): st.session_state[k] = int(v)
                        elif isinstance(default_val, float): st.session_state[k] = float(v)
                        else: st.session_state[k] = str(v)
        except Exception as e:
            st.error(f"Error parsing config file: {e}")

# --- SIDEBAR ---
st.sidebar.title("Simulation Mode")
mode = st.sidebar.selectbox("Select Mode:", ["Wind Simulator", "LOS Explorer"], index=0)

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
    v_c = st.number_input("Sun Circular Velocity (km/s)", key='v_c', step=10)
    sun_pos = np.array([sun_x, sun_y, sun_z])

live_params = {'a': a, 'b': b, 'c': c, 'z0': z0, 'polar_angle': polar_angle, 'az_angle': az_angle}

# ==========================================
# MODE 1: Wind Simulator
# ==========================================
if mode == "Wind Simulator":
    
    # Sidebar for wind kinematic parameters 
    st.sidebar.divider()
    st.sidebar.header("☄️ Wind Kinematics")
    N = st.sidebar.number_input("Number of Particles (N)", min_value=1, max_value=200000, step=500, key='N')

    min_lat = st.sidebar.number_input("Minimum Latitude |b|", min_value=0.0, max_value=90.0, step=1.0, key='min_lat')
    max_lat = st.sidebar.number_input("Maximum Latitude |b|", min_value=0.0, max_value=90.0, step=10.0, key='max_lat')

    if min_lat > max_lat:
        st.sidebar.error("Minimum latitude cannot be larger than maximum latitude.")
    
    distribution_mode = st.sidebar.radio("Particle Distribution", ["Volume Filling", "Edge Confined"], key='distribution_mode')
    if distribution_mode == "Volume Filling":
        density_profile = st.sidebar.radio("Density Profile",
                ["Constant per Volume", "Constant per Z-bin"], key='density_profile')
    else:
        density_profile = "N/A"
    
    kinematic_model = st.sidebar.radio("Flow Geometry", ["Radial Outflow", "Ellipsoidal Streamlines"], key='kinematic_model')
    wind_profile = st.sidebar.radio("Velocity Profile", ["Constant Velocity Wind", "Accelerating Wind"], key='wind_profile')
    
    if wind_profile == "Constant Velocity Wind":
        calc_v_r_const = st.sidebar.number_input("Constant Radial Velocity", value=default_params['v_r_const'], step=50.0, key='v_r_const')
        calc_m_slope, calc_v_r_max = 0.0, 0.0
    else:
        calc_v_r_const = 0.0
        calc_m_slope = st.sidebar.number_input("Acceleration Slope", value=default_params['m_slope'], step=10.0, key='m_slope')
        calc_v_r_max = st.sidebar.number_input("Maximum Velocity", value=default_params['v_r_max'], step=50.0, key='v_r_max')
    
    if st.sidebar.button("Calculate model", type="primary"):
        if min_lat <= max_lat:
            with st.spinner("Generating Wind Particles..."):
                # Calculating model
                df_wind = generate_wind_particles(N, a, b, c, z0, sun_pos, v_c, wind_profile, calc_v_r_const, calc_m_slope, \
                                                  calc_v_r_max, min_lat, max_lat, distribution_mode, density_profile, kinematic_model, \
                                                  polar_angle, az_angle)
                st.session_state['calc_state'] = {
                    'data': df_wind, 'sample_data': df_wind.sample(min(N, 2000)), 'N': N, **live_params, 'sun_pos': sun_pos }

    # Quantities that are calculated and can be plotted
    plot_options = ['l', 'b', 'V_LSR', 'V_GSR', 'd_Sun', 'x', 'y', 'z', 'R', 'theta', 'r', 'phi', 'V_x', 'V_y', 'V_z', 'V_R', 'V_r', 'V_mag']

    cs = st.session_state['calc_state']
    plot_df = cs['data']
    
    # Setting up 3D plot
    if plot_df is None:
        # Model not calculated yet, just showing geometry
        st.subheader("3D Geometry Preview (No data calculated)")
        color_col_3d = 'V_LSR' # Fallback for empty plot
    else:
        # Show also models
        st.subheader(f"3D Particle Distribution (N={cs['N']})")
        if any(cs[k] != live_params[k] for k in live_params):
            st.warning("⚠️ Geometry altered, particles h The plotted particles reflect the old configuration. Click 'Calculate model' to sync.")

        color_col_3d = st.selectbox("Color 3D Particles By:", plot_options, index=plot_options.index('V_LSR'))
    
    # If there is a selection in 2D plot, create selected particle dataframe
    curr_x_axis = st.session_state.get('x_axis_sel', 'b')
    curr_abs_x = st.session_state.get('abs_x', False)
    curr_x_col = f"|{curr_x_axis}|" if curr_abs_x else curr_x_axis
    
    curr_y_axis = st.session_state.get('y_axis_sel', 'V_LSR')
    curr_abs_y = st.session_state.get('abs_y', False)
    curr_y_col = f"|{curr_y_axis}|" if curr_abs_y else curr_y_axis
    
    expected_plot_key = f"scatter_2d_{curr_x_col}_{curr_y_col}"
    # Extract data using the predicted key for this exact run
    selected_particles_df = get_selected_particles(cs['sample_data'], st.session_state.get(expected_plot_key))
    
    if cs['N'] > 2000:
        st.caption(f"Showing a representative sample of 2,000 particles (out of {cs['N']:,}) for 3D performance.")
        
    # Producing 3D plot
    fig_wind = create_3d_wind_plot(
            cs['sample_data'],
            live_params,
            sun_pos,
            color_col=color_col_3d,
            selected_particles=selected_particles_df
    )
    st.plotly_chart(fig_wind, width='stretch')
    
    
    # --- 2D SCATTER / HISTOGRAM ---
    #@st.fragment
    def render_2d_analysis_plot(df):
        st.divider()
        st.subheader("Kinematic Analysis Plot")
        working_df = df.copy()
        plot_type = st.radio("Plot Type:", ["Scatter Plot", "Histogram"], horizontal=True)
        
        if plot_type == "Scatter Plot":
            col1, col2, col3 = st.columns(3)
            x_axis = col1.selectbox("X-Axis", plot_options, index=plot_options.index('b'), key='x_axis_sel')
            abs_x = col1.checkbox(f"Absolute |{x_axis}|", key='abs_x')
            y_axis = col2.selectbox("Y-Axis", plot_options, index=plot_options.index('V_LSR'), key='y_axis_sel')
            abs_y = col2.checkbox(f"Absolute |{y_axis}|", key='abs_y')
            color_var = col3.selectbox("Color By", plot_options, index=plot_options.index('l'), key='color_var_sel')
            abs_c = col3.checkbox(f"Absolute |{color_var}|", key='abs_c')
                        
            x_col = f"|{x_axis}|" if abs_x else x_axis
            y_col = f"|{y_axis}|" if abs_y else y_axis
            c_col = f"|{color_var}|" if abs_c else color_var

        else:
            col1, col2 = st.columns(2)
            hist_var = col1.selectbox("Quantity to Histogram", plot_options, index=plot_options.index('V_LSR'))
            abs_hist = col1.checkbox(f"Absolute |{hist_var}|", key='abs_hist')
            bins = col2.number_input("Number of Bins", min_value=5, max_value=500, value=50, step=5)
            h_col = f"|{hist_var}|" if abs_hist else hist_var

        mask_query = st.text_input("Filter data (e.g., `(x > 2) & (x < 4)` or `abs(V_LSR) > 50)`:", value="")
        if mask_query.strip():
            try:
                working_df = working_df.query(mask_query)
                st.success(f"Mask applied! Points remaining: {len(working_df)} / {len(df)}")
            except Exception as e:
                st.error(f"Invalid query syntax. Error: {e}")
                working_df = df.copy()

        if working_df.empty:
            st.warning("All points filtered out.")
            return

        if plot_type == "Scatter Plot":
            if abs_x: working_df[x_col] = working_df[x_axis].abs()
            if abs_y: working_df[y_col] = working_df[y_axis].abs()
            if abs_c: working_df[c_col] = working_df[color_var].abs()
            
            plot_key = f"scatter_2d_{x_col}_{y_col}"
            
            st.plotly_chart(
                create_2d_scatter_plot(working_df, x_col, y_col, c_col), width='stretch',
                                       on_select="rerun", key=plot_key)
        else:
            if abs_hist: working_df[h_col] = working_df[hist_var].abs()
            st.plotly_chart(create_2d_histogram(working_df, h_col, bins), width='stretch')


        # Export Particle data
        st.divider()
        st.subheader("Export Particle Data")
        
        filter_active = len(mask_query.strip()) > 0
        if not filter_active:
            st.session_state["export_masked_state"] = False
        
        export_masked = st.checkbox("Export masked data", disabled = not filter_active, key="export_masked_state")
        if export_masked:
            export_df = working_df[plot_options]
            export_filename = "simulated_wind_particles_masked.csv"
        else:
            export_df = df[plot_options]
            export_filename = "simulated_wind_particles.csv"

        csv_data = export_df.to_csv(index=False, float_format='%.3g').encode('utf-8')

        st.download_button(label="💾 Download Data as CSV", \
                           data=csv_data, file_name=export_filename, mime="text/csv")

    if plot_df is not None:
        render_2d_analysis_plot(plot_df)
    else:
        st.info("Configure kinematics in the sidebar and click 'Calculate model' to generate analysis tools.")
    
        
    # ==========================================
    # CONFIG MANAGER
    # ==========================================
    st.sidebar.divider()
    st.sidebar.header("⚙️ Config Manager")
    st.sidebar.file_uploader("Import Parameters (.txt)", type=["txt"], key='config_uploader', on_change=process_uploaded_config)

    current_params = {k: st.session_state[k] for k in default_params.keys()}
    export_lines = ["# Fermi Bubble Explorer Parameters", "# Exported Config"] + [f"{k}={v}" for k, v in current_params.items()]
    st.sidebar.download_button("Export Current Config", data="\n".join(export_lines), file_name="wind_model_parameters.txt", mime="text/plain", width='stretch')
    
# ==========================================
# MODE 2: LOS Explorer
# ==========================================
elif mode == "LOS Explorer":
    LOS_explorer(sun_pos, [a, b, c, z0, polar_angle, az_angle], live_params)

