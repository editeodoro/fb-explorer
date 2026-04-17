####################################################################################################
# Interactive FB modelling with StreamLit.
# Run it in browser with:
# > streamlit run FB_model.py
####################################################################################################

import streamlit as st
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import json

st.set_page_config(layout="wide", page_title="Fermi Bubble Explorer")

# --- INITIALIZE DEFAULT PARAMETERS IN SESSION STATE ---
# This allows us to overwrite them dynamically via file import
default_params = {
    'a': 6.0, 'b': 4.0, 'c': 4.0, 'z0': 5.0, 'az_angle': 0.0,
    'sun_x': -8.275, 'sun_y': 0.0, 'sun_z': 0.0,
    'N': 5000,
    'distribution_mode': "Volume Filling",
    'kinematic_model': "Radial Outflow",
    'wind_profile': "Constant Velocity Wind",
    'v_r_const': 500.0,
    'm_slope': 125.0,
    'v_r_max': 500.0,
    'v_c': 240.0,
    'min_lat': 0.0,
    'max_lat': 90.0
}

for k, v in default_params.items():
    if k not in st.session_state:
        st.session_state[k] = v

# Initialize Session State for locked calculations
if 'calc_state' not in st.session_state:
    st.session_state['calc_state'] = {
        'data': None,
        'sample_data': None,
        'N': 0,
        'a': 6.0, 'b': 4.0, 'c': 4.0, 'z0': 5.0,
        'az_angle': 0.0,
        'sun_pos': np.array([-8.275, 0.0, 0.0])
    }

# --- SIDEBAR: CONFIG MANAGER ---
st.sidebar.title("⚙️ Config Manager")

# Import Config
uploaded_file = st.sidebar.file_uploader("Import Parameters (.json)", type=["json", "txt"])
if uploaded_file is not None:
    file_content = uploaded_file.getvalue()
    # Prevent infinite reruns by checking if content changed
    if st.session_state.get('last_uploaded_content') != file_content:
        try:
            loaded_params = json.loads(file_content.decode("utf-8"))
            for k, v in loaded_params.items():
                if k in default_params:
                    st.session_state[k] = v
            st.session_state['last_uploaded_content'] = file_content
            st.rerun()
        except Exception as e:
            st.sidebar.error(f"Error loading file: {e}")

# Export Config
current_params = {k: st.session_state[k] for k in default_params.keys()}
json_str = json.dumps(current_params, indent=4)
st.sidebar.download_button(
    label="Export Current Config",
    data=json_str,
    file_name="wind_model_parameters.json",
    mime="application/json",
    use_container_width=True
)

st.sidebar.divider()

# --- SIDEBAR: GLOBAL SETTINGS ---
st.sidebar.title("Simulation Mode")
mode = st.sidebar.radio("Select Mode:", ["1. Wind Simulator", "2. LOS Explorer"], index=0)

st.sidebar.divider()

with st.sidebar.expander("Bubble Geometry", expanded=True):
    # Added keys to bind inputs to session state
    a = st.number_input("Vertical Semi-axis (a) [kpc]", key='a')
    b = st.number_input("Lateral Semi-axis (b) [kpc]", key='b')
    c = st.number_input("Lateral Semi-axis (c) [kpc]", key='c')
    z0 = st.number_input("Center Offset (z0) [kpc]", key='z0')
    az_angle = st.number_input("Azimuthal Rotation [deg]", key='az_angle', help="Rotates the bubble around the vertical Z-axis.")

st.sidebar.divider()

with st.sidebar.expander("Observer (Sun)", expanded=True):
    sun_x = st.number_input("Sun X [kpc]", key='sun_x', format="%.3f")
    sun_y = st.number_input("Sun Y [kpc]", key='sun_y', format="%.3f")
    sun_z = st.number_input("Sun Z [kpc]", key='sun_z', format="%.3f")
    sun_pos = np.array([sun_x, sun_y, sun_z])

# --- SHARED MATH FUNCTIONS ---
def get_ellipsoid_mesh(z_center, a_val, b_val, c_val, sign=1, angle_deg=0.0):
    u, v = np.mgrid[0:2*np.pi:40j, 0:np.pi:40j]
    x_mesh = b_val * np.cos(u) * np.sin(v)
    y_mesh = c_val * np.sin(u) * np.sin(v)
    z_mesh = z_center + a_val * np.cos(v)
    
    # Apply azimuthal rotation
    gamma = np.radians(angle_deg)
    x_rot = x_mesh * np.cos(gamma) - y_mesh * np.sin(gamma)
    y_rot = x_mesh * np.sin(gamma) + y_mesh * np.cos(gamma)
    
    if sign > 0: z_mesh[z_mesh < 0] = np.nan
    else: z_mesh[z_mesh > 0] = np.nan
    return x_rot, y_rot, z_mesh

# ==========================================
# MODE 1: Wind Simulator
# ==========================================
if mode == "1. Wind Simulator":
    st.sidebar.divider()
    st.sidebar.header("Wind Kinematics")
    N = st.sidebar.number_input("Number of Particles/Clouds (N)", min_value=1, max_value=200000, step=500, key='N')
    
    distribution_mode = st.sidebar.radio("Particle Distribution", ["Volume Filling", "Edge Confined"], key='distribution_mode')
    kinematic_model = st.sidebar.radio("Flow Geometry", ["Radial Outflow", "Ellipsoidal Streamlines"], key='kinematic_model')
    wind_profile = st.sidebar.radio("Velocity Profile", ["Constant Velocity Wind", "Accelerating Wind"], key='wind_profile')
    
    # Track active variables safely without breaking Streamlit keys
    if wind_profile == "Constant Velocity Wind":
        v_r_const_val = st.sidebar.number_input("Constant Radial Velocity [km/s]", key='v_r_const')
        calc_m_slope = 0.0
        calc_v_r_max = 0.0
        calc_v_r_const = v_r_const_val
    else:
        calc_v_r_const = 0.0
        m_slope_val = st.sidebar.number_input("Acceleration Slope [(km/s)/kpc]", key='m_slope')
        v_r_max_val = st.sidebar.number_input("Maximum Velocity [km/s]", key='v_r_max')
        calc_m_slope = m_slope_val
        calc_v_r_max = v_r_max_val
    
    v_c = st.sidebar.number_input("Sun Circular Velocity [km/s]", key='v_c', help="Used for LSR conversion")
    
    min_lat = st.sidebar.number_input("Minimum Latitude |b| [deg]", min_value=0.0, max_value=90.0, key='min_lat')
    max_lat = st.sidebar.number_input("Maximum Latitude |b| [deg]", min_value=0.0, max_value=90.0, key='max_lat')

    if min_lat > max_lat:
        st.sidebar.error("Minimum latitude cannot be greater than Maximum latitude.")

    @st.cache_data
    def generate_wind_particles(N, a, b, c, z0, sun_pos, v_c, wind_profile, v_r_const, m_slope, v_r_max, min_lat, max_lat, distribution_mode, kinematic_model, az_angle):
        particles_b = []
        gamma = np.radians(az_angle)
        cos_g, sin_g = np.cos(gamma), np.sin(gamma)
        
        while len(particles_b) < N:
            batch = max(N * 2, 500)
            
            # Generate in Bubble Frame
            if distribution_mode == "Volume Filling":
                x = np.random.uniform(-b, b, batch)
                y = np.random.uniform(-c, c, batch)
                z = np.random.uniform(-(a+z0), a+z0, batch)
                
                mask_N = (x**2/b**2 + y**2/c**2 + (z-z0)**2/a**2 <= 1) & (z >= 0)
                mask_S = (x**2/b**2 + y**2/c**2 + (z+z0)**2/a**2 <= 1) & (z <= 0)
                
                pts = np.vstack((x[mask_N | mask_S], y[mask_N | mask_S], z[mask_N | mask_S])).T
            else:
                u = np.random.normal(0, 1, batch)
                v = np.random.normal(0, 1, batch)
                w = np.random.normal(0, 1, batch)
                norm = np.sqrt(u**2 + v**2 + w**2)
                
                x_unit = u / norm
                y_unit = v / norm
                z_unit = w / norm
                
                x_e = x_unit * b
                y_e = y_unit * c
                z_e = z_unit * a
                
                is_north = np.random.rand(batch) > 0.5
                
                x = x_e
                y = y_e
                z = np.where(is_north, z_e + z0, z_e - z0)
                
                valid_z_mask = (is_north & (z >= 0)) | (~is_north & (z <= 0))
                pts = np.vstack((x[valid_z_mask], y[valid_z_mask], z[valid_z_mask])).T
            
            if len(pts) > 0:
                # Temporarily rotate to Galactic Frame to check LOS latitude limits
                pts_g = np.empty_like(pts)
                pts_g[:,0] = pts[:,0] * cos_g - pts[:,1] * sin_g
                pts_g[:,1] = pts[:,0] * sin_g + pts[:,1] * cos_g
                pts_g[:,2] = pts[:,2]
                
                d_vec_temp = pts_g - sun_pos
                d_temp = np.linalg.norm(d_vec_temp, axis=1)
                b_deg_temp = np.degrees(np.arcsin(d_vec_temp[:,2] / d_temp))
                
                valid_mask = (np.abs(b_deg_temp) >= min_lat) & (np.abs(b_deg_temp) <= max_lat)
                # Keep valid points in their original Bubble Frame coordinates
                valid_pts_b = pts[valid_mask]
                particles_b.extend(valid_pts_b)
                
            if len(particles_b) >= N:
                particles_b = particles_b[:N]
                break
                
        particles_b = np.array(particles_b)
        x_b, y_b, z_b = particles_b[:,0], particles_b[:,1], particles_b[:,2]
        
        # Calculate distances (Invariant to rotation)
        r = np.sqrt(x_b**2 + y_b**2 + z_b**2)
        r_safe = np.maximum(r, 1e-9)

        # Determine scalar velocity magnitude
        if wind_profile == "Constant Velocity Wind":
            V_mag = np.full(N, v_r_const)
        else:
            V_mag = np.minimum(m_slope * r_safe, v_r_max)
        
        # Calculate Velocity Vectors in Bubble Frame
        if kinematic_model == "Ellipsoidal Streamlines":
            z_c = np.where(z_b > 0, z0, -z0)
            z_prime = z_b - z_c
            R_ell_sq = (x_b/b)**2 + (y_b/c)**2
            sign_z = np.where(z_b > 0, 1.0, -1.0)
            
            Dx = -x_b * z_prime * sign_z
            Dy = -y_b * z_prime * sign_z
            Dz = (a**2) * R_ell_sq * sign_z
            
            norm_D = np.sqrt(Dx**2 + Dy**2 + Dz**2)
            norm_D = np.where(norm_D == 0, 1e-9, norm_D)
            
            Vx_b = V_mag * (Dx / norm_D)
            Vy_b = V_mag * (Dy / norm_D)
            Vz_b = V_mag * (Dz / norm_D)
        else:
            Vx_b = V_mag * (x_b / r_safe)
            Vy_b = V_mag * (y_b / r_safe)
            Vz_b = V_mag * (z_b / r_safe)
        
        # Rotate Position and Velocity coordinates to Galactic Frame
        x = x_b * cos_g - y_b * sin_g
        y = x_b * sin_g + y_b * cos_g
        z = z_b
        
        Vx = Vx_b * cos_g - Vy_b * sin_g
        Vy = Vx_b * sin_g + Vy_b * cos_g
        Vz = Vz_b
        
        particles = np.vstack((x, y, z)).T
        
        # Calculate final observable parameters in Galactic Frame
        R = np.sqrt(x**2 + y**2)
        R_safe = np.maximum(R, 1e-9)
        theta_deg = np.degrees(np.arctan2(y, x))
        phi_deg = np.degrees(np.arccos(z / r_safe))
        
        V_R = (x * Vx + y * Vy) / R_safe
        V_r = (x * Vx + y * Vy + z * Vz) / r_safe
        
        v_vec = np.vstack((Vx, Vy, Vz)).T
        d_vec = particles - sun_pos
        d = np.linalg.norm(d_vec, axis=1)
        u_vec = d_vec / d[:, np.newaxis]
        
        v_gsr = np.sum(v_vec * u_vec, axis=1)
        
        l_rad = np.arctan2(d_vec[:,1], d_vec[:,0])
        b_rad = np.arcsin(d_vec[:,2] / d)
        
        l_deg = np.degrees(l_rad)
        l_deg = (l_deg + 180) % 360 - 180
        b_deg = np.degrees(b_rad)
        
        v_lsr = v_gsr - (v_c * np.sin(l_rad) * np.cos(b_rad))
        
        return pd.DataFrame({
            'l': l_deg, 'b': b_deg,
            'V_LSR': v_lsr, 'V_GSR': v_gsr,
            'd_Sun': d,
            'x': x, 'y': y, 'z': z,
            'R': R, 'θ': theta_deg, 'r': r, 'φ': phi_deg,
            'V_x': Vx, 'V_y': Vy, 'V_z': Vz,
            'V_R': V_R, 'V_r': V_r, 'V_mag': V_mag
        })

    if st.sidebar.button("Calculate model", type="primary"):
        if min_lat <= max_lat:
            with st.spinner("Generating Wind Particles and Kinematics..."):
                df_wind = generate_wind_particles(N, a, b, c, z0, sun_pos, v_c, wind_profile, calc_v_r_const, calc_m_slope, calc_v_r_max, min_lat, max_lat, distribution_mode, kinematic_model, az_angle)
                sample_size = min(N, 2000)
                
                st.session_state['calc_state'] = {
                    'data': df_wind,
                    'sample_data': df_wind.sample(sample_size),
                    'N': N,
                    'a': a, 'b': b, 'c': c, 'z0': z0, 'az_angle': az_angle,
                    'sun_pos': sun_pos
                }
        else:
            st.error("Cannot calculate model: Minimum latitude must be less than or equal to Maximum latitude.")

    cs = st.session_state['calc_state']
    plot_df = cs['data']
    plot_sample = cs['sample_data']
    plot_N = cs['N']
    plot_az = cs.get('az_angle', 0.0)

    if plot_df is not None:
        st.subheader(f"3D Particle Distribution (N={plot_N})")
    else:
        st.subheader("3D Geometry Preview (No data calculated)")

    fig_wind = go.Figure()
    
    limit = 15
    ax_range = [-limit, limit]
    for coords in [([ax_range, [0,0], [0,0]]), ([[0,0], ax_range, [0,0]]), ([[0,0], [0,0], ax_range])]:
        fig_wind.add_trace(go.Scatter3d(x=coords[0], y=coords[1], z=coords[2], mode='lines', line=dict(color='white', width=6), showlegend=False))
    
    for z_c, s in [(z0, 1), (-z0, -1)]:
        bx_mesh, by_mesh, bz_mesh = get_ellipsoid_mesh(z_c, a, b, c, s, plot_az if plot_df is not None else az_angle)
        fig_wind.add_trace(go.Surface(
            x=bx_mesh, y=by_mesh, z=bz_mesh,
            colorscale=[[0, 'white'], [1, 'white']],
            opacity=0.1, showscale=False, hoverinfo='skip'
        ))

    if plot_sample is not None:
        fig_wind.add_trace(go.Scatter3d(
            x=plot_sample['x'], y=plot_sample['y'], z=plot_sample['z'],
            mode='markers',
            marker=dict(size=3, color=plot_sample['V_LSR'], colorscale='RdBu_r',
                        colorbar=dict(title="V_LSR"), opacity=0.8),
            name='Particles'
        ))

    gx, gy = np.meshgrid(np.linspace(-limit, limit, 10), np.linspace(-limit, limit, 10))
    fig_wind.add_trace(go.Surface(x=gx, y=gy, z=np.zeros_like(gx), colorscale=[[0, 'blue'], [1, 'blue']], opacity=0.15, showscale=False, name='Galactic Plane'))
    
    fig_wind.add_trace(go.Scatter3d(
        x=[sun_pos[0]], y=[sun_pos[1]], z=[sun_pos[2]],
        mode='markers+text', text=["Sun"], textposition="top center",
        textfont=dict(color='orange', size=14), marker=dict(size=12, color='orange'), showlegend=False
    ))

    fig_wind.update_layout(scene=dict(aspectmode='manual', aspectratio=dict(x=1, y=1, z=1), xaxis=dict(range=[-limit, limit]), yaxis=dict(range=[-limit, limit]), zaxis=dict(range=[-limit, limit])), template="plotly_dark", height=600, margin=dict(l=0, r=0, b=0, t=0), uirevision='constant')
    st.plotly_chart(fig_wind, width='stretch')

    # --- 2D SCATTER / HISTOGRAM PLOT (Isolated via st.fragment) ---
    @st.fragment
    def render_2d_analysis_plot(df):
        st.divider()
        st.subheader("Kinematic Analysis Plot")
        
        working_df = df.copy()
        
        plot_type = st.radio("Plot Type:", ["Scatter Plot", "Histogram"], horizontal=True)
        options = ['l', 'b', 'V_LSR', 'V_GSR', 'd_Sun', 'x', 'y', 'z', 'R', 'θ', 'r', 'φ', 'V_x', 'V_y', 'V_z', 'V_R', 'V_r', 'V_mag']
        
        if plot_type == "Scatter Plot":
            col1, col2, col3 = st.columns(3)
            x_axis = col1.selectbox("X-Axis", options, index=options.index('b'))
            abs_x = col1.checkbox(f"Absolute |{x_axis}|", key='abs_x')
            y_axis = col2.selectbox("Y-Axis", options, index=options.index('V_LSR'))
            abs_y = col2.checkbox(f"Absolute |{y_axis}|", key='abs_y')
            color_var = col3.selectbox("Color By", options, index=options.index('l'))
            abs_c = col3.checkbox(f"Absolute |{color_var}|", key='abs_c')

            x_col = f"|{x_axis}|" if abs_x else x_axis
            y_col = f"|{y_axis}|" if abs_y else y_axis
            c_col = f"|{color_var}|" if abs_c else color_var
            
        else:
            col1, col2 = st.columns(2)
            hist_var = col1.selectbox("Quantity to Histogram", options, index=options.index('V_LSR'))
            abs_hist = col1.checkbox(f"Absolute |{hist_var}|", key='abs_hist')
            bins = col2.number_input("Number of Bins", min_value=5, max_value=500, value=50, step=5)
            h_col = f"|{hist_var}|" if abs_hist else hist_var

        st.markdown("**Data Masking**")
        mask_query = st.text_input("Filter data using Python/Pandas syntax (e.g., `(x > 2) & (x < 4)`):", value="")

        if mask_query.strip():
            try:
                working_df = working_df.query(mask_query)
                st.success(f"Mask applied! Points remaining: {len(working_df)} / {len(df)}")
            except Exception as e:
                st.error(f"Invalid query syntax. Error: {e}")
                working_df = df.copy()

        if working_df.empty:
            st.warning("The current mask filtered out all data points. Please adjust your query.")
            return

        if plot_type == "Scatter Plot":
            if abs_x: working_df[x_col] = working_df[x_axis].abs()
            if abs_y: working_df[y_col] = working_df[y_axis].abs()
            if abs_c: working_df[c_col] = working_df[color_var].abs()

            fig_2d = px.scatter(
                working_df, x=x_col, y=y_col, color=c_col,
                color_continuous_scale='RdBu_r', hover_data=['x', 'y', 'z']
            )
            fig_2d.update_traces(marker=dict(size=4, opacity=0.7))
            fig_2d.update_xaxes(showgrid=True, gridwidth=1, gridcolor='rgba(255, 255, 255, 0.2)')
        
        else:
            if abs_hist: working_df[h_col] = working_df[hist_var].abs()
            min_val = working_df[h_col].min()
            max_val = working_df[h_col].max()
            bin_size = (max_val - min_val) / bins if max_val > min_val else 1.0
            
            fig_2d = px.histogram(working_df, x=h_col, color_discrete_sequence=['#00FFFF'])
            fig_2d.update_traces(xbins=dict(start=min_val, end=max_val, size=bin_size), autobinx=False)
            fig_2d.update_layout(bargap=0.1)

        fig_2d.update_layout(template="plotly_dark", height=600)
        st.plotly_chart(fig_2d, use_container_width=True)
        
        st.divider()
        st.subheader("Export Particle Data")
        df_export = working_df[options]
        csv_data = df_export.to_csv(index=False, float_format='%.3g').encode('utf-8')
        st.download_button(label="Download Masked Data as CSV", data=csv_data, file_name="simulated_wind_particles_masked.csv", mime="text/csv")
    
    if plot_df is not None:
        render_2d_analysis_plot(plot_df)
    else:
        st.info("Configure your wind kinematics in the sidebar and click 'Calculate model' to generate analysis tools.")

# ==========================================
# MODE 2: LOS Explorer
# ==========================================
elif mode == "2. LOS Explorer":
    with st.spinner("Rendering Sight-line Geometry..."):
        st.sidebar.divider()
        st.sidebar.header("Sight-line Manager")
        num_los = st.sidebar.slider("Number of Sight-lines", 1, 5, 1)

        los_configs = []
        colors = ['#FFFF00', '#00FF00', '#FF00FF', '#00FFFF', '#FFA500']

        for i in range(num_los):
            with st.sidebar.expander(f"Sight-line {i+1}", expanded=(i==0)):
                l_val = st.number_input(f"l° (LOS {i+1})", min_value=-180.0, max_value=180.0, value=0.0, key=f"l{i}")
                lat_val = st.number_input(f"b° (LOS {i+1})", min_value=-90.0, max_value=90.0, value=30.0 - (i*5), key=f"b{i}")
                los_configs.append({'l': l_val, 'b': lat_val, 'color': colors[i]})

        def calculate_intersections(S_gal, d_vec_gal, z_offset, angle_deg):
            gamma = np.radians(angle_deg)
            cos_g, sin_g = np.cos(gamma), np.sin(gamma)
            
            # Map Ray to Bubble Frame
            S = np.array([
                S_gal[0] * cos_g + S_gal[1] * sin_g,
                -S_gal[0] * sin_g + S_gal[1] * cos_g,
                S_gal[2]
            ])
            d_vec_loc = np.array([
                d_vec_gal[0] * cos_g + d_vec_gal[1] * sin_g,
                -d_vec_gal[0] * sin_g + d_vec_gal[1] * cos_g,
                d_vec_gal[2]
            ])
            
            A = (d_vec_loc[0]**2 / b**2) + (d_vec_loc[1]**2 / c**2) + (d_vec_loc[2]**2 / a**2)
            B = 2 * ( (S[0]*d_vec_loc[0]/b**2) + (S[1]*d_vec_loc[1]/c**2) + (d_vec_loc[2]*(S[2]-z_offset)/a**2) )
            C = (S[0]**2 / b**2) + (S[1]**2 / c**2) + ((S[2]-z_offset)**2 / a**2) - 1
            delta = B**2 - 4*A*C
            
            if delta < 0: return []
            t_vals = [(-B - np.sqrt(delta)) / (2*A), (-B + np.sqrt(delta)) / (2*A)]
            valid = []
            for t in t_vals:
                if t > 0:
                    # Map collision back out to Galactic Frame
                    pt_gal = S_gal + t * d_vec_gal
                    if (z_offset > 0 and pt_gal[2] >= -0.01) or (z_offset < 0 and pt_gal[2] <= 0.01):
                        valid.append((t, pt_gal))
            return valid

        all_los_data = []
        for i, config in enumerate(los_configs):
            l_rad, b_rad = np.radians(config['l']), np.radians(config['b'])
            d_vec = np.array([np.cos(b_rad)*np.cos(l_rad), np.cos(b_rad)*np.sin(l_rad), np.sin(b_rad)])
            inters = sorted(calculate_intersections(sun_pos, d_vec, z0, az_angle) + calculate_intersections(sun_pos, d_vec, -z0, az_angle), key=lambda x: x[0])
            all_los_data.append({'id': i+1, 'config': config, 'd_vec': d_vec, 'inters': inters})

        # --- 3D PLOT ---
        fig = go.Figure()
        limit = 15
        ax_range = [-limit, limit]
        for coords in [([ax_range, [0,0], [0,0]]), ([[0,0], ax_range, [0,0]]), ([[0,0], [0,0], ax_range])]:
            fig.add_trace(go.Scatter3d(x=coords[0], y=coords[1], z=coords[2], mode='lines', line=dict(color='white', width=6), showlegend=False))
        
        gx, gy = np.meshgrid(np.linspace(-limit, limit, 10), np.linspace(-limit, limit, 10))
        fig.add_trace(go.Surface(x=gx, y=gy, z=np.zeros_like(gx), colorscale=[[0, 'blue'], [1, 'blue']], opacity=0.15, showscale=False))

        for z_c, s in [(z0, 1), (-z0, -1)]:
            bx_mesh, by_mesh, bz_mesh = get_ellipsoid_mesh(z_c, a, b, c, s, az_angle)
            fig.add_trace(go.Surface(x=bx_mesh, y=by_mesh, z=bz_mesh, colorscale=[[0, 'red'], [1, 'red']], opacity=0.2, showscale=False))

        for data in all_los_data:
            l_end = sun_pos + 40 * data['d_vec']
            fig.add_trace(go.Scatter3d(x=[sun_pos[0], l_end[0]], y=[sun_pos[1], l_end[1]], z=[sun_pos[2], l_end[2]], mode='lines', line=dict(color=data['config']['color'], width=5), name=f"LOS {data['id']}"))
            if data['inters']:
                pts = np.array([pt[1] for pt in data['inters']])
                fig.add_trace(go.Scatter3d(x=pts[:,0], y=pts[:,1], z=pts[:,2], mode='markers', marker=dict(size=6, color=data['config']['color'], symbol='diamond', line=dict(color='white', width=1)), showlegend=False))

        fig.add_trace(go.Scatter3d(x=[sun_pos[0]], y=[sun_pos[1]], z=[sun_pos[2]], mode='markers+text', text=["Sun"], textposition="top center", textfont=dict(color='orange', size=14), marker=dict(size=12, color='orange', symbol='circle'), name='Sun'))

        fig.update_layout(scene=dict(aspectmode='manual', aspectratio=dict(x=1, y=1, z=1), xaxis=dict(range=[-limit, limit]), yaxis=dict(range=[-limit, limit]), zaxis=dict(range=[-limit, limit])), template="plotly_dark", height=700, margin=dict(l=0, r=0, b=0, t=0), uirevision='constant')
        st.plotly_chart(fig, use_container_width=True)

        # --- LOS DATA TABLE ---
        st.subheader("Combined Intersection Data")
        table_rows = []
        for data in all_los_data:
            inters = data['inters']
            if len(inters) == 2: path_in = abs(inters[1][0] - inters[0][0])
            elif len(inters) == 4: path_in = abs(inters[1][0] - inters[0][0]) + abs(inters[3][0] - inters[2][0])
            else: path_in = 0
            for i, inter in enumerate(inters):
                table_rows.append({"LOS": f"LOS {data['id']}", "Point": f"P{i+1}", "Dist (kpc)": round(inter[0], 3), "X": round(inter[1][0], 3), "Y": round(inter[1][1], 3), "Z": round(inter[1][2], 3), "Path Length (kpc)": round(path_in, 3)})

        if table_rows: st.table(pd.DataFrame(table_rows))
        else: st.warning("No intersections detected.")

        active_inters = [d for d in all_los_data if len(d['inters']) >= 2]
        if active_inters:
            st.subheader("Unified Geometry Analysis")
            unified_fig = make_subplots(specs=[[{"secondary_y": True}]])
            for data in active_inters:
                s_vals = np.linspace(data['inters'][0][0], data['inters'][-1][0], 150)
                beta_vals, cos_beta_vals = [], []
                for s in s_vals:
                    P = sun_pos + s * data['d_vec']
                    cb = np.dot(data['d_vec'], P / np.linalg.norm(P))
                    cos_beta_vals.append(cb)
                    beta_vals.append(np.degrees(np.arccos(np.clip(cb, -1.0, 1.0))))
                unified_fig.add_trace(go.Scatter(x=s_vals, y=beta_vals, name=f"LOS {data['id']} Beta", line=dict(color=data['config']['color'], width=3)), secondary_y=False)
                unified_fig.add_trace(go.Scatter(x=s_vals, y=cos_beta_vals, name=f"LOS {data['id']} CosB", line=dict(color=data['config']['color'], width=2, dash='dot')), secondary_y=True)
            unified_fig.update_layout(template="plotly_dark", height=500, hovermode="x unified")
            unified_fig.update_yaxes(title_text="Beta (degrees)", secondary_y=False)
            unified_fig.update_yaxes(title_text="Cos(Beta)", secondary_y=True)
            st.plotly_chart(unified_fig, use_container_width=True)