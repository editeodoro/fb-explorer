import numpy as np
import pandas as pd
import streamlit as st
from geometry import apply_rotation, calculate_intersections

# =========================================================================
# SHARED PHYSICS ENGINE
# =========================================================================
def _get_cylindrical_spherical(x,y,z,Vx,Vy,Vz):

    R = np.sqrt(x**2 + y**2)
    r = np.sqrt(x**2 + y**2 + z**2)
    R_safe = np.maximum(R, 1e-9)
    theta_deg = np.degrees(np.arctan2(y, x))
    r_safe = np.maximum(r, 1e-9)
    phi_deg = np.degrees(np.arccos(z / r_safe))
    
    V_R = (x * Vx + y * Vy) / R_safe
    V_r = (x * Vx + y * Vy + z * Vz) / r_safe
    
    V_theta = (-y * Vx + x * Vy) / R_safe
    V_phi = (z * (x * Vx + y * Vy) - R_safe**2 * Vz) / (R_safe * r_safe)

    return R, r, theta_deg, phi_deg, V_R, V_r, V_theta, V_phi


def _get_observables(x,y,z,Vx,Vy,Vz,sun_pos, sun_v_c):

    particles = np.vstack((x, y, z)).T
    d_vec = particles - sun_pos
    d_Sun = np.linalg.norm(d_vec, axis=1)
    v_gsr = np.sum(np.vstack((Vx, Vy, Vz)).T * (d_vec / d_Sun[:, np.newaxis]), axis=1)
    
    l_rad = np.arctan2(d_vec[:,1], d_vec[:,0])
    b_rad = np.arcsin(np.clip(d_vec[:,2] / np.maximum(d_Sun, 1e-9), -1.0, 1.0))
    
    l_deg = ((np.degrees(l_rad) + 180) % 360) - 180
    b_deg = np.degrees(b_rad)
    v_lsr = v_gsr - (sun_v_c * np.sin(l_rad) * np.cos(b_rad))

    return l_deg, b_deg, v_lsr, v_gsr, d_Sun




def _get_kinematics(x_b, y_b, z_b, a, b, c, z0, kin_params):
    """Calculates the velocity vector for any point in the unrotated frame."""
    r = np.sqrt(x_b**2 + y_b**2 + z_b**2)
    r_safe = np.maximum(r, 1e-9)

    # 1. Wind Profile (Magnitude)
    if kin_params['wind_profile'] == "Constant Velocity Wind":
        V_mag = np.full(len(x_b), kin_params['v_r_const'])
    else:
        V_mag = np.minimum(kin_params['m_slope'] * r_safe, kin_params['v_r_max'])
    
    # 2. Kinematic Model (Direction)
    if kin_params['outflow_model'] == "Ellipsoidal Streamlines":
        z_c = np.where(z_b > 0, z0, -z0)
        z_prime = z_b - z_c
        R_ell_sq = (x_b/b)**2 + (y_b/c)**2
        sign_z = np.where(z_b > 0, 1.0, -1.0)
        
        Dx, Dy, Dz = -x_b * z_prime * sign_z, -y_b * z_prime * sign_z, (a**2) * R_ell_sq * sign_z
        norm_D = np.where((norm:=np.sqrt(Dx**2 + Dy**2 + Dz**2)) == 0, 1e-9, norm)
        
        Vx_b, Vy_b, Vz_b = V_mag * (Dx / norm_D), V_mag * (Dy / norm_D), V_mag * (Dz / norm_D)
    else: # Radial Outflow (From Galactic Center)
        Vx_b, Vy_b, Vz_b = V_mag * (x_b / r_safe), V_mag * (y_b / r_safe), V_mag * (z_b / r_safe)
        
    return Vx_b, Vy_b, Vz_b, V_mag


def _get_advanced_kinematics(x, y, z, formulas, system):
    # --- 1. Prepare All Coordinates ---
    r = np.sqrt(x**2 + y**2 + z**2)
    R = np.sqrt(x**2 + y**2)
    theta = np.arctan2(y, x)            # Azimuthal
    phi = np.arccos(np.clip(z/r, -1, 1))  # Polar (0 at North Pole)
    
    # Dictionary of everything the user might type
    context = {
        'x': x, 'y': y, 'z': z,
        'r': r, 'rho': R, 'phi': phi, 'theta': theta,
        'np': np, 'pi': np.pi, 'exp': np.exp, 'sqrt': np.sqrt,
        'sin': np.sin, 'cos': np.cos, 'tan': np.tan
    }

    # --- 2. Evaluate Formulas ---
    try:
        v1 = eval(formulas[0], {"__builtins__": {}}, context)
        v2 = eval(formulas[1], {"__builtins__": {}}, context)
        v3 = eval(formulas[2], {"__builtins__": {}}, context)
        v1 = np.full_like(x, v1) if np.isscalar(v1) else v1
        v2 = np.full_like(x, v2) if np.isscalar(v2) else v2
        v3 = np.full_like(x, v3) if np.isscalar(v3) else v3
    except Exception as e:
        print (e)
        st.error(f"Math Error: {e}")
        return np.zeros_like(x), np.zeros_like(y), np.zeros_like(z), np.zeros_like(x)

    # --- 3. Convert back to Cartesian (Vx, Vy, Vz) ---
    if system == "Cartesian (x,y,z)":
        vx = v1 
        vy = v2 
        vz = v3
    elif system == "Spherical (r,theta,phi)":
        # Standard Spherical -> Cartesian velocity transformation
        vx = v1*np.sin(phi)*np.cos(theta) + v2*np.cos(phi)*np.cos(theta) - v3*np.sin(theta)
        vy = v1*np.sin(phi)*np.sin(theta) + v2*np.cos(phi)*np.sin(theta) + v3*np.cos(theta)
        vz = v1*np.cos(phi) - v2*np.sin(phi)
    elif system == "Cylindrical (R,theta,z)":
        # Cylindrical -> Cartesian velocity transformation
        vx = v1*np.cos(theta) - v2*np.sin(theta)
        vy = v1*np.sin(theta) + v2*np.cos(theta)
        vz = v3
    
    vmag = np.sqrt(vx**2 + vy**2 + vz**2)
    return vx, vy, vz, vmag
    
# =========================================================================
# PRIMARY FUNCTIONS
# =========================================================================
@st.cache_data
def generate_wind_particles(kin_params, live_params):
    
    # Unpack parameters for easier access
    a, b, c, z0, polar_angle, az_angle  = (live_params[k] for k in ('a', 'b', 'c', 'z0', 'polar_angle', 'az_angle'))
    sun_pos, sun_v_c = np.array([live_params['sun_x'], live_params['sun_y'], live_params['sun_z']]), live_params['v_c']
    N, min_lat, max_lat = kin_params['N'], kin_params['min_lat'], kin_params['max_lat']

    particles_b = []
    attempts = 0
    max_attempts = 1000 
    
    while len(particles_b) < N:
        attempts += 1
        if attempts > max_attempts:
            st.warning(f"⚠️ Latitude limits too restrictive! Only found {len(particles_b)} particles out of {N}.")
            break

        batch = max(N * 2, 500)
        
        if kin_params['distribution_mode'] == "Volume Filling":
            if kin_params['density_profile'] == "Constant per Z-bin":
                is_north = np.random.rand(batch) > 0.5
                z_prime = np.random.uniform(-a, a, batch)
                r_xy = np.sqrt(1 - (z_prime/a)**2)
                rho = np.sqrt(np.random.uniform(0, 1, batch))
                phi = np.random.uniform(0, 2*np.pi, batch)
                
                x = b * r_xy * rho * np.cos(phi)
                y = c * r_xy * rho * np.sin(phi)
                z = np.where(is_north, z_prime + z0, z_prime - z0)
                
                valid_z_mask = (is_north & (z >= 0)) | (~is_north & (z <= 0))
                pts = np.vstack((x[valid_z_mask], y[valid_z_mask], z[valid_z_mask])).T
            else:
                x = np.random.uniform(-b, b, batch)
                y = np.random.uniform(-c, c, batch)
                z = np.random.uniform(-(a+z0), a+z0, batch)
                
                mask_N = (x**2/b**2 + y**2/c**2 + (z-z0)**2/a**2 <= 1) & (z >= 0)
                mask_S = (x**2/b**2 + y**2/c**2 + (z+z0)**2/a**2 <= 1) & (z <= 0)
                pts = np.vstack((x[mask_N | mask_S], y[mask_N | mask_S], z[mask_N | mask_S])).T
        else:
            u, v, w = np.random.normal(0, 1, batch), np.random.normal(0, 1, batch), np.random.normal(0, 1, batch)
            norm = np.sqrt(u**2 + v**2 + w**2)
            
            x_e, y_e, z_e = (u / norm) * b, (v / norm) * c, (w / norm) * a
            is_north = np.random.rand(batch) > 0.5
            z = np.where(is_north, z_e + z0, z_e - z0)
            
            valid_z_mask = (is_north & (z >= 0)) | (~is_north & (z <= 0))
            pts = np.vstack((x_e[valid_z_mask], y_e[valid_z_mask], z[valid_z_mask])).T
        
        if len(pts) > 0:
            pts_g_x, pts_g_y, pts_g_z = apply_rotation(pts[:,0], pts[:,1], pts[:,2], polar_angle, az_angle)
            pts_g = np.vstack((pts_g_x, pts_g_y, pts_g_z)).T
            
            d_vec_temp = pts_g - sun_pos
            d_temp_safe = np.maximum(np.linalg.norm(d_vec_temp, axis=1), 1e-9)
            
            b_deg_temp = np.degrees(np.arcsin(np.clip(d_vec_temp[:,2] / d_temp_safe, -1.0, 1.0)))
            valid_mask = (np.abs(b_deg_temp) >= min_lat) & (np.abs(b_deg_temp) <= max_lat)
            particles_b.extend(pts[valid_mask])
            
        if len(particles_b) >= N:
            particles_b = particles_b[:N]
            break
            
    if len(particles_b) == 0:
        return pd.DataFrame()
            
    particles_b = np.array(particles_b)
    x_b, y_b, z_b = particles_b[:,0], particles_b[:,1], particles_b[:,2]
    
    x, y, z = apply_rotation(x_b, y_b, z_b, polar_angle, az_angle)

    if kin_params['wind_profile'] == "Advanced Kinematics":
        Vx, Vy, Vz, V_mag = _get_advanced_kinematics(x, y, z, kin_params['formulas'], kin_params['coord_sys'])
    else:
        Vx_b, Vy_b, Vz_b, V_mag = _get_kinematics(x_b, y_b, z_b, a, b, c, z0, kin_params)
        Vx, Vy, Vz = apply_rotation(Vx_b, Vy_b, Vz_b, polar_angle, az_angle)
    
    R, r, theta_deg, phi_deg, V_R, V_r, V_theta, V_phi = _get_cylindrical_spherical(x, y, z, Vx, Vy, Vz)
    l_deg, b_deg, v_lsr, v_gsr, d_Sun = _get_observables(x, y, z, Vx, Vy, Vz, sun_pos, sun_v_c)
    
    return pd.DataFrame({
        'l': l_deg, 'b': b_deg, 'V_LSR': v_lsr, 'V_GSR': v_gsr, 'd_Sun': d_Sun,
        'x': x, 'y': y, 'z': z, 'R': R, 'theta': theta_deg, 'r': r, 'phi': phi_deg,
        'V_x': Vx, 'V_y': Vy, 'V_z': Vz, 'V_R': V_R, 'V_r': V_r, 'V_theta': V_theta, 'V_phi': V_phi, 'V_mag': V_mag
    })


@st.cache_data
def estimate_observed_properties(obs_df, kin_params, live_params):
    results = []
    a, b, c, z0 = live_params['a'], live_params['b'], live_params['c'], live_params['z0']
    p_deg, a_deg = live_params['polar_angle'], live_params['az_angle']
    sun_pos, sun_v_c = np.array([live_params['sun_x'], live_params['sun_y'], live_params['sun_z']]), live_params['v_c']
    
    for _, row in obs_df.iterrows():
        l_deg, b_deg, v_obs = row['l'], row['b'], row['V_LSR']
        l_rad, b_rad = np.radians(l_deg), np.radians(b_deg)
        d_vec = np.array([np.cos(b_rad)*np.cos(l_rad), np.cos(b_rad)*np.sin(l_rad), np.sin(b_rad)])
        
        i_N = calculate_intersections(sun_pos, d_vec, z0, p_deg, a_deg, a, b, c)
        i_S = calculate_intersections(sun_pos, d_vec, -z0, p_deg, a_deg, a, b, c)
        all_i = sorted(i_N + i_S, key=lambda x: x[0])
        
        if len(all_i) < 2: continue
            
        s_vals = np.linspace(all_i[0][0], all_i[-1][0], 250)
        
        pts = sun_pos + s_vals[:, np.newaxis] * d_vec
        x, y, z = pts[:,0], pts[:,1], pts[:,2]
        
        x_b, y_b, z_b = apply_rotation(x, y, z, p_deg, a_deg, inverse=True)
        
        if kin_params['wind_profile'] == "Advanced Kinematics":
            Vx, Vy, Vz, V_mag = _get_advanced_kinematics(x, y, z, kin_params['formulas'], kin_params['coord_sys'])
        else:
            Vx_b, Vy_b, Vz_b, V_mag = _get_kinematics(x_b, y_b, z_b, a, b, c, z0, kin_params)
            Vx, Vy, Vz = apply_rotation(Vx_b, Vy_b, Vz_b, p_deg, a_deg)

        
        R, r, theta_deg, phi_deg, V_R, V_r, V_theta, V_phi = _get_cylindrical_spherical(x, y, z, Vx, Vy, Vz)

        d_vec_est = pts - sun_pos
        d_est = np.linalg.norm(d_vec_est, axis=1)
        v_gsr = np.sum(np.vstack((Vx, Vy, Vz)).T * (d_vec_est / d_est[:, np.newaxis]), axis=1)
                
        v_lsr = v_gsr - (sun_v_c * np.sin(l_rad) * np.cos(b_rad))
        v_gsr_obs = v_obs + (sun_v_c * np.sin(l_rad) * np.cos(b_rad))

        # Match!
        diffs = np.abs(v_lsr - v_obs)
        best_idx = np.argmin(diffs)
        
        results.append({
            'l': l_deg, 'b': b_deg, 'V_LSR': v_obs, 'V_LSR_mod': v_lsr[best_idx], 'V_GSR' : v_gsr_obs, 'V_GSR_mod': v_gsr[best_idx], 
            'd_Sun': d_est[best_idx], 'x': x[best_idx], 'y': y[best_idx], 'z': z[best_idx], 'R': R[best_idx], 
            'theta': theta_deg[best_idx], 'r': r[best_idx], 'phi': phi_deg[best_idx], 'V_x': Vx[best_idx], 'V_y': Vy[best_idx], 
            'V_z': Vz[best_idx], 'V_R': V_R[best_idx], 'V_r': V_r[best_idx], 'V_theta': V_theta[best_idx], 
            'V_phi': V_phi[best_idx], 'V_mag': V_mag[best_idx], 'V_LSR_diff': diffs[best_idx]
        })
        
    return pd.DataFrame(results) if results else pd.DataFrame()


def get_selected_particles(sample_df, state):
    
    selected_particles_df = None

    if not state:
        return None

    # --- Extract points ---
    if hasattr(state, "selection"):
        points = state.selection.get("points", [])
    elif isinstance(state, dict):
        points = state.get("selection", {}).get("points", [])
    else:
        points = []

    if not points:
        return None

    try:
        selected_real_indices = []

        for p in points:
            cd = p.get("customdata")

            if isinstance(cd, list) and len(cd) > 0:
                selected_real_indices.append(cd[0])
            elif isinstance(cd, dict):
                selected_real_indices.append(
                    cd.get("real_index") or list(cd.values())[0]
                )
            else:
                selected_real_indices.append(cd)

        if selected_real_indices and sample_df is not None:
            selected_particles_df = sample_df.loc[
                sample_df.index.isin(selected_real_indices)
            ]

            if not selected_particles_df.empty:
                st.success(f"🎯 Highlighted {len(selected_particles_df)} particles!")

    except Exception as e:
        st.error(f"⚠️ Data extraction failed: {e}")

    return selected_particles_df

