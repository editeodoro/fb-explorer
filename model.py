import numpy as np
import pandas as pd
import streamlit as st
from geometry import apply_rotation

@st.cache_data
def generate_wind_particles(N, a, b, c, z0, sun_pos, v_c, wind_profile, v_r_const, m_slope, v_r_max, min_lat, max_lat, distribution_mode, density_profile, kinematic_model, polar_angle, az_angle):
    particles_b = []
    
    # 1. FAILSAFE: Prevent infinite loops if latitude limits are too strict
    attempts = 0
    max_attempts = 1000 
    
    while len(particles_b) < N:
        attempts += 1
        if attempts > max_attempts:
            st.warning(f"⚠️ Latitude limits too restrictive! Only found {len(particles_b)} particles out of {N}.")
            break

        batch = max(N * 2, 500)
        
        if distribution_mode == "Volume Filling":
            if density_profile == "Constant per Z-bin":
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
            
            x, y = x_e, y_e
            z = np.where(is_north, z_e + z0, z_e - z0)
            
            valid_z_mask = (is_north & (z >= 0)) | (~is_north & (z <= 0))
            pts = np.vstack((x[valid_z_mask], y[valid_z_mask], z[valid_z_mask])).T
        
        if len(pts) > 0:
            pts_g_x, pts_g_y, pts_g_z = apply_rotation(pts[:,0], pts[:,1], pts[:,2], polar_angle, az_angle)
            pts_g = np.vstack((pts_g_x, pts_g_y, pts_g_z)).T
            
            d_vec_temp = pts_g - sun_pos
            d_temp = np.linalg.norm(d_vec_temp, axis=1)
            d_temp_safe = np.maximum(d_temp, 1e-9) # Prevent division by zero
            
            # 2. FAILSAFE: Clip values to strictly [-1, 1] to prevent arcsin NaN errors
            z_ratio = np.clip(d_vec_temp[:,2] / d_temp_safe, -1.0, 1.0)
            b_deg_temp = np.degrees(np.arcsin(z_ratio))
            
            valid_mask = (np.abs(b_deg_temp) >= min_lat) & (np.abs(b_deg_temp) <= max_lat)
            particles_b.extend(pts[valid_mask])
            
        if len(particles_b) >= N:
            particles_b = particles_b[:N]
            break
            
    # 3. FAILSAFE: If no particles are found, return an empty DataFrame to avoid IndexError
    if len(particles_b) == 0:
        st.error("No particles exist in this latitude range with the current geometry. Widening the latitude limits.")
        return pd.DataFrame(columns=[
            'l', 'b', 'V_LSR', 'V_GSR', 'd_Sun', 'x', 'y', 'z', 'R', 'theta', 'r', 'phi',
            'V_x', 'V_y', 'V_z', 'V_R', 'V_r', 'V_mag'
        ])
            
    particles_b = np.array(particles_b)
    x_b, y_b, z_b = particles_b[:,0], particles_b[:,1], particles_b[:,2]
    
    r = np.sqrt(x_b**2 + y_b**2 + z_b**2)
    r_safe = np.maximum(r, 1e-9)

    if wind_profile == "Constant Velocity Wind":
        V_mag = np.full(len(particles_b), v_r_const)
    else:
        V_mag = np.minimum(m_slope * r_safe, v_r_max)
    
    if kinematic_model == "Ellipsoidal Streamlines":
        z_c = np.where(z_b > 0, z0, -z0)
        z_prime = z_b - z_c
        R_ell_sq = (x_b/b)**2 + (y_b/c)**2
        sign_z = np.where(z_b > 0, 1.0, -1.0)
        
        Dx, Dy, Dz = -x_b * z_prime * sign_z, -y_b * z_prime * sign_z, (a**2) * R_ell_sq * sign_z
        norm_D = np.where((norm:=np.sqrt(Dx**2 + Dy**2 + Dz**2)) == 0, 1e-9, norm)
        
        Vx_b, Vy_b, Vz_b = V_mag * (Dx / norm_D), V_mag * (Dy / norm_D), V_mag * (Dz / norm_D)
    else:
        Vx_b, Vy_b, Vz_b = V_mag * (x_b / r_safe), V_mag * (y_b / r_safe), V_mag * (z_b / r_safe)
    
    x, y, z = apply_rotation(x_b, y_b, z_b, polar_angle, az_angle)
    Vx, Vy, Vz = apply_rotation(Vx_b, Vy_b, Vz_b, polar_angle, az_angle)
    
    particles = np.vstack((x, y, z)).T
    R_safe = np.maximum(np.sqrt(x**2 + y**2), 1e-9)
    theta_deg = np.degrees(np.arctan2(y, x))
    phi_deg = np.degrees(np.arccos(z / r_safe))
    
    V_R = (x * Vx + y * Vy) / R_safe
    V_r = (x * Vx + y * Vy + z * Vz) / r_safe
    
    d_vec = particles - sun_pos
    d = np.linalg.norm(d_vec, axis=1)
    v_gsr = np.sum(np.vstack((Vx, Vy, Vz)).T * (d_vec / d[:, np.newaxis]), axis=1)
    
    l_rad = np.arctan2(d_vec[:,1], d_vec[:,0])
    b_rad = np.arcsin(np.clip(d_vec[:,2] / np.maximum(d, 1e-9), -1.0, 1.0))
    
    l_deg = ((np.degrees(l_rad) + 180) % 360) - 180
    b_deg = np.degrees(b_rad)
    v_lsr = v_gsr - (v_c * np.sin(l_rad) * np.cos(b_rad))
    
    return pd.DataFrame({
        'l': l_deg, 'b': b_deg, 'V_LSR': v_lsr, 'V_GSR': v_gsr, 'd_Sun': d,
        'x': x, 'y': y, 'z': z, 'R': np.sqrt(x**2 + y**2), 'theta': theta_deg, 'r': r, 'phi': phi_deg,
        'V_x': Vx, 'V_y': Vy, 'V_z': Vz, 'V_R': V_R, 'V_r': V_r, 'V_mag': V_mag
    })


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