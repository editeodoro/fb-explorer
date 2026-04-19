import numpy as np

def apply_rotation(x, y, z, polar_deg, az_deg, inverse=False):
    """Applies a 3D rotation matrix: X-axis rotation (polar) then Z-axis rotation (azimuthal)."""
    theta = np.radians(polar_deg)
    gamma = np.radians(az_deg)
    
    c_t, s_t = np.cos(theta), np.sin(theta)
    c_g, s_g = np.cos(gamma), np.sin(gamma)
    
    # Rx rotates around X (tilt), Rz rotates around Z (azimuth)
    Rx = np.array([[1, 0, 0], [0, c_t, -s_t], [0, s_t, c_t]])
    Rz = np.array([[c_g, -s_g, 0], [s_g, c_g, 0], [0, 0, 1]])
    
    R = Rz @ Rx
    if inverse:
        R = np.linalg.inv(R)
        
    x_rot = R[0,0]*x + R[0,1]*y + R[0,2]*z
    y_rot = R[1,0]*x + R[1,1]*y + R[1,2]*z
    z_rot = R[2,0]*x + R[2,1]*y + R[2,2]*z
    
    return x_rot, y_rot, z_rot

def get_ellipsoid_mesh(z_center, a_val, b_val, c_val, sign, polar_deg=0.0, az_deg=0.0):
    # 1. Calculate the exact angle where the bubble intersects the local z=0 plane
    # Equation: z_center + a_val * cos(v) = 0
    cos_v_cut = np.clip(-z_center / a_val, -1.0, 1.0)
    v_cut = np.arccos(cos_v_cut)
    
    # 2. Set mesh boundaries depending on whether it's the North or South bubble
    if sign > 0:
        v_start, v_end = 0.0, v_cut      # North: From top down to the cut
    else:
        v_start, v_end = v_cut, np.pi    # South: From the cut down to the bottom
        
    # 3. Generate the mesh perfectly up to the cut
    u, v = np.mgrid[0:2*np.pi:40j, v_start:v_end:40j]
    
    x_mesh = b_val * np.cos(u) * np.sin(v)
    y_mesh = c_val * np.sin(u) * np.sin(v)
    z_mesh = z_center + a_val * np.cos(v)
    
    # 4. Apply user rotations
    x_rot, y_rot, z_rot = apply_rotation(x_mesh, y_mesh, z_mesh, polar_deg, az_deg)
    return x_rot, y_rot, z_rot

def calculate_intersections(S_gal, d_vec_gal, z_offset, p_deg, a_deg, a, b, c):
    """Calculates intersections with the bubble. Variables a, b, c explicitly passed."""
    S_x, S_y, S_z = apply_rotation(S_gal[0], S_gal[1], S_gal[2], p_deg, a_deg, inverse=True)
    d_x, d_y, d_z = apply_rotation(d_vec_gal[0], d_vec_gal[1], d_vec_gal[2], p_deg, a_deg, inverse=True)
    
    S = np.array([S_x, S_y, S_z])
    d_vec_loc = np.array([d_x, d_y, d_z])
    
    A = (d_vec_loc[0]**2 / b**2) + (d_vec_loc[1]**2 / c**2) + (d_vec_loc[2]**2 / a**2)
    B = 2 * ( (S[0]*d_vec_loc[0]/b**2) + (S[1]*d_vec_loc[1]/c**2) + (d_vec_loc[2]*(S[2]-z_offset)/a**2) )
    C = (S[0]**2 / b**2) + (S[1]**2 / c**2) + ((S[2]-z_offset)**2 / a**2) - 1
    delta = B**2 - 4*A*C
    
    if delta < 0: return []
    t_vals = [(-B - np.sqrt(delta)) / (2*A), (-B + np.sqrt(delta)) / (2*A)]
    valid = []
    for t in t_vals:
        if t > 0:
            pt_gal = S_gal + t * d_vec_gal
            if (z_offset > 0 and pt_gal[2] >= -0.01) or (z_offset < 0 and pt_gal[2] <= 0.01):
                valid.append((t, pt_gal))
    return valid