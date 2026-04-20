import streamlit as st
import numpy as np
import pandas as pd
from geometry import calculate_intersections
from plotting import create_3d_los_plot, create_los_unified_plot

def LOS_explorer(sun_pos, bubble_geometry, live_params):
    
    a, b, c, z0, polar_angle, az_angle = bubble_geometry
        
    with st.spinner("Rendering Sight-line Geometry..."):
        st.sidebar.divider()
        st.sidebar.header("Sight-line Manager")
        num_los = st.sidebar.slider("Number of Sight-lines", 1, 5, 1)

        los_configs = []
        colors = ['#FFFF00', '#00FF00', '#FF00FF', '#00FFFF', '#FFA500']
        for i in range(num_los):
            with st.sidebar.expander(f"Sight-line {i+1}", expanded=(i==0)):
                l_val = st.number_input(f"l° (LOS {i+1})", -180.0, 180.0, 0.0, key=f"l{i}")
                lat_val = st.number_input(f"b° (LOS {i+1})", -90.0, 90.0, 30.0 - (i*5), key=f"b{i}")
                los_configs.append({'l': l_val, 'b': lat_val, 'color': colors[i]})

        all_los_data = []
        for i, config in enumerate(los_configs):
            l_rad, b_rad = np.radians(config['l']), np.radians(config['b'])
            d_vec = np.array([np.cos(b_rad)*np.cos(l_rad), np.cos(b_rad)*np.sin(l_rad), np.sin(b_rad)])
            inters = sorted(
                calculate_intersections(sun_pos, d_vec, z0, polar_angle, az_angle, a, b, c) +
                calculate_intersections(sun_pos, d_vec, -z0, polar_angle, az_angle, a, b, c),
                key=lambda x: x[0]
            )
            all_los_data.append({'id': i+1, 'config': config, 'd_vec': d_vec, 'inters': inters})
        
        # 3D Plot
        st.subheader("3D Geometry and Line of Sight")
        st.plotly_chart(create_3d_los_plot(all_los_data, live_params, sun_pos), width='stretch')
        

        st.subheader("Combined Intersection Data")
        table_rows = []
        for data in all_los_data:
            inters = data['inters']
            path_in = 0
            if len(inters) == 2: path_in = abs(inters[1][0] - inters[0][0])
            elif len(inters) == 4: path_in = abs(inters[1][0] - inters[0][0]) + abs(inters[3][0] - inters[2][0])
        
            for i, inter in enumerate(inters):
                table_rows.append({"LOS": f"LOS {data['id']}", "Point": f"P{i+1}", "Dist (kpc)": round(inter[0], 3), "X": round(inter[1][0], 3), "Y": round(inter[1][1], 3), "Z": round(inter[1][2], 3), "Path Length (kpc)": round(path_in, 3)})

        if table_rows: st.table(pd.DataFrame(table_rows))
        else: st.warning("No intersections detected.")

        active_inters = [d for d in all_los_data if len(d['inters']) >= 2]
        if active_inters:
            st.subheader("Unified Geometry Analysis")
            st.plotly_chart(create_los_unified_plot(active_inters, sun_pos), width='stretch')