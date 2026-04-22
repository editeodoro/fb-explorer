import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import streamlit as st
from geometry import get_ellipsoid_mesh

@st.cache_data
def get_base_geometry(live_params, sun_pos):
    
    """Generates the static environment once and caches it as a dictionary."""
    fig_base = go.Figure()
    limit = 15
    
    # Galactic Plane
    gx, gy = np.meshgrid([-limit, limit], [-limit, limit])
    fig_base.add_trace(go.Surface(
        x=gx, y=gy, z=np.zeros_like(gx), colorscale=[[0, 'blue'], [1, 'blue']], 
        opacity=0.15, showscale=False, name='Galactic Plane', hoverinfo='skip'
    ))
    
    # Combined Axes and Sun
    x_comb = [-limit, limit, None, 0, 0, None, 0, 0, None, sun_pos[0]]
    y_comb = [0, 0, None, -limit, limit, None, 0, 0, None, sun_pos[1]]
    z_comb = [0, 0, None, 0, 0, None, -limit, limit, None, sun_pos[2]]
    
    m_sizes = [0] * 9 + [20]
    m_colors = ['white'] * 9 + ['orange']
    t_labels = [""] * 9 + ["Sun"]

    fig_base.add_trace(go.Scatter3d(
        x=x_comb, y=y_comb, z=z_comb, mode='lines+markers+text',
        line=dict(color='white', width=6), marker=dict(size=m_sizes, color=m_colors),
        text=t_labels, textposition="top center", textfont=dict(color='orange', size=14),
        showlegend=False, hoverinfo='skip'
    ))
    
    # Live Geometry Surface (Bubbles)
    for z_c, s in [(live_params['z0'], 1), (-live_params['z0'], -1)]:
        bx_mesh, by_mesh, bz_mesh = get_ellipsoid_mesh(
            z_c, live_params['a'], live_params['b'], live_params['c'], s,
            live_params['polar_angle'], live_params['az_angle']
        )
        fig_base.add_trace(go.Surface(
            x=bx_mesh, y=by_mesh, z=bz_mesh, colorscale=[[0, 'white'], [1, 'white']], 
            opacity=0.1, showscale=False, hoverinfo='skip'
        ))

    fig_base.update_layout(
        scene=dict(aspectmode='manual', aspectratio=dict(x=1, y=1, z=1), 
                   xaxis=dict(range=[-limit, limit]), yaxis=dict(range=[-limit, limit]), zaxis=dict(range=[-limit, limit])), 
        template="plotly_dark", height=600, margin=dict(l=0, r=0, b=0, t=0), uirevision='constant'
    )
    
    return fig_base.to_dict()
    

def _get_plotting_limits(sample, minimum_range=1):
    
    Min, Max = sample.min(), sample.max()
    
    # Enforce minimum range to avoid degenerate axes
    if (Max - Min) < minimum_range:
        center = minimum_range/2. * (Max + Min)
        Min, Max = center - minimum_range/2., center + minimum_range/2.
    
    return Min, Max



def create_3d_wind_plot(plot_sample, live_params, sun_pos, color_col='V_LSR', selected_particles=None, obs_df=None):

    # 1. Load the pre-calculated base geometry
    base_fig_dict = get_base_geometry(live_params, sun_pos)
    
    # 2. Reconstruct the figure 
    fig_wind = go.Figure(base_fig_dict)

    # 3. Add ONLY the dynamic particles
    if plot_sample is not None:
        c_min, c_max = _get_plotting_limits(plot_sample[color_col]) if obs_df is None else _get_plotting_limits(obs_df[color_col])

        cscale = 'RdBu_r' if 'V_' in color_col else 'Plasma'
        
        if selected_particles is not None and not selected_particles.empty:
            fig_wind.add_trace(go.Scatter3d(
                x=plot_sample['x'], y=plot_sample['y'], z=plot_sample['z'],
                mode='markers', marker=dict(size=2, color='gray', opacity=0.2),
                name='Unselected', hoverinfo='skip'
            ))
            
            fig_wind.add_trace(go.Scatter3d(
                x=selected_particles['x'], y=selected_particles['y'], z=selected_particles['z'],
                mode='markers',
                marker=dict(
                    size=5, color=selected_particles[color_col], colorscale=cscale,
                    colorbar=dict(title=color_col, x=-0.15), opacity=1.0,
                    line=dict(color='white', width=1), cmin=c_min, cmax=c_max
                ),
                name='Selected'
            ))
        else:
            fig_wind.add_trace(go.Scatter3d(
                x=plot_sample['x'], y=plot_sample['y'], z=plot_sample['z'],
                mode='markers',
                marker=dict(size=3, color=plot_sample[color_col], colorscale=cscale, colorbar=dict(title=color_col, x=-0.15), 
                            opacity=0.8, cmin=c_min, cmax=c_max), 
                name='Particles'
            ))
    
        # Plot Observations on top in 3D
        if obs_df is not None and not obs_df.empty:
            fig_wind.add_trace(go.Scatter3d(
                x=obs_df['x'], y=obs_df['y'], z=obs_df['z'],
                mode='markers',
                marker=dict(size=8, color=obs_df[color_col], colorscale=cscale, symbol='diamond', line=dict(color='yellow', width=4), cmin=c_min, cmax=c_max),
                name='Observations',
                    hovertext=[f"Model V_LSR: {v:.1f} | Diff: {d:.1f}" for v, d in zip(obs_df.get('V_LSR_mod', []), obs_df.get('V_LSR_diff', []))]
            ))

    return fig_wind


def create_2d_scatter_plot(working_df, x_col, y_col, c_col, obs_df=None):

    # Safely inject the pandas index as a real column
    temp_df = working_df.copy()
    temp_df['real_index'] = temp_df.index
    
    xlims = _get_plotting_limits(temp_df[x_col])
    ylims = _get_plotting_limits(temp_df[y_col])
    c_min, c_max = _get_plotting_limits(temp_df[c_col])

    # Passing 'real_index' into custom_data is CRITICAL for the lasso tool to work
    fig_2d = px.scatter(
        temp_df,
        x=x_col,
        y=y_col,
        color=c_col,
        color_continuous_scale='RdBu_r',
        hover_data=['x', 'y', 'z'],
        custom_data=['real_index']  # <-- This links the 2D plot to the 3D plot
    )
    
    fig_2d.update_traces(marker=dict(size=4, opacity=0.7))

    # Plot Observations on top in 2D
    if obs_df is not None and not obs_df.empty:
        fig_2d.add_trace(go.Scatter(
            x=obs_df[x_col], y=obs_df[y_col],
            mode='markers',
            marker=dict(size=12, color=obs_df[c_col], colorscale='RdBu_r', symbol='diamond', line=dict(color='yellow', width=1.)),
            name='Observations',
            hovertext=[f"Model V_LSR: {v:.1f} | Diff: {d:.1f}" for v, d in zip(obs_df.get('V_LSR_mod', []), obs_df.get('V_LSR_diff', []))]
        ))
        c_min, c_max = _get_plotting_limits(obs_df[c_col])

    fig_2d.update_xaxes(range=xlims)
    fig_2d.update_yaxes(range=ylims)
    fig_2d.update_layout(template="plotly_dark", height=600, coloraxis=dict(cmin=c_min, cmax=c_max))
    return fig_2d


def create_2d_histogram(working_df, h_col, bins, obs_df=None):
    # 1. Find min and max across BOTH datasets to ensure bins cover everything
    min_val = working_df[h_col].min()
    max_val = working_df[h_col].max()
    
    has_obs = obs_df is not None and not obs_df.empty and h_col in obs_df.columns
    
    if has_obs:
        min_val = min(min_val, obs_df[h_col].min())
        max_val = max(max_val, obs_df[h_col].max())

    bin_size = (max_val - min_val) / bins if max_val > min_val else 1.0
    
    # 2. Create the base Simulated histogram
    fig_2d = px.histogram(working_df, x=h_col, color_discrete_sequence=['#00FFFF'])
    
    # Update simulated trace for transparency and name
    fig_2d.update_traces(
        xbins=dict(start=min_val, end=max_val, size=bin_size), 
        autobinx=False,
        opacity=0.7,         # Transparency 
        name="Simulated",
        showlegend=True      # Force legend to show
    )
    
    # 3. Add the Observed histogram on top
    if has_obs:
        fig_2d.add_trace(go.Histogram(
            x=obs_df[h_col],
            xbins=dict(start=min_val, end=max_val, size=bin_size),
            autobinx=False,
            marker_color='yellow', # Matches the yellow diamonds used elsewhere
            opacity=0.8,           # Slightly more opaque so it stands out
            name="Observations",
            showlegend=True
        ))

    # 4. Set barmode to 'overlay' so they overlap instead of stacking
    fig_2d.update_layout(
        barmode='overlay', 
        bargap=0.1, 
        template="plotly_dark", 
        height=600,
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01) # Position legend inside plot
    )
    
    return fig_2d


def create_3d_los_plot(all_los_data, live_params, sun_pos):
    
    base_fig_dict = get_base_geometry(live_params, sun_pos)
    
    fig = go.Figure(base_fig_dict)
    
    for data in all_los_data:
        l_end = sun_pos + 40 * data['d_vec']
        fig.add_trace(go.Scatter3d(x=[sun_pos[0], l_end[0]], y=[sun_pos[1], l_end[1]], z=[sun_pos[2], l_end[2]], mode='lines', line=dict(color=data['config']['color'], width=5), name=f"LOS {data['id']}"))
        if data['inters']:
            pts = np.array([pt[1] for pt in data['inters']])
            fig.add_trace(go.Scatter3d(x=pts[:,0], y=pts[:,1], z=pts[:,2], mode='markers', marker=dict(size=6, color=data['config']['color'], symbol='diamond', line=dict(color='white', width=1)), showlegend=False))
    
    
    return fig 


def create_los_unified_plot(active_inters, sun_pos):
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
    unified_fig.update_xaxes(title_text="Distance from Sun (kpc)")
    
    return unified_fig
