import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from geometry import get_ellipsoid_mesh

def create_3d_wind_plot(plot_sample, live_params, sun_pos):
    fig_wind = go.Figure()
    limit = 15
    ax_range = [-limit, limit]
    
    # Axes
    for coords in [([ax_range, [0,0], [0,0]]), ([[0,0], ax_range, [0,0]]), ([[0,0], [0,0], ax_range])]:
        fig_wind.add_trace(go.Scatter3d(x=coords[0], y=coords[1], z=coords[2], mode='lines', line=dict(color='white', width=6), showlegend=False))
    
    # Live Geometry Surface
    for z_c, s in [(live_params['z0'], 1), (-live_params['z0'], -1)]:
        bx_mesh, by_mesh, bz_mesh = get_ellipsoid_mesh(
            z_c, live_params['a'], live_params['b'], live_params['c'], s, 
            live_params['polar_angle'], live_params['az_angle']
        )
        fig_wind.add_trace(go.Surface(x=bx_mesh, y=by_mesh, z=bz_mesh, colorscale=[[0, 'white'], [1, 'white']], opacity=0.1, showscale=False, hoverinfo='skip'))

    # Calculated Particles
    if plot_sample is not None:
        fig_wind.add_trace(go.Scatter3d(
            x=plot_sample['x'], y=plot_sample['y'], z=plot_sample['z'],
            mode='markers',
            marker=dict(size=3, color=plot_sample['V_LSR'], colorscale='RdBu_r', colorbar=dict(title="V_LSR"), opacity=0.8),
            name='Particles'
        ))

    # Galactic Plane & Sun
    gx, gy = np.meshgrid(np.linspace(-limit, limit, 10), np.linspace(-limit, limit, 10))
    fig_wind.add_trace(go.Surface(x=gx, y=gy, z=np.zeros_like(gx), colorscale=[[0, 'blue'], [1, 'blue']], opacity=0.15, showscale=False, name='Galactic Plane'))
    fig_wind.add_trace(go.Scatter3d(x=[sun_pos[0]], y=[sun_pos[1]], z=[sun_pos[2]], mode='markers+text', text=["Sun"], textposition="top center", textfont=dict(color='orange', size=14), marker=dict(size=12, color='orange'), showlegend=False))

    fig_wind.update_layout(scene=dict(aspectmode='manual', aspectratio=dict(x=1, y=1, z=1), xaxis=dict(range=[-limit, limit]), yaxis=dict(range=[-limit, limit]), zaxis=dict(range=[-limit, limit])), template="plotly_dark", height=600, margin=dict(l=0, r=0, b=0, t=0), uirevision='constant')
    return fig_wind

def create_2d_scatter_plot(working_df, x_col, y_col, c_col):
    fig_2d = px.scatter(working_df, x=x_col, y=y_col, color=c_col, color_continuous_scale='RdBu_r', hover_data=['x', 'y', 'z'])
    fig_2d.update_traces(marker=dict(size=4, opacity=0.7))
    fig_2d.update_xaxes(showgrid=True, gridwidth=1, gridcolor='rgba(255, 255, 255, 0.2)')
    fig_2d.update_layout(template="plotly_dark", height=600)
    return fig_2d

def create_2d_histogram(working_df, h_col, bins):
    min_val, max_val = working_df[h_col].min(), working_df[h_col].max()
    bin_size = (max_val - min_val) / bins if max_val > min_val else 1.0
    fig_2d = px.histogram(working_df, x=h_col, color_discrete_sequence=['#00FFFF'])
    fig_2d.update_traces(xbins=dict(start=min_val, end=max_val, size=bin_size), autobinx=False)
    fig_2d.update_layout(bargap=0.1, template="plotly_dark", height=600)
    return fig_2d

def create_3d_los_plot(all_los_data, live_params, sun_pos):
    fig = go.Figure()
    limit = 15
    ax_range = [-limit, limit]
    
    for coords in [([ax_range, [0,0], [0,0]]), ([[0,0], ax_range, [0,0]]), ([[0,0], [0,0], ax_range])]:
        fig.add_trace(go.Scatter3d(x=coords[0], y=coords[1], z=coords[2], mode='lines', line=dict(color='white', width=6), showlegend=False))
        
    gx, gy = np.meshgrid(np.linspace(-limit, limit, 10), np.linspace(-limit, limit, 10))
    fig.add_trace(go.Surface(x=gx, y=gy, z=np.zeros_like(gx), colorscale=[[0, 'blue'], [1, 'blue']], opacity=0.15, showscale=False))

    for z_c, s in [(live_params['z0'], 1), (-live_params['z0'], -1)]:
        bx_mesh, by_mesh, bz_mesh = get_ellipsoid_mesh(
            z_c, live_params['a'], live_params['b'], live_params['c'], s, 
            live_params['polar_angle'], live_params['az_angle']
        )
        fig.add_trace(go.Surface(x=bx_mesh, y=by_mesh, z=bz_mesh, colorscale=[[0, 'red'], [1, 'red']], opacity=0.2, showscale=False))

    for data in all_los_data:
        l_end = sun_pos + 40 * data['d_vec']
        fig.add_trace(go.Scatter3d(x=[sun_pos[0], l_end[0]], y=[sun_pos[1], l_end[1]], z=[sun_pos[2], l_end[2]], mode='lines', line=dict(color=data['config']['color'], width=5), name=f"LOS {data['id']}"))
        if data['inters']:
            pts = np.array([pt[1] for pt in data['inters']])
            fig.add_trace(go.Scatter3d(x=pts[:,0], y=pts[:,1], z=pts[:,2], mode='markers', marker=dict(size=6, color=data['config']['color'], symbol='diamond', line=dict(color='white', width=1)), showlegend=False))

    fig.add_trace(go.Scatter3d(x=[sun_pos[0]], y=[sun_pos[1]], z=[sun_pos[2]], mode='markers+text', text=["Sun"], textposition="top center", textfont=dict(color='orange', size=14), marker=dict(size=12, color='orange', symbol='circle'), name='Sun'))
    fig.update_layout(scene=dict(aspectmode='manual', aspectratio=dict(x=1, y=1, z=1), xaxis=dict(range=[-limit, limit]), yaxis=dict(range=[-limit, limit]), zaxis=dict(range=[-limit, limit])), template="plotly_dark", height=700, margin=dict(l=0, r=0, b=0, t=0), uirevision='constant')
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
    return unified_fig