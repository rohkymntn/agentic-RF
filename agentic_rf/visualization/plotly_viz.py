"""
Interactive Plotly-based visualizations for the web frontend.

All functions return Plotly figure objects (go.Figure) suitable
for display in Streamlit via st.plotly_chart().
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from ..geometry.base import AntennaGeometry
from ..solver.base import SimulationResult


def plot_geometry_3d(geometry: AntennaGeometry, title: str = "Antenna Geometry") -> go.Figure:
    """Interactive 3D antenna geometry view."""
    fig = go.Figure()

    # Plot traces as 3D lines
    for i, trace in enumerate(geometry.traces):
        fig.add_trace(go.Scatter3d(
            x=[trace.start.x * 1e3, trace.end.x * 1e3],
            y=[trace.start.y * 1e3, trace.end.y * 1e3],
            z=[trace.start.z * 1e3, trace.end.z * 1e3],
            mode='lines',
            line=dict(color='#FF6600', width=max(2, trace.width * 1e3 * 2)),
            name=f'Trace' if i == 0 else None,
            showlegend=(i == 0),
            legendgroup='traces',
        ))

    # Plot boxes as wireframes
    for i, box in enumerate(geometry.boxes):
        o = box.origin
        s = box.size
        color = '#FFB800' if box.material.is_conductor else '#00CC66'
        label = box.material.name

        # 12 edges of the box
        corners = [
            [o.x, o.y, o.z],
            [o.x + s.x, o.y, o.z],
            [o.x + s.x, o.y + s.y, o.z],
            [o.x, o.y + s.y, o.z],
            [o.x, o.y, o.z + s.z],
            [o.x + s.x, o.y, o.z + s.z],
            [o.x + s.x, o.y + s.y, o.z + s.z],
            [o.x, o.y + s.y, o.z + s.z],
        ]
        edges = [
            (0, 1), (1, 2), (2, 3), (3, 0),
            (4, 5), (5, 6), (6, 7), (7, 4),
            (0, 4), (1, 5), (2, 6), (3, 7),
        ]

        for j, (a, b) in enumerate(edges):
            fig.add_trace(go.Scatter3d(
                x=[corners[a][0] * 1e3, corners[b][0] * 1e3],
                y=[corners[a][1] * 1e3, corners[b][1] * 1e3],
                z=[corners[a][2] * 1e3, corners[b][2] * 1e3],
                mode='lines',
                line=dict(color=color, width=1),
                name=label if j == 0 else None,
                showlegend=(j == 0 and i < 3),
                legendgroup=f'box_{i}',
                hoverinfo='text',
                text=f'{label}',
            ))

    # Plot ports
    for port in geometry.ports:
        fig.add_trace(go.Scatter3d(
            x=[port.start.x * 1e3, port.end.x * 1e3],
            y=[port.start.y * 1e3, port.end.y * 1e3],
            z=[port.start.z * 1e3, port.end.z * 1e3],
            mode='lines+markers',
            line=dict(color='red', width=6),
            marker=dict(size=4, color='red'),
            name=f'Port {port.port_number}',
        ))

    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title='X (mm)',
            yaxis_title='Y (mm)',
            zaxis_title='Z (mm)',
            aspectmode='data',
        ),
        height=600,
        margin=dict(l=0, r=0, t=40, b=0),
    )

    return fig


def plot_s11_interactive(result: SimulationResult,
                          target_band: tuple | None = None,
                          title: str = "S11 (Return Loss)") -> go.Figure:
    """Interactive S11 plot with hover data."""
    fig = go.Figure()

    freqs_mhz = result.frequencies_hz / 1e6

    fig.add_trace(go.Scatter(
        x=freqs_mhz, y=result.s11,
        mode='lines', name='S11',
        line=dict(color='#1f77b4', width=2),
        hovertemplate='%{x:.1f} MHz<br>S11: %{y:.2f} dB<extra></extra>',
    ))

    # Threshold
    fig.add_hline(y=-10, line_dash="dash", line_color="red",
                  annotation_text="-10 dB (VSWR 2:1)")
    fig.add_hline(y=-6, line_dash="dot", line_color="orange",
                  annotation_text="-6 dB (VSWR 3:1)")

    if target_band:
        fig.add_vrect(x0=target_band[0] / 1e6, x1=target_band[1] / 1e6,
                       fillcolor="green", opacity=0.1, line_width=0,
                       annotation_text="Target Band")

    fig.update_layout(
        title=title,
        xaxis_title="Frequency (MHz)",
        yaxis_title="S11 (dB)",
        yaxis=dict(range=[min(-30, min(result.s11) - 5), 2]),
        height=400,
        hovermode='x unified',
    )
    return fig


def plot_vswr_interactive(result: SimulationResult) -> go.Figure:
    """Interactive VSWR plot."""
    fig = go.Figure()
    freqs_mhz = result.frequencies_hz / 1e6

    fig.add_trace(go.Scatter(
        x=freqs_mhz, y=result.vswr,
        mode='lines', name='VSWR',
        line=dict(color='#2ca02c', width=2),
        hovertemplate='%{x:.1f} MHz<br>VSWR: %{y:.2f}:1<extra></extra>',
    ))

    fig.add_hline(y=2.0, line_dash="dash", line_color="red", annotation_text="2:1")
    fig.add_hline(y=3.0, line_dash="dot", line_color="orange", annotation_text="3:1")

    fig.update_layout(
        title="VSWR",
        xaxis_title="Frequency (MHz)",
        yaxis_title="VSWR",
        yaxis=dict(range=[1, min(10, max(result.vswr[:10].max(), 5))]),
        height=400,
    )
    return fig


def plot_impedance_interactive(result: SimulationResult) -> go.Figure:
    """Interactive impedance plot (R + jX)."""
    if result.z_input.size == 0:
        return go.Figure()

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        subplot_titles=("Resistance (R)", "Reactance (X)"))

    freqs_mhz = result.frequencies_hz / 1e6
    z = result.z_input[:, 0]

    fig.add_trace(go.Scatter(
        x=freqs_mhz, y=np.real(z),
        mode='lines', name='R (Ω)',
        line=dict(color='#1f77b4', width=2),
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=freqs_mhz, y=[50] * len(freqs_mhz),
        mode='lines', name='50 Ω',
        line=dict(color='red', dash='dash', width=1),
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=freqs_mhz, y=np.imag(z),
        mode='lines', name='X (Ω)',
        line=dict(color='#ff7f0e', width=2),
    ), row=2, col=1)

    fig.add_trace(go.Scatter(
        x=freqs_mhz, y=[0] * len(freqs_mhz),
        mode='lines', name='0 Ω',
        line=dict(color='gray', dash='dash', width=1),
        showlegend=False,
    ), row=2, col=1)

    fig.update_layout(height=500, title="Input Impedance")
    fig.update_xaxes(title_text="Frequency (MHz)", row=2, col=1)
    return fig


def plot_smith_interactive(result: SimulationResult,
                            title: str = "Smith Chart") -> go.Figure:
    """Interactive Smith chart using Plotly."""
    fig = go.Figure()

    # Smith chart background circles
    theta = np.linspace(0, 2 * np.pi, 200)

    # Unit circle
    fig.add_trace(go.Scatter(
        x=np.cos(theta), y=np.sin(theta),
        mode='lines', line=dict(color='black', width=1),
        showlegend=False, hoverinfo='skip',
    ))

    # Constant R circles
    for r in [0, 0.2, 0.5, 1, 2, 5]:
        cx = r / (1 + r)
        cr = 1 / (1 + r)
        x = cx + cr * np.cos(theta)
        y = cr * np.sin(theta)
        mask = x**2 + y**2 <= 1.01
        x[~mask] = np.nan
        fig.add_trace(go.Scatter(
            x=x, y=y, mode='lines',
            line=dict(color='lightgray', width=0.5),
            showlegend=False, hoverinfo='skip',
        ))

    # Constant X arcs
    for xv in [0.2, 0.5, 1, 2, 5]:
        for sign in [1, -1]:
            cy = sign / xv
            cr = 1 / xv
            x = 1 + cr * np.cos(theta)
            y = cy + cr * np.sin(theta)
            mask = (x**2 + y**2 <= 1.01) & (x >= -1)
            x[~mask] = np.nan
            fig.add_trace(go.Scatter(
                x=x, y=y, mode='lines',
                line=dict(color='lightgray', width=0.5),
                showlegend=False, hoverinfo='skip',
            ))

    # Real axis
    fig.add_trace(go.Scatter(
        x=[-1, 1], y=[0, 0], mode='lines',
        line=dict(color='lightgray', width=0.5),
        showlegend=False, hoverinfo='skip',
    ))

    # Plot S11 data
    s11 = result.s11_complex
    freqs = result.frequencies_hz

    hover_text = [f'{f/1e6:.1f} MHz<br>|S11|={20*np.log10(abs(s)+1e-12):.1f} dB'
                  for f, s in zip(freqs, s11)]

    fig.add_trace(go.Scatter(
        x=np.real(s11), y=np.imag(s11),
        mode='lines+markers',
        marker=dict(size=3, color=freqs / 1e6, colorscale='Viridis',
                    colorbar=dict(title="MHz"), showscale=True),
        line=dict(color='#1f77b4', width=2),
        name='S11',
        text=hover_text, hoverinfo='text',
    ))

    # Mark start and end
    fig.add_trace(go.Scatter(
        x=[np.real(s11[0])], y=[np.imag(s11[0])],
        mode='markers', marker=dict(size=10, color='green', symbol='circle'),
        name=f'Start ({freqs[0]/1e6:.0f} MHz)',
    ))
    fig.add_trace(go.Scatter(
        x=[np.real(s11[-1])], y=[np.imag(s11[-1])],
        mode='markers', marker=dict(size=10, color='red', symbol='square'),
        name=f'Stop ({freqs[-1]/1e6:.0f} MHz)',
    ))

    fig.update_layout(
        title=title,
        xaxis=dict(range=[-1.15, 1.15], scaleanchor="y", title="Real(Γ)"),
        yaxis=dict(range=[-1.15, 1.15], title="Imag(Γ)"),
        height=550, width=550,
    )
    return fig


def plot_field_heatmap(field_data: dict, field_type: str = "B_mag",
                        title: str = "Field Distribution") -> go.Figure:
    """
    Plot near-field distribution as interactive heatmap.

    field_type: "E_mag", "H_mag", "B_mag", "Ex", "Ey", "Ez", "Hx", "Hy", "Hz", etc.
    """
    C1 = field_data["coord1"]
    C2 = field_data["coord2"]
    labels = field_data["labels"]

    data = field_data.get(field_type, field_data["E_mag"])
    if np.iscomplexobj(data):
        data = np.abs(data)

    # Log scale for better visualization
    data_db = 20 * np.log10(np.maximum(data, 1e-12))
    vmax = np.max(data_db)
    vmin = max(vmax - 60, np.min(data_db))

    fig = go.Figure(data=go.Heatmap(
        x=C1[0, :],
        y=C2[:, 0],
        z=data_db,
        colorscale='Hot',
        zmin=vmin,
        zmax=vmax,
        colorbar=dict(title=f"|{field_type}| (dB rel.)"),
        hovertemplate=f'{labels[0]}: %{{x:.1f}} mm<br>{labels[1]}: %{{y:.1f}} mm<br>Value: %{{z:.1f}} dB<extra></extra>',
    ))

    fig.update_layout(
        title=title,
        xaxis_title=labels[0],
        yaxis_title=labels[1],
        height=500,
        yaxis=dict(scaleanchor="x"),
    )
    return fig


def plot_field_arrows(field_data: dict, field_component: str = "H",
                       title: str = "Vector Field") -> go.Figure:
    """Plot field as quiver (arrow) plot."""
    C1 = field_data["coord1"]
    C2 = field_data["coord2"]
    labels = field_data["labels"]
    plane = field_data["plane"]

    # Select field components based on plane
    if plane == "xz":
        if field_component == "H":
            U = np.real(field_data["Hx"])
            V = np.real(field_data["Hz"])
        else:
            U = np.real(field_data["Ex"])
            V = np.real(field_data["Ez"])
    elif plane == "xy":
        if field_component == "H":
            U = np.real(field_data["Hx"])
            V = np.real(field_data["Hy"])
        else:
            U = np.real(field_data["Ex"])
            V = np.real(field_data["Ey"])
    else:  # yz
        if field_component == "H":
            U = np.real(field_data["Hy"])
            V = np.real(field_data["Hz"])
        else:
            U = np.real(field_data["Ey"])
            V = np.real(field_data["Ez"])

    # Subsample for readable arrows
    step = max(1, len(C1) // 20)
    x = C1[::step, ::step].flatten()
    y = C2[::step, ::step].flatten()
    u = U[::step, ::step].flatten()
    v = V[::step, ::step].flatten()

    # Normalize for display
    mag = np.sqrt(u**2 + v**2)
    max_mag = np.max(mag) + 1e-12
    scale = (C1[0, 1] - C1[0, 0]) * step * 0.8
    u_norm = u / max_mag * scale
    v_norm = v / max_mag * scale

    # Background: magnitude heatmap
    mag_field = field_data[f"{field_component}_mag"]
    fig = go.Figure(data=go.Heatmap(
        x=C1[0, :], y=C2[:, 0],
        z=20 * np.log10(np.maximum(np.abs(mag_field), 1e-12)),
        colorscale='Blues', opacity=0.5,
        showscale=True,
        colorbar=dict(title=f"|{field_component}| (dB)"),
    ))

    # Add arrows using annotations
    for xi, yi, ui, vi, mi in zip(x, y, u_norm, v_norm, mag):
        if mi / max_mag > 0.05:  # skip tiny arrows
            fig.add_annotation(
                x=xi + ui, y=yi + vi,
                ax=xi, ay=yi,
                xref="x", yref="y",
                axref="x", ayref="y",
                showarrow=True,
                arrowhead=2, arrowsize=1.5, arrowwidth=1.5,
                arrowcolor=f'rgba(255,{int(100*(1-mi/max_mag))},0,0.8)',
            )

    fig.update_layout(
        title=title,
        xaxis_title=labels[0],
        yaxis_title=labels[1],
        height=550,
        yaxis=dict(scaleanchor="x"),
    )
    return fig


def plot_radiation_3d(pattern_data: dict, title: str = "3D Radiation Pattern") -> go.Figure:
    """Interactive 3D radiation pattern surface."""
    fig = go.Figure(data=go.Surface(
        x=pattern_data["X"],
        y=pattern_data["Y"],
        z=pattern_data["Z"],
        surfacecolor=pattern_data["gain_dbi"],
        colorscale='Jet',
        colorbar=dict(title="Gain (dBi)"),
        opacity=0.85,
        hovertemplate='Gain: %{surfacecolor:.1f} dBi<extra></extra>',
    ))

    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title='X', yaxis_title='Y', zaxis_title='Z',
            aspectmode='data',
        ),
        height=550,
        margin=dict(l=0, r=0, t=40, b=0),
    )
    return fig


def plot_convergence_interactive(history: list[float],
                                  title: str = "Optimization Convergence") -> go.Figure:
    """Interactive optimization convergence plot."""
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=list(range(1, len(history) + 1)),
        y=history,
        mode='lines',
        line=dict(color='#1f77b4', width=2),
        name='Best Score',
        hovertemplate='Iteration %{x}<br>Score: %{y:.4f}<extra></extra>',
    ))

    fig.update_layout(
        title=title,
        xaxis_title="Iteration",
        yaxis_title="Objective Score (lower = better)",
        height=400,
    )
    return fig


def plot_pareto_interactive(results: list, x_attr: str = "radius_mm",
                             y_attr: str = "s11_db",
                             color_attr: str = "bandwidth_hz") -> go.Figure:
    """Interactive Pareto front scatter plot."""
    x = [getattr(r, x_attr) for r in results]
    y = [getattr(r, y_attr) for r in results]
    c = [getattr(r, color_attr, 0) for r in results]

    fig = go.Figure(data=go.Scatter(
        x=x, y=y,
        mode='markers',
        marker=dict(
            size=10, color=c, colorscale='Viridis',
            colorbar=dict(title=color_attr.replace('_', ' ')),
            showscale=True,
        ),
        hovertemplate=(
            f'{x_attr}: %{{x:.2f}}<br>'
            f'{y_attr}: %{{y:.2f}}<br>'
            f'{color_attr}: %{{marker.color:.2f}}<extra></extra>'
        ),
    ))

    fig.update_layout(
        title="Pareto Front (Multi-Objective Tradeoffs)",
        xaxis_title=x_attr.replace('_', ' ').title(),
        yaxis_title=y_attr.replace('_', ' ').title(),
        height=450,
    )
    return fig
