"""
Comprehensive antenna visualization.

Provides publication-quality plots for:
  - S-parameters (magnitude, phase)
  - Smith chart
  - VSWR
  - Impedance (R + jX vs frequency)
  - Radiation patterns (polar, 3D)
  - 3D antenna geometry
  - Optimization convergence
  - Pareto fronts
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Optional

import numpy as np

from ..solver.base import SimulationResult
from ..geometry.base import AntennaGeometry
from ..optimizer.objectives import ObjectiveResult


class AntennaPlotter:
    """All-in-one plotting for antenna design results."""

    def __init__(self, style: str = "default", figsize: tuple = (10, 6)):
        """
        Args:
            style: Matplotlib style ('default', 'dark_background', 'seaborn-v0_8', etc.)
            figsize: Default figure size
        """
        self.style = style
        self.figsize = figsize
        self._setup_style()

    def _setup_style(self):
        """Configure matplotlib style."""
        import matplotlib.pyplot as plt
        try:
            plt.style.use(self.style)
        except OSError:
            pass
        plt.rcParams.update({
            'font.size': 11,
            'axes.grid': True,
            'grid.alpha': 0.3,
            'figure.dpi': 100,
        })

    # ═══════════════════════════════════════════════════════════════════════
    # S-Parameter Plots
    # ═══════════════════════════════════════════════════════════════════════

    def plot_s11(self, result: SimulationResult,
                 title: str = "S11 (Return Loss)",
                 target_band: tuple | None = None,
                 threshold_db: float = -10.0,
                 save_path: str | None = None):
        """
        Plot S11 magnitude in dB vs frequency.

        Args:
            result: Simulation result
            title: Plot title
            target_band: (f_start, f_stop) in Hz to highlight
            threshold_db: S11 threshold line
            save_path: Save plot to file
        """
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=self.figsize)

        freqs_mhz = result.frequencies_hz / 1e6
        ax.plot(freqs_mhz, result.s11, 'b-', linewidth=2, label='S11')

        # Threshold line
        ax.axhline(y=threshold_db, color='r', linestyle='--', alpha=0.7,
                    label=f'{threshold_db} dB')

        # Target band highlight
        if target_band:
            ax.axvspan(target_band[0] / 1e6, target_band[1] / 1e6,
                        alpha=0.15, color='green', label='Target band')

        ax.set_xlabel('Frequency (MHz)')
        ax.set_ylabel('S11 (dB)')
        ax.set_title(title)
        ax.legend(loc='best')
        ax.set_ylim(top=0)

        plt.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches='tight')
        return fig

    def plot_vswr(self, result: SimulationResult,
                  title: str = "VSWR",
                  save_path: str | None = None):
        """Plot VSWR vs frequency."""
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=self.figsize)

        freqs_mhz = result.frequencies_hz / 1e6
        ax.plot(freqs_mhz, result.vswr, 'b-', linewidth=2)
        ax.axhline(y=2.0, color='r', linestyle='--', alpha=0.7, label='VSWR = 2:1')
        ax.axhline(y=3.0, color='orange', linestyle='--', alpha=0.5, label='VSWR = 3:1')

        ax.set_xlabel('Frequency (MHz)')
        ax.set_ylabel('VSWR')
        ax.set_title(title)
        ax.set_ylim(1, 10)
        ax.legend()

        plt.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches='tight')
        return fig

    # ═══════════════════════════════════════════════════════════════════════
    # Smith Chart
    # ═══════════════════════════════════════════════════════════════════════

    def plot_smith(self, result: SimulationResult,
                   title: str = "Smith Chart",
                   annotate_freqs: list[float] | None = None,
                   save_path: str | None = None):
        """
        Plot impedance trajectory on Smith chart.

        Args:
            result: Simulation result with S-parameters
            annotate_freqs: List of frequencies [Hz] to annotate
            save_path: Save path
        """
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(8, 8))
        ax.set_aspect('equal')

        # Draw Smith chart grid
        self._draw_smith_grid(ax)

        # Plot data
        s11 = result.s11_complex
        ax.plot(s11.real, s11.imag, 'b-', linewidth=2, zorder=10)
        ax.plot(s11[0].real, s11[0].imag, 'go', markersize=8,
                label=f'Start: {result.frequencies_hz[0]/1e6:.0f} MHz', zorder=11)
        ax.plot(s11[-1].real, s11[-1].imag, 'rs', markersize=8,
                label=f'Stop: {result.frequencies_hz[-1]/1e6:.0f} MHz', zorder=11)

        # Annotate specific frequencies
        if annotate_freqs:
            for f in annotate_freqs:
                idx = np.argmin(np.abs(result.frequencies_hz - f))
                pt = s11[idx]
                ax.annotate(f'{f/1e6:.0f} MHz',
                            xy=(pt.real, pt.imag),
                            xytext=(10, 10), textcoords='offset points',
                            fontsize=9, color='red',
                            arrowprops=dict(arrowstyle='->', color='red'))

        ax.set_title(title)
        ax.legend(loc='lower right')
        ax.set_xlim(-1.1, 1.1)
        ax.set_ylim(-1.1, 1.1)

        plt.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches='tight')
        return fig

    def _draw_smith_grid(self, ax):
        """Draw Smith chart constant-R and constant-X circles."""
        theta = np.linspace(0, 2 * np.pi, 200)

        # Unit circle (|Γ| = 1)
        ax.plot(np.cos(theta), np.sin(theta), 'k-', linewidth=1)
        ax.axhline(y=0, color='k', linewidth=0.5)

        # Constant resistance circles
        for r in [0, 0.2, 0.5, 1, 2, 5]:
            center_x = r / (1 + r)
            radius = 1 / (1 + r)
            circle_x = center_x + radius * np.cos(theta)
            circle_y = radius * np.sin(theta)
            # Clip to unit circle
            mask = circle_x**2 + circle_y**2 <= 1.01
            ax.plot(circle_x[mask], circle_y[mask], 'gray', linewidth=0.4, alpha=0.5)

        # Constant reactance arcs
        for x in [0.2, 0.5, 1, 2, 5]:
            for sign in [1, -1]:
                center_y = sign / x if x != 0 else 0
                radius = 1 / abs(x) if x != 0 else 0
                arc_x = 1 + radius * np.cos(theta)
                arc_y = center_y + radius * np.sin(theta)
                mask = (arc_x**2 + arc_y**2 <= 1.01) & (arc_x >= -1)
                if np.any(mask):
                    ax.plot(arc_x[mask], arc_y[mask], 'gray', linewidth=0.4, alpha=0.5)

    # ═══════════════════════════════════════════════════════════════════════
    # Impedance Plot
    # ═══════════════════════════════════════════════════════════════════════

    def plot_impedance(self, result: SimulationResult,
                       title: str = "Input Impedance",
                       save_path: str | None = None):
        """Plot real and imaginary impedance vs frequency."""
        import matplotlib.pyplot as plt

        if result.z_input.size == 0:
            return None

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(self.figsize[0], self.figsize[1] * 1.2),
                                         sharex=True)

        freqs_mhz = result.frequencies_hz / 1e6
        z = result.z_input[:, 0]

        ax1.plot(freqs_mhz, np.real(z), 'b-', linewidth=2, label='R (Ω)')
        ax1.axhline(y=50, color='r', linestyle='--', alpha=0.5, label='50 Ω')
        ax1.set_ylabel('Resistance (Ω)')
        ax1.legend()
        ax1.set_title(title)

        ax2.plot(freqs_mhz, np.imag(z), 'r-', linewidth=2, label='X (Ω)')
        ax2.axhline(y=0, color='gray', linestyle='-', alpha=0.5)
        ax2.set_xlabel('Frequency (MHz)')
        ax2.set_ylabel('Reactance (Ω)')
        ax2.legend()

        plt.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches='tight')
        return fig

    # ═══════════════════════════════════════════════════════════════════════
    # Radiation Pattern
    # ═══════════════════════════════════════════════════════════════════════

    def plot_pattern_polar(self, result: SimulationResult,
                            freq_hz: float,
                            plane: str = "E",
                            title: str = "Radiation Pattern",
                            save_path: str | None = None):
        """
        Plot radiation pattern in polar coordinates.

        Args:
            result: Simulation with pattern data
            freq_hz: Frequency to plot
            plane: "E" (phi=0) or "H" (phi=90) plane
        """
        import matplotlib.pyplot as plt

        if result.gain_pattern is None or result.theta_angles is None:
            return None

        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

        freq_idx = np.argmin(np.abs(result.frequencies_hz - freq_hz))
        theta = result.theta_angles

        if plane == "E":
            phi_idx = 0
            label = "E-plane (φ=0°)"
        else:
            phi_idx = len(result.phi_angles) // 4  # 90°
            label = "H-plane (φ=90°)"

        gain = result.gain_pattern[freq_idx, :, phi_idx]
        gain_db = 10 * np.log10(np.maximum(gain, 1e-12))
        gain_db_norm = gain_db - np.max(gain_db)  # normalize to 0 dB peak

        ax.plot(theta, gain_db_norm, 'b-', linewidth=2, label=label)
        ax.set_theta_zero_location('N')
        ax.set_theta_direction(-1)
        ax.set_rlim(-30, 5)
        ax.set_title(f"{title}\nf = {freq_hz/1e6:.1f} MHz", pad=20)
        ax.legend(loc='lower right')

        plt.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches='tight')
        return fig

    # ═══════════════════════════════════════════════════════════════════════
    # 3D Geometry
    # ═══════════════════════════════════════════════════════════════════════

    def plot_geometry_3d(self, geometry: AntennaGeometry,
                          title: str = "Antenna Geometry",
                          save_path: str | None = None):
        """Plot antenna geometry in 3D."""
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        from mpl_toolkits.mplot3d.art3d import Line3DCollection

        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')

        # Plot traces
        for trace in geometry.traces:
            xs = [trace.start.x * 1e3, trace.end.x * 1e3]
            ys = [trace.start.y * 1e3, trace.end.y * 1e3]
            zs = [trace.start.z * 1e3, trace.end.z * 1e3]
            ax.plot(xs, ys, zs, 'b-', linewidth=max(1, trace.width * 1e3))

        # Plot boxes (as wireframes)
        for box in geometry.boxes:
            o = box.origin
            s = box.size
            color = 'orange' if box.material.is_conductor else 'green'
            alpha = 0.3

            # Draw box edges
            x = [o.x, o.x + s.x]
            y = [o.y, o.y + s.y]
            z = [o.z, o.z + s.z]

            for i in range(2):
                for j in range(2):
                    ax.plot([x[0]*1e3, x[1]*1e3], [y[i]*1e3, y[i]*1e3],
                            [z[j]*1e3, z[j]*1e3], color=color, alpha=alpha, linewidth=0.5)
                    ax.plot([x[i]*1e3, x[i]*1e3], [y[0]*1e3, y[1]*1e3],
                            [z[j]*1e3, z[j]*1e3], color=color, alpha=alpha, linewidth=0.5)
                    ax.plot([x[i]*1e3, x[i]*1e3], [y[j]*1e3, y[j]*1e3],
                            [z[0]*1e3, z[1]*1e3], color=color, alpha=alpha, linewidth=0.5)

        # Plot ports
        for port in geometry.ports:
            ax.plot(
                [port.start.x * 1e3, port.end.x * 1e3],
                [port.start.y * 1e3, port.end.y * 1e3],
                [port.start.z * 1e3, port.end.z * 1e3],
                'r-', linewidth=3, label=f'Port {port.port_number}'
            )

        ax.set_xlabel('X (mm)')
        ax.set_ylabel('Y (mm)')
        ax.set_zlabel('Z (mm)')
        ax.set_title(title)

        plt.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches='tight')
        return fig

    # ═══════════════════════════════════════════════════════════════════════
    # Optimization Plots
    # ═══════════════════════════════════════════════════════════════════════

    def plot_convergence(self, convergence_history: list[float],
                          title: str = "Optimization Convergence",
                          save_path: str | None = None):
        """Plot optimization convergence curve."""
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=self.figsize)
        ax.plot(range(1, len(convergence_history) + 1), convergence_history,
                'b-', linewidth=2)
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Best Objective Score')
        ax.set_title(title)
        ax.set_yscale('log' if min(convergence_history) > 0 else 'linear')

        plt.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches='tight')
        return fig

    def plot_pareto_front(self, results: list[ObjectiveResult],
                           x_metric: str = "radius_mm",
                           y_metric: str = "s11_db",
                           title: str = "Pareto Front",
                           save_path: str | None = None):
        """
        Plot Pareto front from multi-objective optimization.

        Args:
            results: List of Pareto-optimal ObjectiveResult
            x_metric: Attribute name for x-axis
            y_metric: Attribute name for y-axis
        """
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=self.figsize)

        x_vals = [getattr(r, x_metric) for r in results]
        y_vals = [getattr(r, y_metric) for r in results]

        ax.scatter(x_vals, y_vals, c='blue', s=60, edgecolors='navy', zorder=10)

        # Sort and connect Pareto front
        sorted_idx = np.argsort(x_vals)
        ax.plot([x_vals[i] for i in sorted_idx],
                [y_vals[i] for i in sorted_idx],
                'b--', alpha=0.5)

        ax.set_xlabel(x_metric.replace('_', ' ').title())
        ax.set_ylabel(y_metric.replace('_', ' ').title())
        ax.set_title(title)

        plt.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches='tight')
        return fig

    # ═══════════════════════════════════════════════════════════════════════
    # Comparison Plots
    # ═══════════════════════════════════════════════════════════════════════

    def plot_sim_vs_measurement(self, simulated: SimulationResult,
                                  measured: SimulationResult,
                                  title: str = "Simulation vs Measurement",
                                  save_path: str | None = None):
        """Compare simulated and measured S11."""
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=self.figsize)

        ax.plot(simulated.frequencies_hz / 1e6, simulated.s11,
                'b-', linewidth=2, label='Simulated')
        ax.plot(measured.frequencies_hz / 1e6, measured.s11,
                'r--', linewidth=2, label='Measured')

        ax.set_xlabel('Frequency (MHz)')
        ax.set_ylabel('S11 (dB)')
        ax.set_title(title)
        ax.legend()
        ax.set_ylim(top=0)

        plt.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches='tight')
        return fig
