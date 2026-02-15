"""
Analytical EM solver — fast screening engine.

This uses the analytical models built into each antenna template to quickly
evaluate designs without running full-wave simulation. Results are approximate
but orders of magnitude faster than FDTD/MoM.

Use case: initial population screening in optimization, quick "what-if" exploration.
Accuracy: ±20-30% for resonant frequency, ±5 dB for S11, directivity order-of-magnitude.
"""

from __future__ import annotations

import time
import math

import numpy as np

from .base import SolverBase, SimulationResult
from ..geometry.base import AntennaGeometry, AntennaTemplate
from ..utils.constants import chu_limit_q, chu_limit_bandwidth, C_0, ETA_0
from ..utils.units import z_to_gamma


class AnalyticalSolver(SolverBase):
    """
    Fast analytical solver using template impedance models.

    Requires the antenna template (for impedance/resonance estimation)
    in addition to the geometry.
    """

    def __init__(self, template: AntennaTemplate | None = None):
        self._template = template

    @property
    def name(self) -> str:
        return "Analytical (fast screen)"

    def is_available(self) -> bool:
        return True  # always available

    def set_template(self, template: AntennaTemplate):
        """Set the antenna template for analytical models."""
        self._template = template

    def simulate(
        self,
        geometry: AntennaGeometry,
        freq_start_hz: float,
        freq_stop_hz: float,
        n_freq: int = 101,
        z0: float = 50.0,
        include_pattern: bool = False,
        tissue_loading: float = 1.0,
        **kwargs,
    ) -> SimulationResult:
        """
        Run analytical frequency sweep.

        Args:
            geometry: Antenna geometry (used for size/bounding box)
            freq_start_hz: Start frequency
            freq_stop_hz: Stop frequency
            n_freq: Number of points
            z0: Reference impedance
            include_pattern: Whether to generate approximate pattern
            tissue_loading: Multiplicative factor for εr loading (>1 for implant)
        """
        if self._template is None:
            raise RuntimeError("AnalyticalSolver requires an antenna template. Call set_template() first.")

        t_start = time.time()

        freqs = np.linspace(freq_start_hz, freq_stop_hz, n_freq)
        params = geometry.parameters

        # Compute impedance at each frequency
        z_arr = np.zeros(n_freq, dtype=complex)
        s11_arr = np.zeros(n_freq, dtype=complex)

        for i, f in enumerate(freqs):
            z = self._template.estimate_impedance(params, f)
            z_arr[i] = z
            s11_arr[i] = z_to_gamma(z, z0)

        # S-parameter matrix (1-port)
        s_params = s11_arr.reshape(n_freq, 1, 1)

        # Efficiency estimate
        # For small antennas: η_rad ≈ R_rad / (R_rad + R_loss)
        radiation_eff = np.zeros(n_freq)
        total_eff = np.zeros(n_freq)
        for i, f in enumerate(freqs):
            z = z_arr[i]
            R = z.real
            # Rough split: radiation resistance vs loss
            lam = C_0 / f
            ka = geometry.electrical_size_at(f)

            # For electrically small: R_rad ≈ 20π²(ka)⁴ (loop-like) or 20π²(ka)² (dipole-like)
            if ka < 0.5:
                R_rad_est = 20 * math.pi**2 * ka**4 * ETA_0
                R_loss_est = max(R - R_rad_est, 0.5)
                eta_rad = R_rad_est / (R_rad_est + R_loss_est) if R > 0 else 0.5
            else:
                eta_rad = 0.8  # reasonable for electrically large

            # Tissue loading reduces efficiency
            eta_rad *= 1.0 / tissue_loading

            radiation_eff[i] = min(eta_rad, 1.0)
            mismatch = 1.0 - abs(s11_arr[i])**2
            total_eff[i] = radiation_eff[i] * mismatch

        # Gain/directivity estimate
        peak_gain = np.zeros(n_freq)
        peak_dir = np.zeros(n_freq)
        for i, f in enumerate(freqs):
            ka = geometry.electrical_size_at(f)
            if ka < 0.5:
                D = 1.5  # small antenna directivity ≈ 1.5 (short dipole)
            else:
                D = 3.0 + 4.0 * ka  # rough scaling
            peak_dir[i] = 10 * math.log10(D)
            peak_gain[i] = peak_dir[i] + 10 * math.log10(max(radiation_eff[i], 1e-6))

        # Approximate far-field pattern (if requested)
        pattern_data = {}
        if include_pattern:
            n_theta = 37
            n_phi = 73
            theta = np.linspace(0, math.pi, n_theta)
            phi = np.linspace(0, 2 * math.pi, n_phi)
            THETA, PHI = np.meshgrid(theta, phi, indexing='ij')

            # Simple dipole-like pattern: sin(θ) for E-plane
            gain_pattern = np.zeros((n_freq, n_theta, n_phi))
            for i in range(n_freq):
                D_max = 10 ** (peak_dir[i] / 10)
                gain_pattern[i] = D_max * np.sin(THETA) ** 2 * radiation_eff[i]

            pattern_data = {
                "gain_pattern": gain_pattern,
                "theta_angles": theta,
                "phi_angles": phi,
            }

        solve_time = time.time() - t_start

        result = SimulationResult(
            frequencies_hz=freqs,
            s_parameters=s_params,
            z_input=z_arr.reshape(n_freq, 1),
            radiation_efficiency=radiation_eff,
            total_efficiency=total_eff,
            peak_gain_dbi=peak_gain,
            peak_directivity_dbi=peak_dir,
            solver_name=self.name,
            solve_time_seconds=solve_time,
            converged=True,
            notes="Analytical estimates. Use full-wave solver for accurate results.",
            **pattern_data,
        )

        return result


class MultiportAnalyticalSolver(AnalyticalSolver):
    """
    Analytical solver for 2-port problems (e.g., WPT link).

    Given two antenna templates and coupling coefficient,
    computes S21 (link efficiency) analytically.
    """

    def __init__(self, template_tx: AntennaTemplate | None = None,
                 template_rx: AntennaTemplate | None = None):
        super().__init__(template_tx)
        self._template_tx = template_tx
        self._template_rx = template_rx

    def simulate_link(
        self,
        geom_tx: AntennaGeometry,
        geom_rx: AntennaGeometry,
        coupling_k: float,
        freq_start_hz: float,
        freq_stop_hz: float,
        n_freq: int = 101,
        z0: float = 50.0,
    ) -> SimulationResult:
        """
        Simulate a 2-port WPT / near-field link.

        Uses coupled resonator model:
        S21 = j·k·√(Q₁Q₂) / (1 + j·Q₁·δ₁)(1 + j·Q₂·δ₂) + k²Q₁Q₂
        """
        if self._template_tx is None or self._template_rx is None:
            raise RuntimeError("Need both Tx and Rx templates")

        t_start = time.time()
        freqs = np.linspace(freq_start_hz, freq_stop_hz, n_freq)

        # Get resonance frequencies
        f0_tx = self._template_tx.estimate_resonance_hz(geom_tx.parameters)
        f0_rx = self._template_rx.estimate_resonance_hz(geom_rx.parameters)

        # Estimate Q factors
        z_tx = self._template_tx.estimate_impedance(geom_tx.parameters, f0_tx)
        z_rx = self._template_rx.estimate_impedance(geom_rx.parameters, f0_rx)
        Q_tx = max(abs(z_tx.imag) / max(z_tx.real, 0.1), 10)
        Q_rx = max(abs(z_rx.imag) / max(z_rx.real, 0.1), 10)

        s_params = np.zeros((n_freq, 2, 2), dtype=complex)

        for i, f in enumerate(freqs):
            delta_tx = (f - f0_tx) / f0_tx
            delta_rx = (f - f0_rx) / f0_rx

            denom = ((1 + 1j * Q_tx * 2 * delta_tx) *
                     (1 + 1j * Q_rx * 2 * delta_rx) +
                     coupling_k**2 * Q_tx * Q_rx)

            s21 = 1j * coupling_k * math.sqrt(Q_tx * Q_rx) / denom
            s11 = 1 - abs(s21)**2  # simplified

            s_params[i, 0, 0] = math.sqrt(max(s11, 0)) * np.exp(1j * np.angle(s21))
            s_params[i, 0, 1] = s21
            s_params[i, 1, 0] = s21
            s_params[i, 1, 1] = s_params[i, 0, 0]

        solve_time = time.time() - t_start

        return SimulationResult(
            frequencies_hz=freqs,
            s_parameters=s_params,
            solver_name="Analytical 2-port Link",
            solve_time_seconds=solve_time,
            converged=True,
            notes=f"Coupled resonator model, k={coupling_k:.4f}, Q_tx={Q_tx:.1f}, Q_rx={Q_rx:.1f}",
        )
