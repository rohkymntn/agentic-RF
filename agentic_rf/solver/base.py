"""
Abstract base for all EM solvers.

Each solver takes an AntennaGeometry + frequency sweep and returns
S-parameters, impedance, efficiency, patterns, etc.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from ..geometry.base import AntennaGeometry


@dataclass
class SimulationResult:
    """
    Complete results from an EM simulation.

    All frequency-domain quantities are stored as arrays indexed by frequency.
    """
    # Frequency sweep
    frequencies_hz: np.ndarray = field(default_factory=lambda: np.array([]))

    # S-parameters (complex, shape: [n_freq, n_ports, n_ports])
    s_parameters: np.ndarray = field(default_factory=lambda: np.array([]))

    # Input impedance per port (complex, shape: [n_freq, n_ports])
    z_input: np.ndarray = field(default_factory=lambda: np.array([]))

    # Radiation efficiency (linear, shape: [n_freq])
    radiation_efficiency: np.ndarray = field(default_factory=lambda: np.array([]))

    # Total efficiency (includes mismatch, shape: [n_freq])
    total_efficiency: np.ndarray = field(default_factory=lambda: np.array([]))

    # Peak gain (dBi, shape: [n_freq])
    peak_gain_dbi: np.ndarray = field(default_factory=lambda: np.array([]))

    # Peak directivity (dBi, shape: [n_freq])
    peak_directivity_dbi: np.ndarray = field(default_factory=lambda: np.array([]))

    # Far-field pattern (optional, shape: [n_freq, n_theta, n_phi])
    pattern_theta: Optional[np.ndarray] = None
    pattern_phi: Optional[np.ndarray] = None
    gain_pattern: Optional[np.ndarray] = None
    theta_angles: Optional[np.ndarray] = None
    phi_angles: Optional[np.ndarray] = None

    # Near-field data (optional)
    near_field_e: Optional[np.ndarray] = None
    near_field_h: Optional[np.ndarray] = None

    # SAR data (optional, shape: [nx, ny, nz])
    sar_data: Optional[np.ndarray] = None
    peak_sar_1g: Optional[float] = None
    peak_sar_10g: Optional[float] = None

    # Metadata
    solver_name: str = ""
    solve_time_seconds: float = 0.0
    converged: bool = True
    notes: str = ""

    # ─── Convenience accessors ───────────────────────────────────────────

    @property
    def s11(self) -> np.ndarray:
        """S11 in dB."""
        if self.s_parameters.size == 0:
            return np.array([])
        s11_complex = self.s_parameters[:, 0, 0]
        return 20 * np.log10(np.abs(s11_complex) + 1e-12)

    @property
    def s11_complex(self) -> np.ndarray:
        """Complex S11."""
        if self.s_parameters.size == 0:
            return np.array([])
        return self.s_parameters[:, 0, 0]

    @property
    def vswr(self) -> np.ndarray:
        """VSWR from S11."""
        if self.s_parameters.size == 0:
            return np.array([])
        s11_mag = np.abs(self.s_parameters[:, 0, 0])
        return (1 + s11_mag) / (1 - np.clip(s11_mag, 0, 0.9999))

    def resonance_freq(self, threshold_db: float = -10.0) -> float:
        """Find frequency of deepest S11 null (resonance)."""
        if self.s11.size == 0:
            return 0.0
        idx = np.argmin(self.s11)
        return float(self.frequencies_hz[idx])

    def bandwidth_hz(self, threshold_db: float = -10.0) -> float:
        """
        Bandwidth where S11 < threshold.

        Returns bandwidth in Hz. Uses -10 dB (VSWR 2:1) by default.
        """
        if self.s11.size == 0:
            return 0.0
        below = self.s11 < threshold_db
        if not np.any(below):
            return 0.0
        indices = np.where(below)[0]
        f_low = self.frequencies_hz[indices[0]]
        f_high = self.frequencies_hz[indices[-1]]
        return float(f_high - f_low)

    def fractional_bandwidth(self, threshold_db: float = -10.0) -> float:
        """Fractional bandwidth (BW / f_center)."""
        bw = self.bandwidth_hz(threshold_db)
        fc = self.resonance_freq()
        if fc <= 0:
            return 0.0
        return bw / fc

    def s11_at_freq(self, freq_hz: float) -> float:
        """S11 in dB at nearest frequency point."""
        if self.s11.size == 0:
            return 0.0
        idx = np.argmin(np.abs(self.frequencies_hz - freq_hz))
        return float(self.s11[idx])

    def efficiency_at_freq(self, freq_hz: float) -> float:
        """Total efficiency at nearest frequency point."""
        if self.total_efficiency.size == 0:
            return 0.0
        idx = np.argmin(np.abs(self.frequencies_hz - freq_hz))
        return float(self.total_efficiency[idx])

    def summary(self) -> dict:
        """Return a compact summary dict of key performance metrics."""
        f_res = self.resonance_freq()
        return {
            "resonance_mhz": f_res / 1e6 if f_res > 0 else None,
            "min_s11_db": float(np.min(self.s11)) if self.s11.size > 0 else None,
            "bandwidth_mhz": self.bandwidth_hz() / 1e6,
            "fractional_bw_pct": self.fractional_bandwidth() * 100,
            "peak_gain_dbi": float(np.max(self.peak_gain_dbi)) if self.peak_gain_dbi.size > 0 else None,
            "peak_efficiency_pct": (float(np.max(self.total_efficiency)) * 100
                                     if self.total_efficiency.size > 0 else None),
            "peak_sar_1g": self.peak_sar_1g,
            "solver": self.solver_name,
            "solve_time_s": self.solve_time_seconds,
        }


class SolverBase(ABC):
    """Abstract EM solver interface."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Solver name."""
        ...

    @abstractmethod
    def simulate(
        self,
        geometry: AntennaGeometry,
        freq_start_hz: float,
        freq_stop_hz: float,
        n_freq: int = 101,
        **kwargs,
    ) -> SimulationResult:
        """
        Run EM simulation over a frequency sweep.

        Args:
            geometry: Complete antenna geometry
            freq_start_hz: Start frequency
            freq_stop_hz: Stop frequency
            n_freq: Number of frequency points
            **kwargs: Solver-specific options

        Returns:
            SimulationResult with all computed quantities
        """
        ...

    @abstractmethod
    def is_available(self) -> bool:
        """Check if solver is installed and ready."""
        ...
