"""
Impedance analysis and matching network design.

Provides:
  - Smith chart data generation
  - L/C matching network synthesis
  - Quality factor extraction
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from ..utils.constants import Z0_DEFAULT
from ..utils.units import z_to_gamma, gamma_to_z


@dataclass
class MatchingNetwork:
    """A synthesized matching network."""
    topology: str           # "L_shunt_series", "L_series_shunt", "Pi", "T"
    components: list[dict]  # [{"type": "L"|"C", "value": float, "position": "series"|"shunt"}]
    freq_hz: float          # Design frequency
    z_source: complex       # Source impedance
    z_load: complex         # Load impedance (antenna)
    notes: str = ""

    def describe(self) -> str:
        """Human-readable description."""
        parts = []
        for c in self.components:
            if c["type"] == "L":
                val_nh = c["value"] * 1e9
                parts.append(f"{c['position']} L = {val_nh:.2f} nH")
            else:
                val_pf = c["value"] * 1e12
                parts.append(f"{c['position']} C = {val_pf:.2f} pF")
        return f"{self.topology}: " + " → ".join(parts)


class ImpedanceAnalyzer:
    """Impedance analysis and matching network synthesis."""

    def __init__(self, z0: float = Z0_DEFAULT):
        self.z0 = z0

    # ─── Smith chart data ────────────────────────────────────────────────

    def to_smith(self, z: np.ndarray | complex) -> np.ndarray | complex:
        """Convert impedance to reflection coefficient (Smith chart coordinates)."""
        return (z - self.z0) / (z + self.z0)

    def from_smith(self, gamma: np.ndarray | complex) -> np.ndarray | complex:
        """Convert reflection coefficient to impedance."""
        return self.z0 * (1 + gamma) / (1 - gamma)

    def smith_chart_data(self, z_array: np.ndarray) -> dict:
        """
        Generate Smith chart plot data.

        Returns dict with real/imag parts of reflection coefficient,
        plus constant-R and constant-X circles for the chart grid.
        """
        gamma = self.to_smith(z_array)
        return {
            "gamma_real": np.real(gamma),
            "gamma_imag": np.imag(gamma),
            "gamma_mag": np.abs(gamma),
            "gamma_phase_deg": np.degrees(np.angle(gamma)),
        }

    # ─── Q factor extraction ────────────────────────────────────────────

    def extract_q(self, frequencies_hz: np.ndarray, z_array: np.ndarray) -> dict:
        """
        Extract loaded Q factor from impedance data near resonance.

        Q = f₀ / Δf₃dB, or equivalently from the rate of reactance change.
        """
        # Find resonance (where X crosses zero)
        X = np.imag(z_array)
        R = np.real(z_array)

        # Zero-crossing of reactance
        sign_changes = np.where(np.diff(np.sign(X)))[0]
        if len(sign_changes) == 0:
            # No resonance found, estimate from minimum |X|
            idx = np.argmin(np.abs(X))
        else:
            idx = sign_changes[0]

        f0 = frequencies_hz[idx]
        R0 = R[idx]

        # Q from reactance slope: Q = (f₀/2R₀) · |dX/df|
        if idx > 0 and idx < len(frequencies_hz) - 1:
            dX = X[idx + 1] - X[idx - 1]
            df = frequencies_hz[idx + 1] - frequencies_hz[idx - 1]
            dX_df = abs(dX / df)
            Q = f0 * dX_df / (2 * max(R0, 0.1))
        else:
            Q = 10.0  # fallback

        return {
            "f0_hz": f0,
            "f0_mhz": f0 / 1e6,
            "R_at_resonance": R0,
            "Q_loaded": Q,
            "estimated_bandwidth_hz": f0 / max(Q, 1),
        }

    # ─── Matching Network Synthesis ──────────────────────────────────────

    def synthesize_l_match(self, z_load: complex, freq_hz: float,
                            z_source: float | None = None) -> list[MatchingNetwork]:
        """
        Synthesize L-section matching networks.

        Returns up to 2 solutions (high-pass and low-pass variants).
        """
        z_s = z_source if z_source else self.z0
        R_s = z_s if isinstance(z_s, (int, float)) else z_s.real
        R_L = z_load.real
        X_L = z_load.imag
        omega = 2 * math.pi * freq_hz

        solutions = []

        if R_L <= 0:
            return solutions

        # Case 1: R_L > R_s (need to transform down)
        # Case 2: R_L < R_s (need to transform up)

        Q_req_sq = R_L / R_s - 1 if R_L > R_s else R_s / R_L - 1
        if Q_req_sq < 0:
            Q_req_sq = 0
        Q_req = math.sqrt(Q_req_sq)

        if R_L > R_s:
            # Shunt element at load side, series element at source side
            # Two solutions: one with shunt C / series L, one with shunt L / series C

            # Solution 1: Shunt C at load, Series L
            B_shunt = Q_req / R_L
            X_series = Q_req * R_s - X_L

            C_shunt = B_shunt / omega if B_shunt > 0 else None
            L_series = X_series / omega if X_series > 0 else None

            if C_shunt and C_shunt > 0 and L_series and L_series > 0:
                solutions.append(MatchingNetwork(
                    topology="L_shunt_C_series_L",
                    components=[
                        {"type": "C", "value": C_shunt, "position": "shunt"},
                        {"type": "L", "value": L_series, "position": "series"},
                    ],
                    freq_hz=freq_hz,
                    z_source=complex(R_s, 0),
                    z_load=z_load,
                    notes="Low-pass L-match (shunt C, series L)",
                ))

            # Solution 2: Shunt L at load, Series C
            B_shunt = -Q_req / R_L
            X_series = -(Q_req * R_s + X_L)

            L_shunt = -1.0 / (B_shunt * omega) if B_shunt < 0 else None
            C_series = -1.0 / (X_series * omega) if X_series < 0 else None

            if L_shunt and L_shunt > 0 and C_series and C_series > 0:
                solutions.append(MatchingNetwork(
                    topology="L_shunt_L_series_C",
                    components=[
                        {"type": "L", "value": L_shunt, "position": "shunt"},
                        {"type": "C", "value": C_series, "position": "series"},
                    ],
                    freq_hz=freq_hz,
                    z_source=complex(R_s, 0),
                    z_load=z_load,
                    notes="High-pass L-match (shunt L, series C)",
                ))
        else:
            # Series element at load side, shunt element at source side
            X_series = Q_req * R_L - X_L
            B_shunt = Q_req / R_s

            L_series = X_series / omega if X_series > 0 else None
            C_shunt = B_shunt / omega if B_shunt > 0 else None

            if L_series and L_series > 0 and C_shunt and C_shunt > 0:
                solutions.append(MatchingNetwork(
                    topology="L_series_L_shunt_C",
                    components=[
                        {"type": "L", "value": L_series, "position": "series"},
                        {"type": "C", "value": C_shunt, "position": "shunt"},
                    ],
                    freq_hz=freq_hz,
                    z_source=complex(R_s, 0),
                    z_load=z_load,
                ))

        return solutions

    def compensate_reactance(self, z_load: complex, freq_hz: float) -> dict:
        """
        Determine a single series or shunt component to cancel reactance.

        Returns the component needed to make the antenna impedance purely real.
        """
        X = z_load.imag
        omega = 2 * math.pi * freq_hz

        if abs(X) < 0.1:
            return {"needed": False, "note": "Impedance already ~real"}

        if X > 0:
            # Inductive: need series capacitor to cancel
            C = 1.0 / (omega * X)
            return {
                "needed": True,
                "component": "C",
                "position": "series",
                "value_F": C,
                "value_pF": C * 1e12,
                "note": f"Series C = {C*1e12:.2f} pF to cancel +j{X:.1f} Ω",
            }
        else:
            # Capacitive: need series inductor
            L = abs(X) / omega
            return {
                "needed": True,
                "component": "L",
                "position": "series",
                "value_H": L,
                "value_nH": L * 1e9,
                "note": f"Series L = {L*1e9:.2f} nH to cancel -j{abs(X):.1f} Ω",
            }
