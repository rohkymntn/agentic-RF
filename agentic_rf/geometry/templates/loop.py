"""
Loop Antenna template.

Loop antennas are fundamental for near-field wireless power transfer
and implant communication. Small loops (circumference << λ) are
current-mode antennas with well-understood analytical models.

Key features:
  - Excellent for inductive coupling / WPT
  - Well-defined magnetic dipole behavior when electrically small
  - Self-resonant with added capacitance
  - Common for MICS/MedRadio implant links
"""

from __future__ import annotations
import math

import numpy as np

from ..base import (
    AntennaTemplate, DesignParameter, AntennaGeometry,
    Point3D, TraceSegment, Box3D, Port,
)
from ...utils.constants import C_0, MU_0, EPS_0
from ...utils.materials import COPPER


class LoopAntennaTemplate(AntennaTemplate):

    @property
    def name(self) -> str:
        return "Loop Antenna"

    @property
    def description(self) -> str:
        return (
            "Circular or rectangular loop antenna. For electrically small loops, "
            "the radiation resistance is R_rad = 31171·(A/λ²)² Ω. Multi-turn loops "
            "increase inductance and coupling for WPT. Widely used in implant antennas, "
            "NFC, RFID, and wireless power transfer."
        )

    @property
    def parameters(self) -> list[DesignParameter]:
        return [
            DesignParameter("radius_mm", 1.0, 100.0, 15.0, "mm",
                            "Loop radius (for circular loop)"),
            DesignParameter("num_turns", 1, 20, 1, "",
                            "Number of turns", is_integer=True),
            DesignParameter("trace_width_mm", 0.2, 5.0, 1.0, "mm",
                            "Trace/wire width"),
            DesignParameter("turn_spacing_mm", 0.2, 5.0, 0.5, "mm",
                            "Spacing between turns (multi-turn)"),
            DesignParameter("feed_gap_mm", 0.5, 5.0, 1.0, "mm",
                            "Feed gap (where port connects)"),
            DesignParameter("substrate_er", 1.0, 12.0, 1.0, "",
                            "Substrate permittivity (1=free space)"),
            DesignParameter("height_mm", 0.0, 50.0, 0.0, "mm",
                            "Height above ground (0 = no ground)"),
        ]

    def _loop_inductance(self, radius_m: float, n_turns: int,
                          wire_radius_m: float) -> float:
        """
        Inductance of a circular loop (Neumann formula).

        For a single turn: L = μ₀·a·(ln(8a/b) - 2)
        For N turns: L_total ≈ N²·L_single (tightly wound)
        """
        a = radius_m
        b = max(wire_radius_m, 1e-6)
        L_single = MU_0 * a * (math.log(8.0 * a / b) - 2.0)
        # Multi-turn: Neumann gives N² scaling with mutual coupling correction
        if n_turns <= 1:
            return L_single
        # Rough correction for spacing
        return n_turns ** 2 * L_single * 0.85  # 0.85 accounts for imperfect coupling

    def _loop_capacitance(self, radius_m: float, n_turns: int,
                           trace_width_m: float, gap_m: float, er: float) -> float:
        """Estimated distributed + gap capacitance."""
        # Distributed capacitance of trace
        circumference = 2 * math.pi * radius_m * n_turns
        C_dist = EPS_0 * er * trace_width_m / max(gap_m, 1e-6) * 1e-2
        # Feed gap capacitance
        C_gap = EPS_0 * er * trace_width_m * trace_width_m / max(gap_m, 1e-6)
        return C_dist + C_gap

    def estimate_resonance_hz(self, params: dict[str, float]) -> float:
        """
        Self-resonance of the loop: f = 1 / (2π√(LC)).

        For small loops without added capacitance, this is typically
        much higher than operating frequency.
        """
        p = self.validate_params(params)
        r = p["radius_mm"] * 1e-3
        n = int(p["num_turns"])
        tw = p["trace_width_mm"] * 1e-3
        gap = p["feed_gap_mm"] * 1e-3
        er = p["substrate_er"]

        L = self._loop_inductance(r, n, tw / 2)
        C = self._loop_capacitance(r, n, tw, gap, er)

        if L <= 0 or C <= 0:
            return 1e9
        return 1.0 / (2.0 * math.pi * math.sqrt(L * C))

    def estimate_impedance(self, params: dict[str, float], freq_hz: float) -> complex:
        """
        Input impedance of a loop antenna.

        Z = R_loss + R_rad + jωL - j/(ωC)
        """
        p = self.validate_params(params)
        r = p["radius_mm"] * 1e-3
        n = int(p["num_turns"])
        tw = p["trace_width_mm"] * 1e-3
        gap = p["feed_gap_mm"] * 1e-3
        er = p["substrate_er"]

        omega = 2.0 * math.pi * freq_hz
        lam = C_0 / freq_hz
        area = math.pi * r ** 2

        L = self._loop_inductance(r, n, tw / 2)

        # Radiation resistance: R_rad = 31171 · N² · (A/λ²)²
        R_rad = 31171.0 * n**2 * (area / lam**2)**2

        # Loss resistance (conductor loss)
        from ...utils.constants import skin_depth
        delta = skin_depth(freq_hz, 5.8e7)  # copper
        circumference = 2 * math.pi * r * n
        R_loss = circumference / (tw * 5.8e7 * delta)

        # Total resistance
        R = R_rad + R_loss

        # Reactance: predominantly inductive for small loops
        X_L = omega * L
        C = self._loop_capacitance(r, n, tw, gap, er)
        X_C = 1.0 / (omega * C) if C > 0 else 0

        return complex(R, X_L - X_C)

    def coupling_coefficient(self, params: dict[str, float],
                              other_params: dict[str, float],
                              distance_m: float) -> float:
        """
        Estimate coupling coefficient k between two coaxial loops.

        k = M / √(L₁·L₂), where M depends on geometry and separation.
        For coaxial loops: M = μ₀·π·a₁²·a₂²·N₁·N₂ / (2·(a²+d²)^(3/2))
        where a = max(a1, a2) and d = distance.
        """
        p1 = self.validate_params(params)
        p2 = self.validate_params(other_params)

        r1 = p1["radius_mm"] * 1e-3
        r2 = p2["radius_mm"] * 1e-3
        n1 = int(p1["num_turns"])
        n2 = int(p2["num_turns"])
        tw1 = p1["trace_width_mm"] * 1e-3
        tw2 = p2["trace_width_mm"] * 1e-3

        # Mutual inductance (coaxial, approximate)
        a_eff = max(r1, r2)
        M = (MU_0 * math.pi * r1**2 * r2**2 * n1 * n2 /
             (2.0 * (a_eff**2 + distance_m**2) ** 1.5))

        L1 = self._loop_inductance(r1, n1, tw1 / 2)
        L2 = self._loop_inductance(r2, n2, tw2 / 2)

        k = M / math.sqrt(L1 * L2)
        return min(k, 1.0)

    def generate_geometry(self, params: dict[str, float]) -> AntennaGeometry:
        """Generate loop geometry as segmented circle."""
        p = self.validate_params(params)
        mm = 1e-3
        t = 35e-6

        r = p["radius_mm"] * mm
        n = int(p["num_turns"])
        tw = p["trace_width_mm"] * mm
        ts = p["turn_spacing_mm"] * mm
        gap = p["feed_gap_mm"] * mm
        h = p["height_mm"] * mm

        geom = AntennaGeometry(template_name="Loop", parameters=p)

        # Generate circular loop as line segments
        n_segs = 36  # segments per turn
        gap_angle = gap / r if r > 0 else 0.1  # angle subtended by feed gap

        for turn in range(n):
            r_turn = r + turn * ts
            for i in range(n_segs):
                theta1 = 2 * math.pi * i / n_segs + gap_angle / 2
                theta2 = 2 * math.pi * (i + 1) / n_segs + gap_angle / 2

                if theta2 > 2 * math.pi - gap_angle / 2 and turn == 0 and i == n_segs - 1:
                    break  # leave gap for feed

                x1, y1 = r_turn * math.cos(theta1), r_turn * math.sin(theta1)
                x2, y2 = r_turn * math.cos(theta2), r_turn * math.sin(theta2)

                geom.traces.append(TraceSegment(
                    start=Point3D(x1, y1, h),
                    end=Point3D(x2, y2, h),
                    width=tw,
                    thickness=t,
                ))

        # Port across feed gap
        theta_start = gap_angle / 2
        theta_end = 2 * math.pi - gap_angle / 2
        geom.ports.append(Port(
            start=Point3D(r * math.cos(theta_end), r * math.sin(theta_end), h),
            end=Point3D(r * math.cos(theta_start), r * math.sin(theta_start), h),
            impedance=50.0,
        ))

        geom.compute_bounding_box()
        return geom
