"""
Planar Spiral Antenna template.

Planar spirals are critical for:
  - Wireless power transfer (inductive coupling coils)
  - Broadband antennas (Archimedean/equiangular spirals)
  - Implant antennas (miniaturized spirals in high-εr substrates)

Key features:
  - Multi-turn spirals provide high inductance for WPT
  - Archimedean spiral: broadband, CP radiation
  - Can be rectangular or circular
  - Mutual inductance for WPT calculable from geometry
"""

from __future__ import annotations
import math

import numpy as np

from ..base import (
    AntennaTemplate, DesignParameter, AntennaGeometry,
    Point3D, TraceSegment, Box3D, Port,
)
from ...utils.constants import C_0, MU_0, EPS_0
from ...utils.materials import COPPER, Material


class SpiralAntennaTemplate(AntennaTemplate):

    @property
    def name(self) -> str:
        return "Planar Spiral Antenna"

    @property
    def description(self) -> str:
        return (
            "Planar spiral antenna / coil. Archimedean spiral for broadband radiation "
            "or multi-turn coil for wireless power transfer. Inner and outer radii, "
            "number of turns, and trace dimensions control performance. "
            "Excellent for WPT Tx/Rx coils, implant antennas, and broadband sensors."
        )

    @property
    def parameters(self) -> list[DesignParameter]:
        return [
            DesignParameter("inner_radius_mm", 0.5, 50.0, 3.0, "mm",
                            "Inner radius of spiral"),
            DesignParameter("outer_radius_mm", 5.0, 150.0, 25.0, "mm",
                            "Outer radius of spiral"),
            DesignParameter("num_turns", 1, 50, 5, "",
                            "Number of spiral turns", is_integer=True),
            DesignParameter("trace_width_mm", 0.15, 5.0, 1.0, "mm",
                            "Trace width"),
            DesignParameter("trace_gap_mm", 0.15, 5.0, 0.5, "mm",
                            "Gap between turns"),
            DesignParameter("substrate_er", 1.0, 12.0, 4.4, "",
                            "Substrate permittivity"),
            DesignParameter("substrate_height_mm", 0.2, 5.0, 1.6, "mm",
                            "Substrate thickness"),
        ]

    def _spiral_inductance(self, params: dict[str, float]) -> float:
        """
        Inductance of planar spiral (Wheeler / modified Wheeler).

        L = μ₀·N²·d_avg·c₁ / (1 + c₂·ρ)
        where ρ = (d_out - d_in) / (d_out + d_in) is the fill ratio.
        """
        p = params
        r_in = p["inner_radius_mm"] * 1e-3
        r_out = p["outer_radius_mm"] * 1e-3
        n = int(p["num_turns"])

        d_avg = (r_in + r_out)  # average diameter (using radius)
        rho = (r_out - r_in) / max(r_out + r_in, 1e-6)

        # Modified Wheeler coefficients (for circular spiral)
        c1 = 2.46
        c2 = 3.6

        L = MU_0 * n**2 * d_avg * c1 / (1 + c2 * rho)
        return max(L, 1e-12)

    def _spiral_capacitance(self, params: dict[str, float]) -> float:
        """Distributed capacitance between turns."""
        p = params
        r_in = p["inner_radius_mm"] * 1e-3
        r_out = p["outer_radius_mm"] * 1e-3
        n = int(p["num_turns"])
        tw = p["trace_width_mm"] * 1e-3
        tg = p["trace_gap_mm"] * 1e-3
        er = p["substrate_er"]
        sub_h = p["substrate_height_mm"] * 1e-3

        # Approximate: capacitance between adjacent turns
        avg_circumference = math.pi * (r_in + r_out)
        C_turn = EPS_0 * er * avg_circumference * tw / max(tg, 1e-6)

        # Total distributed capacitance (series combination)
        if n > 1:
            return C_turn / (n - 1)
        return C_turn * 0.1  # small residual

    def _spiral_resistance(self, params: dict[str, float], freq_hz: float) -> float:
        """DC + AC resistance of spiral trace."""
        p = params
        r_in = p["inner_radius_mm"] * 1e-3
        r_out = p["outer_radius_mm"] * 1e-3
        n = int(p["num_turns"])
        tw = p["trace_width_mm"] * 1e-3

        # Total trace length
        total_length = 0
        for i in range(n):
            r = r_in + (r_out - r_in) * (i + 0.5) / n
            total_length += 2 * math.pi * r

        # AC resistance with skin effect
        from ...utils.constants import skin_depth
        delta = skin_depth(freq_hz, 5.8e7)  # copper
        t_eff = min(delta, 35e-6)  # effective thickness

        sigma_cu = 5.8e7
        R_dc = total_length / (sigma_cu * tw * 35e-6)
        R_ac = total_length / (sigma_cu * tw * t_eff)

        return R_ac

    def estimate_resonance_hz(self, params: dict[str, float]) -> float:
        """Self-resonant frequency: f = 1/(2π√LC)."""
        p = self.validate_params(params)
        L = self._spiral_inductance(p)
        C = self._spiral_capacitance(p)
        return 1.0 / (2 * math.pi * math.sqrt(max(L * C, 1e-30)))

    def estimate_impedance(self, params: dict[str, float], freq_hz: float) -> complex:
        """Spiral impedance: Z = R + j(ωL - 1/ωC)."""
        p = self.validate_params(params)
        omega = 2 * math.pi * freq_hz

        L = self._spiral_inductance(p)
        C = self._spiral_capacitance(p)
        R = self._spiral_resistance(p, freq_hz)

        X_L = omega * L
        X_C = 1.0 / (omega * C) if C > 0 else 0

        return complex(R, X_L - X_C)

    def estimate_inductance_nh(self, params: dict[str, float]) -> float:
        """Return inductance in nH."""
        p = self.validate_params(params)
        return self._spiral_inductance(p) * 1e9

    def estimate_q_factor(self, params: dict[str, float], freq_hz: float) -> float:
        """Quality factor Q = ωL/R."""
        p = self.validate_params(params)
        L = self._spiral_inductance(p)
        R = self._spiral_resistance(p, freq_hz)
        omega = 2 * math.pi * freq_hz
        return omega * L / max(R, 1e-6)

    def mutual_inductance(self, params1: dict[str, float],
                          params2: dict[str, float],
                          distance_m: float) -> float:
        """
        Estimate mutual inductance between two coaxial spirals.

        Uses Neumann formula approximation for coaxial circular coils.
        M = μ₀·π·N₁·N₂·r₁²·r₂² / (2·(r_max² + d²)^(3/2))
        """
        p1 = self.validate_params(params1)
        p2 = self.validate_params(params2)

        r1 = (p1["inner_radius_mm"] + p1["outer_radius_mm"]) / 2 * 1e-3
        r2 = (p2["inner_radius_mm"] + p2["outer_radius_mm"]) / 2 * 1e-3
        n1 = int(p1["num_turns"])
        n2 = int(p2["num_turns"])

        r_max = max(r1, r2)
        M = (MU_0 * math.pi * n1 * n2 * r1**2 * r2**2 /
             (2.0 * (r_max**2 + distance_m**2) ** 1.5))
        return M

    def coupling_coefficient(self, params1: dict[str, float],
                              params2: dict[str, float],
                              distance_m: float) -> float:
        """k = M / √(L₁·L₂)."""
        M = self.mutual_inductance(params1, params2, distance_m)
        L1 = self._spiral_inductance(self.validate_params(params1))
        L2 = self._spiral_inductance(self.validate_params(params2))
        k = M / math.sqrt(max(L1 * L2, 1e-30))
        return min(k, 1.0)

    def wpt_link_efficiency(self, params_tx: dict[str, float],
                             params_rx: dict[str, float],
                             distance_m: float,
                             freq_hz: float) -> float:
        """
        Estimate wireless power transfer link efficiency.

        η = (k²·Q₁·Q₂) / (1 + k²·Q₁·Q₂)
        (simplified resonant coupling model)
        """
        k = self.coupling_coefficient(params_tx, params_rx, distance_m)
        Q1 = self.estimate_q_factor(params_tx, freq_hz)
        Q2 = self.estimate_q_factor(params_rx, freq_hz)

        kQ = k**2 * Q1 * Q2
        return kQ / (1.0 + kQ)

    def generate_geometry(self, params: dict[str, float]) -> AntennaGeometry:
        """Generate Archimedean spiral geometry."""
        p = self.validate_params(params)
        mm = 1e-3
        t = 35e-6

        r_in = p["inner_radius_mm"] * mm
        r_out = p["outer_radius_mm"] * mm
        n = int(p["num_turns"])
        tw = p["trace_width_mm"] * mm
        sub_h = p["substrate_height_mm"] * mm
        er = p["substrate_er"]

        geom = AntennaGeometry(template_name="Spiral", parameters=p)

        # Substrate
        sub_mat = Material(f"Sub_er{er:.1f}", eps_r=er, tan_d=0.02)
        side = 2 * r_out * 1.2
        geom.boxes.append(Box3D(
            origin=Point3D(-side / 2, -side / 2, 0),
            size=Point3D(side, side, sub_h),
            material=sub_mat,
        ))

        # Spiral trace (Archimedean: r = r_in + (r_out-r_in)·θ/(2π·N))
        segs_per_turn = 36
        total_segs = segs_per_turn * n
        z = sub_h + t

        for i in range(total_segs):
            frac1 = i / total_segs
            frac2 = (i + 1) / total_segs

            theta1 = 2 * math.pi * n * frac1
            theta2 = 2 * math.pi * n * frac2

            r1 = r_in + (r_out - r_in) * frac1
            r2 = r_in + (r_out - r_in) * frac2

            geom.traces.append(TraceSegment(
                start=Point3D(r1 * math.cos(theta1), r1 * math.sin(theta1), z),
                end=Point3D(r2 * math.cos(theta2), r2 * math.sin(theta2), z),
                width=tw,
                thickness=t,
            ))

        # Port at inner radius
        geom.ports.append(Port(
            start=Point3D(r_in, 0, 0),
            end=Point3D(r_in, 0, z),
            impedance=50.0,
        ))

        geom.compute_bounding_box()
        return geom
