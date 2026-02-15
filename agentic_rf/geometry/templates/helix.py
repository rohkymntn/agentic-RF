"""
Helical Antenna template.

Helical antennas operate in two primary modes:
  - Normal mode (small helix, D << λ): omnidirectional, linearly polarized
  - Axial mode (C ≈ λ): directional, circularly polarized, high gain

Key features:
  - Normal mode: excellent for implants and compact devices
  - Axial mode: high gain, CP, good for satellite/links
  - 3D structure naturally provides volume → better bandwidth than planar
"""

from __future__ import annotations
import math

import numpy as np

from ..base import (
    AntennaTemplate, DesignParameter, AntennaGeometry,
    Point3D, TraceSegment, Port,
)
from ...utils.constants import C_0, MU_0, ETA_0
from ...utils.materials import COPPER


class HelixAntennaTemplate(AntennaTemplate):

    @property
    def name(self) -> str:
        return "Helical Antenna"

    @property
    def description(self) -> str:
        return (
            "Helical antenna supporting normal mode (omnidirectional, small helix) "
            "and axial mode (directional, CP, circumference ≈ λ). Normal mode helices "
            "are excellent for implant and miniature antennas. Axial mode provides "
            "high gain with circular polarization."
        )

    @property
    def parameters(self) -> list[DesignParameter]:
        return [
            DesignParameter("radius_mm", 1.0, 50.0, 10.0, "mm",
                            "Helix radius"),
            DesignParameter("pitch_mm", 1.0, 50.0, 10.0, "mm",
                            "Turn-to-turn pitch (spacing)"),
            DesignParameter("num_turns", 1, 30, 5, "",
                            "Number of turns", is_integer=True),
            DesignParameter("wire_diameter_mm", 0.1, 3.0, 0.5, "mm",
                            "Wire diameter"),
            DesignParameter("ground_radius_mm", 5.0, 100.0, 30.0, "mm",
                            "Ground plane radius (0 = no ground)"),
        ]

    def _is_axial_mode(self, params: dict[str, float], freq_hz: float) -> bool:
        """Check if helix is in axial mode (C ≈ λ)."""
        r = params["radius_mm"] * 1e-3
        C = 2 * math.pi * r
        lam = C_0 / freq_hz
        return 0.75 < C / lam < 1.33

    def estimate_resonance_hz(self, params: dict[str, float]) -> float:
        """
        Estimate resonance frequency.

        Normal mode: total wire length ≈ λ/4 or λ/2
        Axial mode: circumference ≈ λ
        """
        p = self.validate_params(params)
        r = p["radius_mm"] * 1e-3
        pitch = p["pitch_mm"] * 1e-3
        n = int(p["num_turns"])

        C = 2 * math.pi * r
        wire_per_turn = math.sqrt(C**2 + pitch**2)
        total_wire = wire_per_turn * n

        # For normal mode (small helix): resonate when total length ≈ λ/4
        # For a helix over ground, this is the dominant mode
        f_normal = C_0 / (4.0 * total_wire)

        # For axial mode: C ≈ λ
        f_axial = C_0 / C

        # Return whichever is more likely given the geometry
        # If helix is short (few turns, small), normal mode
        # If helix is long with large circumference, axial mode
        if n <= 3 and C < 0.05:  # small helix
            return f_normal
        return f_axial

    def estimate_impedance(self, params: dict[str, float], freq_hz: float) -> complex:
        """Helix impedance estimate."""
        p = self.validate_params(params)
        r = p["radius_mm"] * 1e-3
        pitch = p["pitch_mm"] * 1e-3
        n = int(p["num_turns"])
        f0 = self.estimate_resonance_hz(p)

        C_circ = 2 * math.pi * r
        lam = C_0 / freq_hz

        if self._is_axial_mode(p, freq_hz):
            # Axial mode input impedance (Kraus): R ≈ 140·(C/λ)
            R = 140.0 * (C_circ / lam)
            alpha = math.atan(pitch / C_circ)
            Q = 10.0
        else:
            # Normal mode: small helix over ground
            # Radiation resistance of short helix
            height = n * pitch
            area = math.pi * r**2
            R_rad = 20.0 * (math.pi**2) * n**2 * (area / lam**2)**2
            R_loss = 2.0
            R = R_rad + R_loss
            Q = 30.0

        delta_f = (freq_hz - f0) / f0
        X = R * Q * 2.0 * delta_f

        return complex(max(R, 0.5), X)

    def estimate_gain_dbi(self, params: dict[str, float], freq_hz: float) -> float:
        """Estimate peak gain in dBi."""
        p = self.validate_params(params)
        r = p["radius_mm"] * 1e-3
        pitch = p["pitch_mm"] * 1e-3
        n = int(p["num_turns"])

        C = 2 * math.pi * r
        lam = C_0 / freq_hz
        S = pitch

        if self._is_axial_mode(p, freq_hz):
            # Kraus formula: D ≈ 15·N·(C/λ)²·(S/λ)
            D = 15.0 * n * (C / lam)**2 * (S / lam)
            return 10.0 * math.log10(max(D, 1.0))
        else:
            # Normal mode: like a short dipole, D ≈ 1.5
            return 1.76  # dBi

    def generate_geometry(self, params: dict[str, float]) -> AntennaGeometry:
        """Generate helical antenna as segmented wire."""
        p = self.validate_params(params)
        mm = 1e-3

        r = p["radius_mm"] * mm
        pitch = p["pitch_mm"] * mm
        n = int(p["num_turns"])
        wire_d = p["wire_diameter_mm"] * mm
        gr = p["ground_radius_mm"] * mm

        geom = AntennaGeometry(template_name="Helix", parameters=p)

        # Generate helix as segments
        segs_per_turn = 24
        total_segs = segs_per_turn * n

        for i in range(total_segs):
            t1 = i / segs_per_turn
            t2 = (i + 1) / segs_per_turn

            theta1 = 2 * math.pi * t1
            theta2 = 2 * math.pi * t2

            x1 = r * math.cos(theta1)
            y1 = r * math.sin(theta1)
            z1 = t1 * pitch

            x2 = r * math.cos(theta2)
            y2 = r * math.sin(theta2)
            z2 = t2 * pitch

            geom.traces.append(TraceSegment(
                start=Point3D(x1, y1, z1),
                end=Point3D(x2, y2, z2),
                width=wire_d,
                thickness=wire_d,
            ))

        # Feed: from ground (z=0) to start of helix
        geom.traces.append(TraceSegment(
            start=Point3D(r, 0, 0),
            end=Point3D(r, 0, 0),
            width=wire_d,
            thickness=wire_d,
        ))

        geom.ports.append(Port(
            start=Point3D(r, 0, -0.001),
            end=Point3D(r, 0, 0),
            impedance=50.0,
        ))

        geom.compute_bounding_box()
        return geom
