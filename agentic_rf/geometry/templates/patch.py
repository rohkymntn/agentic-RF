"""
Microstrip Patch Antenna template.

The rectangular patch is the canonical planar antenna. Well-understood
cavity model provides excellent analytical estimates.

Key features:
  - Resonance: L ≈ λ/(2√εr_eff) for fundamental TM₁₀ mode
  - Directional (broadside radiation)
  - Moderate bandwidth (2-8% typical)
  - Can be miniaturized with high-εr substrates
"""

from __future__ import annotations
import math

from ..base import (
    AntennaTemplate, DesignParameter, AntennaGeometry,
    Point3D, TraceSegment, Box3D, Port,
)
from ...utils.constants import C_0, ETA_0
from ...utils.materials import COPPER, Material


class PatchAntennaTemplate(AntennaTemplate):

    @property
    def name(self) -> str:
        return "Microstrip Patch Antenna"

    @property
    def description(self) -> str:
        return (
            "Rectangular microstrip patch antenna. Resonance at L ≈ λ/(2√εr_eff). "
            "Broadside radiation pattern with ~6-8 dBi gain. Bandwidth is controlled "
            "by substrate height and permittivity. Inset feed controls impedance match."
        )

    @property
    def parameters(self) -> list[DesignParameter]:
        return [
            DesignParameter("patch_length_mm", 5.0, 200.0, 36.0, "mm",
                            "Patch length (resonant dimension)"),
            DesignParameter("patch_width_mm", 5.0, 200.0, 44.0, "mm",
                            "Patch width (controls impedance and BW)"),
            DesignParameter("substrate_height_mm", 0.5, 10.0, 1.6, "mm",
                            "Substrate height"),
            DesignParameter("substrate_er", 1.5, 12.0, 4.4, "",
                            "Substrate relative permittivity"),
            DesignParameter("feed_inset_mm", 0.0, 30.0, 8.0, "mm",
                            "Inset feed depth (for impedance matching)"),
            DesignParameter("feed_width_mm", 0.5, 5.0, 1.5, "mm",
                            "Feed line width"),
            DesignParameter("ground_margin_mm", 5.0, 30.0, 10.0, "mm",
                            "Ground plane margin beyond patch edges"),
        ]

    def _effective_er(self, er: float, h_mm: float, w_mm: float) -> float:
        """Hammerstad–Jensen effective permittivity."""
        ratio = w_mm / max(h_mm, 0.01)
        if ratio >= 1.0:
            return (er + 1) / 2 + (er - 1) / (2 * math.sqrt(1 + 12 / ratio))
        else:
            return (er + 1) / 2 + (er - 1) / 2 * (
                1 / math.sqrt(1 + 12 / ratio) + 0.04 * (1 - ratio) ** 2
            )

    def _fringe_extension(self, er_eff: float, h_mm: float, w_mm: float) -> float:
        """Fringing field extension ΔL (Hammerstad)."""
        ratio = w_mm / max(h_mm, 0.01)
        return 0.412 * h_mm * (
            (er_eff + 0.3) * (ratio + 0.264) /
            ((er_eff - 0.258) * (ratio + 0.8))
        )

    def estimate_resonance_hz(self, params: dict[str, float]) -> float:
        """
        Patch resonance (TM₁₀ mode):

        f = C₀ / (2·L_eff·√εr_eff)
        L_eff = L + 2·ΔL (fringe extension)
        """
        p = self.validate_params(params)
        L = p["patch_length_mm"]
        W = p["patch_width_mm"]
        h = p["substrate_height_mm"]
        er = p["substrate_er"]

        er_eff = self._effective_er(er, h, W)
        dL = self._fringe_extension(er_eff, h, W)
        L_eff = (L + 2 * dL) * 1e-3  # to meters

        return C_0 / (2.0 * L_eff * math.sqrt(er_eff))

    def estimate_impedance(self, params: dict[str, float], freq_hz: float) -> complex:
        """
        Patch impedance using cavity model.

        At resonance edge: Z_edge = 1/(2·G₁) where G₁ = slot conductance.
        Inset feed: Z_in = Z_edge · cos²(π·y₀/L).
        """
        p = self.validate_params(params)
        f0 = self.estimate_resonance_hz(p)

        L = p["patch_length_mm"] * 1e-3
        W = p["patch_width_mm"] * 1e-3
        h = p["substrate_height_mm"] * 1e-3
        er = p["substrate_er"]
        inset = p["feed_inset_mm"] * 1e-3

        lam = C_0 / freq_hz
        k0 = 2 * math.pi / lam

        # Slot conductance (Balanis approximation for W > λ)
        if k0 * W < 2.5:
            G1 = (W / (120.0 * lam)) * (1.0 - (k0 * h)**2 / 24.0)
        else:
            G1 = W / (120.0 * lam)

        G1 = max(G1, 1e-6)

        # Edge impedance
        R_edge = 1.0 / (2.0 * G1)

        # Inset feed tapping
        R_in = R_edge * math.cos(math.pi * inset / L) ** 2

        # Q factor and reactance
        er_eff = self._effective_er(er, p["substrate_height_mm"], p["patch_width_mm"])
        Q = C_0 * math.sqrt(er_eff) / (4.0 * freq_hz * h)
        Q = max(Q, 5.0)

        delta_f = (freq_hz - f0) / f0
        X = R_in * Q * 2.0 * delta_f

        return complex(max(R_in, 1.0), X)

    def estimate_directivity(self, params: dict[str, float], freq_hz: float) -> float:
        """Broadside directivity estimate in dBi."""
        p = self.validate_params(params)
        W = p["patch_width_mm"] * 1e-3
        lam = C_0 / freq_hz

        # Approximate directivity: D ≈ 4·π·W·L / λ² (scaled)
        # Simplified: for typical patches, D ≈ 6-8 dBi
        D_linear = 6.6 * (W / lam)
        D_linear = max(D_linear, 3.0)
        return 10.0 * math.log10(D_linear)

    def generate_geometry(self, params: dict[str, float]) -> AntennaGeometry:
        """Generate patch antenna geometry."""
        p = self.validate_params(params)
        mm = 1e-3
        t = 35e-6

        pl = p["patch_length_mm"] * mm
        pw = p["patch_width_mm"] * mm
        h = p["substrate_height_mm"] * mm
        er = p["substrate_er"]
        inset = p["feed_inset_mm"] * mm
        fw = p["feed_width_mm"] * mm
        margin = p["ground_margin_mm"] * mm

        gl = pl + 2 * margin
        gw = pw + 2 * margin

        geom = AntennaGeometry(template_name="Patch", parameters=p)

        # Ground plane
        geom.boxes.append(Box3D(
            origin=Point3D(0, 0, 0),
            size=Point3D(gl, gw, t),
            material=COPPER,
        ))

        # Substrate
        sub_mat = Material(f"Sub_er{er:.1f}", eps_r=er, tan_d=0.02)
        geom.boxes.append(Box3D(
            origin=Point3D(0, 0, t),
            size=Point3D(gl, gw, h),
            material=sub_mat,
        ))

        # Patch
        px0 = margin
        py0 = margin
        geom.boxes.append(Box3D(
            origin=Point3D(px0, py0, t + h),
            size=Point3D(pl, pw, t),
            material=COPPER,
        ))

        # Feed line (microstrip from edge to inset point)
        feed_y = py0 + pw / 2
        geom.traces.append(TraceSegment(
            start=Point3D(0, feed_y, t + h),
            end=Point3D(px0 + inset, feed_y, t + h),
            width=fw,
            thickness=t,
        ))

        # Port (at substrate edge)
        geom.ports.append(Port(
            start=Point3D(0, feed_y, t),
            end=Point3D(0, feed_y, t + h),
            impedance=50.0,
        ))

        geom.compute_bounding_box()
        return geom
