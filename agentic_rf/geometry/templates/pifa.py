"""
Planar Inverted-F Antenna (PIFA) template.

The PIFA is the workhorse of mobile antennas. A conductive patch is
suspended above a ground plane, shorted at one edge and fed via a pin.

Key features:
  - Resonance when L + W ≈ λ/4
  - Very low profile (height ≈ λ/40 typical)
  - Impedance controlled by feed-to-short distance
  - Used in phones, laptops, wearables
"""

from __future__ import annotations
import math

from ..base import (
    AntennaTemplate, DesignParameter, AntennaGeometry,
    Point3D, TraceSegment, Box3D, Port,
)
from ...utils.constants import C_0
from ...utils.materials import COPPER, FR4, Material


class PIFATemplate(AntennaTemplate):

    @property
    def name(self) -> str:
        return "Planar Inverted-F Antenna (PIFA)"

    @property
    def description(self) -> str:
        return (
            "Low-profile patch antenna with shorting wall/pin to ground. "
            "Resonance set by L+W ≈ λ/4. Feed position tunes impedance. "
            "Widely used in mobile, wearable, and IoT devices."
        )

    @property
    def parameters(self) -> list[DesignParameter]:
        return [
            DesignParameter("patch_length_mm", 5.0, 100.0, 30.0, "mm",
                            "Patch length along resonant direction"),
            DesignParameter("patch_width_mm", 5.0, 80.0, 20.0, "mm",
                            "Patch width"),
            DesignParameter("height_mm", 1.0, 15.0, 5.0, "mm",
                            "Height above ground"),
            DesignParameter("feed_x_mm", 1.0, 50.0, 10.0, "mm",
                            "Feed position along length from short edge"),
            DesignParameter("feed_y_mm", 1.0, 40.0, 10.0, "mm",
                            "Feed position along width"),
            DesignParameter("short_width_mm", 1.0, 20.0, 5.0, "mm",
                            "Shorting wall/strip width"),
            DesignParameter("ground_length_mm", 20.0, 150.0, 60.0, "mm",
                            "Ground plane length"),
            DesignParameter("ground_width_mm", 20.0, 100.0, 40.0, "mm",
                            "Ground plane width"),
            DesignParameter("substrate_er", 1.0, 12.0, 1.0, "",
                            "Fill material permittivity (1=air)"),
        ]

    def estimate_resonance_hz(self, params: dict[str, float]) -> float:
        """
        PIFA resonance: f ≈ C₀ / (4 · (L + W) · √εr_eff)

        The shorting wall makes the effective length ≈ λ/4, and the
        resonant path wraps around L + W.
        """
        p = self.validate_params(params)
        L = p["patch_length_mm"] * 1e-3
        W = p["patch_width_mm"] * 1e-3
        er = p["substrate_er"]

        # Effective permittivity (simplified for suspended patch)
        h = p["height_mm"] * 1e-3
        er_eff = (er + 1) / 2  # rough for air-filled PIFA

        return C_0 / (4.0 * (L + W) * math.sqrt(er_eff))

    def estimate_impedance(self, params: dict[str, float], freq_hz: float) -> complex:
        """PIFA impedance estimate based on cavity model."""
        p = self.validate_params(params)
        f0 = self.estimate_resonance_hz(p)

        L = p["patch_length_mm"] * 1e-3
        W = p["patch_width_mm"] * 1e-3
        h = p["height_mm"] * 1e-3
        fx = p["feed_x_mm"] * 1e-3

        # Edge impedance of patch (simplified cavity model)
        lam = C_0 / freq_hz
        G_slot = (W * h) / (120.0 * lam)  # slot conductance
        R_edge = 1.0 / (2.0 * G_slot) if G_slot > 0 else 300.0
        R_edge = min(R_edge, 500.0)

        # Impedance tapping: Z ∝ sin²(π·x_feed/L)
        tap_ratio = math.sin(math.pi * fx / L) ** 2
        R_in = R_edge * tap_ratio

        # Reactance near resonance
        Q = 15.0 + 50.0 * (h / lam)  # Q decreases with height
        Q = max(Q, 5.0)
        delta_f = (freq_hz - f0) / f0
        X = R_in * Q * 2.0 * delta_f

        return complex(max(R_in, 1.0), X)

    def generate_geometry(self, params: dict[str, float]) -> AntennaGeometry:
        """Generate PIFA geometry."""
        p = self.validate_params(params)
        mm = 1e-3
        t = 35e-6  # metal thickness

        pl = p["patch_length_mm"] * mm
        pw = p["patch_width_mm"] * mm
        h = p["height_mm"] * mm
        fx = p["feed_x_mm"] * mm
        fy = p["feed_y_mm"] * mm
        sw = p["short_width_mm"] * mm
        gl = p["ground_length_mm"] * mm
        gw = p["ground_width_mm"] * mm
        er = p["substrate_er"]

        geom = AntennaGeometry(template_name="PIFA", parameters=p)

        # Ground plane
        geom.boxes.append(Box3D(
            origin=Point3D(0, 0, 0),
            size=Point3D(gl, gw, t),
            material=COPPER,
        ))

        # Patch (top plate)
        patch_x0 = 0  # patch starts at x=0 (short edge)
        patch_y0 = (gw - pw) / 2  # centered in y
        geom.boxes.append(Box3D(
            origin=Point3D(patch_x0, patch_y0, h),
            size=Point3D(pl, pw, t),
            material=COPPER,
        ))

        # Shorting wall
        geom.boxes.append(Box3D(
            origin=Point3D(0, patch_y0 + (pw - sw) / 2, t),
            size=Point3D(t, sw, h - t),
            material=COPPER,
        ))

        # Fill material (if not air)
        if er > 1.05:
            fill_mat = Material(f"Fill_er{er:.1f}", eps_r=er, tan_d=0.005)
            geom.boxes.append(Box3D(
                origin=Point3D(0, patch_y0, t),
                size=Point3D(pl, pw, h - t),
                material=fill_mat,
            ))

        # Feed pin
        feed_pt_bottom = Point3D(fx, patch_y0 + fy, t)
        feed_pt_top = Point3D(fx, patch_y0 + fy, h)
        geom.traces.append(TraceSegment(
            start=feed_pt_bottom,
            end=feed_pt_top,
            width=1.0 * mm,
            thickness=t,
        ))

        # Port
        geom.ports.append(Port(
            start=feed_pt_bottom,
            end=feed_pt_top,
            impedance=50.0,
        ))

        geom.compute_bounding_box()
        return geom
