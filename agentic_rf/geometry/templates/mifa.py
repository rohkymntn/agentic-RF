"""
Meandered Inverted-F Antenna (MIFA) template.

The MIFA is one of the most common antennas for compact wireless devices.
A quarter-wave resonator is folded into meander lines above a ground plane,
fed via a pin and shorted via another pin.

Key features:
  - Very compact (meander folds the λ/4 path)
  - Works well in small PCB form factors
  - Bandwidth limited by height above ground
  - Common in IoT, wearables, medical devices

Parameters:
  - total_meander_length: total conductor path length (≈ λ/4 at resonance)
  - meander_width: width of each meander arm
  - meander_gap: spacing between meander turns
  - height: height above ground plane
  - trace_width: conductor width
  - feed_offset: distance from short to feed pin
  - ground_length: ground plane length
  - ground_width: ground plane width
  - substrate_er: substrate relative permittivity
  - substrate_height: substrate thickness
"""

from __future__ import annotations

import math

import numpy as np

from ..base import (
    AntennaTemplate, DesignParameter, AntennaGeometry,
    Point3D, TraceSegment, Box3D, Port,
)
from ...utils.constants import C_0, freq_to_wavelength
from ...utils.materials import COPPER, FR4, Material


class MIFATemplate(AntennaTemplate):

    @property
    def name(self) -> str:
        return "Meandered Inverted-F Antenna (MIFA)"

    @property
    def description(self) -> str:
        return (
            "Compact quarter-wave antenna using meander folding above a ground plane. "
            "Excellent for IoT, wearable, and implant applications where PCB area is limited. "
            "Resonance is primarily set by total meander path length ≈ λ/4 in the effective medium. "
            "Bandwidth increases with height above ground but degrades with more aggressive meandering."
        )

    @property
    def parameters(self) -> list[DesignParameter]:
        return [
            DesignParameter("total_length_mm", 10.0, 200.0, 50.0, "mm",
                            "Total meander conductor path length"),
            DesignParameter("meander_width_mm", 2.0, 30.0, 8.0, "mm",
                            "Width of each meander arm"),
            DesignParameter("meander_gap_mm", 0.3, 5.0, 1.0, "mm",
                            "Gap between meander turns"),
            DesignParameter("height_mm", 1.0, 20.0, 5.0, "mm",
                            "Height above ground plane"),
            DesignParameter("trace_width_mm", 0.2, 3.0, 0.5, "mm",
                            "Trace width"),
            DesignParameter("feed_offset_mm", 0.5, 10.0, 2.0, "mm",
                            "Distance from short pin to feed pin"),
            DesignParameter("ground_length_mm", 10.0, 100.0, 40.0, "mm",
                            "Ground plane length"),
            DesignParameter("ground_width_mm", 10.0, 60.0, 30.0, "mm",
                            "Ground plane width"),
            DesignParameter("substrate_er", 1.0, 12.0, 4.4, "",
                            "Substrate relative permittivity"),
            DesignParameter("substrate_height_mm", 0.2, 5.0, 1.6, "mm",
                            "Substrate thickness"),
        ]

    def _effective_er(self, er: float, h_mm: float, w_mm: float) -> float:
        """Approximate effective permittivity for microstrip-like geometry."""
        # Simplified Hammerstad–Jensen
        ratio = w_mm / max(h_mm, 0.01)
        if ratio >= 1.0:
            return (er + 1) / 2 + (er - 1) / (2 * math.sqrt(1 + 12 / ratio))
        else:
            return (er + 1) / 2 + (er - 1) / 2 * (
                1 / math.sqrt(1 + 12 / ratio) + 0.04 * (1 - ratio) ** 2
            )

    def estimate_resonance_hz(self, params: dict[str, float]) -> float:
        """
        Resonance when total path ≈ λ_eff / 4.

        f ≈ C₀ / (4 · L_total · √εr_eff) with meander slowdown factor.
        """
        p = self.validate_params(params)
        L = p["total_length_mm"] * 1e-3
        er = p["substrate_er"]
        h = p["substrate_height_mm"]
        w = p["trace_width_mm"]

        er_eff = self._effective_er(er, h, w)

        # Meander slowdown: mutual coupling between adjacent segments
        # reduces the effective electrical length by ~10-20%
        gap = p["meander_gap_mm"]
        meander_factor = 1.0 - 0.1 * math.exp(-gap / 1.0)  # less coupling = less slowdown

        return C_0 / (4.0 * L * math.sqrt(er_eff) * meander_factor)

    def estimate_impedance(self, params: dict[str, float], freq_hz: float) -> complex:
        """
        Analytical impedance estimate for MIFA.

        Near resonance, the IFA behaves as a parallel RLC.
        Feed-to-short spacing controls the impedance tap.
        """
        p = self.validate_params(params)
        f0 = self.estimate_resonance_hz(p)

        # Radiation resistance of a short monopole over ground
        h_m = p["height_mm"] * 1e-3
        lam = C_0 / freq_hz
        # IFA radiation resistance scales with (h/λ)²
        R_rad = 40.0 * (math.pi * h_m / lam) ** 2

        # Loss resistance (conductor + dielectric)
        R_loss = 1.0  # ~1 Ω for copper traces at UHF

        # Total resistance at resonance
        R_total = R_rad + R_loss

        # Impedance tapping ratio: feed_offset / total gives fraction of max Z
        tap = p["feed_offset_mm"] / max(p["total_length_mm"], 1.0)
        R_at_feed = R_total / max(tap ** 2, 0.001)
        R_at_feed = min(R_at_feed, 500.0)  # physical limit

        # Reactance: inductive below resonance, capacitive above
        Q = 20.0  # typical Q for MIFA
        delta_f = (freq_hz - f0) / f0
        X = R_at_feed * Q * 2.0 * delta_f

        return complex(R_at_feed, X)

    def generate_geometry(self, params: dict[str, float]) -> AntennaGeometry:
        """Generate MIFA 3D geometry with meander trace."""
        p = self.validate_params(params)

        mm = 1e-3  # conversion
        total_len = p["total_length_mm"] * mm
        mw = p["meander_width_mm"] * mm
        mg = p["meander_gap_mm"] * mm
        h = p["height_mm"] * mm
        tw = p["trace_width_mm"] * mm
        t_metal = 35e-6  # 1 oz copper
        fo = p["feed_offset_mm"] * mm
        gl = p["ground_length_mm"] * mm
        gw = p["ground_width_mm"] * mm
        sub_h = p["substrate_height_mm"] * mm

        geom = AntennaGeometry(
            template_name="MIFA",
            parameters=p,
        )

        # --- Ground plane ---
        geom.boxes.append(Box3D(
            origin=Point3D(-gl / 2, -gw / 2, 0),
            size=Point3D(gl, gw, t_metal),
            material=COPPER,
        ))

        # --- Substrate ---
        er = p["substrate_er"]
        substrate_mat = Material(f"Substrate_er{er:.1f}", eps_r=er, tan_d=0.02)
        geom.boxes.append(Box3D(
            origin=Point3D(-gl / 2, -gw / 2, t_metal),
            size=Point3D(gl, gw, sub_h),
            material=substrate_mat,
        ))

        # --- Generate meander trace ---
        z_trace = t_metal + sub_h + h  # top of antenna
        traces = []

        # Shorting pin (vertical)
        traces.append(TraceSegment(
            start=Point3D(-gl / 2, 0, t_metal),
            end=Point3D(-gl / 2, 0, z_trace),
            width=tw, thickness=t_metal,
        ))

        # Feed pin (vertical)
        feed_x = -gl / 2 + fo
        traces.append(TraceSegment(
            start=Point3D(feed_x, 0, t_metal),
            end=Point3D(feed_x, 0, z_trace),
            width=tw, thickness=t_metal,
        ))

        # Horizontal meander path
        x = -gl / 2
        y = 0.0
        direction = 1  # +y direction
        remaining = total_len
        pitch = tw + mg

        while remaining > 0:
            # Horizontal segment along x
            seg_len = min(mw, remaining)
            traces.append(TraceSegment(
                start=Point3D(x, y, z_trace),
                end=Point3D(x + seg_len, y, z_trace),
                width=tw, thickness=t_metal,
            ))
            x += seg_len
            remaining -= seg_len
            if remaining <= 0:
                break

            # Vertical connector
            dy = direction * pitch
            seg_len = min(abs(dy), remaining)
            dy = math.copysign(seg_len, dy)
            traces.append(TraceSegment(
                start=Point3D(x, y, z_trace),
                end=Point3D(x, y + dy, z_trace),
                width=tw, thickness=t_metal,
            ))
            y += dy
            remaining -= abs(dy)
            direction *= -1

        geom.traces = traces

        # --- Port ---
        geom.ports.append(Port(
            start=Point3D(feed_x, 0, t_metal),
            end=Point3D(feed_x, 0, z_trace),
            impedance=50.0,
        ))

        geom.compute_bounding_box()
        return geom
