"""
Meander Line Antenna template.

Meander line antennas are the go-to for miniaturized planar antennas.
The meandering folds a resonant length into a compact footprint,
trading bandwidth for size reduction.

Key features:
  - Extreme miniaturization possible (λ/10 or smaller footprint)
  - Narrowband (high Q due to small size)
  - Planar, easy to manufacture on PCB
  - Widely used for implant and IoT antennas
"""

from __future__ import annotations
import math

from ..base import (
    AntennaTemplate, DesignParameter, AntennaGeometry,
    Point3D, TraceSegment, Box3D, Port,
)
from ...utils.constants import C_0
from ...utils.materials import COPPER, Material


class MeanderAntennaTemplate(AntennaTemplate):

    @property
    def name(self) -> str:
        return "Meander Line Antenna"

    @property
    def description(self) -> str:
        return (
            "Planar meander line antenna for extreme miniaturization. Folds a resonant "
            "conductor path into a small footprint. Bandwidth is limited by Chu limit "
            "(small ka). Excellent for implant antennas (MICS/MedRadio), IoT, and any "
            "application needing very small size."
        )

    @property
    def parameters(self) -> list[DesignParameter]:
        return [
            DesignParameter("num_sections", 3, 30, 15, "",
                            "Number of meander sections", is_integer=True),
            DesignParameter("section_length_mm", 1.0, 30.0, 12.0, "mm",
                            "Length of each vertical meander arm"),
            DesignParameter("trace_width_mm", 0.15, 3.0, 0.5, "mm",
                            "Conductor trace width"),
            DesignParameter("trace_gap_mm", 0.15, 3.0, 0.5, "mm",
                            "Gap between adjacent traces"),
            DesignParameter("substrate_er", 1.0, 80.0, 4.4, "",
                            "Substrate permittivity (high εr for implant)"),
            DesignParameter("substrate_height_mm", 0.2, 5.0, 1.6, "mm",
                            "Substrate thickness"),
            DesignParameter("ground_plane", 0, 1, 1, "",
                            "Has ground plane (1=yes, 0=no)", is_integer=True),
            DesignParameter("ground_length_mm", 5.0, 60.0, 20.0, "mm",
                            "Ground plane length"),
            DesignParameter("ground_width_mm", 5.0, 60.0, 20.0, "mm",
                            "Ground plane width"),
        ]

    def _total_conductor_length(self, params: dict[str, float]) -> float:
        """Total physical conductor length [m]."""
        p = params
        n = int(p["num_sections"])
        sl = p["section_length_mm"] * 1e-3
        tw = p["trace_width_mm"] * 1e-3
        tg = p["trace_gap_mm"] * 1e-3

        # Each section: one vertical arm + one horizontal connector
        pitch = tw + tg
        L_vertical = n * sl
        L_horizontal = (n - 1) * pitch
        return L_vertical + L_horizontal

    def estimate_resonance_hz(self, params: dict[str, float]) -> float:
        """
        Meander resonance: total path ≈ λ_eff / 4 (monopole) or λ_eff / 2 (dipole).

        With ground plane: quarter-wave (λ/4) resonance.
        Without ground plane: half-wave (λ/2) resonance.
        Mutual coupling between adjacent arms shortens effective length.
        """
        p = self.validate_params(params)
        L_total = self._total_conductor_length(p)

        er = p["substrate_er"]
        h = p["substrate_height_mm"]
        w = p["trace_width_mm"]
        has_gnd = int(p["ground_plane"])

        # Effective permittivity
        if has_gnd and er > 1:
            ratio = w / max(h, 0.01)
            er_eff = (er + 1) / 2 + (er - 1) / (2 * math.sqrt(1 + 12 / max(ratio, 0.01)))
        else:
            er_eff = (er + 1) / 2  # rough for no ground

        # Meander coupling factor: tightly spaced traces couple,
        # effectively shortening the electrical length
        gap = p["trace_gap_mm"]
        coupling_factor = 1.0 - 0.15 * math.exp(-gap / 0.5)

        divisor = 4.0 if has_gnd else 2.0
        return C_0 / (divisor * L_total * math.sqrt(er_eff) * coupling_factor)

    def estimate_impedance(self, params: dict[str, float], freq_hz: float) -> complex:
        """Meander antenna input impedance."""
        p = self.validate_params(params)
        f0 = self.estimate_resonance_hz(p)

        er = p["substrate_er"]
        n_sec = int(p["num_sections"])
        sl = p["section_length_mm"] * 1e-3

        lam = C_0 / freq_hz

        # Radiation resistance for small meander
        # Approximate as short monopole: R_rad ≈ 40π²(h_eff/λ)²
        h_eff = sl  # effective height ≈ one arm length
        R_rad = 40.0 * (math.pi * h_eff / lam) ** 2

        # Loss (especially important for high-εr implant substrates)
        R_loss = 2.0 + 0.5 * er  # loss increases with εr (roughly)

        R = R_rad + R_loss

        # Q is dominated by Chu limit for electrically small antennas
        # Typical meander Q: 20-100 depending on size
        footprint = n_sec * (p["trace_width_mm"] + p["trace_gap_mm"]) * 1e-3
        ka = 2 * math.pi * max(footprint, sl) / lam
        Q = max(1 / ka**3 + 1 / ka, 10.0) if ka > 0 else 100.0

        delta_f = (freq_hz - f0) / f0
        X = R * Q * 2.0 * delta_f

        return complex(max(R, 0.5), X)

    def generate_geometry(self, params: dict[str, float]) -> AntennaGeometry:
        """Generate planar meander line geometry."""
        p = self.validate_params(params)
        mm = 1e-3
        t = 35e-6

        n_sec = int(p["num_sections"])
        sl = p["section_length_mm"] * mm
        tw = p["trace_width_mm"] * mm
        tg = p["trace_gap_mm"] * mm
        sub_h = p["substrate_height_mm"] * mm
        er = p["substrate_er"]
        has_gnd = int(p["ground_plane"])
        gl = p["ground_length_mm"] * mm
        gw = p["ground_width_mm"] * mm

        geom = AntennaGeometry(template_name="Meander", parameters=p)

        pitch = tw + tg
        z_trace = t + sub_h if has_gnd else sub_h

        # Ground plane
        if has_gnd:
            geom.boxes.append(Box3D(
                origin=Point3D(0, 0, 0),
                size=Point3D(gl, gw, t),
                material=COPPER,
            ))

        # Substrate
        sub_mat = Material(f"Sub_er{er:.1f}", eps_r=er, tan_d=0.02)
        geom.boxes.append(Box3D(
            origin=Point3D(0, 0, t if has_gnd else 0),
            size=Point3D(gl, gw, sub_h),
            material=sub_mat,
        ))

        # Meander trace
        x = pitch  # start offset from edge
        direction = 1  # 1 = up, -1 = down
        y_base = (gw - sl) / 2  # center vertically

        for i in range(n_sec):
            # Vertical arm
            y_start = y_base if direction == 1 else y_base + sl
            y_end = y_base + sl if direction == 1 else y_base

            geom.traces.append(TraceSegment(
                start=Point3D(x, y_start, z_trace),
                end=Point3D(x, y_end, z_trace),
                width=tw, thickness=t,
            ))

            # Horizontal connector to next arm
            if i < n_sec - 1:
                y_conn = y_end
                geom.traces.append(TraceSegment(
                    start=Point3D(x, y_conn, z_trace),
                    end=Point3D(x + pitch, y_conn, z_trace),
                    width=tw, thickness=t,
                ))

            x += pitch
            direction *= -1

        # Port at start of first arm
        feed_x = pitch
        if has_gnd:
            geom.ports.append(Port(
                start=Point3D(feed_x, y_base, t),
                end=Point3D(feed_x, y_base, z_trace),
                impedance=50.0,
            ))
        else:
            geom.ports.append(Port(
                start=Point3D(feed_x, y_base - 0.001, z_trace),
                end=Point3D(feed_x, y_base, z_trace),
                impedance=50.0,
            ))

        geom.compute_bounding_box()
        return geom
