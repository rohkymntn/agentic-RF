"""
Base classes for parametric antenna templates.

Every antenna family (PIFA, loop, patch, etc.) inherits from AntennaTemplate.
The template defines:
  - Named parameters with bounds
  - A method to generate physical geometry (for EM solvers)
  - A method to estimate resonant frequency (analytical fast-screen)
  - Constraints (keepouts, manufacturing rules)
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np

from ..utils.constants import C_0, freq_to_wavelength, electrical_size
from ..utils.materials import Material, FR4, COPPER, AIR


# ═══════════════════════════════════════════════════════════════════════════════
# Design Parameter
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class DesignParameter:
    """A single design parameter with bounds and metadata."""
    name: str
    min_val: float
    max_val: float
    default: float
    unit: str = "m"
    description: str = ""
    is_integer: bool = False

    def validate(self, value: float) -> float:
        """Clamp value to bounds."""
        if self.is_integer:
            value = round(value)
        return max(self.min_val, min(self.max_val, value))

    def normalize(self, value: float) -> float:
        """Map value to [0, 1] range."""
        span = self.max_val - self.min_val
        if span <= 0:
            return 0.5
        return (value - self.min_val) / span

    def denormalize(self, norm: float) -> float:
        """Map [0, 1] back to parameter range."""
        value = self.min_val + norm * (self.max_val - self.min_val)
        return self.validate(value)


# ═══════════════════════════════════════════════════════════════════════════════
# Geometry Primitives
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class Point3D:
    """3D point in meters."""
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0

    def to_array(self) -> np.ndarray:
        return np.array([self.x, self.y, self.z])

    def __add__(self, other: Point3D) -> Point3D:
        return Point3D(self.x + other.x, self.y + other.y, self.z + other.z)

    def __sub__(self, other: Point3D) -> Point3D:
        return Point3D(self.x - other.x, self.y - other.y, self.z - other.z)


@dataclass
class TraceSegment:
    """A conductive trace segment (for generating solver geometry)."""
    start: Point3D
    end: Point3D
    width: float        # trace width [m]
    thickness: float    # metal thickness [m]
    material: Material = field(default_factory=lambda: COPPER)
    layer: int = 0      # PCB layer index


@dataclass
class Box3D:
    """Axis-aligned 3D box (for substrates, ground planes, etc.)."""
    origin: Point3D     # corner (min x, min y, min z)
    size: Point3D       # dimensions (dx, dy, dz)
    material: Material = field(default_factory=lambda: FR4)

    @property
    def center(self) -> Point3D:
        return Point3D(
            self.origin.x + self.size.x / 2,
            self.origin.y + self.size.y / 2,
            self.origin.z + self.size.z / 2,
        )


@dataclass
class Port:
    """Excitation port (feed point)."""
    start: Point3D
    end: Point3D
    impedance: float = 50.0
    port_number: int = 1


@dataclass
class AntennaGeometry:
    """
    Complete physical geometry description of an antenna, ready for a solver.

    This is solver-agnostic: each solver adapter translates this into
    its native format.
    """
    traces: list[TraceSegment] = field(default_factory=list)
    boxes: list[Box3D] = field(default_factory=list)
    ports: list[Port] = field(default_factory=list)
    bounding_box: Optional[tuple[Point3D, Point3D]] = None

    # Metadata
    template_name: str = ""
    parameters: dict[str, float] = field(default_factory=dict)

    def compute_bounding_box(self) -> tuple[Point3D, Point3D]:
        """Compute axis-aligned bounding box of all geometry."""
        all_points = []
        for t in self.traces:
            all_points.extend([t.start, t.end])
        for b in self.boxes:
            all_points.append(b.origin)
            all_points.append(b.origin + b.size)
        for p in self.ports:
            all_points.extend([p.start, p.end])

        if not all_points:
            return (Point3D(), Point3D())

        xs = [p.x for p in all_points]
        ys = [p.y for p in all_points]
        zs = [p.z for p in all_points]
        self.bounding_box = (
            Point3D(min(xs), min(ys), min(zs)),
            Point3D(max(xs), max(ys), max(zs)),
        )
        return self.bounding_box

    @property
    def enclosing_radius(self) -> float:
        """Radius of smallest enclosing sphere (for Chu limit)."""
        bb = self.bounding_box or self.compute_bounding_box()
        dx = bb[1].x - bb[0].x
        dy = bb[1].y - bb[0].y
        dz = bb[1].z - bb[0].z
        return math.sqrt(dx**2 + dy**2 + dz**2) / 2.0

    def electrical_size_at(self, freq_hz: float) -> float:
        """Return ka at given frequency."""
        return electrical_size(freq_hz, self.enclosing_radius)

    def get_trace_vertices(self) -> list[list[tuple[float, float, float]]]:
        """Return trace paths as lists of (x,y,z) tuples for visualization."""
        paths = []
        for t in self.traces:
            paths.append([
                (t.start.x, t.start.y, t.start.z),
                (t.end.x, t.end.y, t.end.z),
            ])
        return paths


# ═══════════════════════════════════════════════════════════════════════════════
# Abstract Antenna Template
# ═══════════════════════════════════════════════════════════════════════════════

class AntennaTemplate(ABC):
    """
    Abstract base for all parametric antenna families.

    Subclasses must implement:
      - parameters(): list of DesignParameter
      - generate_geometry(params): build AntennaGeometry from param dict
      - estimate_resonance(params): fast analytical resonance estimate
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable template name."""
        ...

    @property
    @abstractmethod
    def description(self) -> str:
        """Template description for the LLM agent."""
        ...

    @property
    @abstractmethod
    def parameters(self) -> list[DesignParameter]:
        """List of design parameters with bounds."""
        ...

    @abstractmethod
    def generate_geometry(self, params: dict[str, float]) -> AntennaGeometry:
        """
        Generate full 3D geometry from parameter values.

        Args:
            params: dict mapping parameter name → value

        Returns:
            AntennaGeometry ready for EM solver
        """
        ...

    @abstractmethod
    def estimate_resonance_hz(self, params: dict[str, float]) -> float:
        """
        Fast analytical estimate of resonant frequency [Hz].

        This is for screening / initial guess — not a substitute for EM simulation.
        """
        ...

    @abstractmethod
    def estimate_impedance(self, params: dict[str, float], freq_hz: float) -> complex:
        """
        Fast analytical estimate of input impedance at a frequency.

        Returns complex impedance Z = R + jX.
        """
        ...

    # ─── Convenience methods ─────────────────────────────────────────────────

    def get_default_params(self) -> dict[str, float]:
        """Return default parameter values."""
        return {p.name: p.default for p in self.parameters}

    def get_bounds(self) -> list[tuple[float, float]]:
        """Return parameter bounds as list of (min, max) tuples."""
        return [(p.min_val, p.max_val) for p in self.parameters]

    def get_param_names(self) -> list[str]:
        """Return parameter names in order."""
        return [p.name for p in self.parameters]

    def validate_params(self, params: dict[str, float]) -> dict[str, float]:
        """Validate and clamp all parameters to bounds."""
        param_lookup = {p.name: p for p in self.parameters}
        validated = {}
        for name, value in params.items():
            if name in param_lookup:
                validated[name] = param_lookup[name].validate(value)
        # Fill missing with defaults
        for p in self.parameters:
            if p.name not in validated:
                validated[p.name] = p.default
        return validated

    def params_to_vector(self, params: dict[str, float]) -> np.ndarray:
        """Convert param dict to normalized [0,1] vector (for optimizers)."""
        param_lookup = {p.name: p for p in self.parameters}
        return np.array([
            param_lookup[name].normalize(params.get(name, param_lookup[name].default))
            for name in self.get_param_names()
        ])

    def vector_to_params(self, x: np.ndarray) -> dict[str, float]:
        """Convert normalized [0,1] vector back to param dict."""
        params = {}
        for i, p in enumerate(self.parameters):
            params[p.name] = p.denormalize(float(x[i]))
        return params

    def estimate_bandwidth(self, params: dict[str, float]) -> float:
        """
        Estimate fractional bandwidth from Chu limit + Q estimate.

        Returns fractional bandwidth (e.g., 0.02 = 2%).
        """
        geom = self.generate_geometry(params)
        f0 = self.estimate_resonance_hz(params)
        ka = geom.electrical_size_at(f0)
        if ka <= 0:
            return 0.0
        from ..utils.constants import chu_limit_bandwidth
        return chu_limit_bandwidth(ka, vswr_max=2.0)

    def quick_evaluate(self, params: dict[str, float], target_freq_hz: float,
                       z0: float = 50.0) -> dict[str, float]:
        """
        Quick analytical evaluation (no EM sim) returning key metrics.

        Returns dict with: f_res, s11_db, vswr, bandwidth_pct, ka, q_min
        """
        from ..utils.constants import chu_limit_q, chu_limit_bandwidth
        from ..utils.units import z_to_gamma, s11_to_vswr, mag_to_db

        params = self.validate_params(params)
        f_res = self.estimate_resonance_hz(params)
        z_in = self.estimate_impedance(params, target_freq_hz)
        gamma = z_to_gamma(z_in, z0)
        s11 = abs(gamma)
        geom = self.generate_geometry(params)
        ka = geom.electrical_size_at(target_freq_hz)

        q_min = chu_limit_q(ka) if ka > 0 else float('inf')
        bw = chu_limit_bandwidth(ka) if ka > 0 else 0.0

        return {
            "f_res_hz": f_res,
            "f_res_mhz": f_res / 1e6,
            "s11_db": mag_to_db(s11),
            "vswr": s11_to_vswr(s11),
            "bandwidth_frac": bw,
            "bandwidth_pct": bw * 100.0,
            "ka": ka,
            "q_min": q_min,
            "z_real": z_in.real,
            "z_imag": z_in.imag,
            "enclosing_radius_mm": geom.enclosing_radius * 1e3,
        }
