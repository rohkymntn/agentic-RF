"""
Antenna design objective functions.

Maps simulation results to scalar objectives for optimization.
Supports multi-objective with weights and constraints.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np

from ..solver.base import SimulationResult
from ..geometry.base import AntennaTemplate, AntennaGeometry
from ..analysis.safety import SARAnalyzer, SAR_LIMIT_FCC_1G


@dataclass
class ObjectiveConfig:
    """Configuration for what the optimizer should optimize."""

    # Target frequency band
    target_freq_hz: float = 403.5e6          # center of MICS band
    freq_band_start_hz: float = 402e6
    freq_band_stop_hz: float = 405e6

    # S11 / matching objective
    s11_weight: float = 1.0                   # weight for S11 minimization
    s11_threshold_db: float = -10.0           # minimum acceptable S11

    # Bandwidth objective
    bandwidth_weight: float = 0.5
    min_bandwidth_hz: float = 3e6             # minimum 3 MHz BW

    # Efficiency objective
    efficiency_weight: float = 0.8
    min_efficiency: float = 0.1               # minimum 10% efficiency

    # Size objective (minimize)
    size_weight: float = 0.3
    max_radius_mm: float = 15.0               # maximum enclosing radius

    # Gain objective (maximize for far-field, ignore for NF)
    gain_weight: float = 0.0                  # 0 = don't care about gain

    # SAR constraint (for implants)
    sar_constraint: bool = False
    max_input_power_w: float = 0.025          # 25 mW (typical implant)

    # Robustness
    robustness_weight: float = 0.2
    detuning_pct: float = 5.0                 # test ±5% frequency shift

    # Reference impedance
    z0: float = 50.0


@dataclass
class ObjectiveResult:
    """Result of evaluating a single design."""
    # Individual objectives (lower is better for all)
    s11_score: float = 0.0           # penalized S11 (lower = better match)
    bandwidth_score: float = 0.0     # penalized bandwidth (lower = wider)
    efficiency_score: float = 0.0    # penalized efficiency (lower = higher η)
    size_score: float = 0.0          # penalized size (lower = smaller)
    gain_score: float = 0.0          # penalized gain (lower = higher gain)
    robustness_score: float = 0.0    # penalized robustness
    sar_penalty: float = 0.0         # SAR constraint penalty

    # Aggregated
    total_score: float = 0.0         # weighted sum (lower is better)

    # Raw metrics (for Pareto analysis)
    s11_db: float = 0.0
    bandwidth_hz: float = 0.0
    efficiency: float = 0.0
    radius_mm: float = 0.0
    peak_gain_dbi: float = 0.0
    sar_1g: float = 0.0

    # Feasibility
    feasible: bool = True
    constraint_violations: list[str] = field(default_factory=list)

    # Parameters that produced this result
    parameters: dict[str, float] = field(default_factory=dict)


class AntennaObjective:
    """
    Evaluates antenna designs against multi-objective criteria.

    Can use either analytical or full-wave solver.
    """

    def __init__(self, config: ObjectiveConfig, template: AntennaTemplate,
                 solver=None):
        """
        Args:
            config: Objective configuration
            template: Antenna template for geometry generation
            solver: EM solver (None = use analytical estimates from template)
        """
        self.config = config
        self.template = template
        self.solver = solver
        self._eval_count = 0
        self._best_score = float('inf')
        self._history: list[ObjectiveResult] = []

    @property
    def n_params(self) -> int:
        return len(self.template.parameters)

    @property
    def bounds(self) -> list[tuple[float, float]]:
        return self.template.get_bounds()

    @property
    def param_names(self) -> list[str]:
        return self.template.get_param_names()

    def evaluate(self, params: dict[str, float]) -> ObjectiveResult:
        """
        Evaluate a single antenna design.

        Returns ObjectiveResult with all individual scores and total.
        """
        self._eval_count += 1
        cfg = self.config
        params = self.template.validate_params(params)

        result = ObjectiveResult(parameters=params)

        # Generate geometry
        geom = self.template.generate_geometry(params)
        radius = geom.enclosing_radius

        if self.solver is not None:
            # Full-wave simulation
            sim = self.solver.simulate(
                geom,
                cfg.freq_band_start_hz * 0.8,
                cfg.freq_band_stop_hz * 1.2,
                n_freq=51,
                z0=cfg.z0,
            )
            result.s11_db = sim.s11_at_freq(cfg.target_freq_hz)
            result.bandwidth_hz = sim.bandwidth_hz(cfg.s11_threshold_db)
            result.efficiency = sim.efficiency_at_freq(cfg.target_freq_hz)
            result.peak_gain_dbi = (float(np.max(sim.peak_gain_dbi))
                                     if sim.peak_gain_dbi.size > 0 else 0)
        else:
            # Analytical fast evaluation
            quick = self.template.quick_evaluate(params, cfg.target_freq_hz, cfg.z0)
            result.s11_db = quick["s11_db"]
            result.bandwidth_hz = quick["bandwidth_frac"] * cfg.target_freq_hz
            result.efficiency = 0.5  # rough estimate
            result.peak_gain_dbi = 1.5  # short dipole estimate

        result.radius_mm = radius * 1e3

        # ─── Score each objective ────────────────────────────────────────

        # S11: want to minimize (more negative = better)
        # Score: 0 if S11 < threshold, penalize linearly above
        if result.s11_db < cfg.s11_threshold_db:
            result.s11_score = result.s11_db / cfg.s11_threshold_db  # normalized, <1 is good
        else:
            result.s11_score = 1.0 + (result.s11_db - cfg.s11_threshold_db) / 10.0  # penalty

        # Bandwidth: want to maximize
        bw_ratio = result.bandwidth_hz / max(cfg.min_bandwidth_hz, 1)
        if bw_ratio >= 1.0:
            result.bandwidth_score = 1.0 / bw_ratio  # <1 is good
        else:
            result.bandwidth_score = 1.0 + (1.0 - bw_ratio) * 5.0  # heavy penalty

        # Efficiency: want to maximize
        if result.efficiency >= cfg.min_efficiency:
            result.efficiency_score = 1.0 - result.efficiency
        else:
            result.efficiency_score = 2.0 + (cfg.min_efficiency - result.efficiency) * 10.0

        # Size: want to minimize
        size_ratio = result.radius_mm / cfg.max_radius_mm
        if size_ratio <= 1.0:
            result.size_score = size_ratio
        else:
            result.size_score = 1.0 + (size_ratio - 1.0) * 5.0  # penalty for exceeding

        # Gain: want to maximize (if weight > 0)
        result.gain_score = max(0, 8.0 - result.peak_gain_dbi) / 8.0  # normalize to ~[0,1]

        # Robustness: check detuning
        if cfg.robustness_weight > 0:
            f_shift = cfg.target_freq_hz * cfg.detuning_pct / 100
            z_up = self.template.estimate_impedance(params, cfg.target_freq_hz + f_shift)
            z_down = self.template.estimate_impedance(params, cfg.target_freq_hz - f_shift)
            from ..utils.units import z_to_gamma, mag_to_db
            s11_up = mag_to_db(abs(z_to_gamma(z_up, cfg.z0)))
            s11_down = mag_to_db(abs(z_to_gamma(z_down, cfg.z0)))
            worst = max(s11_up, s11_down)
            if worst < -6.0:
                result.robustness_score = 0.0  # robust
            else:
                result.robustness_score = (worst + 6.0) / 6.0  # penalty

        # SAR constraint
        if cfg.sar_constraint:
            analyzer = SARAnalyzer()
            from ..utils.materials import MUSCLE
            sar_result = analyzer.estimate_sar(
                cfg.max_input_power_w, max(result.efficiency, 0.01),
                cfg.target_freq_hz, MUSCLE,
                antenna_area_m2=math.pi * (radius) ** 2,
            )
            result.sar_1g = sar_result.peak_sar_1g
            if not sar_result.compliant_fcc:
                result.sar_penalty = 10.0 * (result.sar_1g / SAR_LIMIT_FCC_1G - 1.0)
                result.feasible = False
                result.constraint_violations.append(
                    f"SAR {result.sar_1g:.2f} W/kg exceeds {SAR_LIMIT_FCC_1G} W/kg limit"
                )

        # ─── Constraints ─────────────────────────────────────────────────
        if result.radius_mm > cfg.max_radius_mm:
            result.feasible = False
            result.constraint_violations.append(
                f"Size {result.radius_mm:.1f} mm exceeds {cfg.max_radius_mm} mm limit"
            )

        # ─── Weighted total ──────────────────────────────────────────────
        result.total_score = (
            cfg.s11_weight * result.s11_score +
            cfg.bandwidth_weight * result.bandwidth_score +
            cfg.efficiency_weight * result.efficiency_score +
            cfg.size_weight * result.size_score +
            cfg.gain_weight * result.gain_score +
            cfg.robustness_weight * result.robustness_score +
            result.sar_penalty
        )

        # Track best
        if result.total_score < self._best_score:
            self._best_score = result.total_score

        self._history.append(result)
        return result

    def evaluate_vector(self, x: np.ndarray) -> float:
        """Evaluate from normalized [0,1] parameter vector. Returns total_score."""
        params = self.template.vector_to_params(x)
        result = self.evaluate(params)
        return result.total_score

    def evaluate_vector_multi(self, x: np.ndarray) -> np.ndarray:
        """
        Multi-objective evaluation for Pareto optimizers (NSGA-II).

        Returns array of objectives to MINIMIZE:
        [s11_penalty, -bandwidth, -efficiency, size]
        """
        params = self.template.vector_to_params(x)
        result = self.evaluate(params)

        return np.array([
            -result.s11_db,          # minimize (want most negative S11, so negate)
            -result.bandwidth_hz,    # minimize (want max BW)
            1.0 - result.efficiency, # minimize (want max efficiency)
            result.radius_mm,        # minimize size
        ])

    @property
    def eval_count(self) -> int:
        return self._eval_count

    @property
    def best_result(self) -> ObjectiveResult | None:
        if not self._history:
            return None
        return min(self._history, key=lambda r: r.total_score)

    @property
    def pareto_front(self) -> list[ObjectiveResult]:
        """Extract Pareto-optimal designs from history."""
        if not self._history:
            return []

        # Simple Pareto extraction on (s11, bandwidth, efficiency, size)
        front = []
        for r in self._history:
            dominated = False
            for other in self._history:
                if (other.s11_db <= r.s11_db and
                    other.bandwidth_hz >= r.bandwidth_hz and
                    other.efficiency >= r.efficiency and
                    other.radius_mm <= r.radius_mm and
                    (other.s11_db < r.s11_db or
                     other.bandwidth_hz > r.bandwidth_hz or
                     other.efficiency > r.efficiency or
                     other.radius_mm < r.radius_mm)):
                    dominated = True
                    break
            if not dominated:
                front.append(r)
        return front
