"""
Bayesian optimization for antenna design.

Bayesian optimization is ideal when each evaluation is expensive
(full-wave EM simulation can take minutes to hours). It builds a
surrogate model (Gaussian Process) of the objective landscape and
uses acquisition functions to decide where to sample next.

Excellent for: refining designs with full-wave solvers.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional, Callable

import numpy as np

from .objectives import AntennaObjective, ObjectiveResult


@dataclass
class BayesianOptimizerConfig:
    """Bayesian optimization configuration."""
    n_initial: int = 10            # Number of initial random samples
    n_iterations: int = 50         # Number of optimization iterations
    acquisition: str = "EI"        # "EI" (Expected Improvement), "LCB", "PI"
    kappa: float = 2.576           # LCB exploration parameter
    xi: float = 0.01               # EI/PI exploitation parameter
    n_restarts: int = 5            # Restarts for acquisition optimization
    random_seed: int | None = None
    verbose: bool = True


@dataclass
class OptimizationResult:
    """Result of optimization run."""
    best_params: dict[str, float]
    best_score: float
    best_result: ObjectiveResult
    all_results: list[ObjectiveResult]
    n_evaluations: int
    wall_time_seconds: float
    convergence_history: list[float]  # best score at each iteration


class BayesianOptimizer:
    """
    Bayesian optimization engine for antenna design.

    Uses scikit-optimize (skopt) for the GP surrogate model.
    Falls back to random search if skopt is not available.
    """

    def __init__(self, config: BayesianOptimizerConfig | None = None):
        self.config = config or BayesianOptimizerConfig()

    def optimize(self, objective: AntennaObjective,
                 callback: Callable | None = None) -> OptimizationResult:
        """
        Run Bayesian optimization.

        Args:
            objective: The antenna objective function
            callback: Optional callback(iteration, params, score) per iteration

        Returns:
            OptimizationResult with best design and history
        """
        t_start = time.time()
        cfg = self.config

        try:
            return self._optimize_skopt(objective, callback)
        except ImportError:
            print("[BayesianOptimizer] scikit-optimize not available, falling back to random search")
            return self._optimize_random(objective, callback)

    def _optimize_skopt(self, objective: AntennaObjective,
                         callback: Callable | None) -> OptimizationResult:
        """Bayesian optimization using scikit-optimize."""
        from skopt import gp_minimize
        from skopt.space import Real, Integer

        cfg = self.config
        t_start = time.time()

        # Build search space
        dimensions = []
        for p in objective.template.parameters:
            if p.is_integer:
                dimensions.append(Integer(int(p.min_val), int(p.max_val), name=p.name))
            else:
                dimensions.append(Real(p.min_val, p.max_val, name=p.name))

        convergence = []
        best_so_far = float('inf')

        def eval_func(x):
            nonlocal best_so_far
            params = {p.name: v for p, v in zip(objective.template.parameters, x)}
            result = objective.evaluate(params)

            best_so_far = min(best_so_far, result.total_score)
            convergence.append(best_so_far)

            if callback:
                callback(len(convergence), params, result.total_score)

            if cfg.verbose and len(convergence) % 10 == 0:
                print(f"  Iter {len(convergence)}: score={result.total_score:.4f}, "
                      f"S11={result.s11_db:.1f} dB, best={best_so_far:.4f}")

            return result.total_score

        acq_map = {"EI": "EI", "LCB": "LCB", "PI": "PI"}
        acq_func = acq_map.get(cfg.acquisition, "EI")

        result = gp_minimize(
            eval_func,
            dimensions,
            n_calls=cfg.n_initial + cfg.n_iterations,
            n_initial_points=cfg.n_initial,
            acq_func=acq_func,
            acq_func_kwargs={"xi": cfg.xi, "kappa": cfg.kappa},
            random_state=cfg.random_seed,
            verbose=False,
        )

        # Extract best
        best_params = {p.name: v for p, v in
                       zip(objective.template.parameters, result.x)}
        best_obj = objective.evaluate(best_params)

        return OptimizationResult(
            best_params=best_params,
            best_score=result.fun,
            best_result=best_obj,
            all_results=list(objective._history),
            n_evaluations=objective.eval_count,
            wall_time_seconds=time.time() - t_start,
            convergence_history=convergence,
        )

    def _optimize_random(self, objective: AntennaObjective,
                          callback: Callable | None) -> OptimizationResult:
        """Fallback: random search with local refinement."""
        cfg = self.config
        t_start = time.time()
        rng = np.random.RandomState(cfg.random_seed)

        total_evals = cfg.n_initial + cfg.n_iterations
        convergence = []
        best_score = float('inf')
        best_params = objective.template.get_default_params()

        for i in range(total_evals):
            if i < cfg.n_initial:
                # Random sample
                x = rng.uniform(0, 1, objective.n_params)
                params = objective.template.vector_to_params(x)
            else:
                # Perturb best with decreasing step
                x_best = objective.template.params_to_vector(best_params)
                step = 0.1 * (1.0 - (i - cfg.n_initial) / max(cfg.n_iterations, 1))
                x = x_best + rng.normal(0, step, objective.n_params)
                x = np.clip(x, 0, 1)
                params = objective.template.vector_to_params(x)

            result = objective.evaluate(params)

            if result.total_score < best_score:
                best_score = result.total_score
                best_params = params.copy()

            convergence.append(best_score)

            if callback:
                callback(i + 1, params, result.total_score)

        best_result = objective.evaluate(best_params)

        return OptimizationResult(
            best_params=best_params,
            best_score=best_score,
            best_result=best_result,
            all_results=list(objective._history),
            n_evaluations=objective.eval_count,
            wall_time_seconds=time.time() - t_start,
            convergence_history=convergence,
        )
