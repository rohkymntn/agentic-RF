"""
CMA-ES (Covariance Matrix Adaptation Evolution Strategy) optimizer.

CMA-ES is excellent for:
  - Nonconvex, non-separable optimization
  - Moderate dimensions (5-100 parameters)
  - When the landscape has ridges, plateaus, or multimodal structure

It adapts its search distribution (covariance matrix) to the landscape,
making it very effective for antenna optimization where parameters interact.

Uses pycma library.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from .objectives import AntennaObjective, ObjectiveResult


@dataclass
class CMAESConfig:
    """CMA-ES configuration."""
    sigma0: float = 0.3          # Initial step size (in normalized [0,1] space)
    max_evaluations: int = 500
    tol_fun: float = 1e-6        # Function value tolerance
    random_seed: int | None = None
    verbose: bool = True


class CMAESOptimizer:
    """
    CMA-ES optimizer for antenna design.

    Works in the normalized [0,1] parameter space.
    """

    def __init__(self, config: CMAESConfig | None = None):
        self.config = config or CMAESConfig()

    def optimize(self, objective: AntennaObjective,
                 x0: np.ndarray | None = None) -> dict:
        """
        Run CMA-ES optimization.

        Args:
            objective: Antenna objective function
            x0: Initial point in normalized [0,1] space. None = center.

        Returns:
            dict with best_params, best_score, convergence_history, etc.
        """
        try:
            return self._optimize_cma(objective, x0)
        except ImportError:
            print("[CMA-ES] pycma not available, using Nelder-Mead fallback")
            return self._optimize_scipy(objective, x0)

    def _optimize_cma(self, objective: AntennaObjective,
                       x0: np.ndarray | None) -> dict:
        """CMA-ES via pycma."""
        import cma

        cfg = self.config
        t_start = time.time()

        n = objective.n_params
        if x0 is None:
            x0 = np.full(n, 0.5)  # center of [0,1]^n

        convergence = []
        best_score = float('inf')

        def eval_func(x):
            nonlocal best_score
            # Clamp to [0,1]
            x = np.clip(x, 0, 1)
            score = objective.evaluate_vector(x)
            best_score = min(best_score, score)
            convergence.append(best_score)
            return score

        opts = cma.CMAOptions()
        opts['maxfevals'] = cfg.max_evaluations
        opts['tolfun'] = cfg.tol_fun
        opts['bounds'] = [0, 1]
        opts['verbose'] = -1 if not cfg.verbose else 1
        if cfg.random_seed is not None:
            opts['seed'] = cfg.random_seed

        es = cma.CMAEvolutionStrategy(x0, cfg.sigma0, opts)
        es.optimize(eval_func)

        best_x = np.clip(es.result.xbest, 0, 1)
        best_params = objective.template.vector_to_params(best_x)
        best_result = objective.evaluate(best_params)

        return {
            "best_params": best_params,
            "best_score": es.result.fbest,
            "best_result": best_result,
            "all_results": list(objective._history),
            "n_evaluations": objective.eval_count,
            "wall_time_seconds": time.time() - t_start,
            "convergence_history": convergence,
        }

    def _optimize_scipy(self, objective: AntennaObjective,
                         x0: np.ndarray | None) -> dict:
        """Fallback: multi-start scipy differential_evolution + Nelder-Mead polish."""
        from scipy.optimize import differential_evolution

        cfg = self.config
        t_start = time.time()

        n = objective.n_params
        convergence = []
        best_score = float('inf')

        def eval_func(x):
            nonlocal best_score
            x = np.clip(x, 0, 1)
            score = objective.evaluate_vector(x)
            best_score = min(best_score, score)
            convergence.append(best_score)
            return score

        bounds = [(0, 1)] * n

        result = differential_evolution(
            eval_func,
            bounds,
            maxiter=cfg.max_evaluations // (n * 4),
            popsize=max(10, n * 2),
            tol=cfg.tol_fun,
            polish=True,
            seed=cfg.random_seed,
        )

        best_x = np.clip(result.x, 0, 1)
        best_params = objective.template.vector_to_params(best_x)
        best_result = objective.evaluate(best_params)

        return {
            "best_params": best_params,
            "best_score": result.fun,
            "best_result": best_result,
            "all_results": list(objective._history),
            "n_evaluations": objective.eval_count,
            "wall_time_seconds": time.time() - t_start,
            "convergence_history": convergence,
        }
