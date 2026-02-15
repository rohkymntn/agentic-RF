"""
NSGA-II multi-objective optimizer for antenna design.

NSGA-II (Non-dominated Sorting Genetic Algorithm II) is the standard
for multi-objective optimization. It produces a Pareto front showing
the tradeoff between competing objectives (e.g., size vs bandwidth).

Uses pymoo library for the implementation.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from .objectives import AntennaObjective, ObjectiveResult, ObjectiveConfig


@dataclass
class NSGAIIConfig:
    """NSGA-II configuration."""
    pop_size: int = 40           # Population size
    n_generations: int = 50      # Number of generations
    crossover_prob: float = 0.9
    mutation_prob: float = 0.1   # Per-gene mutation probability
    random_seed: int | None = None
    verbose: bool = True


class NSGAIIOptimizer:
    """
    Multi-objective optimizer using NSGA-II.

    Produces a Pareto front of designs trading off:
    - S11 (match quality)
    - Bandwidth
    - Efficiency
    - Size

    Uses pymoo; falls back to simple multi-objective random search if unavailable.
    """

    def __init__(self, config: NSGAIIConfig | None = None):
        self.config = config or NSGAIIConfig()

    def optimize(self, objective: AntennaObjective) -> dict:
        """
        Run NSGA-II optimization.

        Returns dict with:
            - pareto_front: list of ObjectiveResult on the Pareto front
            - all_results: complete history
            - wall_time_seconds: total time
        """
        try:
            return self._optimize_pymoo(objective)
        except ImportError:
            print("[NSGA-II] pymoo not available, using simple multi-objective search")
            return self._optimize_simple(objective)

    def _optimize_pymoo(self, objective: AntennaObjective) -> dict:
        """NSGA-II via pymoo."""
        from pymoo.core.problem import Problem
        from pymoo.algorithms.moo.nsga2 import NSGA2
        from pymoo.operators.crossover.sbx import SBX
        from pymoo.operators.mutation.pm import PM
        from pymoo.optimize import minimize as pymoo_minimize

        cfg = self.config
        t_start = time.time()

        bounds = objective.bounds
        n_var = len(bounds)
        xl = np.array([b[0] for b in bounds])
        xu = np.array([b[1] for b in bounds])

        class AntennaProblem(Problem):
            def __init__(self):
                super().__init__(
                    n_var=n_var,
                    n_obj=4,  # s11, bandwidth, efficiency, size
                    n_constr=0,
                    xl=xl,
                    xu=xu,
                )

            def _evaluate(self, X, out, *args, **kwargs):
                F = np.zeros((len(X), 4))
                for i, x in enumerate(X):
                    params = {p.name: v for p, v in
                              zip(objective.template.parameters, x)}
                    result = objective.evaluate(params)
                    F[i] = [
                        -result.s11_db,          # minimize (more negative s11 = better)
                        -result.bandwidth_hz,    # minimize (max BW)
                        1.0 - result.efficiency, # minimize (max efficiency)
                        result.radius_mm,        # minimize size
                    ]
                out["F"] = F

        problem = AntennaProblem()

        algorithm = NSGA2(
            pop_size=cfg.pop_size,
            crossover=SBX(prob=cfg.crossover_prob, eta=15),
            mutation=PM(prob=cfg.mutation_prob, eta=20),
            eliminate_duplicates=True,
        )

        res = pymoo_minimize(
            problem,
            algorithm,
            ("n_gen", cfg.n_generations),
            seed=cfg.random_seed,
            verbose=cfg.verbose,
        )

        # Extract Pareto front
        pareto_results = []
        if res.X is not None:
            for x in res.X:
                params = {p.name: v for p, v in
                          zip(objective.template.parameters, x)}
                result = objective.evaluate(params)
                pareto_results.append(result)

        wall_time = time.time() - t_start

        return {
            "pareto_front": pareto_results,
            "all_results": list(objective._history),
            "n_evaluations": objective.eval_count,
            "wall_time_seconds": wall_time,
        }

    def _optimize_simple(self, objective: AntennaObjective) -> dict:
        """Simple multi-objective search (fallback)."""
        cfg = self.config
        t_start = time.time()
        rng = np.random.RandomState(cfg.random_seed)

        total_evals = cfg.pop_size * cfg.n_generations

        for _ in range(total_evals):
            x = rng.uniform(0, 1, objective.n_params)
            params = objective.template.vector_to_params(x)
            objective.evaluate(params)

        pareto = objective.pareto_front

        return {
            "pareto_front": pareto,
            "all_results": list(objective._history),
            "n_evaluations": objective.eval_count,
            "wall_time_seconds": time.time() - t_start,
        }
