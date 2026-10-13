# -*- coding: utf-8 -*-
"""True two-level optimizer for the wind-turbine pile foundation research model.

Outer level (12 variables): global cap and pile-system geometry.
Inner level (14 variables): ten reinforcement variables plus four anchor variables.
The reinforcement search is deterministic coordinate descent; the anchor search
uses the existing deterministic discrete solver. No inner genetic algorithm is
started for each outer individual.
"""

from copy import deepcopy
import json
import math
import multiprocessing
import os
import random
import time

import numpy as np

import foundation_config as cfg
import optimize_foundation_ga as legacy
from anchor_cage_solver import reset_anchor_solver_stats, get_anchor_solver_stats


_PARTITION_CONFIG = cfg.CONFIG["BILEVEL_OPT"]
OUTER_INDICES = tuple(_PARTITION_CONFIG["OUTER_INDICES"])
INNER_REBAR_INDICES = tuple(_PARTITION_CONFIG["INNER_REBAR_INDICES"])
ANCHOR_VARIABLES = tuple(_PARTITION_CONFIG["INNER_ANCHOR_VARIABLES"])

VARIABLE_PARTITION = {
    "outer": {
        "indices": OUTER_INDICES,
        "names": (
            "R_cap", "H_depth", "h1", "h2", "h3", "r_col",
            "L_pile", "D_pile", "n_pile_inner", "n_pile_outer",
            "pile_ring_inner_ratio", "pile_ring_outer_ratio",
        ),
    },
    "inner_rebar": {
        "indices": INNER_REBAR_INDICES,
        "names": (
            "n_bottom_radial", "d_bottom_through", "d_bottom_short",
            "d_bottom_ring_1", "d_bottom_ring_2", "n_top_radial",
            "d_top_radial", "d_top_ring", "n_pile_main", "d_pile_main",
        ),
    },
    "inner_anchor": {"names": ANCHOR_VARIABLES},
}


def split_full_gene(full_gene):
    return (
        [float(full_gene[i]) for i in OUTER_INDICES],
        [float(full_gene[i]) for i in INNER_REBAR_INDICES],
    )


def compose_full_gene(outer_gene, inner_rebar_gene):
    if len(outer_gene) != len(OUTER_INDICES):
        raise ValueError("outer gene must contain 12 variables")
    if len(inner_rebar_gene) != len(INNER_REBAR_INDICES):
        raise ValueError("inner rebar gene must contain 10 variables")
    full = [0.0] * len(legacy.VAR_RANGES)
    for index, value in zip(OUTER_INDICES, outer_gene):
        full[index] = float(value)
    for index, value in zip(INNER_REBAR_INDICES, inner_rebar_gene):
        full[index] = float(value)
    return full


def _rank(details):
    violation = float(details.get("total_violation", 1e10))
    cost = float(details.get("total_cost_rmb", details.get("objective", 1e10)))
    return (violation > 1e-4, violation if violation > 1e-4 else 0.0, cost)


def _inner_seed(fraction):
    result = []
    for index in INNER_REBAR_INDICES:
        low, high = legacy.VAR_RANGES[index]
        value = low + float(fraction) * (high - low)
        result.append(float(int(round(value))))
    return result


class DeterministicInnerSolver:
    """Coordinate-search reinforcement, then refine the anchor cage."""

    def __init__(self, fatigue_moment_ratio=0.10, pcd_samples=8,
                 coordinate_passes=2, seed_fractions=(0.25, 0.50, 0.75),
                 anchor_lengths=None):
        self.fatigue_moment_ratio = float(fatigue_moment_ratio)
        self.pcd_samples = int(pcd_samples)
        self.coordinate_passes = int(coordinate_passes)
        self.seed_fractions = tuple(float(x) for x in seed_fractions)
        configured_lengths = cfg.CONFIG["ANCHOR_OPT"]["VARIABLES"].get(
            "L_TOTAL_VALUES_M", [cfg.CONFIG["ANCHOR_OPT"]["FIXED"]["L_TOTAL_M"]])
        self.anchor_lengths = tuple(float(x) for x in (
            configured_lengths if anchor_lengths is None else anchor_lengths))
        if not self.anchor_lengths or any(x <= 0 for x in self.anchor_lengths):
            raise ValueError("anchor_lengths must contain positive values")
        self._evaluation_cache = {}
        self._solution_cache = {}
        self.stats = {
            "outer_calls": 0,
            "outer_cache_hits": 0,
            "combined_evaluations": 0,
            "coordinate_moves": 0,
        }

    def _set_mode(self, mode):
        legacy.ANCHOR_MODE = mode
        legacy.ANCHOR_FATIGUE_MOMENT_RATIO = self.fatigue_moment_ratio
        legacy.ANCHOR_PCD_SAMPLES = self.pcd_samples
        legacy.ANCHOR_FULL_PCD_GRID = False
        cfg.CONFIG["ANCHOR_OPT"]["FATIGUE_MOMENT_RATIO"] = self.fatigue_moment_ratio
        cfg.CONFIG["ANCHOR_OPT"]["SOLVER"]["PCD_PAIRS_PER_COMBO"] = self.pcd_samples
        cfg.CONFIG["ANCHOR_OPT"]["SOLVER"]["FULL_PCD_GRID"] = False
        cfg.CONFIG["ANCHOR_OPT"]["VARIABLES"]["L_TOTAL_VALUES_M"] = list(self.anchor_lengths)

    def _evaluate(self, full_gene, mode):
        key = (mode, tuple(round(float(x), 8) for x in full_gene),
               self.fatigue_moment_ratio, self.pcd_samples, self.anchor_lengths)
        if key not in self._evaluation_cache:
            self._set_mode(mode)
            self._evaluation_cache[key] = legacy.evaluate_detailed(full_gene)
            self.stats["combined_evaluations"] += 1
        return self._evaluation_cache[key]

    @staticmethod
    def _neighbor_values(full_index, current):
        low, high = legacy.VAR_RANGES[full_index]
        current = int(round(current))
        return [float(value) for value in (current - 1, current + 1)
                if low <= value <= high]

    def solve(self, outer_gene):
        self.stats["outer_calls"] += 1
        outer_key = tuple(round(float(x), 8) for x in outer_gene)
        if outer_key in self._solution_cache:
            self.stats["outer_cache_hits"] += 1
            return deepcopy(self._solution_cache[outer_key])

        local_optima = []
        for fraction in self.seed_fractions:
            inner = _inner_seed(fraction)
            full = compose_full_gene(outer_gene, inner)
            details = self._evaluate(full, "nested_2")
            for _ in range(max(1, self.coordinate_passes)):
                improved = False
                for position, full_index in enumerate(INNER_REBAR_INDICES):
                    best_inner = inner
                    best_details = details
                    for value in self._neighbor_values(full_index, inner[position]):
                        candidate = list(inner)
                        candidate[position] = value
                        candidate_details = self._evaluate(
                            compose_full_gene(outer_gene, candidate), "nested_2")
                        if _rank(candidate_details) < _rank(best_details):
                            best_inner, best_details = candidate, candidate_details
                    if best_inner is not inner:
                        inner, details = best_inner, best_details
                        self.stats["coordinate_moves"] += 1
                        improved = True
                if not improved:
                    break
            local_optima.append((inner, details))

        # Refine each deterministic local optimum with all configured anchor variables.
        refined = []
        for inner, _ in local_optima:
            details = self._evaluate(compose_full_gene(outer_gene, inner), "nested")
            refined.append((inner, details))
        inner, details = min(refined, key=lambda item: _rank(item[1]))
        result = {
            "outer_gene": list(outer_gene),
            "inner_rebar_gene": list(inner),
            "full_gene": compose_full_gene(outer_gene, inner),
            "details": deepcopy(details),
        }
        self._solution_cache[outer_key] = deepcopy(result)
        return result


_WORKER_INNER_SOLVER = None
_WORKER_SOLVER_SETTINGS = None
_WORKER_OUTER_CACHE_ENABLED = True


def _init_bilevel_worker(fatigue_moment_ratio, pcd_samples,
                         coordinate_passes, seed_fractions,
                         outer_cache_enabled, anchor_lengths):
    """Create one persistent deterministic inner solver per worker process."""
    global _WORKER_INNER_SOLVER, _WORKER_SOLVER_SETTINGS
    global _WORKER_OUTER_CACHE_ENABLED
    legacy.init_worker(True, None, "nested", fatigue_moment_ratio,
                       pcd_samples, False)
    _WORKER_SOLVER_SETTINGS = dict(
        fatigue_moment_ratio=fatigue_moment_ratio,
        pcd_samples=pcd_samples,
        coordinate_passes=coordinate_passes,
        seed_fractions=seed_fractions,
        anchor_lengths=anchor_lengths,
    )
    _WORKER_OUTER_CACHE_ENABLED = bool(outer_cache_enabled)
    _WORKER_INNER_SOLVER = DeterministicInnerSolver(**_WORKER_SOLVER_SETTINGS)


def _solve_outer_worker(outer_gene):
    solver = _WORKER_INNER_SOLVER
    if not _WORKER_OUTER_CACHE_ENABLED:
        # Fresh per outer request: keeps the necessary within-solve memoization,
        # but removes cross-generation/cross-individual solution reuse.
        solver = DeterministicInnerSolver(**_WORKER_SOLVER_SETTINGS)
    before = dict(solver.stats)
    result = solver.solve(outer_gene)
    after = solver.stats
    delta = {name: after[name] - before.get(name, 0) for name in after}
    return result, delta


def _mutate_outer(gene, rng, probability=0.20):
    child = list(gene)
    for position, full_index in enumerate(OUTER_INDICES):
        if rng.random() >= probability:
            continue
        low, high = legacy.VAR_RANGES[full_index]
        if full_index in legacy.DISCRETE_MAP:
            child[position] = float(max(low, min(high,
                int(round(child[position])) + rng.choice((-1, 1)))))
        else:
            child[position] = float(max(low, min(high,
                child[position] + rng.gauss(0.0, 0.08 * (high - low)))))
    return child


def run_bilevel_optimization(seed=0, pop_size=8, generations=2,
                             fatigue_moment_ratio=0.10, pcd_samples=8,
                             coordinate_passes=2, cores=1,
                             outer_cache_enabled=True, coarse_to_fine=False,
                             refine_top_k=8, anchor_lengths=None, verbose=True):
    """Run a compact outer GA whose every fitness is an optimized inner solution."""
    rng = random.Random(int(seed))
    random.seed(int(seed))
    np.random.seed(int(seed))
    legacy.init_worker(True, None, "nested", fatigue_moment_ratio, pcd_samples, False)
    reset_anchor_solver_stats()
    configured_lengths = cfg.CONFIG["ANCHOR_OPT"]["VARIABLES"].get(
        "L_TOTAL_VALUES_M", [cfg.CONFIG["ANCHOR_OPT"]["FIXED"]["L_TOTAL_M"]])
    use_anchor_lengths = tuple(float(x) for x in (
        configured_lengths if anchor_lengths is None else anchor_lengths))
    if not use_anchor_lengths or any(x <= 0 for x in use_anchor_lengths):
        raise ValueError("anchor_lengths must contain positive values")
    full_seed_fractions = tuple(cfg.CONFIG["BILEVEL_OPT"].get(
        "SEED_FRACTIONS", (0.25, 0.50, 0.75)))
    search_seed_fractions = ((0.50,) if coarse_to_fine else full_seed_fractions)
    search_pcd_samples = min(2, int(pcd_samples)) if coarse_to_fine else int(pcd_samples)
    search_coordinate_passes = 1 if coarse_to_fine else int(coordinate_passes)
    use_cores = max(1, min(int(cores), os.cpu_count() or 1))
    inner_solver = None
    pool = None
    if use_cores == 1:
        inner_solver = DeterministicInnerSolver(
            fatigue_moment_ratio=fatigue_moment_ratio,
            pcd_samples=search_pcd_samples,
            coordinate_passes=search_coordinate_passes,
            seed_fractions=search_seed_fractions,
            anchor_lengths=use_anchor_lengths,
        )
    else:
        pool = multiprocessing.Pool(
            processes=use_cores,
            initializer=_init_bilevel_worker,
            initargs=(fatigue_moment_ratio, search_pcd_samples,
                      search_coordinate_passes, search_seed_fractions,
                      outer_cache_enabled, use_anchor_lengths),
        )
    start = time.perf_counter()

    # Reuse the project's geometric/capacity pre-screening, then discard inner genes.
    initial_full = legacy.feasible_init_population(int(pop_size))
    population = [split_full_gene(ind)[0] for ind in initial_full]
    history = []
    evaluations = 0
    unique_evaluations = 0
    result_cache = {}
    candidate_archive = {}
    aggregate_inner_stats = {
        "outer_calls": 0,
        "outer_cache_hits": 0,
        "combined_evaluations": 0,
        "coordinate_moves": 0,
        "parent_result_cache_hits": 0,
    }

    def evaluate_population(genes):
        nonlocal evaluations, unique_evaluations
        evaluations += len(genes)
        keys = [tuple(round(float(x), 8) for x in gene) for gene in genes]
        if not outer_cache_enabled:
            if pool is None:
                solved = []
                for gene in genes:
                    fresh_solver = DeterministicInnerSolver(
                        fatigue_moment_ratio=fatigue_moment_ratio,
                        pcd_samples=search_pcd_samples,
                        coordinate_passes=search_coordinate_passes,
                        seed_fractions=search_seed_fractions,
                        anchor_lengths=use_anchor_lengths,
                    )
                    item = fresh_solver.solve(gene)
                    solved.append((item, dict(fresh_solver.stats)))
            else:
                solved = pool.map(_solve_outer_worker, [list(g) for g in genes])
            unique_evaluations += len(genes)
            for _, delta in solved:
                for name, value in delta.items():
                    aggregate_inner_stats[name] += value
            output = [deepcopy(item) for item, _ in solved]
            for item in output:
                key = tuple(round(float(x), 8) for x in item["outer_gene"])
                candidate_archive[key] = deepcopy(item)
            return output
        missing = []
        missing_keys = []
        pending = set()
        for key, gene in zip(keys, genes):
            if key in result_cache:
                aggregate_inner_stats["parent_result_cache_hits"] += 1
            elif key not in pending:
                missing_keys.append(key)
                missing.append(list(gene))
                pending.add(key)
            else:
                aggregate_inner_stats["parent_result_cache_hits"] += 1
        if missing:
            if pool is None:
                solved = []
                for gene in missing:
                    before = dict(inner_solver.stats)
                    item = inner_solver.solve(gene)
                    delta = {name: inner_solver.stats[name] - before.get(name, 0)
                             for name in inner_solver.stats}
                    solved.append((item, delta))
            else:
                solved = pool.map(_solve_outer_worker, missing)
            unique_evaluations += len(missing)
            for key, (item, delta) in zip(missing_keys, solved):
                result_cache[key] = item
                candidate_archive[key] = deepcopy(item)
                for name, value in delta.items():
                    aggregate_inner_stats[name] += value
        return [deepcopy(result_cache[key]) for key in keys]

    results = evaluate_population(population)
    for generation in range(int(generations) + 1):
        feasible = [x for x in results if _rank(x["details"])[0] is False]
        best = min(results, key=lambda x: _rank(x["details"]))
        history.append({
            "generation": generation,
            "outer_evaluations": evaluations,
            "feasible_count": len(feasible),
            "feasible_ratio": len(feasible) / len(results),
            "best_violation": best["details"]["total_violation"],
            "best_total_cost_rmb": best["details"]["total_cost_rmb"],
            "elapsed_sec": time.perf_counter() - start,
        })
        if generation >= int(generations):
            break
        order = sorted(range(len(population)), key=lambda i: _rank(results[i]["details"]))
        elite_count = max(1, len(population) // 5)
        next_population = [list(population[i]) for i in order[:elite_count]]
        while len(next_population) < len(population):
            parent_pool = order[:max(2, len(order) // 2)]
            first = population[rng.choice(parent_pool)]
            second = population[rng.choice(parent_pool)]
            cut = rng.randrange(1, len(OUTER_INDICES))
            child = list(first[:cut]) + list(second[cut:])
            next_population.append(_mutate_outer(child, rng))
        population = next_population
        results = evaluate_population(population)
        if verbose:
            print(f"[bilevel] generation={generation + 1} "
                  f"feasible={history[-1]['feasible_ratio']:.1%} "
                  f"best={history[-1]['best_total_cost_rmb']:.2f}")

    if pool is not None:
        pool.close()
        pool.join()

    search_best = min(results, key=lambda x: _rank(x["details"]))
    refinement = None
    if coarse_to_fine:
        top_k = max(1, min(int(refine_top_k), len(candidate_archive)))
        shortlisted = sorted(candidate_archive.values(),
                             key=lambda x: _rank(x["details"]))[:top_k]
        refine_genes = [item["outer_gene"] for item in shortlisted]
        refine_start = time.perf_counter()
        if use_cores == 1:
            refine_solver = DeterministicInnerSolver(
                fatigue_moment_ratio=fatigue_moment_ratio,
                pcd_samples=pcd_samples,
                coordinate_passes=coordinate_passes,
                seed_fractions=full_seed_fractions,
                anchor_lengths=use_anchor_lengths,
            )
            refined = [refine_solver.solve(gene) for gene in refine_genes]
            refine_stats = dict(refine_solver.stats)
        else:
            refine_pool = multiprocessing.Pool(
                processes=use_cores,
                initializer=_init_bilevel_worker,
                initargs=(fatigue_moment_ratio, pcd_samples,
                          coordinate_passes, full_seed_fractions, True,
                          use_anchor_lengths),
            )
            solved = refine_pool.map(_solve_outer_worker, refine_genes)
            refine_pool.close()
            refine_pool.join()
            refined = [item for item, _ in solved]
            refine_stats = {
                "outer_calls": sum(delta["outer_calls"] for _, delta in solved),
                "outer_cache_hits": sum(delta["outer_cache_hits"] for _, delta in solved),
                "combined_evaluations": sum(delta["combined_evaluations"] for _, delta in solved),
                "coordinate_moves": sum(delta["coordinate_moves"] for _, delta in solved),
            }
        best = min(refined, key=lambda x: _rank(x["details"]))
        refinement = {
            "shortlist_size": top_k,
            "runtime_sec": time.perf_counter() - refine_start,
            "search_best_cost_rmb": search_best["details"]["total_cost_rmb"],
            "refined_best_cost_rmb": best["details"]["total_cost_rmb"],
            "stats": refine_stats,
        }
    else:
        best = search_best
    best_details = best["details"]
    best_params = legacy.decode_gene(best["full_gene"])
    best_params["anchor"] = best_details["anchor"]
    return {
        "status": "feasible" if best_details["total_violation"] <= 1e-4 else "infeasible_best",
        "seed": int(seed),
        "partition": VARIABLE_PARTITION,
        "outer_variable_count": len(OUTER_INDICES),
        "inner_variable_count": len(INNER_REBAR_INDICES) + len(ANCHOR_VARIABLES),
        "best_outer_gene": best["outer_gene"],
        "best_inner_rebar_gene": best["inner_rebar_gene"],
        "best_full_gene": best["full_gene"],
        "best_anchor": best_details["anchor"],
        "best_params": best_params,
        "best_violation": best_details["total_violation"],
        "foundation_cost_rmb": best_details["foundation_cost_rmb"],
        "anchor_cost_rmb": best_details["anchor_cost_rmb"],
        "total_cost_rmb": best_details["total_cost_rmb"],
        "constraint_details": best_details["constraints"],
        "history": history,
        "outer_evaluations": evaluations,
        "outer_unique_evaluations": unique_evaluations,
        "inner_solver_stats": aggregate_inner_stats,
        "anchor_solver_stats": (
            get_anchor_solver_stats() if use_cores == 1
            else {"scope": "worker_local_not_aggregated"}
        ),
        "runtime_sec": time.perf_counter() - start,
        "cores": use_cores,
        "outer_cache_enabled": bool(outer_cache_enabled),
        "fatigue_moment_ratio": float(fatigue_moment_ratio),
        "pcd_samples": int(pcd_samples),
        "coordinate_passes": int(coordinate_passes),
        "anchor_lengths_m": list(use_anchor_lengths),
        "coarse_to_fine": bool(coarse_to_fine),
        "refinement": refinement,
    }


def to_jsonable(value):
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    return value


def dumps_result(result):
    return json.dumps(to_jsonable(result), ensure_ascii=False, indent=2)
