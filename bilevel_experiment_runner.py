# -*- coding: utf-8 -*-
"""Command-line runner for the 12 outer + 14 inner bilevel optimizer."""

import argparse
import json
from pathlib import Path

from bilevel_foundation_optimizer import run_bilevel_optimization, to_jsonable


def main():
    parser = argparse.ArgumentParser(description="风机桩基础12+14双层优化")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--pop-size", type=int, default=8)
    parser.add_argument("--generations", type=int, default=2)
    parser.add_argument("--fatigue-moment-ratio", type=float, default=0.10)
    parser.add_argument("--pcd-samples", type=int, default=8)
    parser.add_argument("--coordinate-passes", type=int, default=2)
    parser.add_argument(
        "--anchor-lengths", type=float, nargs="+", default=None,
        help="锚栓总长度候选；例如对照组使用 --anchor-lengths 4.0",
    )
    parser.add_argument("--cores", type=int, default=1)
    parser.add_argument(
        "--disable-outer-cache",
        action="store_true",
        help="关闭跨代外层方案缓存，用于缓存消融；内层单次求解仍保留必要记忆化",
    )
    parser.add_argument(
        "--coarse-to-fine",
        action="store_true",
        help="用轻量内层探索外层，再对排名靠前的外层方案完整精修",
    )
    parser.add_argument("--refine-top-k", type=int, default=8)
    parser.add_argument("--output", default="experiments_bilevel/smoke_result.json")
    args = parser.parse_args()
    result = run_bilevel_optimization(
        seed=args.seed,
        pop_size=args.pop_size,
        generations=args.generations,
        fatigue_moment_ratio=args.fatigue_moment_ratio,
        pcd_samples=args.pcd_samples,
        coordinate_passes=args.coordinate_passes,
        cores=args.cores,
        outer_cache_enabled=not args.disable_outer_cache,
        coarse_to_fine=args.coarse_to_fine,
        refine_top_k=args.refine_top_k,
        anchor_lengths=args.anchor_lengths,
    )
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(to_jsonable(result), ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({
        "status": result["status"],
        "total_cost_rmb": result["total_cost_rmb"],
        "best_violation": result["best_violation"],
        "runtime_sec": result["runtime_sec"],
        "outer_evaluations": result["outer_evaluations"],
        "outer_unique_evaluations": result["outer_unique_evaluations"],
        "cores": result["cores"],
        "outer_cache_enabled": result["outer_cache_enabled"],
        "coarse_to_fine": result["coarse_to_fine"],
        "refinement": result["refinement"],
        "inner_solver_stats": result["inner_solver_stats"],
        "output": str(path),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
