# -*- coding: utf-8 -*-
"""
批量实验入口。

用途：
    1. 不依赖人工 input()，直接调用优化入口。
    2. 固定随机种子，保存每一代统计数据和最终结果。
    3. 为后续算法消融实验提供统一的比较基线。
"""

import argparse
import csv
import json
import math
import multiprocessing
import os
import statistics
import traceback
from datetime import datetime

import numpy as np

import optimize_foundation_ga as opt


HISTORY_COLUMNS = [
    "generation",
    "elapsed_sec",
    "ga_evaluations",
    "best_objective",
    "best_feasible_objective",
    "best_violation",
    "mean_violation",
    "feasible_count",
    "feasible_ratio",
    "epsilon",
]


def to_jsonable(value):
    """递归转换 NumPy 类型，确保结果可以写入 JSON。"""
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    return value


def write_history(path, history):
    with open(path, "w", encoding="utf-8", newline="") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=HISTORY_COLUMNS)
        writer.writeheader()
        for row in history:
            writer.writerow({column: row.get(column) for column in HISTORY_COLUMNS})


def write_json(path, payload):
    with open(path, "w", encoding="utf-8") as file_obj:
        json.dump(to_jsonable(payload), file_obj, ensure_ascii=False, indent=2)


def collect_summary(results):
    successful = [
        result for result in results
        if result.get("best_objective") is not None
        and result.get("status") != "error"
    ]
    feasible = [result for result in successful if result.get("best_feasible")]

    objectives = [float(result["best_objective"]) for result in successful]
    runtimes = [float(result["runtime_sec"]) for result in successful]
    evaluations = [int(result["ga_evaluations"]) for result in successful]
    final_ratios = [
        float(result["history"][-1]["feasible_ratio"])
        for result in successful
        if result.get("history")
    ]

    summary = {
        "n_trials": len(results),
        "n_successful": len(successful),
        "n_feasible": len(feasible),
        "feasible_rate": float(len(feasible) / len(results)) if results else 0.0,
        "best_objective": min(objectives) if objectives else None,
        "mean_objective": statistics.mean(objectives) if objectives else None,
        "std_objective": statistics.stdev(objectives) if len(objectives) > 1 else 0.0,
        "worst_objective": max(objectives) if objectives else None,
        "mean_runtime_sec": statistics.mean(runtimes) if runtimes else None,
        "mean_ga_evaluations": statistics.mean(evaluations) if evaluations else None,
        "mean_final_feasible_ratio": (
            statistics.mean(final_ratios) if final_ratios else None
        ),
    }
    for name in ("foundation_cost_rmb", "anchor_cost_rmb", "total_cost_rmb"):
        values = [float(r[name]) for r in successful if r.get(name) is not None]
        feasible_values = [float(r[name]) for r in feasible if r.get(name) is not None]
        summary["mean_" + name] = statistics.mean(values) if values else None
        summary["best_feasible_" + name] = min(feasible_values) if feasible_values else None
    return summary


def parse_args():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    parser = argparse.ArgumentParser(description="风电桩基优化批量实验入口")
    parser.add_argument("--variant", default="baseline", help="实验版本名称")
    parser.add_argument("--seeds", nargs="+", type=int, default=[0], help="随机种子列表")
    parser.add_argument("--pop-size", type=int, default=None, help="覆盖种群规模")
    parser.add_argument("--generations", type=int, default=None, help="覆盖进化代数")
    parser.add_argument("--cores", type=int, default=16, help="并行核心数")
    parser.add_argument(
        "--allow-tension",
        choices=["yes", "no"],
        default="yes",
        help="是否允许出现受拉桩",
    )
    parser.add_argument(
        "--fixed-pile-length",
        type=float,
        default=None,
        help="锁定桩长；不提供则由算法寻优",
    )
    parser.add_argument(
        "--anchor-mode",
        choices=["fixed", "nested_2", "nested"],
        default="fixed",
        help="fixed基线、nested_2仅优化规格数量、nested完整22+4",
    )
    parser.add_argument(
        "--fatigue-moment-ratio",
        type=float,
        default=None,
        help="研究用疲劳弯矩幅/极限弯矩比；正式设计须改用厂家疲劳谱",
    )
    parser.add_argument(
        "--anchor-pcd-samples",
        type=int,
        default=None,
        help="完整nested中每个规格/数量组合的PCD样本对数",
    )
    parser.add_argument(
        "--anchor-full-pcd-grid",
        action="store_true",
        help="完整nested使用剪枝后的全部PCD网格，仅建议用于收敛性检查",
    )
    parser.add_argument(
        "--output-dir",
        default=os.path.join(script_dir, "experiments"),
        help="实验结果输出目录",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="显示优化过程输出",
    )
    parser.add_argument(
        "--adaptive-epsilon",
        action="store_true",
        help="启用自适应 epsilon 约束处理",
    )
    parser.add_argument("--epsilon0", type=float, default=0.10, help="初始 epsilon")
    parser.add_argument("--epsilon-power", type=float, default=2.0, help="epsilon 衰减指数")
    parser.add_argument(
        "--epsilon-decay-fraction",
        type=float,
        default=1.0,
        help="epsilon 在前多少比例的代数内衰减到 0",
    )
    parser.add_argument(
        "--mixed-variables",
        action="store_true",
        help="启用连续变量与离散工程变量混合算子",
    )
    parser.add_argument(
        "--mixed-mutation-only",
        action="store_true",
        help="仅替换离散变量变异，交叉仍使用原始两点交叉",
    )
    parser.add_argument(
        "--mixed-mutation-indpb",
        type=float,
        default=0.15,
        help="混合变异的单变量扰动概率",
    )
    parser.add_argument(
        "--mixed-mutation-step",
        type=int,
        default=2,
        help="离散变量单次最大跳跃档位数",
    )
    parser.add_argument(
        "--feasible-local-search",
        action="store_true",
        help="启用严格可行坐标局部搜索",
    )
    parser.add_argument(
        "--local-search-passes",
        type=int,
        default=6,
        help="坐标局部搜索最大轮数",
    )
    parser.add_argument(
        "--local-search-max-evaluations",
        type=int,
        default=1200,
        help="坐标局部搜索最大评估次数",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    variant_dir = os.path.join(args.output_dir, args.variant)
    os.makedirs(variant_dir, exist_ok=True)

    metadata = {
        "variant": args.variant,
        "seeds": args.seeds,
        "pop_size": args.pop_size,
        "generations": args.generations,
        "cores": args.cores,
        "allow_tension": args.allow_tension == "yes",
        "fixed_pile_length": args.fixed_pile_length,
        "anchor_mode": args.anchor_mode,
        "fatigue_moment_ratio": args.fatigue_moment_ratio,
        "anchor_pcd_samples": args.anchor_pcd_samples,
        "anchor_full_pcd_grid": args.anchor_full_pcd_grid,
        "anchor_config": opt.cfg.CONFIG["ANCHOR_OPT"],
        "adaptive_epsilon": args.adaptive_epsilon,
        "epsilon0": args.epsilon0,
        "epsilon_power": args.epsilon_power,
        "epsilon_decay_fraction": args.epsilon_decay_fraction,
        "mixed_variables": args.mixed_variables,
        "mixed_mutation_only": args.mixed_mutation_only,
        "mixed_mutation_indpb": args.mixed_mutation_indpb,
        "mixed_mutation_step": args.mixed_mutation_step,
        "feasible_local_search": args.feasible_local_search,
        "local_search_passes": args.local_search_passes,
        "local_search_max_evaluations": args.local_search_max_evaluations,
        "started_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    write_json(os.path.join(variant_dir, "run_config.json"), metadata)

    results = []
    for seed in args.seeds:
        run_dir = os.path.join(variant_dir, f"seed_{seed:03d}")
        os.makedirs(run_dir, exist_ok=True)
        started = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        print(f"[{args.variant}] seed={seed} 开始运行")
        try:
            result = opt.run_optimization(
                allow_tension=args.allow_tension == "yes",
                fixed_pile_length=args.fixed_pile_length,
                seed=seed,
                pop_size=args.pop_size,
                n_gen=args.generations,
                cores=args.cores,
                verbose=args.verbose,
                save_artifacts=False,
                use_adaptive_epsilon=args.adaptive_epsilon,
                epsilon0=args.epsilon0,
                epsilon_power=args.epsilon_power,
                epsilon_decay_fraction=args.epsilon_decay_fraction,
                use_mixed_variables=args.mixed_variables,
                mixed_crossover=not args.mixed_mutation_only,
                mixed_mutation_indpb=args.mixed_mutation_indpb,
                mixed_mutation_step=args.mixed_mutation_step,
                use_feasible_local_search=args.feasible_local_search,
                local_search_passes=args.local_search_passes,
                local_search_max_evaluations=args.local_search_max_evaluations,
                anchor_mode=args.anchor_mode,
                fatigue_moment_ratio=args.fatigue_moment_ratio,
                anchor_pcd_samples=args.anchor_pcd_samples,
                anchor_full_pcd_grid=args.anchor_full_pcd_grid,
            )
        except Exception as exc:
            result = {
                "seed": seed,
                "status": "error",
                "error": str(exc),
                "traceback": traceback.format_exc(),
                "best_objective": None,
                "best_violation": None,
                "best_feasible": False,
                "best_params": None,
                "runtime_sec": None,
                "ga_evaluations": 0,
                "history": [],
            }

        result["variant"] = args.variant
        result["started_at"] = started
        result["finished_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        write_history(
            os.path.join(run_dir, "history.csv"),
            result.get("history", []),
        )
        write_json(os.path.join(run_dir, "result.json"), result)
        results.append(result)

        if result["status"] == "error":
            print(f"[{args.variant}] seed={seed} 失败: {result['error']}")
        else:
            print(
                f"[{args.variant}] seed={seed} 完成 "
                f"status={result['status']} "
                f"objective={result['best_objective']} "
                f"violation={result['best_violation']}"
            )

    summary = collect_summary(results)
    summary["finished_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    write_json(os.path.join(variant_dir, "summary.json"), summary)

    summary_path = os.path.join(variant_dir, "summary.csv")
    with open(summary_path, "w", encoding="utf-8-sig", newline="") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=list(summary.keys()))
        writer.writeheader()
        writer.writerow(summary)

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if summary["n_successful"] == summary["n_trials"] else 1


if __name__ == "__main__":
    multiprocessing.freeze_support()
    raise SystemExit(main())
