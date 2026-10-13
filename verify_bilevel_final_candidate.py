"""Recheck a saved bilevel candidate with the complete anchor PCD grid."""

import argparse
import json
import time
from pathlib import Path

import optimize_foundation_ga as legacy
from anchor_cage_solver import get_anchor_solver_stats, reset_anchor_solver_stats
from bilevel_foundation_optimizer import to_jsonable


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    source_path = Path(args.input)
    source = json.loads(source_path.read_text(encoding="utf-8"))
    gene = source["best_full_gene"]
    fatigue_ratio = float(source.get("fatigue_moment_ratio", 0.10))

    legacy.init_worker(True, None, "nested", fatigue_ratio, 1, True)
    reset_anchor_solver_stats()
    start = time.perf_counter()
    details = legacy.evaluate_detailed(gene)
    runtime = time.perf_counter() - start

    constraints = details.get("constraints", {})
    failed = {
        name: item for name, item in constraints.items()
        if float(item.get("violation", 0.0)) > 1e-9
    }
    result = {
        "source_file": str(source_path),
        "verification_mode": "nested_complete_pcd_grid",
        "pcd_step_m": 0.05,
        "fatigue_moment_ratio": fatigue_ratio,
        "best_full_gene": gene,
        "sampled_search": {
            "anchor": source["best_anchor"],
            "foundation_cost_rmb": source["foundation_cost_rmb"],
            "anchor_cost_rmb": source["anchor_cost_rmb"],
            "total_cost_rmb": source["total_cost_rmb"],
            "total_violation": source["best_violation"],
        },
        "full_grid": {
            "status": details.get("status"),
            "anchor": details.get("anchor"),
            "foundation_cost_rmb": details.get("foundation_cost_rmb"),
            "anchor_cost_rmb": details.get("anchor_cost_rmb"),
            "total_cost_rmb": details.get("total_cost_rmb"),
            "total_violation": details.get("total_violation"),
            "constraint_count": len(constraints),
            "failed_constraint_count": len(failed),
            "failed_constraints": failed,
            "constraints": constraints,
        },
        "anchor_solver_stats": get_anchor_solver_stats(),
        "runtime_sec": runtime,
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(to_jsonable(result), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps({
        "status": details.get("status"),
        "total_violation": details.get("total_violation"),
        "total_cost_rmb": details.get("total_cost_rmb"),
        "anchor": details.get("anchor"),
        "failed_constraint_count": len(failed),
        "anchor_solver_stats": get_anchor_solver_stats(),
        "runtime_sec": runtime,
        "output": str(output_path),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
