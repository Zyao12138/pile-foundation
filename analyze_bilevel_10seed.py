"""Aggregate the 10-seed comparison and generate presentation figures."""

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent
EXP = ROOT / "experiments_bilevel_10seed"
FIG = EXP / "ppt_figures"
FIG.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.sans-serif": ["Microsoft YaHei", "SimHei", "Arial Unicode MS", "DejaVu Sans"],
    "axes.unicode_minus": False,
    "font.size": 12,
    "axes.titlesize": 17,
    "axes.labelsize": 13,
    "legend.fontsize": 11,
    "figure.dpi": 150,
    "savefig.dpi": 300,
})

COLORS = {"direct": "#7A869A", "single": "#2F80ED", "two": "#F2994A"}


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def direct_doc(seed):
    if seed < 5:
        path = ROOT / f"experiments_anchor_ablation/ablation_nested_f0p10/seed_{seed:03d}/result.json"
    else:
        path = EXP / f"direct22plus4_10seed_extension/seed_{seed:03d}/result.json"
    return load(path)


def single_doc(seed):
    if seed < 5:
        return load(ROOT / f"experiments_bilevel_cache_ablation/cached_seed_{seed}.json")
    return load(EXP / f"single_stage_seed_{seed}.json")


def two_doc(seed):
    if seed < 5:
        name = "time_matched_seed1.json" if seed == 1 else f"time_matched_seed_{seed}.json"
        return load(ROOT / "experiments_bilevel_two_stage" / name)
    return load(EXP / f"two_stage_seed_{seed}.json")


seeds = list(range(10))
docs = {
    "direct": [direct_doc(i) for i in seeds],
    "single": [single_doc(i) for i in seeds],
    "two": [two_doc(i) for i in seeds],
}
costs = {name: np.array([d["total_cost_rmb"] for d in items]) for name, items in docs.items()}
runtimes = {name: np.array([d["runtime_sec"] for d in items]) for name, items in docs.items()}
feasible = {
    "direct": np.array([d.get("best_violation", d.get("violation", 1e10)) <= 1e-4 for d in docs["direct"]]),
    "single": np.array([d["best_violation"] <= 1e-4 for d in docs["single"]]),
    "two": np.array([d["best_violation"] <= 1e-4 for d in docs["two"]]),
}


def stats(name):
    values = costs[name]
    return {
        "feasible_count": int(feasible[name].sum()),
        "trial_count": len(values),
        "mean_cost_rmb": float(values.mean()),
        "median_cost_rmb": float(np.median(values)),
        "std_cost_rmb": float(values.std()),
        "min_cost_rmb": float(values.min()),
        "max_cost_rmb": float(values.max()),
        "mean_runtime_sec": float(runtimes[name].mean()),
    }


summary = {name: stats(name) for name in costs}
summary["two_vs_direct"] = {
    "paired_savings_percent": ((costs["direct"] - costs["two"]) / costs["direct"] * 100).tolist(),
    "mean_savings_percent": float(((costs["direct"] - costs["two"]) / costs["direct"] * 100).mean()),
    "median_savings_percent": float(np.median((costs["direct"] - costs["two"]) / costs["direct"] * 100)),
    "wins": int((costs["two"] < costs["direct"]).sum()),
}
summary["two_vs_single"] = {
    "paired_savings_percent": ((costs["single"] - costs["two"]) / costs["single"] * 100).tolist(),
    "mean_savings_percent": float(((costs["single"] - costs["two"]) / costs["single"] * 100).mean()),
    "median_savings_percent": float(np.median((costs["single"] - costs["two"]) / costs["single"] * 100)),
    "wins": int((costs["two"] < costs["single"]).sum()),
}
(EXP / "TEN_SEED_SUMMARY.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

with (EXP / "TEN_SEED_RESULTS.csv").open("w", newline="", encoding="utf-8-sig") as f:
    writer = csv.writer(f)
    writer.writerow(["seed", "direct22plus4_rmb", "single_stage_rmb", "two_stage_rmb", "two_vs_direct_percent"])
    savings = np.array(summary["two_vs_direct"]["paired_savings_percent"])
    for i in seeds:
        writer.writerow([i, costs["direct"][i], costs["single"][i], costs["two"][i], savings[i]])


def save(fig, name):
    fig.tight_layout()
    fig.savefig(FIG / name, bbox_inches="tight", facecolor="white")
    plt.close(fig)


# Figure 1: all ten paired results.
fig, ax = plt.subplots(figsize=(12, 6.4))
x = np.arange(10)
w = 0.25
ax.bar(x - w, costs["direct"] / 10000, w, label="直接22+4", color=COLORS["direct"])
ax.bar(x, costs["single"] / 10000, w, label="单阶段12+14", color=COLORS["single"])
ax.bar(x + w, costs["two"] / 10000, w, label="两阶段12+14", color=COLORS["two"])
ax.set_title("10个随机种子的同时间成本对比", weight="bold")
ax.set_xlabel("随机种子")
ax.set_ylabel("最优可行总成本（万元）")
ax.set_xticks(x, [str(i) for i in seeds])
ax.set_ylim(230, max(v.max() for v in costs.values()) / 10000 + 16)
ax.grid(axis="y", alpha=0.22)
ax.legend(frameon=False, ncol=3, loc="upper center")
save(fig, "01_ten_seed_cost_comparison.png")


# Figure 2: distribution as boxplot plus all observations.
fig, ax = plt.subplots(figsize=(9.5, 6.3))
data = [costs["direct"] / 10000, costs["single"] / 10000, costs["two"] / 10000]
bp = ax.boxplot(data, patch_artist=True, widths=0.52, tick_labels=["直接22+4", "单阶段双层", "两阶段双层"], showmeans=True,
                meanprops={"marker": "D", "markerfacecolor": "white", "markeredgecolor": "black", "markersize": 7})
for box, color in zip(bp["boxes"], [COLORS["direct"], COLORS["single"], COLORS["two"]]):
    box.set_facecolor(color)
    box.set_alpha(0.72)
for i, values in enumerate(data, start=1):
    jitter = np.linspace(-0.10, 0.10, len(values))
    ax.scatter(np.full(len(values), i) + jitter, values, color="#222222", s=25, alpha=0.72, zorder=3)
ax.set_title("10次独立运行的成本分布", weight="bold")
ax.set_ylabel("最优可行总成本（万元）")
ax.grid(axis="y", alpha=0.22)
save(fig, "02_ten_seed_cost_distribution.png")


# Figure 3: mean, median and standard deviation.
names = ["direct", "single", "two"]
labels = ["直接22+4", "单阶段双层", "两阶段双层"]
means = np.array([summary[n]["mean_cost_rmb"] for n in names]) / 10000
medians = np.array([summary[n]["median_cost_rmb"] for n in names]) / 10000
stds = np.array([summary[n]["std_cost_rmb"] for n in names]) / 10000
fig, ax = plt.subplots(figsize=(9.7, 6.3))
x = np.arange(3)
w = 0.34
b1 = ax.bar(x - w / 2, means, w, yerr=stds, capsize=6, label="平均成本±标准差",
            color=[COLORS[n] for n in names])
b2 = ax.bar(x + w / 2, medians, w, label="中位成本", color=[COLORS[n] for n in names], alpha=0.42,
            edgecolor="black", linewidth=0.7)
for bars in (b1, b2):
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.2, f"{bar.get_height():.2f}", ha="center", fontsize=10)
ax.set_title("10种子汇总成本与离散程度", weight="bold")
ax.set_ylabel("总成本（万元）")
ax.set_xticks(x, labels)
ax.set_ylim(230, max((means + stds).max(), medians.max()) + 14)
ax.grid(axis="y", alpha=0.22)
ax.legend(frameon=False, ncol=2)
save(fig, "03_ten_seed_summary.png")


# Figure 4: paired savings versus direct.
saving = (costs["direct"] - costs["two"]) / costs["direct"] * 100
fig, ax = plt.subplots(figsize=(11, 5.8))
bar_colors = ["#27AE60" if v >= 0 else "#EB5757" for v in saving]
bars = ax.bar(x := np.arange(10), saving, color=bar_colors, width=0.65)
ax.axhline(0, color="#333333", linewidth=1)
ax.axhline(saving.mean(), color=COLORS["two"], linestyle="--", linewidth=2,
           label=f"10次平均降本 {saving.mean():.2f}%")
ax.set_title("两阶段双层相对直接22+4的逐种子降本率", weight="bold")
ax.set_xlabel("随机种子")
ax.set_ylabel("降本率（%）")
ax.set_xticks(x, [str(i) for i in seeds])
ax.set_ylim(min(-8, saving.min() - 2), max(10, saving.max() + 2))
ax.grid(axis="y", alpha=0.22)
ax.legend(frameon=False)
for bar, value in zip(bars, saving):
    ax.text(bar.get_x() + bar.get_width() / 2, value + (0.3 if value >= 0 else -0.65), f"{value:.1f}%",
            ha="center", va="bottom" if value >= 0 else "top", fontsize=9)
save(fig, "04_ten_seed_paired_savings.png")


print(json.dumps(summary, ensure_ascii=False, indent=2))
