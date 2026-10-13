"""Generate presentation-ready figures from the saved bilevel experiments."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "experiments_bilevel_two_stage" / "ppt_figures"
OUT.mkdir(parents=True, exist_ok=True)

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

COLORS = {"direct": "#7A869A", "single": "#2F80ED", "two": "#F2994A", "final": "#27AE60"}


def load(path):
    return json.loads((ROOT / path).read_text(encoding="utf-8"))


def save(fig, name):
    fig.tight_layout()
    fig.savefig(OUT / name, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def add_labels(ax, bars, fmt="{:.1f}", dy=0):
    for bar in bars:
        value = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, value + dy, fmt.format(value),
                ha="center", va="bottom", fontsize=10)


direct_docs = [load(f"experiments_anchor_ablation/ablation_nested_f0p10/seed_{i:03d}/result.json") for i in range(5)]
direct = np.array([d["total_cost_rmb"] for d in direct_docs]) / 10000
single = np.array([load(f"experiments_bilevel_cache_ablation/cached_seed_{i}.json")["total_cost_rmb"] for i in range(5)]) / 10000
two_paths = [
    "experiments_bilevel_two_stage/time_matched_seed_0.json",
    "experiments_bilevel_two_stage/time_matched_seed1.json",
    "experiments_bilevel_two_stage/time_matched_seed_2.json",
    "experiments_bilevel_two_stage/time_matched_seed_3.json",
    "experiments_bilevel_two_stage/time_matched_seed_4.json",
]
two_docs = [load(p) for p in two_paths]
two = np.array([d["total_cost_rmb"] for d in two_docs]) / 10000


# 1. Representative convergence and feasibility history.
d3 = two_docs[3]
history = d3["history"]
gen = np.array([h["generation"] for h in history])
cost = np.array([h["best_total_cost_rmb"] for h in history]) / 10000
ratio = np.array([h["feasible_ratio"] for h in history]) * 100
refined = d3["refinement"]["refined_best_cost_rmb"] / 10000

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7.2), sharex=True, gridspec_kw={"height_ratios": [1.45, 1]})
ax1.plot(gen, cost, color=COLORS["two"], marker="o", linewidth=2.6, markersize=7, label="粗搜索阶段最优成本")
ax1.scatter([gen[-1] + 1], [refined], s=90, color=COLORS["final"], zorder=4, label="前8名完整精修")
ax1.plot([gen[-1], gen[-1] + 1], [cost[-1], refined], color=COLORS["final"], linestyle="--", linewidth=2)
ax1.set_ylabel("最优总成本（万元）")
ax1.set_title("两阶段双层优化收敛过程（种子3）", weight="bold")
ax1.grid(axis="y", alpha=0.25)
ax1.legend(frameon=False)
ax1.annotate(f"精修后 {refined:.2f} 万元", (gen[-1] + 1, refined), xytext=(-125, 8),
             textcoords="offset points", color=COLORS["final"], fontsize=11)

ax2.plot(gen, ratio, color=COLORS["single"], marker="s", linewidth=2.3, markersize=6)
ax2.set_xlabel("外层代数（最右侧为精修阶段）")
ax2.set_ylabel("种群可行率（%）")
ax2.set_xticks(np.arange(0, gen[-1] + 2))
ax2.set_xticklabels([str(i) for i in gen] + ["精修"])
ax2.set_ylim(0, 100)
ax2.grid(axis="y", alpha=0.25)
save(fig, "01_two_stage_convergence.png")


# 2. Seed-by-seed comparison.
fig, ax = plt.subplots(figsize=(10.5, 6.2))
x = np.arange(5)
w = 0.24
b1 = ax.bar(x - w, direct, w, label="直接22+4", color=COLORS["direct"])
b2 = ax.bar(x, single, w, label="单阶段12+14", color=COLORS["single"])
b3 = ax.bar(x + w, two, w, label="两阶段12+14", color=COLORS["two"])
ax.set_title("5个随机种子的同时间成本对比", weight="bold")
ax.set_xlabel("随机种子")
ax.set_ylabel("最优可行总成本（万元）")
ax.set_xticks(x, [str(i) for i in range(5)])
ax.set_ylim(230, max(direct.max(), single.max(), two.max()) + 18)
ax.grid(axis="y", alpha=0.22)
ax.legend(frameon=False, ncol=3, loc="upper center")
add_labels(ax, b3, "{:.1f}", 1.2)
save(fig, "02_multiseed_cost_comparison.png")


# 3. Aggregate mean and median comparison.
means = np.array([direct.mean(), single.mean(), two.mean()])
medians = np.array([np.median(direct), np.median(single), np.median(two)])
labels = ["直接22+4", "单阶段双层", "两阶段双层"]
fig, ax = plt.subplots(figsize=(9.5, 6.1))
x = np.arange(3)
w = 0.34
b1 = ax.bar(x - w / 2, means, w, label="平均成本", color=[COLORS["direct"], COLORS["single"], COLORS["two"]], alpha=0.95)
b2 = ax.bar(x + w / 2, medians, w, label="中位成本", color=[COLORS["direct"], COLORS["single"], COLORS["two"]], alpha=0.48, edgecolor="black", linewidth=0.6)
ax.set_title("两阶段方法同时降低平均成本与中位成本", weight="bold")
ax.set_ylabel("总成本（万元）")
ax.set_xticks(x, labels)
ax.set_ylim(240, 285)
ax.grid(axis="y", alpha=0.22)
ax.legend(frameon=False, ncol=2)
add_labels(ax, b1, "{:.2f}", 0.7)
add_labels(ax, b2, "{:.2f}", 0.7)
ax.text(2, 242.5, "相对直接22+4：平均降低3.81%，中位降低6.16%", ha="center", color="#A65300", fontsize=11)
save(fig, "03_aggregate_cost_comparison.png")


# 4. Paired savings by seed.
saving = (direct - two) / direct * 100
fig, ax = plt.subplots(figsize=(9.8, 5.8))
bar_colors = [COLORS["final"] if v >= 0 else "#EB5757" for v in saving]
bars = ax.bar(np.arange(5), saving, color=bar_colors, width=0.62)
ax.axhline(0, color="#333333", linewidth=1)
ax.axhline(saving.mean(), color=COLORS["two"], linestyle="--", linewidth=2, label=f"平均降本 {saving.mean():.2f}%")
ax.set_title("两阶段双层相对直接22+4的逐种子降本率", weight="bold")
ax.set_xlabel("随机种子")
ax.set_ylabel("降本率（%）")
ax.set_xticks(np.arange(5), [str(i) for i in range(5)])
ax.set_ylim(min(-4.5, saving.min() - 1), saving.max() + 2)
ax.grid(axis="y", alpha=0.22)
ax.legend(frameon=False)
for bar, value in zip(bars, saving):
    ax.text(bar.get_x() + bar.get_width() / 2, value + (0.25 if value >= 0 else -0.55),
            f"{value:.2f}%", ha="center", va="bottom" if value >= 0 else "top", fontsize=11)
save(fig, "04_paired_savings.png")


# 5. Sampled versus full-grid final candidate cost breakdown.
final_check = load("experiments_bilevel_two_stage/final_candidate_full_grid.json")
sample = final_check["sampled_search"]
full = final_check["full_grid"]
foundation = np.array([sample["foundation_cost_rmb"], full["foundation_cost_rmb"]]) / 10000
anchor = np.array([sample["anchor_cost_rmb"], full["anchor_cost_rmb"]]) / 10000
fig, ax = plt.subplots(figsize=(8.4, 6.2))
x = np.arange(2)
b1 = ax.bar(x, foundation, width=0.55, color=COLORS["single"], label="基础主体")
b2 = ax.bar(x, anchor, width=0.55, bottom=foundation, color=COLORS["two"], label="锚栓系统")
totals = foundation + anchor
for i, value in enumerate(totals):
    ax.text(i, value + 2, f"{value:.2f}万元", ha="center", fontsize=12, weight="bold")
ax.set_title("最终候选完整PCD网格复核", weight="bold")
ax.set_ylabel("总成本（万元）")
ax.set_xticks(x, ["8组PCD采样", "18,001个候选完整网格"])
ax.set_ylim(0, max(totals) + 28)
ax.legend(frameon=False, ncol=2, loc="upper center")
ax.text(0.5, 18, "完整网格再降低0.77万元", ha="center", color=COLORS["final"], fontsize=12)
save(fig, "05_full_grid_cost_breakdown.png")


# 6. Governing constraint utilizations.
constraints = full["constraints"]
selected = [
    ("锚栓疲劳应力", constraints["anchor_fatigue_stress"]["value"] / constraints["anchor_fatigue_stress"]["limit"]),
    ("锚栓极限拉力", constraints["anchor_uls_tension"]["value"] / constraints["anchor_uls_tension"]["limit"]),
    ("桩抗弯承载", constraints["pile_bending_uls"]["utilization"]),
    ("桩负摩阻力", constraints["pile_negative_friction_uls"]["utilization"]),
    ("最小桩间距", constraints["pile_spacing_minimum"]["utilization"]),
    ("桩边距", constraints["pile_edge_clearance"]["utilization"]),
]
selected.sort(key=lambda item: item[1])
names = [x[0] for x in selected]
values = np.array([x[1] for x in selected]) * 100
fig, ax = plt.subplots(figsize=(9.4, 5.8))
colors = ["#EB5757" if v >= 95 else COLORS["two"] if v >= 85 else COLORS["single"] for v in values]
bars = ax.barh(names, values, color=colors, height=0.62)
ax.axvline(100, color="#333333", linestyle="--", linewidth=1.5, label="约束限值")
ax.set_title("最终候选的主要控制约束", weight="bold")
ax.set_xlabel("约束利用率（%）")
ax.set_xlim(0, 106)
ax.grid(axis="x", alpha=0.22)
ax.legend(frameon=False, loc="lower right")
for bar, value in zip(bars, values):
    ax.text(value + 1, bar.get_y() + bar.get_height() / 2, f"{value:.1f}%", va="center", fontsize=11)
save(fig, "06_governing_constraints.png")

print(f"Generated 6 figures in: {OUT}")
