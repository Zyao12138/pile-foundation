"""Aggregate the time-matched 4-variable versus 5-variable anchor comparison."""

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent
BASE = ROOT / "experiments_anchor_5var" / "comparison"
FIG = BASE / "ppt_figures"
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


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def four_doc(seed):
    if seed < 5:
        name = "time_matched_seed1.json" if seed == 1 else f"time_matched_seed_{seed}.json"
        return load(ROOT / "experiments_bilevel_two_stage" / name)
    return load(ROOT / f"experiments_bilevel_10seed/two_stage_seed_{seed}.json")


four_docs = [four_doc(i) for i in range(10)]
five_docs = [load(BASE / f"time_matched_final_seed_{i}.json") for i in range(10)]
four = np.array([d["total_cost_rmb"] for d in four_docs])
five = np.array([d["total_cost_rmb"] for d in five_docs])
four_time = np.array([d["runtime_sec"] for d in four_docs])
five_time = np.array([d["runtime_sec"] for d in five_docs])
lengths = np.array([d["best_anchor"]["L_total_m"] for d in five_docs])
savings = (four - five) / four * 100


def stats(values, times):
    return {
        "feasible_count": 10,
        "trial_count": 10,
        "mean_cost_rmb": float(values.mean()),
        "median_cost_rmb": float(np.median(values)),
        "std_cost_rmb": float(values.std()),
        "mean_runtime_sec": float(times.mean()),
    }


summary = {
    "four_variable": stats(four, four_time),
    "five_variable": stats(five, five_time),
    "five_vs_four": {
        "paired_savings_percent": savings.tolist(),
        "mean_savings_percent": float(savings.mean()),
        "median_savings_percent": float(np.median(savings)),
        "wins": int((savings > 0).sum()),
        "selected_lengths_m": lengths.tolist(),
    },
    "settings": {
        "four_variable": "population=20, generations=5, refine_top_k=8",
        "five_variable": "population=16, generations=4, refine_top_k=4",
        "reason": "time-matched wall-clock budget",
    },
}
(BASE / "SUMMARY_4VAR_VS_5VAR.json").write_text(
    json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

with (BASE / "RESULTS_4VAR_VS_5VAR.csv").open("w", newline="", encoding="utf-8-sig") as f:
    writer = csv.writer(f)
    writer.writerow(["seed", "four_variable_rmb", "five_variable_rmb", "five_vs_four_savings_percent",
                     "selected_length_m", "four_runtime_sec", "five_runtime_sec"])
    for i in range(10):
        writer.writerow([i, four[i], five[i], savings[i], lengths[i], four_time[i], five_time[i]])


def save(fig, name):
    fig.tight_layout()
    fig.savefig(FIG / name, bbox_inches="tight", facecolor="white")
    plt.close(fig)


# 1. Paired costs.
fig, ax = plt.subplots(figsize=(11.5, 6.2))
x = np.arange(10)
w = 0.36
ax.bar(x - w / 2, four / 10000, w, color="#2F80ED", label="4变量锚栓")
ax.bar(x + w / 2, five / 10000, w, color="#F2994A", label="5变量锚栓")
ax.set_title("4变量与5变量锚栓优化的10种子同时间对比", weight="bold")
ax.set_xlabel("随机种子")
ax.set_ylabel("最优可行总成本（万元）")
ax.set_xticks(x, [str(i) for i in range(10)])
ax.set_ylim(240, max(four.max(), five.max()) / 10000 + 12)
ax.grid(axis="y", alpha=0.22)
ax.legend(frameon=False, ncol=2, loc="upper center")
save(fig, "01_4var_vs_5var_cost.png")


# 2. Aggregate summary.
means = np.array([four.mean(), five.mean()]) / 10000
medians = np.array([np.median(four), np.median(five)]) / 10000
stds = np.array([four.std(), five.std()]) / 10000
fig, ax = plt.subplots(figsize=(8.8, 6.1))
x = np.arange(2)
w = 0.34
b1 = ax.bar(x - w / 2, means, w, yerr=stds, capsize=6, color=["#2F80ED", "#F2994A"], label="平均成本±标准差")
b2 = ax.bar(x + w / 2, medians, w, color=["#9DC1F4", "#F7C999"], edgecolor="#555555", label="中位成本")
for bars in (b1, b2):
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.0,
                f"{bar.get_height():.2f}", ha="center", fontsize=11)
ax.set_title("5变量在同时间预算下未降低总体成本", weight="bold")
ax.set_ylabel("总成本（万元）")
ax.set_xticks(x, ["4变量锚栓", "5变量锚栓"])
ax.set_ylim(240, max((means + stds).max(), medians.max()) + 12)
ax.grid(axis="y", alpha=0.22)
ax.legend(frameon=False)
save(fig, "02_4var_vs_5var_summary.png")


# 3. Paired savings.
fig, ax = plt.subplots(figsize=(10.8, 5.8))
colors = ["#27AE60" if v > 0 else "#EB5757" for v in savings]
bars = ax.bar(np.arange(10), savings, color=colors, width=0.65)
ax.axhline(0, color="#333333", linewidth=1)
ax.axhline(savings.mean(), color="#F2994A", linestyle="--", linewidth=2,
           label=f"平均变化：成本升高 {-savings.mean():.2f}%")
ax.set_title("5变量相对4变量的逐种子成本变化", weight="bold")
ax.set_xlabel("随机种子")
ax.set_ylabel("降本率（%）")
ax.set_xticks(np.arange(10), [str(i) for i in range(10)])
ax.set_ylim(min(-10, savings.min() - 2), max(10, savings.max() + 2))
ax.grid(axis="y", alpha=0.22)
ax.legend(frameon=False)
for bar, value in zip(bars, savings):
    ax.text(bar.get_x() + bar.get_width() / 2, value + (0.3 if value >= 0 else -0.65),
            f"{value:.1f}%", ha="center", va="bottom" if value >= 0 else "top", fontsize=9)
save(fig, "03_4var_vs_5var_paired_change.png")


# 4. Runtime comparison and selected length.
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.5, 5.4))
bars = ax1.bar([0, 1], [four_time.mean(), five_time.mean()], color=["#2F80ED", "#F2994A"], width=0.58)
ax1.set_title("平均运行时间")
ax1.set_ylabel("秒")
ax1.set_xticks([0, 1], ["4变量", "5变量"])
ax1.set_ylim(0, max(four_time.mean(), five_time.mean()) + 15)
for bar in bars:
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height()+1, f"{bar.get_height():.2f}s", ha="center")
unique, counts = np.unique(lengths, return_counts=True)
ax2.bar([str(x) for x in unique], counts, color="#F2994A", width=0.55)
ax2.set_title("5变量组最终长度选择")
ax2.set_xlabel("锚栓总长度（m）")
ax2.set_ylabel("选择次数")
ax2.set_ylim(0, 11)
for i, count in enumerate(counts):
    ax2.text(i, count + 0.25, f"{count}次", ha="center")
save(fig, "04_runtime_and_length_selection.png")


print(json.dumps(summary, ensure_ascii=False, indent=2))
