# -*- coding: utf-8 -*-
"""
文件名: analyze_anchor_sensitivity.py
功能: 针对双圈锚栓笼开展多参数敏感性分析，生成学术论文专属折线图 (高算力强化版)
"""

import math
import random
import matplotlib.pyplot as plt
from deap import base, creator, tools, algorithms

# ==========================================
# 1. 基础设定与材料库
# ==========================================
F_T_DESIGN = 500.0
FLANGE_COST_PER_M = 15000.0
ALLOWABLE_DELTA_SIGMA = 50.0


def get_bolt_specs(bolt_idx):
    raw_data = {
        0: {'d': 42.0, 'P': 4.5, 'd_sleeve': 55.0, 'unit_cost': 1200.0},
        1: {'d': 48.0, 'P': 5.0, 'd_sleeve': 65.0, 'unit_cost': 1800.0},
        2: {'d': 56.0, 'P': 5.5, 'd_sleeve': 75.0, 'unit_cost': 2600.0},
        3: {'d': 64.0, 'P': 6.0, 'd_sleeve': 85.0, 'unit_cost': 3500.0}
    }
    b = raw_data[bolt_idx]
    Ae = (math.pi / 4.0) * ((b['d'] - 0.9382 * b['P']) ** 2)
    return {'type': f"M{int(b['d'])}", 'd': b['d'], 'Ae': Ae,
            'T_allow': (Ae * F_T_DESIGN) / 1000.0, 'd_sleeve': b['d_sleeve'], 'unit_cost': b['unit_cost']}


# 全局注册 DEAP Creator
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)


# ==========================================
# 2. 核心评估与高算力算法封装
# ==========================================
def evaluate_anchor(individual, M, N_load, delta_M, max_rebar_dia, r_col, H_cap):
    PCD_A, PCD_B = round(individual[0] / 0.01) * 0.01, round(individual[1] / 0.01) * 0.01
    PCD_out, PCD_in = max(PCD_A, PCD_B), min(PCD_A, PCD_B)
    N_ring = int(round(individual[2] / 4.0) * 4)
    bolt_idx = max(0, min(3, int(round(individual[3]))))
    bolt_specs = get_bolt_specs(bolt_idx)

    total_cost = (2 * N_ring) * bolt_specs['unit_cost'] + 3.0 * (
                math.pi * ((PCD_out + PCD_in) / 2.0) * FLANGE_COST_PER_M)
    penalty = 0.0

    T_max_out = (4 * M * PCD_out) / (N_ring * (PCD_out ** 2 + PCD_in ** 2)) - (N_load / (2 * N_ring))
    if T_max_out > bolt_specs['T_allow']: penalty += (T_max_out - bolt_specs['T_allow']) * 50000.0

    delta_sigma = ((4 * delta_M * PCD_out) / (N_ring * (PCD_out ** 2 + PCD_in ** 2)) * 1000.0) / bolt_specs['Ae']
    if delta_sigma > ALLOWABLE_DELTA_SIGMA: penalty += (delta_sigma - ALLOWABLE_DELTA_SIGMA) * 100000.0

    if ((math.pi * PCD_in * 1000.0 / N_ring) - bolt_specs['d_sleeve']) < (max_rebar_dia + 50.0): penalty += 1e6
    if ((PCD_out - PCD_in) / 2.0 * 1000.0) < (bolt_specs['d_sleeve'] + 100.0): penalty += 1e6

    actual_cover = r_col - (PCD_out / 2.0) - (bolt_specs['d_sleeve'] / 2000.0)
    if actual_cover < max(0.25, 4.0 * bolt_specs['d'] / 1000.0): penalty += 1e7
    if 35.0 * (bolt_specs['d'] / 1000.0) > (H_cap - 0.2): penalty += 1e6
    if PCD_in < r_col * 0.6: penalty += 1e6

    return (total_cost + 1e7 + penalty,) if penalty > 0 else (total_cost,)


def run_optimization(M, N_load, delta_M, max_rebar_dia, r_col, H_cap):
    """运行强化版单次 GA 并返回最优结果"""
    toolbox = base.Toolbox()
    max_pcd, min_pcd = (r_col - 0.25) * 2.0, r_col * 0.6
    toolbox.register("attr_pcd1", random.uniform, min_pcd, max_pcd)
    toolbox.register("attr_pcd2", random.uniform, min_pcd, max_pcd)
    toolbox.register("attr_nring", random.randint, 60, 150)  # 放宽数量上限，给予算法更大喘息空间
    toolbox.register("attr_type", random.randint, 0, 3)
    toolbox.register("individual", tools.initCycle, creator.Individual,
                     (toolbox.attr_pcd1, toolbox.attr_pcd2, toolbox.attr_nring, toolbox.attr_type), n=1)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    toolbox.register("evaluate", evaluate_anchor, M=M, N_load=N_load, delta_M=delta_M,
                     max_rebar_dia=max_rebar_dia, r_col=r_col, H_cap=H_cap)
    toolbox.register("mate", tools.cxBlend, alpha=0.5)
    toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=1.0, indpb=0.2)
    toolbox.register("select", tools.selTournament, tournsize=3)

    # 🎯 【核心升级】：种群 100，代数 80，彻底消灭未收敛跳点
    pop = toolbox.population(n=100)
    hof = tools.HallOfFame(1)
    algorithms.eaSimple(pop, toolbox, cxpb=0.7, mutpb=0.2, ngen=80, halloffame=hof, verbose=False)

    best = hof[0]
    best_PCD_out = max(round(best[0] / 0.01) * 0.01, round(best[1] / 0.01) * 0.01)
    best_N_total = int(round(best[2] / 4.0) * 4) * 2
    best_cost = evaluate_anchor(best, M, N_load, delta_M, max_rebar_dia, r_col, H_cap)[0]

    return best_cost, best_PCD_out, best_N_total


# ==========================================
# 3. 开展敏感性分析循环
# ==========================================
if __name__ == "__main__":
    print("\n🚀 启动双圈锚栓笼参数敏感性分析 (高保真强化版)...")

    # 基准参数 (Base Case)
    BASE_M, BASE_N, BASE_DELTA_M = 85000.0, 15000.0, 25500.0
    BASE_R_COL, BASE_H_CAP, MAX_REBAR = 3.3, 2.8, 32.0

    variations = [-0.20, -0.10, 0.0, 0.10, 0.20]
    labels = ["-20%", "-10%", "Base", "+10%", "+20%"]

    results = {
        'M': {'cost': [], 'pcd': [], 'n_bolts': []},
        'Delta_M': {'cost': [], 'pcd': [], 'n_bolts': []},
        'r_col': {'cost': [], 'pcd': [], 'n_bolts': []}
    }

    total_runs = len(variations) * 3
    current_run = 0

    # 1. 测试极限弯矩 M 的敏感性
    print("\n>>> [1/3] 正在分析极限弯矩 (Limit Moment) 敏感性...")
    for v in variations:
        current_run += 1
        print(f"  进度 {current_run}/{total_runs}: M = {BASE_M * (1 + v):.0f} kNm")
        c, p, n = run_optimization(BASE_M * (1 + v), BASE_N, BASE_DELTA_M, MAX_REBAR, BASE_R_COL, BASE_H_CAP)
        results['M']['cost'].append(c / 10000.0)  # 转为万元
        results['M']['pcd'].append(p)
        results['M']['n_bolts'].append(n)

    # 2. 测试疲劳弯矩 Delta_M 的敏感性
    print("\n>>> [2/3] 正在分析疲劳弯矩 (Fatigue Moment) 敏感性...")
    for v in variations:
        current_run += 1
        print(f"  进度 {current_run}/{total_runs}: Delta_M = {BASE_DELTA_M * (1 + v):.0f} kNm")
        c, p, n = run_optimization(BASE_M, BASE_N, BASE_DELTA_M * (1 + v), MAX_REBAR, BASE_R_COL, BASE_H_CAP)
        results['Delta_M']['cost'].append(c / 10000.0)
        results['Delta_M']['pcd'].append(p)
        results['Delta_M']['n_bolts'].append(n)

    # 3. 测试台柱半径 r_col 的敏感性 (几何约束墙)
    print("\n>>> [3/3] 正在分析台柱半径 (Pedestal Radius) 敏感性...")
    for v in variations:
        current_run += 1
        print(f"  进度 {current_run}/{total_runs}: r_col = {BASE_R_COL * (1 + v):.2f} m")
        c, p, n = run_optimization(BASE_M, BASE_N, BASE_DELTA_M, MAX_REBAR, BASE_R_COL * (1 + v), BASE_H_CAP)
        results['r_col']['cost'].append(c / 10000.0)
        results['r_col']['pcd'].append(p)
        results['r_col']['n_bolts'].append(n)

    print("\n✅ 所有敏感性计算完毕，正在生成顶刊级图表...")

    # ==========================================
    # 4. 绘制顶刊级敏感性雷达图
    # ==========================================
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman']
    plt.rcParams['figure.dpi'] = 300

    fig, axs = plt.subplots(1, 3, figsize=(16, 5), facecolor='white')
    colors = {'M': '#E74C3C', 'Delta_M': '#2980B9', 'r_col': '#27AE60'}
    legends = {'M': r'Limit Moment ($M$)', 'Delta_M': r'Fatigue Moment ($\Delta M$)',
               'r_col': r'Pedestal Radius ($r_{col}$)'}

    x_axis = [v * 100 for v in variations]

    # 提取所有造价用于智能判定 Y 轴区间
    all_costs = []
    for key in results:
        all_costs.extend(results[key]['cost'])

    # 图 A: 总造价敏感性
    for key in results:
        axs[0].plot(x_axis, results[key]['cost'], marker='s', linewidth=2.5, color=colors[key], label=legends[key],
                    zorder=3)
    axs[0].set_title("(a) Sensitivity of Total Cost", fontsize=14, fontweight='bold')
    axs[0].set_ylabel("Optimal Cost (10,000 RMB)", fontsize=12)

    # 🎯 【核心升级】：智能 Y 轴截断。
    # 如果有超过 500 万的死刑惩罚解，将 Y 轴上限锁定在正常可行解的顶部区域，
    # 使得线条指向图外，既能说明该点已不可行，又不会压缩正常造价的观察空间。
    valid_costs = [c for c in all_costs if c < 500]
    if valid_costs:
        axs[0].set_ylim(bottom=min(valid_costs) * 0.95, top=max(valid_costs) * 1.3)

    # 图 B: 锚栓总数量敏感性
    for key in results:
        axs[1].plot(x_axis, results[key]['n_bolts'], marker='^', linewidth=2.5, color=colors[key], label=legends[key],
                    zorder=3)
    axs[1].set_title("(b) Sensitivity of Bolt Quantity", fontsize=14, fontweight='bold')
    axs[1].set_ylabel("Total Number of Bolts", fontsize=12)

    # 图 C: 外圈 PCD 敏感性
    for key in results:
        axs[2].plot(x_axis, results[key]['pcd'], marker='o', linewidth=2.5, color=colors[key], label=legends[key],
                    zorder=3)
    axs[2].set_title("(c) Sensitivity of Outer PCD", fontsize=14, fontweight='bold')
    axs[2].set_ylabel("Outer PCD (m)", fontsize=12)

    for ax in axs:
        ax.set_xticks(x_axis)
        ax.set_xticklabels(labels)
        ax.set_xlabel("Parameter Variation (%)", fontsize=12)
        ax.grid(True, linestyle='--', color='#BDC3C7', alpha=0.6, zorder=0)
        ax.legend(loc='best', fontsize=10, framealpha=0.9)

    plt.tight_layout()
    plt.show()
