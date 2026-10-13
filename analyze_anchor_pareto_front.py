# -*- coding: utf-8 -*-
"""
文件名: analyze_anchor_pareto_front.py
功能: 基于 NSGA-II 算法的双圈锚栓笼多目标帕累托前沿分析 (总造价 vs 疲劳应力幅)
"""

import math
import random
import numpy as np
import matplotlib.pyplot as plt
from deap import base, creator, tools, algorithms
import json
import sys


# ==========================================
# 1. 基础数据读取与材料库
# ==========================================
def load_foundation_data(json_path="Optimized_Rebar_Data.json"):
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        bc = data.get("Boundary_Conditions", {})
        M = float(bc.get("Limit_M_kNm", 85000.0))
        N = float(bc.get("Limit_N_kN", 15000.0))
        delta_M = float(bc.get("Fatigue_Delta_M_kNm", M * 0.3))

        geom = data.get("Foundation_Geometry", {})
        r_col = float(geom.get("Pedestal_Radius_m", 3.3))
        H_cap = float(geom.get("Thickness_m", 2.8))

        bot_radial = data.get("Bottom_Plate_Rebar", {}).get("Radial", {})
        top_radial = data.get("Top_Plate_Rebar", {}).get("Radial", {})
        d2 = float(bot_radial.get("Reinforced_Bars", {}).get("Diameter_mm", 0.032))
        d4 = float(top_radial.get("Through_Bars", {}).get("Diameter_mm", 0.025))
        max_rebar_dia = max(d2 * 1000.0 if d2 < 1.0 else d2, d4 * 1000.0 if d4 < 1.0 else d4)

        return M, N, delta_M, max_rebar_dia, r_col, H_cap
    except Exception as e:
        print(f"\n❌ 读取 JSON 发生异常: {e}")
        sys.exit(1)


F_T_DESIGN = 500.0
FLANGE_COST_PER_M = 15000.0


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


# 🎯 【多目标核心】：FitnessMulti 定义为极小化两个目标 (权重皆为 -1.0)
# Obj 1: Total Cost, Obj 2: Fatigue Delta Sigma
creator.create("FitnessMulti", base.Fitness, weights=(-1.0, -1.0))
creator.create("Individual", list, fitness=creator.FitnessMulti)


# ==========================================
# 2. 多目标适应度评估函数
# ==========================================
def evaluate_moo(individual, M, N_load, delta_M, max_rebar_dia, r_col, H_cap):
    PCD_A, PCD_B = round(individual[0] / 0.01) * 0.01, round(individual[1] / 0.01) * 0.01
    PCD_out, PCD_in = max(PCD_A, PCD_B), min(PCD_A, PCD_B)
    N_ring = int(round(individual[2] / 4.0) * 4)
    bolt_idx = max(0, min(3, int(round(individual[3]))))
    bolt_specs = get_bolt_specs(bolt_idx)

    total_cost = (2 * N_ring) * bolt_specs['unit_cost'] + 3.0 * (
                math.pi * ((PCD_out + PCD_in) / 2.0) * FLANGE_COST_PER_M)
    delta_T_out = (4 * delta_M * PCD_out) / (N_ring * (PCD_out ** 2 + PCD_in ** 2))
    delta_sigma = (delta_T_out * 1000.0) / bolt_specs['Ae']

    penalty = 0.0

    # 红线校验
    T_max_out = (4 * M * PCD_out) / (N_ring * (PCD_out ** 2 + PCD_in ** 2)) - (N_load / (2 * N_ring))
    if T_max_out > bolt_specs['T_allow']: penalty += (T_max_out - bolt_specs['T_allow']) * 50000.0
    if ((math.pi * PCD_in * 1000.0 / N_ring) - bolt_specs['d_sleeve']) < (max_rebar_dia + 50.0): penalty += 1e6
    if ((PCD_out - PCD_in) / 2.0 * 1000.0) < (bolt_specs['d_sleeve'] + 100.0): penalty += 1e6
    if (r_col - (PCD_out / 2.0) - (bolt_specs['d_sleeve'] / 2000.0)) < max(0.25, 4.0 * bolt_specs[
        'd'] / 1000.0): penalty += 1e7
    if 35.0 * (bolt_specs['d'] / 1000.0) > (H_cap - 0.2): penalty += 1e6
    if PCD_in < r_col * 0.6: penalty += 1e6

    # 🎯 【多目标惩罚】：如果违规，双目标同时施加重罚，将其踢出帕累托前沿
    if penalty > 0:
        return (total_cost + 1e7 + penalty, delta_sigma + 1000.0 + penalty / 10000.0)

    return (total_cost, delta_sigma)


# ==========================================
# 3. 运行 NSGA-II 算法
# ==========================================
if __name__ == "__main__":
    print("\n🚀 启动双圈锚栓笼 NSGA-II 多目标帕累托前沿寻优...")
    M, N_load, delta_M, max_dia, r_col, H_cap = load_foundation_data("Optimized_Rebar_Data.json")

    toolbox = base.Toolbox()
    max_pcd, min_pcd = (r_col - 0.25) * 2.0, r_col * 0.6
    toolbox.register("attr_pcd1", random.uniform, min_pcd, max_pcd)
    toolbox.register("attr_pcd2", random.uniform, min_pcd, max_pcd)
    toolbox.register("attr_nring", random.randint, 60, 160)
    toolbox.register("attr_type", random.randint, 0, 3)

    toolbox.register("individual", tools.initCycle, creator.Individual,
                     (toolbox.attr_pcd1, toolbox.attr_pcd2, toolbox.attr_nring, toolbox.attr_type), n=1)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    toolbox.register("evaluate", evaluate_moo, M=M, N_load=N_load, delta_M=delta_M,
                     max_rebar_dia=max_dia, r_col=r_col, H_cap=H_cap)

    toolbox.register("mate", tools.cxBlend, alpha=0.5)
    toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=1.0, indpb=0.2)
    # 🎯 【多目标核心】：选择算法更改为 NSGA-II (非支配排序算法)
    toolbox.register("select", tools.selNSGA2)

    # 为了画出好看的帕累托前沿，需要极大的种群基数和代数
    pop = toolbox.population(n=200)

    print("  正在执行多目标演化计算 (包含非支配排序与拥挤度计算)....")
    # 使用 eaMuPlusLambda 执行 NSGA-II
    pop, logbook = algorithms.eaMuPlusLambda(pop, toolbox, mu=200, lambda_=200,
                                             cxpb=0.7, mutpb=0.2, ngen=100, verbose=False)

    # 提取所有评估过的个体，并滤除带有惩罚分 (即 cost > 1e7) 的无效解
    valid_inds = [ind for ind in pop if ind.fitness.values[0] < 1e7]

    if not valid_inds:
        print("❌ 警告：未找到任何可行解，请检查约束是否过于苛刻！")
        sys.exit()

    # 提取第一层帕累托前沿 (First Pareto Front)
    pareto_front = tools.sortNondominated(valid_inds, len(valid_inds), first_front_only=True)[0]

    # 整理数据用于绘图
    costs = [ind.fitness.values[0] / 10000.0 for ind in pareto_front]  # 转为万元
    stresses = [ind.fitness.values[1] for ind in pareto_front]

    # 按造价从小到大排序，方便连线
    sorted_idx = np.argsort(costs)
    pareto_costs = np.array(costs)[sorted_idx]
    pareto_stresses = np.array(stresses)[sorted_idx]

    # 全局非支配解 (搜索空间的云图背景)
    all_costs = [ind.fitness.values[0] / 10000.0 for ind in valid_inds]
    all_stresses = [ind.fitness.values[1] for ind in valid_inds]

    print("\n✅ NSGA-II 计算完毕，正在生成学术级帕累托前沿图表...")

    # ==========================================
    # 4. 绘制顶刊级帕累托前沿图 (精致美化版)
    # ==========================================
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman']
    plt.rcParams['figure.dpi'] = 300

    fig, ax = plt.subplots(figsize=(9, 6), facecolor='white')

    # 1. 缩小背景探索点的尺寸 (s=10)
    ax.scatter(all_costs, all_stresses, color='#BDC3C7', alpha=0.4, s=10, label='Feasible Search Space')

    # 2. 缩小帕累托前沿点的尺寸 (markersize=4) 和线宽 (linewidth=1.5)
    ax.plot(pareto_costs, pareto_stresses, color='#E74C3C', marker='o', linestyle='-',
            linewidth=1.5, markersize=4, label='Pareto Front (Optimal Trade-offs)')

    # 3. 标注三大关键决策点
    if len(pareto_costs) >= 3:
        ax.annotate('Cost-driven Optimal',
                    xy=(pareto_costs[0], pareto_stresses[0]),
                    xytext=(pareto_costs[0] + 5, pareto_stresses[0] + 15),  # 微调文字位置防遮挡
                    arrowprops=dict(facecolor='#2C3E50', arrowstyle='->', lw=1.2), fontsize=10, fontweight='bold')

        ax.annotate('Safety-driven Optimal',
                    xy=(pareto_costs[-1], pareto_stresses[-1]),
                    xytext=(pareto_costs[-1] - 20, pareto_stresses[-1] + 15),
                    arrowprops=dict(facecolor='#2C3E50', arrowstyle='->', lw=1.2), fontsize=10, fontweight='bold')

        norm_c = (pareto_costs - np.min(pareto_costs)) / (np.max(pareto_costs) - np.min(pareto_costs) + 1e-6)
        norm_s = (pareto_stresses - np.min(pareto_stresses)) / (
                    np.max(pareto_stresses) - np.min(pareto_stresses) + 1e-6)
        knee_idx = np.argmin(np.sqrt(norm_c ** 2 + norm_s ** 2))
        ax.annotate('Balanced (Knee Point)',
                    xy=(pareto_costs[knee_idx], pareto_stresses[knee_idx]),
                    xytext=(pareto_costs[knee_idx] + 5, pareto_stresses[knee_idx] + 15),
                    arrowprops=dict(facecolor='#27AE60', arrowstyle='->', lw=1.2), fontsize=10, fontweight='bold',
                    color='#27AE60')

    ax.axhline(y=50.0, color='#C0392B', linestyle='--', linewidth=1.2, label='Code Limit (50 MPa)')

    ax.set_title("Multi-Objective Pareto Front via NSGA-II\n(Total Cost vs. Fatigue Stress)", fontsize=15,
                 fontweight='bold', pad=15)
    ax.set_xlabel("Foundation Anchor Cage Total Cost (10,000 RMB)", fontsize=12)
    ax.set_ylabel(r"Maximum Fatigue Stress Amplitude $\Delta\sigma$ (MPa)", fontsize=12)

    ax.grid(True, linestyle=':', color='#BDC3C7', alpha=0.8)
    legend = ax.legend(loc='upper right', framealpha=0.95, edgecolor='#BDC3C7', fontsize=10)

    plt.tight_layout()
    plt.show()