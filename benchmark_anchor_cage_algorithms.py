# -*- coding: utf-8 -*-
"""
独立模块：锚栓笼多算法竞速对标框架 (合规升级版)
功能：结合 NB/T 10311-2019 等规范，对比 网格搜索(Grid Search)、粒子群(PSO) 和 遗传算法(GA)
"""

import numpy as np
import matplotlib.pyplot as plt
import time
import math

# 设置绘图中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# ================= 1. 物理问题定义与边界条件 =================
# 外部极限输入载荷 (假设 5MW-6MW 级别风机)
M_EXTREME = 85000.0  # 极限弯矩 (kN.m)
N_EXTREME = 4000.0  # 极限下压轴力 (kN)
D_REBAR = 0.032  # 底板主筋直径 (m)

# 【规范修正 1】施工振捣净距：大体积混凝土建议至少 100mm (0.10m)
MIN_CLEARANCE = 0.10

# 【规范修正 2】锚栓材料力学库 (以风电常用的 42CrMo, 10.9级高强锚栓为例)
# T_yield: 屈服拉力(kN) = 截面积(mm2) * 900MPa / 1000
# 预紧力设定为屈服拉力的 70%
BOLT_DB = {
    0: {'name': 'M42', 'T_yield': 1008.0, 'd_sleeve': 0.065, 'price': 1.0},
    1: {'name': 'M48', 'T_yield': 1323.0, 'd_sleeve': 0.075, 'price': 1.3},
    2: {'name': 'M56', 'T_yield': 1827.0, 'd_sleeve': 0.085, 'price': 1.8}
}

# 设计变量空间 (Bounds)
# X = [PCD, N_bolts, Bolt_Type]
BOUNDS = [
    (4.0, 6.0),  # PCD: 4.0m ~ 6.0m
    (80, 200),  # N_bolts: 80 ~ 200 (离散，4的倍数)
    (0, 2)  # Bolt_Type: 0, 1, 2
]

# ================= 2. 核心适应度函数 (Fitness Function) =================
EVAL_COUNT = 0


def evaluate_design(x):
    global EVAL_COUNT
    EVAL_COUNT += 1

    # 基因解码
    pcd = x[0]
    n_bolts = int(round(x[1] / 4.0) * 4)
    bolt_type_idx = int(round(max(0, min(2, x[2]))))
    bolt_prop = BOLT_DB[bolt_type_idx]

    # 【规范修正 3】目标函数：锚栓造价 + 法兰盘非线性材料成本
    # 法兰体积与 PCD^2 呈正相关(因外径增加且需要更厚以抗弯)，这里用平方关系模拟造价激增
    cost = n_bolts * bolt_prop['price'] + 150.0 * (pcd / 4.0) ** 2

    penalty = 0.0

    # 【规范修正 4】力学红线：预应力与法兰脱开验算 (刚度等效法)
    # 计算单根锚栓承受的最大外部等效拉力
    t_ext = (4.0 * M_EXTREME / pcd - N_EXTREME) / n_bolts

    # 锚栓预紧力 P0 (通常取屈服强度的 70%)
    p0_pretension = 0.7 * bolt_prop['T_yield']

    # 锚栓-混凝土体系的受拉刚度比 (通常钢:砼的刚度分配系数约为 0.2)
    stiffness_ratio = 0.2

    # 验算 A：防止法兰盘与混凝土受拉脱开 (Decompression Check)
    # 脱缝临界外部拉力 = P0 / (1 - stiffness_ratio)
    t_separation = p0_pretension / (1.0 - stiffness_ratio)
    if t_ext > t_separation:
        penalty += (t_ext - t_separation) * 50000.0  # 惩罚

    # 验算 B：防止锚栓在叠加拉力下屈服失效 (Yield Check)
    # 实际锚栓内力 = 预紧力 + 分担的外部拉力
    f_bolt_actual = p0_pretension + stiffness_ratio * t_ext
    # 安全系数管控：最大内力不得超过屈服强度的 90%
    if f_bolt_actual > 0.9 * bolt_prop['T_yield']:
        penalty += (f_bolt_actual - 0.9 * bolt_prop['T_yield']) * 50000.0

    # 验算 C：施工空间红线 (钢筋穿插与振捣空间)
    arc_spacing = (math.pi * pcd) / n_bolts
    clear_gap = arc_spacing - bolt_prop['d_sleeve']
    required_space = D_REBAR + MIN_CLEARANCE

    if clear_gap < required_space:
        penalty += (required_space - clear_gap) * 100000.0

    return cost + penalty


# ================= 3. 算法 1：上帝视角网格搜索 (Grid Search) =================
def run_grid_search():
    global EVAL_COUNT
    EVAL_COUNT = 0
    best_cost = float('inf')
    best_x = None

    pcd_range = np.arange(BOUNDS[0][0], BOUNDS[0][1] + 0.05, 0.05)
    n_range = np.arange(BOUNDS[1][0], BOUNDS[1][1] + 4, 4)
    type_range = [0, 1, 2]

    start_time = time.time()
    for pcd in pcd_range:
        for n in n_range:
            for t in type_range:
                x = [pcd, n, t]
                fit = evaluate_design(x)
                if fit < best_cost:
                    best_cost = fit
                    best_x = x

    return best_x, best_cost, time.time() - start_time, EVAL_COUNT


# ================= 4. 算法 2：离散粒子群算法 (DPSO) =================
def run_pso(pop_size=30, max_iter=50):
    global EVAL_COUNT
    EVAL_COUNT = 0

    dim = 3
    pos = np.zeros((pop_size, dim))
    pos[:, 0] = np.random.uniform(BOUNDS[0][0], BOUNDS[0][1], pop_size)
    pos[:, 1] = np.random.uniform(BOUNDS[1][0], BOUNDS[1][1], pop_size)
    pos[:, 2] = np.random.uniform(BOUNDS[2][0], BOUNDS[2][1], pop_size)

    vel = np.random.uniform(-1, 1, (pop_size, dim))
    pbest_pos = pos.copy()
    pbest_fit = np.array([evaluate_design(p) for p in pos])
    gbest_pos = pbest_pos[np.argmin(pbest_fit)]
    gbest_fit = np.min(pbest_fit)

    history = []
    start_time = time.time()

    w, c1, c2 = 0.5, 1.5, 1.5
    for i in range(max_iter):
        r1, r2 = np.random.rand(pop_size, dim), np.random.rand(pop_size, dim)
        vel = w * vel + c1 * r1 * (pbest_pos - pos) + c2 * r2 * (gbest_pos - pos)
        pos = pos + vel

        for d in range(dim):
            pos[:, d] = np.clip(pos[:, d], BOUNDS[d][0], BOUNDS[d][1])

        fits = np.array([evaluate_design(p) for p in pos])
        better_mask = fits < pbest_fit
        pbest_pos[better_mask] = pos[better_mask]
        pbest_fit[better_mask] = fits[better_mask]

        if np.min(pbest_fit) < gbest_fit:
            gbest_fit = np.min(pbest_fit)
            gbest_pos = pbest_pos[np.argmin(pbest_fit)]

        history.append(gbest_fit)

    return gbest_pos, gbest_fit, time.time() - start_time, EVAL_COUNT, history


# ================= 5. 算法 3：遗传算法 (GA) =================
def run_ga(pop_size=30, max_iter=50):
    global EVAL_COUNT
    EVAL_COUNT = 0

    dim = 3
    pop = np.zeros((pop_size, dim))
    pop[:, 0] = np.random.uniform(BOUNDS[0][0], BOUNDS[0][1], pop_size)
    pop[:, 1] = np.random.uniform(BOUNDS[1][0], BOUNDS[1][1], pop_size)
    pop[:, 2] = np.random.uniform(BOUNDS[2][0], BOUNDS[2][1], pop_size)

    history = []
    start_time = time.time()

    for i in range(max_iter):
        fits = np.array([evaluate_design(p) for p in pop])
        best_fit = np.min(fits)
        history.append(best_fit)

        inv_fits = 1.0 / (fits + 1e-6)
        probs = inv_fits / np.sum(inv_fits)
        indices = np.random.choice(pop_size, size=pop_size, p=probs)
        new_pop = pop[indices].copy()

        for j in range(0, pop_size, 2):
            if np.random.rand() < 0.8 and j + 1 < pop_size:
                pt = np.random.randint(1, dim)
                temp = new_pop[j, pt:].copy()
                new_pop[j, pt:] = new_pop[j + 1, pt:]
                new_pop[j + 1, pt:] = temp

        for j in range(pop_size):
            if np.random.rand() < 0.2:
                d = np.random.randint(dim)
                new_pop[j, d] += np.random.normal(0, (BOUNDS[d][1] - BOUNDS[d][0]) * 0.1)
                new_pop[j, d] = np.clip(new_pop[j, d], BOUNDS[d][0], BOUNDS[d][1])

        pop = new_pop

    fits = np.array([evaluate_design(p) for p in pop])
    best_idx = np.argmin(fits)
    return pop[best_idx], fits[best_idx], time.time() - start_time, EVAL_COUNT, history


# ================= 6. 运行与结果对比 =================
if __name__ == "__main__":
    print("=" * 75)
    print(" 🚀 锚栓笼接口协同优化：多算法竞速对标测试 (合规工程版)")
    print("=" * 75)

    print("正在运行 全局网格搜索 (寻找绝对最优基准)...")
    gs_x, gs_fit, gs_time, gs_evals = run_grid_search()

    print("正在运行 离散粒子群算法 (DPSO)...")
    pso_x, pso_fit, pso_time, pso_evals, pso_hist = run_pso(pop_size=50, max_iter=80)

    print("正在运行 遗传算法 (GA)...")
    ga_x, ga_fit, ga_time, ga_evals, ga_hist = run_ga(pop_size=50, max_iter=80)


    def decode_and_format(x):
        pcd = x[0]
        n_bolts = int(round(x[1] / 4.0) * 4)
        b_type = BOLT_DB[int(round(max(0, min(2, x[2]))))]['name']
        return f"PCD={pcd:.2f}m, N={n_bolts}根, 型号={b_type}"


    print("\n" + "=" * 75)
    print(
        f"{'算法名称':<12} | {'最终寻优结果 (PCD, 数量, 型号)':<35} | {'造价指数':<8} | {'耗时(s)':<6} | {'评估次数'}")
    print("-" * 75)
    print(f"{'Grid Search':<12} | {decode_and_format(gs_x):<35} | {gs_fit:<8.2f} | {gs_time:<6.2f} | {gs_evals}")
    print(f"{'DPSO':<12} | {decode_and_format(pso_x):<35} | {pso_fit:<8.2f} | {pso_time:<6.2f} | {pso_evals}")
    print(f"{'GA':<12} | {decode_and_format(ga_x):<35} | {ga_fit:<8.2f} | {ga_time:<6.2f} | {ga_evals}")
    print("=" * 75)

    plt.figure(figsize=(10, 6))
    plt.axhline(y=gs_fit, color='r', linestyle='--', linewidth=2, label='Grid Search (True Optimum)')
    plt.plot(pso_hist, label='DPSO Convergence', linewidth=2, marker='o', markersize=4)
    plt.plot(ga_hist, label='GA Convergence', linewidth=2, marker='s', markersize=4)

    plt.title('Algorithm Convergence Comparison (Engineering Compliant Model)', fontsize=14, fontweight='bold')
    plt.xlabel('Iterations / Generations', fontsize=12)
    plt.ylabel('Objective Cost (Lower is better)', fontsize=12)

    # 动态适应 Y 轴范围，过滤严重惩罚值
    plt.ylim(gs_fit * 0.95, gs_fit * 2.0)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.show()