# -*- coding: utf-8 -*-
"""
文件名: optimize_cost_crack_pareto.py (NSGA-II 终极极速云图版 - 经济性 vs 耐久性 升维版)
功能: 风机基础智能优化主程序 - 寻找【直接材料总造价】与【最大裂缝宽度】的真实帕累托前沿
核心重构:
  - [目标升维] 抛弃纯物料对比，重构为 CAPEX(成本) vs SLS(正常使用极限状态裂缝)，极具学术价值。
  - [连续化松弛] 拦截 foundation_config 中的钢筋根数列表，转为连续寻优区间，彻底打破“离散断层”。
  - [量纲归一化] 升级了寻找 Knee Point (拐点) 的正交投影算法，完美消除百万级造价与微米级裂缝的量纲鸿沟。
  - [主控解析] 深度输出各主要受力指标的满载率，自动锁定截面力学瓶颈。
"""

import math
import random
import numpy as np
import matplotlib.pyplot as plt
from deap import base, creator, tools
from tqdm import tqdm

import multiprocessing
import os
import json
from datetime import datetime

import foundation_config as cfg
import analyze_integrated_pile_foundation as model
ABAQUS_OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "abaqus_postprocess")

try:
    from analyze_cap_structure import CapStructuralAnalysis
    from analyze_pile_structure import PileStructuralAnalysis

    HAS_CHECK_MODULES = True
except ImportError:
    HAS_CHECK_MODULES = False
    print("【提示】未检测到验算模块，最终报告将跳过详细验算表。")

if not model.HAS_MODULES:
    print("【警告】Optimization2: 未检测到物理计算模块，优化将基于几何约束运行。")

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# ================= 0. 全局设置 =================
CURRENT_ITER = 0
ALLOW_TENSION = True
FIXED_PILE_LENGTH = None


def init_worker(allow_tension_flag, fixed_pile_length_val):
    global ALLOW_TENSION, FIXED_PILE_LENGTH
    ALLOW_TENSION = allow_tension_flag
    FIXED_PILE_LENGTH = fixed_pile_length_val


# ================= 1. 变量初始化与解析 (连续化松弛) =================
def parse_config_vars():
    raw_config = cfg.CONFIG["var_ranges"]
    var_ranges = []
    int_indices = []
    discrete_map = {}
    for i, item in enumerate(raw_config):
        if isinstance(item, tuple):
            var_ranges.append(item)
        elif isinstance(item, list):
            # 将 [100, 110, 120...300] 等长列表转为连续浮点数范围，抹平离散阶跃
            if len(item) > 5 and item[0] >= 100 and item[-1] >= 200:
                var_ranges.append((float(item[0]), float(item[-1])))
                int_indices.append(i)
            else:
                var_ranges.append((0, len(item) - 1))
                int_indices.append(i)
                discrete_map[i] = item
    return var_ranges, int_indices, discrete_map


VAR_RANGES, INT_INDICES, DISCRETE_MAP = parse_config_vars()


def lhs_sample(n, samples):
    result = np.zeros((samples, n))
    for i in range(n):
        low, up = VAR_RANGES[i]
        d = 1.0 / samples
        col_uniform = np.random.uniform(0, d, samples) + np.arange(samples) * d
        np.random.shuffle(col_uniform)
        result[:, i] = low + col_uniform * (up - low)
    return result


# ================= 2. 基因解码 =================
def decode_gene(ind):
    vals = []
    for i, val in enumerate(ind):
        if np.isnan(val) or np.isinf(val):
            vals.append(DISCRETE_MAP[i][0] if i in DISCRETE_MAP else VAR_RANGES[i][0])
            continue
        if i in DISCRETE_MAP:
            idx = int(round(val))
            idx = max(0, min(len(DISCRETE_MAP[i]) - 1, idx))
            vals.append(DISCRETE_MAP[i][idx])
        else:
            vals.append(val)

    R_cap_m = round((vals[0] / 100.0) * 10) / 10.0
    H_depth_m = round((vals[1] / 100.0) * 10) / 10.0
    h1_edge_m = round((vals[2] / 100.0) * 10) / 10.0
    h2_slope_m = round((vals[3] / 100.0) * 10) / 10.0
    h3_pedestal_m = round((vals[4] / 100.0) * 10) / 10.0
    r_ped_m = round((vals[5] / 100.0) * 10) / 10.0

    def snap_rebar_count(R_m, raw_n, is_edge_doubled=False):
        n = max(10, int(round(raw_n)))  # 允许单根微调，打破10倍数封印
        perimeter_mm = 2 * math.pi * R_m * 1000
        eff_n = n * 2 if is_edge_doubled else n
        if perimeter_mm / eff_n > 300.0:
            req_eff_n = int(math.ceil(perimeter_mm / 300.0))
            n = req_eff_n // 2 if is_edge_doubled else req_eff_n
        elif perimeter_mm / eff_n < 100.0:
            req_eff_n = int(math.floor(perimeter_mm / 100.0))
            n = req_eff_n // 2 if is_edge_doubled else req_eff_n
        return max(10, n)

    n1 = snap_rebar_count(R_cap_m, int(vals[6]), is_edge_doubled=True)
    d1 = vals[7] / 1000.0
    d2 = vals[8] / 1000.0
    D1_ring_1 = vals[9] / 1000.0
    D1_ring_2 = vals[10] / 1000.0
    s1_ring_1 = round(cfg.CONFIG['DESIGN_PARAMS'].get('s1', 150) / 50.0) * 50.0 / 1000.0
    s1_ring_2 = round(cfg.CONFIG['DESIGN_PARAMS'].get('s1_2', 200) / 50.0) * 50.0 / 1000.0
    n2 = n1;
    n3 = n1;
    d3 = d1

    n4 = snap_rebar_count(R_cap_m, int(vals[11]), is_edge_doubled=True)
    d4 = vals[12] / 1000.0
    D2_ring = vals[13] / 1000.0
    s2_ring = round(cfg.CONFIG['DESIGN_PARAMS']['s2'] / 50.0) * 50.0 / 1000.0
    n5 = n4;
    d5 = d4

    L_pile_m = float(FIXED_PILE_LENGTH) if FIXED_PILE_LENGTH is not None else float(round(vals[14]))
    D_pile_m = vals[15] / 100.0
    n_in = int(vals[16])
    n_out = int(vals[17])
    r_in = round(R_cap_m * (vals[18] / 100.0) * 10) / 10.0
    r_out = round(R_cap_m * (vals[19] / 100.0) * 10) / 10.0

    n_pile_main = int(vals[20])
    if n_pile_main % 2 != 0: n_pile_main += 1
    d_pile_main = vals[21] / 1000.0

    geo = {
        'R_cap': R_cap_m, 'H': H_depth_m,
        'h1': h1_edge_m, 'h2': h2_slope_m, 'h3': h3_pedestal_m,
        'r_col': r_ped_m, 'r_top': r_ped_m + 0.2,
        'D_pile': D_pile_m, 'L_pile': L_pile_m,
        'rings': [{'n': n_in, 'r': r_in}, {'n': n_out, 'r': r_out}]
    }
    rebar = {
        'n1': n1, 'd1': d1, 'n2': n2, 'd2': d2, 'n3': n3, 'd3': d3,
        'ring_opt': {'d1': D1_ring_1, 'd2': D1_ring_2, 's1': s1_ring_1 * 1000.0, 's2': s1_ring_2 * 1000.0,
                     's': s1_ring_1 * 1000.0},
        'n4': n4, 'd4': d4, 'n5': n5, 'd5': d5,
        'top_ring_opt': {'d': D2_ring, 's': s2_ring * 1000.0},
        'pile_main_n': n_pile_main, 'pile_main_d': d_pile_main,
    }
    stirrup_consts = cfg.CONFIG['PILE_CONSTANTS']
    stirrup = {'d': stirrup_consts['STIRRUP_D'], 's': stirrup_consts['STIRRUP_S_DENSE'], 'fyv': 360.0}
    mat = cfg.CONFIG['CONCRETE_LIB'][1]

    return {'geo': geo, 'rebar': rebar, 'stirrup': stirrup, 'mat': mat}


# ================= 3. 多目标评估函数 (核心重构：造价 vs 裂缝) =================
def evaluate_moo(ind):
    global ALLOW_TENSION
    if np.any(np.isnan(ind)): return (1e10, 1e10, 1e10)

    try:
        safe_values = []
        for i, val in enumerate(ind):
            low, up = VAR_RANGES[i]
            safe_val = max(low, min(up, val))
            safe_values.append(int(round(safe_val)) if i in INT_INDICES else safe_val)

        params = decode_gene(safe_values)
        geo, rb, mat = params['geo'], params['rebar'], params['mat']

        total_viol, _ = model.assess_design_performance(params)

        ach = cfg.CONFIG.get('ANCHOR_CAGE', {})
        L_tot, bg, te, bc = ach.get('L_total', 4.2), ach.get('bottom_groove', 0.3), ach.get('top_expose', 0.3), ach.get(
            'bottom_cover', 0.2)
        if (geo['h1'] + geo['h2'] + geo['h3'] - bg) < (L_tot - te + bc):
            total_viol += ((L_tot - te + bc) - (geo['h1'] + geo['h2'] + geo['h3'] - bg)) * 50.0

        Do = ach.get('D_outer', 5.1)
        if geo['r_col'] < (Do / 2.0) + 0.6: total_viol += ((Do / 2.0) + 0.6 - geo['r_col']) * 50.0

        if not ALLOW_TENSION:
            pf_check = model.IntegratedPileFoundation(params, cfg.CONFIG['load_cases']['Extreme'], cfg.SOIL_LAYERS,
                                                      (1.35, 1.5))
            _, forces = pf_check.get_individual_pile_forces()
            if forces and min(forces) < 0:
                total_viol += abs(min(forces)) + 10.0

        # --- 提取目标 1: 直接材料总造价 (CAPEX) ---
        pf_basic = model.IntegratedPileFoundation(params, cfg.CONFIG['load_cases']["Normal"], cfg.SOIL_LAYERS)
        Vc, W_cap_steel, _, _ = pf_basic.get_quantities()
        W_pile = pf_basic.N_total * geo['L_pile'] * (rb['pile_main_n'] * math.pi * (rb['pile_main_d'] / 2) ** 2 * 7850)
        Total_Steel = W_cap_steel + W_pile

        Material_Cost = Vc * 500.0 + Total_Steel * 5.5  # 假设 商混500元/方, 钢筋5.5元/kg

        # --- 提取目标 2: 最大裂缝宽度 (Serviceability / Durability) ---
        cap = CapStructuralAnalysis(geo, mat)
        pf_sls = model.IntegratedPileFoundation(params, cfg.CONFIG['load_cases']["Normal"], cfg.SOIL_LAYERS, (1.0, 1.0))
        res_sls, _, _ = pf_sls.get_pile_reactions_detailed()
        _, w_max, _ = cap.check_crack_width(res_sls, rb, detailed=True)

        if total_viol <= 1e-4: total_viol = 0.0

    except Exception as e:
        return (1e10, 1e10, 1e10)

    # 算法同时最小化 (总造价, 最大裂缝)
    return (Material_Cost, w_max, total_viol)


# ================= 3.5 核心重构：带维度隔离的适应度分配引擎 =================
def assign_fitness(ind, cost, crack, viol):
    TOLERANCE = 0.05

    if viol <= TOLERANCE:
        ind.fitness.values = (cost, crack)
        return True
    else:
        BASE_COST = 5000000.0  # 500万
        BASE_CRACK = 2.0  # 2.0 mm 超大裂缝

        penalty_cost = BASE_COST + cost + viol * 1000000.0
        penalty_crack = BASE_CRACK + crack + viol * 10.0

        ind.fitness.values = (penalty_cost, penalty_crack)
        return False


# ================= 4. 手写离散/连续智能变异引擎 =================
def custom_mutate(ind, indpb):
    for i in range(len(ind)):
        if random.random() < indpb:
            low, up = VAR_RANGES[i]
            if i in DISCRETE_MAP or i in INT_INDICES:
                curr = int(round(ind[i]))
                step = random.choice([-1, 1, 2, -2])
                new_val = curr + step
                ind[i] = max(int(low), min(int(up), new_val))
            else:
                sigma = (up - low) * 0.15
                val = ind[i] + random.gauss(0, sigma)
                ind[i] = max(low, min(up, val))
    return ind,


# ================= 5. DEAP 架构注册 =================
creator.create("FitnessMulti", base.Fitness, weights=(-1.0, -1.0))  # 均为极小化目标
creator.create("Individual", list, fitness=creator.FitnessMulti)

toolbox = base.Toolbox()
toolbox.register("population", lambda n: [creator.Individual(ind) for ind in lhs_sample(len(VAR_RANGES), n)])
toolbox.register("evaluate", evaluate_moo)

low_bounds = [b[0] for b in VAR_RANGES]
up_bounds = [b[1] for b in VAR_RANGES]
toolbox.register("mate", tools.cxSimulatedBinaryBounded, low=low_bounds, up=up_bounds, eta=20.0)

toolbox.register("select", tools.selNSGA2)
toolbox.register("mate_select", tools.selTournamentDCD)
toolbox.register("mutate", custom_mutate, indpb=0.2)


# ================= 6. 核心逻辑循环 =================
def main():
    global CURRENT_ITER, ALLOW_TENSION, FIXED_PILE_LENGTH
    print("=" * 60)
    print("   风机基础多目标优化程序 (双目标: 材料造价 vs 结构最大裂缝)")
    print("=" * 60)

    while True:
        u_in = input(">> 是否允许出现受拉桩？(y/n) [默认:y] : ").strip().lower()
        if u_in in ['y', 'yes', '']:
            ALLOW_TENSION = True;
            print("=> 【已选择】允许受拉模式");
            break
        elif u_in in ['n', 'no']:
            ALLOW_TENSION = False;
            print("=> 【已选择】全抗压模式");
            break

    while True:
        l_in = input(">> 是否手动锁定桩长？(输入数字，回车自动) : ").strip()
        if l_in == '':
            FIXED_PILE_LENGTH = None;
            print("=> 【已选择】算法自动寻优");
            break
        else:
            try:
                FIXED_PILE_LENGTH = float(l_in);
                print(f"=> 【已选择】桩长锁定为: {FIXED_PILE_LENGTH} m");
                break
            except ValueError:
                print("输入无效。")

    ga_params = cfg.CONFIG['GA_PARAMS']
    pop_size = max(1000, ga_params['POP_SIZE'])
    ngen = ga_params['N_GEN']

    use_cores = 16  # 根据你的电脑配置调整
    pool = multiprocessing.Pool(processes=use_cores, initializer=init_worker,
                                initargs=(ALLOW_TENSION, FIXED_PILE_LENGTH))
    toolbox.register("map", pool.map)

    pop = toolbox.population(pop_size)
    feasible_archive = []

    print(f"\n--- 启动 NSGA-II 优化引擎 (核心数: {use_cores}, 种群: {pop_size}, 代数: {ngen}) ---")

    # 初代评估
    fitnesses = toolbox.map(toolbox.evaluate, pop)
    for ind, (cost, crack, viol) in zip(pop, fitnesses):
        ind.cost_score, ind.crack_score, ind.viol = cost, crack, viol
        is_feasible = assign_fitness(ind, cost, crack, viol)
        if is_feasible:
            clean_ind = toolbox.clone(ind)
            feasible_archive.append(clean_ind)

    pop = toolbox.select(pop, len(pop))

    pbar = tqdm(range(1, ngen + 1), desc="🚀 NSGA-II 进化中", unit="gen", ncols=120)
    for gen in pbar:
        CURRENT_ITER = gen

        offspring = toolbox.mate_select(pop, len(pop))
        offspring = [toolbox.clone(ind) for ind in offspring]

        for ind1, ind2 in zip(offspring[::2], offspring[1::2]):
            if random.random() <= ga_params['CX_PROB']:
                toolbox.mate(ind1, ind2)
                del ind1.fitness.values;
                del ind2.fitness.values
        for ind in offspring:
            if random.random() <= ga_params['MUT_PROB']:
                toolbox.mutate(ind)
                del ind.fitness.values

        # 子代评估
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
        for ind, (cost, crack, viol) in zip(invalid_ind, fitnesses):
            ind.cost_score, ind.crack_score, ind.viol = cost, crack, viol
            is_feasible = assign_fitness(ind, cost, crack, viol)
            if is_feasible:
                clean_ind = toolbox.clone(ind)
                feasible_archive.append(clean_ind)

        combined_pop = pop + offspring
        pop[:] = toolbox.select(combined_pop, pop_size)

        valid_pop = [ind for ind in pop if ind.viol <= 0.05]
        if valid_pop:
            best_cost = min(valid_pop, key=lambda x: x.cost_score).cost_score
            min_crack = min(valid_pop, key=lambda x: x.crack_score).crack_score
            pbar.set_postfix({"Archive": len(feasible_archive), "Cost(万)": f"{best_cost / 10000:.1f}",
                              "Crack(mm)": f"{min_crack:.3f}"})
        else:
            pbar.set_postfix({"Archive": len(feasible_archive), "Status": "Searching..."})

    pool.close();
    pool.join()

    # ================= 7. 最终极速提取与全景出图 =================
    print("\n\n" + "=" * 80)
    print(f"【优化完成】历史共捕获 {len(feasible_archive)} 个合法解，准备执行极速 O(N log N) 非支配排序...")

    if not feasible_archive:
        print("\n❌ 严重警告：全程未找到完全满足所有规范约束的有效解！请放宽搜索空间。")
        return

    # 去重
    unique_feasible = {}
    for ind in feasible_archive:
        key = (round(ind.crack_score, 3), round(ind.cost_score, 0))
        if key not in unique_feasible:
            unique_feasible[key] = ind

    final_candidates = list(unique_feasible.values())
    print(f"  -> 剔除物理克隆体后，剩余 {len(final_candidates)} 个独立候选方案。")

    # 排序：按裂缝从小到大排序 (裂缝越小，耐久性越好，造价势必越高)
    final_candidates.sort(key=lambda ind: (ind.crack_score, ind.cost_score))

    # ================= 纯净的帕累托前沿提纯逻辑 =================
    true_pareto_front = [final_candidates[0]]
    min_cost = final_candidates[0].cost_score

    for ind in final_candidates[1:]:
        # 裂缝逐渐变大的过程中，必须能实质性地降低造价才算有效解 (至少省 500 块钱)
        if ind.cost_score < min_cost - 500.0:
            true_pareto_front.append(ind)
            min_cost = ind.cost_score

    print(f"  -> 极速提纯完成，最终得到 {len(true_pareto_front)} 个极其密集的帕累托解！")

    # ================= 8. 纯净版：帕累托前沿局部放大可视化 =================
    crack_front = [ind.crack_score for ind in true_pareto_front]
    cost_front = [ind.cost_score / 10000.0 for ind in true_pareto_front]  # 转为万元

    plt.figure(figsize=(10, 6))

    plt.plot(crack_front, cost_front, c='#3498db', linestyle='-', linewidth=2.5, alpha=0.8, zorder=4)
    plt.scatter(crack_front, cost_front, c='#e74c3c', s=60, alpha=1.0, edgecolors='k', linewidths=1.0, zorder=5,
                label=f'Pareto Front (共 {len(true_pareto_front)} 个)')

    # --- 归一化寻找拐点 (Knee Point) 并在图上高亮 ---
    knee_idx = 0
    if len(true_pareto_front) >= 3:
        # 因为造价和裂缝量纲差距巨大(几十 vs 零点几)，必须归一化后再计算正交距离
        pts = np.array([[ind.crack_score, ind.cost_score / 10000.0] for ind in true_pareto_front])
        pts_min = pts.min(axis=0)
        pts_max = pts.max(axis=0)
        range_pts = pts_max - pts_min
        range_pts[range_pts == 0] = 1e-5
        pts_norm = (pts - pts_min) / range_pts  # 归一化到 [0, 1]

        p1, p2 = pts_norm[0], pts_norm[-1]
        line_vec = p2 - p1
        line_len = np.linalg.norm(line_vec)
        if line_len > 1e-5:
            line_unitvec = line_vec / line_len
            vec_from_p1 = pts_norm - p1
            proj_len = np.dot(vec_from_p1, line_unitvec)
            dist = np.linalg.norm(vec_from_p1 - np.outer(proj_len, line_unitvec), axis=1)
            knee_idx = int(np.argmax(dist))

        plt.scatter([crack_front[0], crack_front[knee_idx], crack_front[-1]],
                    [cost_front[0], cost_front[knee_idx], cost_front[-1]],
                    c='yellow', s=350, marker='*', edgecolors='k', linewidths=1.5, zorder=6,
                    label='三大典型分析点 (极致耐久、高性价比、极简成本)')

    plt.title('多目标优化：结构安全性(最大裂缝) vs 直接建造成本', fontsize=16, fontweight='bold')
    plt.xlabel('SLS 工况下承台最大裂缝宽度 (mm)', fontsize=13)
    plt.ylabel('直接材料总造价 (万元)', fontsize=13)

    if len(crack_front) > 1 and len(cost_front) > 1:
        crack_margin = max(0.005, (max(crack_front) - min(crack_front)) * 0.10)
        cost_margin = max(2.0, (max(cost_front) - min(cost_front)) * 0.10)
        plt.xlim(min(crack_front) - crack_margin, max(crack_front) + crack_margin)
        plt.ylim(min(cost_front) - cost_margin, max(cost_front) + cost_margin)

    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(fontsize=12, loc='upper right')
    plt.tight_layout()
    plt.savefig('pareto_front_crack_cost.png', dpi=300)
    plt.show()

    # ================= 9. 方案输出与主控因素解析 =================
    if true_pareto_front:
        timestamp = datetime.now().strftime("%m%d_%H%M")

        if len(true_pareto_front) >= 3:
            p_left = true_pareto_front[0]
            p_knee = true_pareto_front[knee_idx]
            p_right = true_pareto_front[-1]

            print("\n" + "★" * 65)
            print(" 方案 A: 【极致耐久方案】(裂缝极小，但堆砌材料导致造价高昂)")
            print("★" * 65)
            report_and_export_native(p_left, f"abaqus_min_crack_{timestamp}.json", "方案 A (极致耐久)")

            print("\n" + "★" * 65)
            print(" 方案 B: 【高性价比拐点方案】(兼顾经济性与裂缝冗余度的平衡解)")
            print("★" * 65)
            report_and_export_native(p_knee, f"abaqus_knee_{timestamp}.json", "方案 B (拐点方案)")

            print("\n" + "★" * 65)
            print(" 方案 C: 【极致省钱方案】(造价被压榨到极限，裂缝逼近规范红线)")
            print("★" * 65)
            report_and_export_native(p_right, f"abaqus_min_cost_{timestamp}.json", "方案 C (极致省钱)")
        else:
            report_and_export_native(true_pareto_front[0], f"abaqus_min_crack_{timestamp}.json", "极致耐久方案")
            report_and_export_native(true_pareto_front[-1], f"abaqus_min_cost_{timestamp}.json", "极致省钱方案")


def report_and_export_native(ind, export_name, label="最优方案"):
    params = decode_gene(ind)
    Total_Cost_W = ind.cost_score / 10000.0
    w_max_score = ind.crack_score

    geo, rb, mat = params['geo'], params['rebar'], params['mat']
    load_uls = cfg.CONFIG['load_cases']["Extreme"]
    load_sls = cfg.CONFIG['load_cases']["Normal"]

    pf_uls = model.IntegratedPileFoundation(params, load_uls, cfg.SOIL_LAYERS, (1.35, 1.5))
    res_uls, Fxy_uls, _ = pf_uls.get_pile_reactions_detailed()

    pf_sls = model.IntegratedPileFoundation(params, load_sls, cfg.SOIL_LAYERS, (1.0, 1.0))
    res_sls, Fxy_sls, _ = pf_sls.get_pile_reactions_detailed()

    cap = CapStructuralAnalysis(geo, mat)
    pile = PileStructuralAnalysis(geo['D_pile'], rb['pile_main_n'], rb['pile_main_d'], mat, params['stirrup'],
                                  geo.get('cover', 0.06))

    _, m_req, m_prov = cap.check_flexural_reinforcement(res_uls, rb, detailed=True)
    _, v_req, v_prov = cap.check_inclined_section_shear(res_uls, rb['d1'] * 1000, detailed=True)
    _, p1_req, p1_prov = cap.check_corner_pile_punching(res_uls, geo['D_pile'], detailed=True)
    _, p2_req, p2_prov = cap.check_pedestal_punching(res_uls, rb['d1'] * 1000, detailed=True)
    _, w_max, w_limit = cap.check_crack_width(res_sls, rb, detailed=True)

    Ra, _ = pile.calculate_compression_capacity(geo['L_pile'], cfg.SOIL_LAYERS)
    N_max_sls = max([p['N_design'] for p in res_sls])

    N_min_sls = min([p['N_design'] for p in res_sls])
    if N_min_sls < 0:
        piles_coords = [(p['x'], p['y']) for p in res_sls]
        Rt = pile.calculate_uplift_capacity(geo['L_pile'], cfg.SOIL_LAYERS, piles_coords)
        util_uplift = (abs(N_min_sls) / Rt) * 100
    else:
        util_uplift = 0.0

    utils = {
        '承台受剪利用率 (ULS)': (v_req / (v_prov + 1e-5)) * 100,
        '角桩冲切利用率 (ULS)': (p1_req / (p1_prov + 1e-5)) * 100,
        '塔筒冲切利用率 (ULS)': (p2_req / (p2_prov + 1e-5)) * 100,
        '承台抗弯利用率 (ULS)': (m_req / (m_prov + 1e-5)) * 100,
        '最大裂缝逼近率 (SLS)': (w_max / w_limit) * 100,
        '单桩抗压利用率 (SLS)': (N_max_sls / Ra) * 100,
        '单桩抗拔利用率 (SLS)': util_uplift,
    }

    main_factor = max(utils, key=utils.get)
    max_util = utils[main_factor]

    print(f"========== {label} 深度验算与主控分析 ==========")
    print(f"【双目标呈现】 最大裂缝宽度 = {w_max_score:.3f} mm | 直接总造价 = {Total_Cost_W:.1f} 万元")
    print(f"【核心几何图】 承台厚度 H={geo['H']:.1f}m | 基础半径 R={geo['R_cap']:.1f}m | 桩长 L={geo['L_pile']:.1f}m")

    print(f"\n【★ 主控因素锁定 ★】")
    print(f"  >>>> 该方案被卡死在瓶颈: 【{main_factor}】, 满载率高达 {max_util:.2f}% <<<<")

    print(f"\n【各项满载率明细 (雷达图数据)】")
    for k, v in utils.items():
        if v >= 99.0:
            alert = " 🔴 (极限卡脖子!)"
        elif v >= 90.0:
            alert = " 🟡 (高度危险/冗余不足)"
        else:
            alert = " 🟢 (安全余裕充分)"

        print(f"   - {k.ljust(18)}: {v:>6.2f}% {alert}")
    print("=" * 65 + "\n")

    soil_data = [{"name": l.name, "thickness_m": l.thickness, "density_kN_m3": l.gamma, "elastic_modulus_MPa": l.Es,
                  "poisson_ratio": l.v, "cohesion_kPa": l.C, "friction_angle_deg": l.angle, "qsik_kPa": l.qsik,
                  "qpk_kPa": l.qpk} for l in cfg.SOIL_LAYERS]
    export_data = {
        "pile_property": {
            "geometry": {"D_pile_m": geo['D_pile'], "L_pile_m": geo['L_pile'], "cover_thickness_m": 0.06},
            "rebar": {"main_rebar_n": rb['pile_main_n'], "main_rebar_d_m": rb['pile_main_d'],
                      "stirrup_d_m": params['stirrup']['d']},
            "concrete_material": {"fc_MPa": mat.get('fc', 19.1), "ft_MPa": mat.get('ft', 1.71),
                                  "Ec_MPa": mat.get('Ec', 3.25e4)}
        },
        "soil_layers": soil_data,
        "critical_loads_uls": {
            "max_compression_kN": round(max([p['N_design'] for p in res_uls]), 2),
            "max_tension_kN": round(min([0.0] + [p['N_design'] for p in res_uls]), 2),
            "max_shear_kN": round(max([p.get('H_actual', Fxy_uls / pf_uls.N_total) for p in res_uls]), 2)
        }
    }
    os.makedirs(ABAQUS_OUTPUT_DIR, exist_ok=True)
    output_path = export_name if os.path.isabs(export_name) else os.path.join(ABAQUS_OUTPUT_DIR, export_name)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(export_data, f, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
