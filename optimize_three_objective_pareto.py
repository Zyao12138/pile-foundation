# -*- coding: utf-8 -*-
"""
文件名: optimize_three_objective_pareto.py (NSGA-II 终极极速云图版 - 3D 多目标扩展版 v3.4)
功能: 风机基础智能优化主程序 - 寻找混凝土、钢筋与占地面积的真实3D帕累托前沿
核心升级:
  - [三维目标] 引入基础占地面积作为第三个极小化目标，实现真正的 3D 空间寻优。
  - [极速提纯] 引入 DEAP 原生 3D 非支配排序，解决三维空间下的帕累托前沿提取。
  - [全景报告] 修复 17 项全景报告中由于工况错位、漏传 PM 曲线参数导致的虚假红叉。
  - [评估闭环] 彻底消除“评估不对称”：将最终报告的 17 项严苛校验内嵌至 GA 评估循环，彻底绞杀任何违规边缘解。
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
    from analyze_pile_structure import PileStructuralAnalysis, pm_manager

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


# ================= 1. 变量初始化与解析 =================
def parse_config_vars():
    raw_config = cfg.CONFIG["var_ranges"]
    var_ranges = []
    int_indices = []
    discrete_map = {}
    for i, item in enumerate(raw_config):
        if isinstance(item, tuple):
            var_ranges.append(item)
        elif isinstance(item, list):
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
        n = max(10, int(round(raw_n / 10.0) * 10))
        perimeter_mm = 2 * math.pi * R_m * 1000
        eff_n = n * 2 if is_edge_doubled else n
        if perimeter_mm / eff_n > 300.0:
            req_eff_n = int(math.ceil((perimeter_mm / 300.0) / 10.0) * 10)
            n = req_eff_n // 2 if is_edge_doubled else req_eff_n
        elif perimeter_mm / eff_n < 100.0:
            req_eff_n = int(math.floor((perimeter_mm / 100.0) / 10.0) * 10)
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


# ================= 3. 多目标评估函数 (核心重构：内嵌最严苛 17 项终极拦截) =================
def evaluate_moo(ind):
    global ALLOW_TENSION
    if np.any(np.isnan(ind)): return (1e10, 1e10, 1e10, 1e10)

    try:
        safe_values = []
        for i, val in enumerate(ind):
            low, up = VAR_RANGES[i]
            safe_val = max(low, min(up, val))
            safe_values.append(int(round(safe_val)) if i in INT_INDICES else safe_val)

        params = decode_gene(safe_values)
        geo, rb, mat = params['geo'], params['rebar'], params['mat']

        # 1. 基础物理黑盒评估
        total_viol, _ = model.assess_design_performance(params)

        # 2. 附加几何干涉控制
        ach = cfg.CONFIG.get('ANCHOR_CAGE', {})
        L_tot, bg, te, bc = ach.get('L_total', 4.2), ach.get('bottom_groove', 0.3), ach.get('top_expose', 0.3), ach.get(
            'bottom_cover', 0.2)
        if (geo['h1'] + geo['h2'] + geo['h3'] - bg) < (L_tot - te + bc):
            total_viol += ((L_tot - te + bc) - (geo['h1'] + geo['h2'] + geo['h3'] - bg)) * 50.0

        Do = ach.get('D_outer', 5.1)
        if geo['r_col'] < (Do / 2.0) + 0.6: total_viol += ((Do / 2.0) + 0.6 - geo['r_col']) * 50.0

        # =========================================================================
        # 3. [核心注入] 强制将终端报告的 17 项最严校验内嵌到进化循环中，实现评估闭环
        # =========================================================================
        if HAS_CHECK_MODULES:
            load_uls = cfg.CONFIG['load_cases']["Extreme"]
            load_sls = cfg.CONFIG['load_cases']["Normal"]

            pf_uls = model.IntegratedPileFoundation(params, load_uls, cfg.SOIL_LAYERS, (1.35, 1.5))
            pf_sls = model.IntegratedPileFoundation(params, load_sls, cfg.SOIL_LAYERS, (1.0, 1.0))

            res_uls, Fxy_uls, _ = pf_uls.get_pile_reactions_detailed()
            res_sls, Fxy_sls, _ = pf_sls.get_pile_reactions_detailed()

            # [模块 A: 承台严格验算]
            cap_analyzer = CapStructuralAnalysis(geo, mat, cover_mm=50.0)
            total_viol += max(0, cap_analyzer.check_flexural_reinforcement(res_uls, rb))
            total_viol += max(0, cap_analyzer.check_inclined_section_shear(res_uls, rb['d1'] * 1000))
            total_viol += max(0, cap_analyzer.check_crack_width(res_sls, rb))  # SLS工况裂缝直接纳入GA惩罚！
            total_viol += max(0, cap_analyzer.check_corner_pile_punching(res_uls, geo['D_pile']))
            total_viol += max(0, cap_analyzer.check_pedestal_punching(res_uls, rb['d1'] * 1000))

            # [模块 B & C: 桩基严格验算]
            pile_analyzer = PileStructuralAnalysis(geo['D_pile'], rb['pile_main_n'], rb['pile_main_d'], mat,
                                                   params['stirrup'], cover=0.06)
            piles_coords = [(p['x'], p['y']) for p in res_uls]

            Ra_norm, _ = pile_analyzer.calculate_compression_capacity(geo['L_pile'], cfg.SOIL_LAYERS)
            Rt = pile_analyzer.calculate_uplift_capacity(geo['L_pile'], cfg.SOIL_LAYERS, piles_coords)
            Rha_norm, _ = pile_analyzer.calculate_horizontal_capacity(geo['L_pile'], cfg.SOIL_LAYERS)

            N_max_uls = max([p['N_design'] for p in res_uls])
            N_min_uls = min([0.0] + [p['N_design'] for p in res_uls])
            H_uls = Fxy_uls / pf_uls.N_total
            N_max_sls = max([p['N_design'] for p in res_sls])

            if N_max_sls > Ra_norm: total_viol += (N_max_sls - Ra_norm) / Ra_norm * 10.0
            if abs(N_min_uls) > Rt: total_viol += (abs(N_min_uls) - Rt) / Rt * 10.0
            if H_uls > Rha_norm: total_viol += (H_uls - Rha_norm) / Rha_norm * 10.0

            v_weak = pile_analyzer.check_weak_underlying_layer(N_max_sls, 0, geo['L_pile'], cfg.SOIL_LAYERS,
                                                               piles_coords)
            if v_weak > 1.0: total_viol += (v_weak - 1.0) * 10.0

            settle, tilt = pile_analyzer.check_settlement_and_tilt(load_sls['FZ'], load_sls['MXY'], geo['L_pile'],
                                                                   cfg.SOIL_LAYERS, piles_coords)
            if settle > 100.0: total_viol += (settle - 100.0) / 100.0 * 10.0
            if tilt > 0.003: total_viol += (tilt - 0.003) / 0.003 * 10.0

            v_str_comp, _ = pile_analyzer.check_structural_strength(N_max_uls)
            v_str_tens, _ = pile_analyzer.check_tensile_strength(N_min_uls)
            if v_str_comp > 1.0: total_viol += (v_str_comp - 1.0) * 10.0
            if v_str_tens > 1.0: total_viol += (v_str_tens - 1.0) * 10.0

            r_outer = max([p['r'] for p in res_uls]) if res_uls else 0
            util_loc = pile_analyzer.check_pile_top_local_compression(N_max_uls, mat['fc'], geo['R_cap'], r_outer)
            if util_loc > 1.0: total_viol += (util_loc - 1.0) * 10.0

            util_buck, _, _ = pile_analyzer.check_buckling_stability(N_max_uls, geo['L_pile'], cfg.SOIL_LAYERS)
            if util_buck > 1.0: total_viol += (util_buck - 1.0) * 10.0

            m_eq = pile_analyzer._get_equivalent_m_value_internal(cfg.SOIL_LAYERS, geo['L_pile'])
            Rh, x0, M_max_design, alpha = pile_analyzer.calculate_horizontal_mechanics(H_uls, 0, m_eq, piles_coords)

            Mu_cap_real = pm_manager.get_capacity_usage(
                geo['D_pile'], rb['pile_main_n'], rb['pile_main_d'],
                N_max_uls, M_max_design, mat['fc'], 60.0, return_capacity=True
            )
            util_shear, util_bend, _ = pile_analyzer.check_bending_shear_capacity(M_max_design, H_uls, N_max_uls,
                                                                                  Mu_calc_external=Mu_cap_real)
            if util_shear > 1.0: total_viol += (util_shear - 1.0) * 10.0
            if util_bend > 1.0: total_viol += (util_bend - 1.0) * 10.0

            H_sls = Fxy_sls / pf_sls.N_total
            Rh_sls, x0_sls, M_max_sls, alpha_sls = pile_analyzer.calculate_horizontal_mechanics(H_sls, 0, m_eq,
                                                                                                piles_coords)
            w_pile = pile_analyzer.check_crack_width(M_max_sls, N_max_sls)
            if w_pile > 0.2: total_viol += (w_pile - 0.2) / 0.2 * 10.0

            v_rule_viol, _, _ = pile_analyzer.check_rebar_construction_rules()
            total_viol += max(0, v_rule_viol)

            if not ALLOW_TENSION and N_min_uls < 0:
                total_viol += abs(N_min_uls) + 10.0

        # =========================================================================

        # 计算用量目标
        pf_basic = model.IntegratedPileFoundation(params, cfg.CONFIG['load_cases']["Normal"], cfg.SOIL_LAYERS)
        Vc, W_cap_steel, _, _ = pf_basic.get_quantities()
        W_pile = pf_basic.N_total * geo['L_pile'] * (rb['pile_main_n'] * math.pi * (rb['pile_main_d'] / 2) ** 2 * 7850)

        Total_Steel = W_cap_steel + W_pile
        Area = math.pi * (geo['R_cap'] ** 2)

        if total_viol <= 1e-4: total_viol = 0.0

    except Exception as e:
        return (1e10, 1e10, 1e10, 1e10)

    return (Vc, Total_Steel, Area, total_viol)


# ================= 3.5 核心重构：带维度隔离的适应度分配引擎 =================
def assign_fitness(ind, vc, st, area, viol):
    # 【极致过滤】：严格零容忍
    TOLERANCE = 0.0001

    if viol <= TOLERANCE:
        ind.fitness.values = (vc, st, area)
        return True
    else:
        BASE_VC = 20000.0
        BASE_ST = 2000000.0
        BASE_AREA = 10000.0

        penalty_vc = BASE_VC + vc + viol * 100000.0
        penalty_st = BASE_ST + st + viol * 10000000.0
        penalty_area = BASE_AREA + area + viol * 10000.0

        ind.fitness.values = (penalty_vc, penalty_st, penalty_area)
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
creator.create("FitnessMulti", base.Fitness, weights=(-1.0, -1.0, -1.0))
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
    print("   风机基础多目标优化程序 (3D帕累托内嵌严格闭环版 v3.4)")
    print("   优化目标: [1.混凝土用量] [2.钢筋用量] [3.占地面积]")
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

    use_cores = 12
    pool = multiprocessing.Pool(processes=use_cores, initializer=init_worker,
                                initargs=(ALLOW_TENSION, FIXED_PILE_LENGTH))
    toolbox.register("map", pool.map)

    pop = toolbox.population(pop_size)
    feasible_archive = []

    print(f"\n--- 启动 NSGA-II 3D 优化引擎 (核心数: {use_cores}, 种群: {pop_size}, 代数: {ngen}) ---")

    fitnesses = toolbox.map(toolbox.evaluate, pop)
    for ind, (vc, st, area, viol) in zip(pop, fitnesses):
        ind.vc_score, ind.st_score, ind.area_score, ind.viol = vc, st, area, viol
        is_feasible = assign_fitness(ind, vc, st, area, viol)
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

        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
        for ind, (vc, st, area, viol) in zip(invalid_ind, fitnesses):
            ind.vc_score, ind.st_score, ind.area_score, ind.viol = vc, st, area, viol
            is_feasible = assign_fitness(ind, vc, st, area, viol)
            if is_feasible:
                clean_ind = toolbox.clone(ind)
                feasible_archive.append(clean_ind)

        combined_pop = pop + offspring
        pop[:] = toolbox.select(combined_pop, pop_size)

        valid_pop = [ind for ind in pop if ind.viol <= 0.0001]
        if valid_pop:
            best_vc = min(valid_pop, key=lambda x: x.vc_score).vc_score
            pbar.set_postfix({"Archive": len(feasible_archive), "Best_Vc": f"{best_vc:.1f} m³"})
        else:
            pbar.set_postfix({"Archive": len(feasible_archive), "Status": "Searching..."})

    pool.close();
    pool.join()

    # ================= 7. 最终极速提取与全景出图 =================
    print("\n\n" + "=" * 80)
    print(f"【优化完成】历史共捕获 {len(feasible_archive)} 个严格合法解，准备执行 3D 非支配排序...")

    if not feasible_archive:
        print("\n❌ 严重警告：全程未找到完全满足所有规范约束的有效解！")
        print(
            "💡 原因分析：最新版程序开启了最严苛的“评估闭环”，如果地质条件极其恶劣或者搜索空间过小，算法会被全军覆没。请尝试：\n  1. 放大搜索变量的上下界（如承台厚度或配筋范围）\n  2. 换更好的桩端持力层")
        return

    unique_feasible = {}
    for ind in feasible_archive:
        key = (round(ind.vc_score, 2), round(ind.st_score, 1), round(ind.area_score, 2))
        if key not in unique_feasible:
            unique_feasible[key] = ind

    final_candidates = list(unique_feasible.values())
    print(f"  -> 剔除物理克隆体后，剩余 {len(final_candidates)} 个独立候选方案。")

    fronts = tools.sortNondominated(final_candidates, len(final_candidates), first_front_only=True)
    true_pareto_front = fronts[0] if fronts else []
    true_pareto_front.sort(key=lambda ind: ind.vc_score)

    print(f"  -> 提取完成，最终得到 {len(true_pareto_front)} 个三维空间下的高价值帕累托解！")

    # ================= 8. 纯净版：3D帕累托前沿可视化 (恢复出图) =================
    vc_front = [ind.vc_score for ind in true_pareto_front]
    st_front = [ind.st_score for ind in true_pareto_front]
    area_front = [ind.area_score for ind in true_pareto_front]

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    p = ax.scatter(vc_front, st_front, area_front, c=area_front, cmap='viridis',
                   s=60, alpha=0.9, edgecolors='k', linewidths=0.5,
                   label=f'非支配解集 (共 {len(true_pareto_front)} 个)')

    ax.set_title('多目标优化：3D 帕累托前沿空间', fontsize=15, fontweight='bold')
    ax.set_xlabel('混凝土总方量 ($m^3$)', fontsize=11)
    ax.set_ylabel('钢筋总重量 (kg)', fontsize=11)
    ax.set_zlabel('基础占地面积 ($m^2$)', fontsize=11)
    cbar = fig.colorbar(p, ax=ax, shrink=0.7, pad=0.1)
    cbar.set_label('占地面积 ($m^2$)', fontsize=11)

    ax.grid(True, linestyle='--', alpha=0.5)
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig('pareto_front_3d.png', dpi=300)

    print("📊 正在绘制 3D 帕累托图，请查看弹窗...")
    plt.show()  # 恢复弹窗出图

    # ================= 9. 方案输出 (展示17项全景报告) =================
    if true_pareto_front:
        timestamp = datetime.now().strftime("%m%d_%H%M")

        print("\n" + "★" * 50)
        print(" 【极致省商混方案】(三维帕累托极限前沿 - 综合表现最优解)")
        print("★" * 50)
        report_and_export_native(true_pareto_front[0], f"abaqus_min_concrete_{timestamp}.json")


# ================= 10. 【终极重构】17 项全量物理报告输出引擎 =================
def report_and_export_native(ind, export_name):
    params = decode_gene(ind)
    V_conc, Total_Steel_kg, Area_m2 = ind.vc_score, ind.st_score, ind.area_score

    geo, rb, mat = params['geo'], params['rebar'], params['mat']

    load_uls = cfg.CONFIG['load_cases']["Extreme"]
    load_sls = cfg.CONFIG['load_cases']["Normal"]

    pf_uls = model.IntegratedPileFoundation(params, load_uls, cfg.SOIL_LAYERS, (1.35, 1.5))
    pf_sls = model.IntegratedPileFoundation(params, load_sls, cfg.SOIL_LAYERS, (1.0, 1.0))

    res_uls, Fxy_uls, _ = pf_uls.get_pile_reactions_detailed()
    res_sls, Fxy_sls, _ = pf_sls.get_pile_reactions_detailed()

    mode_str = " (允许受拉模式)" if ALLOW_TENSION else " (全抗压模式)"
    print(f"================== 风机基础优化设计报告{mode_str} ==================")
    print(f"【1. 几何与总量】")
    print(f"  [承台] 埋深 H={geo['H']:.1f} m, 底部半径 R={geo['R_cap']:.1f} m")
    print(f"  [桩基] 共 {pf_uls.N_total} 根, 桩径 D={geo['D_pile']:.1f} m, 桩长 L={geo['L_pile']:.1f} m")
    print(f"  >>> 混凝土总量 = {V_conc:.1f} m³ | 钢筋总重量 = {Total_Steel_kg:.1f} kg | 基础占地 = {Area_m2:.1f} m²")

    print(f"\n【2. 结构规范验算详细结果 (17 项核心规范全景对标)】")
    try:
        if HAS_CHECK_MODULES:
            def fmt(passed, text):
                return f"{'✅ 通过' if passed else '❌ 警告/超限'} ({text})"

            cap_analyzer = CapStructuralAnalysis(geo, mat, cover_mm=50.0)

            v1, req1, prov1 = cap_analyzer.check_flexural_reinforcement(res_uls, rb, detailed=True)
            v2, v_des2, v_cap2 = cap_analyzer.check_inclined_section_shear(res_uls, rb['d1'] * 1000, detailed=True)
            v3, w_max3, w_lim3 = cap_analyzer.check_crack_width(res_sls, rb, detailed=True)
            v4, n_des4, f_cap4 = cap_analyzer.check_corner_pile_punching(res_uls, geo['D_pile'], detailed=True)
            v5, f_shr5, f_cap5 = cap_analyzer.check_pedestal_punching(res_uls, rb['d1'] * 1000, detailed=True)

            print(f"  --- 承台主体验算 (Cap Structural) ---")
            print(f"  [ 1] 承台抗弯验算 : {fmt(v1 <= 0, f'需配筋: {req1:.1f} mm²/m, 实配筋: {prov1:.1f} mm²/m')}")
            print(f"  [ 2] 斜截面抗剪   : {fmt(v2 <= 0, f'剪力设计值: {v_des2:.1f} kN, 抗剪承载力: {v_cap2:.1f} kN')}")
            print(f"  [ 3] 承台裂缝宽度 : {fmt(v3 <= 0, f'最大裂缝: {w_max3:.3f} mm, 规范限值: {w_lim3:.2f} mm')}")
            print(f"  [ 4] 角桩冲切验算 : {fmt(v4 <= 0, f'冲切力极值: {n_des4:.1f} kN, 冲切承载力: {f_cap4:.1f} kN')}")
            print(f"  [ 5] 台柱冲切验算 : {fmt(v5 <= 0, f'总剪切荷载: {f_shr5:.1f} kN, 冲切承载力: {f_cap5:.1f} kN')}")

            pile_analyzer = PileStructuralAnalysis(geo['D_pile'], rb['pile_main_n'], rb['pile_main_d'], mat,
                                                   params['stirrup'], cover=0.06)
            piles_coords = [(p['x'], p['y']) for p in res_uls]

            Ra_norm, _ = pile_analyzer.calculate_compression_capacity(geo['L_pile'], cfg.SOIL_LAYERS)
            Rt = pile_analyzer.calculate_uplift_capacity(geo['L_pile'], cfg.SOIL_LAYERS, piles_coords)
            Rha_norm, _ = pile_analyzer.calculate_horizontal_capacity(geo['L_pile'], cfg.SOIL_LAYERS)

            N_max_uls = max([p['N_design'] for p in res_uls])
            N_min_uls = min([0.0] + [p['N_design'] for p in res_uls])
            H_uls = Fxy_uls / pf_uls.N_total

            N_max_sls = max([p['N_design'] for p in res_sls])
            v_weak = pile_analyzer.check_weak_underlying_layer(N_max_sls, 0, geo['L_pile'], cfg.SOIL_LAYERS,
                                                               piles_coords)
            settle, tilt = pile_analyzer.check_settlement_and_tilt(load_sls['FZ'], load_sls['MXY'], geo['L_pile'],
                                                                   cfg.SOIL_LAYERS, piles_coords)

            print(f"  --- 桩基承载力与岩土 (Pile Geotechnical) ---")
            print(
                f"  [ 6] 单桩抗压承载力 : {fmt(N_max_sls <= Ra_norm, f'最大标准轴压: {N_max_sls:.1f} kN, 承压特征值: {Ra_norm:.1f} kN')}")
            if N_min_uls < 0:
                print(
                    f"  [ 7] 单桩抗拔承载力 : {fmt(abs(N_min_uls) <= Rt, f'最大拔力: {abs(N_min_uls):.1f} kN, 抗拔特征值: {Rt:.1f} kN')}")
            else:
                print(f"  [ 7] 单桩抗拔承载力 : ✅ 通过 (地勘与荷载组合下全截面受压，无拉拔力)")
            print(
                f"  [ 8] 单桩水平承载力 : {fmt(H_uls <= Rha_norm, f'单桩水平力: {H_uls:.1f} kN, 水平特征值: {Rha_norm:.1f} kN')}")
            if v_weak > 0:
                print(f"  [ 9] 软弱下卧层验算 : {fmt(v_weak <= 1.0, f'附加应力比值: {v_weak:.3f} ≤ 1.0')}")
            else:
                print(f"  [ 9] 软弱下卧层验算 : ✅ 通过 (桩端下未检测到软弱土层)")
            print(
                f"  [10] 沉降与倾斜验算 : {fmt(settle <= 100.0 and tilt <= 0.003, f'沉降预估: {settle:.1f} mm, 倾斜率: {tilt:.5f}')}")

            v_str_comp, N_cap_str = pile_analyzer.check_structural_strength(N_max_uls)
            v_str_tens, T_cap_str = pile_analyzer.check_tensile_strength(N_min_uls)

            r_outer = max([p['r'] for p in res_uls]) if res_uls else 0
            util_loc = pile_analyzer.check_pile_top_local_compression(N_max_uls, mat['fc'], geo['R_cap'], r_outer)
            util_buck, N_buck, phi_buck = pile_analyzer.check_buckling_stability(N_max_uls, geo['L_pile'],
                                                                                 cfg.SOIL_LAYERS)

            m_eq = pile_analyzer._get_equivalent_m_value_internal(cfg.SOIL_LAYERS, geo['L_pile'])
            Rh, x0, M_max_design, alpha = pile_analyzer.calculate_horizontal_mechanics(H_uls, 0, m_eq, piles_coords)

            Mu_cap_real = pm_manager.get_capacity_usage(
                geo['D_pile'], rb['pile_main_n'], rb['pile_main_d'],
                N_max_uls, M_max_design, mat['fc'], 60.0, return_capacity=True
            )
            util_shear, util_bend, Vu = pile_analyzer.check_bending_shear_capacity(M_max_design, H_uls, N_max_uls,
                                                                                   Mu_calc_external=Mu_cap_real)

            H_sls = Fxy_sls / pf_sls.N_total
            Rh_sls, x0_sls, M_max_sls, alpha_sls = pile_analyzer.calculate_horizontal_mechanics(H_sls, 0, m_eq,
                                                                                                piles_coords)
            w_pile = pile_analyzer.check_crack_width(M_max_sls, N_max_sls)

            v_rule_viol, v_rule_rho, v_rule_s = pile_analyzer.check_rebar_construction_rules()

            print(f"  --- 桩基结构与构造 (Pile Structural) ---")
            print(
                f"  [11] 桩身受压强度   : {fmt(v_str_comp <= 1.0, f'最大轴压: {N_max_uls:.1f} kN, 截面承压能力: {N_cap_str:.1f} kN')}")
            if N_min_uls < 0:
                print(
                    f"  [12] 桩身受拉强度   : {fmt(v_str_tens <= 1.0, f'最大拔力: {abs(N_min_uls):.1f} kN, 截面抗拉能力: {T_cap_str:.1f} kN')}")
            else:
                print(f"  [12] 桩身受拉强度   : ✅ 通过 (全截面受压，无需额外抗拉)")
            print(f"  [13] 桩顶局部受压   : {fmt(util_loc <= 1.0, f'局部受压利用率: {util_loc * 100:.1f}%')}")
            print(
                f"  [14] 压屈稳定验算   : {fmt(util_buck <= 1.0, f'稳定系数φ: {phi_buck:.2f}, 临界承载力: {N_buck:.1f} kN')}")
            print(
                f"  [15] 受弯受剪承载力 : {fmt(util_bend <= 1.0 and util_shear <= 1.0, f'抗弯利用率: {util_bend * 100:.1f}%, 抗剪利用率: {util_shear * 100:.1f}%')}")
            print(f"  [16] 桩身裂缝宽度   : {fmt(w_pile <= 0.20, f'最大裂缝: {w_pile:.3f} mm, 规范限值: 0.20 mm')}")
            print(
                f"  [17] 钢筋构造限值   : {fmt(v_rule_viol <= 0, f'实际配筋率: {v_rule_rho * 100:.2f}%, 钢筋净距: {v_rule_s:.1f} mm')}")

    except Exception as e:
        print(f"  ⚠️ 详细明细提取出现异常: {e}")

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
    print(f"\n[Abaqus 接口] 方案模型已成功导出至: {output_path}\n")


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
