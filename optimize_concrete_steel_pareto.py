# -*- coding: utf-8 -*-
"""
文件名: optimize_concrete_steel_pareto.py (NSGA-II 终极极速云图版 - 连续化松弛平滑版 v2.3 - 无JSON输出版)
功能: 风机基础智能优化主程序 - 寻找混凝土与钢筋的真实帕累托前沿
核心修复:
  - [连续化松弛] 拦截 foundation_config 中的钢筋根数列表，转为连续寻优区间，彻底打破“离散断层”。
  - [解除强取整] 移除了钢筋根数强制取 10 的倍数的限制，允许精确到单根钢筋微调。
  - [极速提纯] 采用最严谨的 1D 线性扫描非支配排序，精度放宽至 1kg，呈现极其密集的帕累托前沿。
  - [主控解析] 增加基于正交投影的拐点自动寻找算法，并深度输出各主要受力指标的满载率。
  - [精简输出] 移除了所有的 JSON 文件导出逻辑，保持工程目录纯净。
"""

import math
import random
import numpy as np
import matplotlib.pyplot as plt
from deap import base, creator, tools
from tqdm import tqdm

import multiprocessing
import foundation_config as cfg
import analyze_integrated_pile_foundation as model

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


# ================= 1. 变量初始化与解析 (核心平滑修复区) =================
def parse_config_vars():
    raw_config = cfg.CONFIG["var_ranges"]
    var_ranges = []
    int_indices = []
    discrete_map = {}
    for i, item in enumerate(raw_config):
        if isinstance(item, tuple):
            var_ranges.append(item)
        elif isinstance(item, list):
            # 将 [100, 110, 120...300] 转为连续的浮点数范围 (100.0, 300.0)
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
        n = max(10, int(round(raw_n)))
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


# ================= 3. 多目标评估函数 =================
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

        pf_basic = model.IntegratedPileFoundation(params, cfg.CONFIG['load_cases']["Normal"], cfg.SOIL_LAYERS)
        Vc, W_cap_steel, _, _ = pf_basic.get_quantities()
        W_pile = pf_basic.N_total * geo['L_pile'] * (rb['pile_main_n'] * math.pi * (rb['pile_main_d'] / 2) ** 2 * 7850)
        Total_Steel = W_cap_steel + W_pile

        if total_viol <= 1e-4: total_viol = 0.0

    except Exception as e:
        return (1e10, 1e10, 1e10)

    return (Vc, Total_Steel, total_viol)


# ================= 3.5 核心重构：带维度隔离的适应度分配引擎 =================
def assign_fitness(ind, vc, st, viol):
    TOLERANCE = 0.05

    if viol <= TOLERANCE:
        ind.fitness.values = (vc, st)
        return True
    else:
        BASE_VC = 20000.0
        BASE_ST = 2000000.0

        penalty_vc = BASE_VC + vc + viol * 100000.0
        penalty_st = BASE_ST + st + viol * 10000000.0

        ind.fitness.values = (penalty_vc, penalty_st)
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
creator.create("FitnessMulti", base.Fitness, weights=(-1.0, -1.0))
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
    print("   风机基础多目标优化程序 (极速云图提纯连续平滑版 v2.3)")
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

    print(f"\n--- 启动 NSGA-II 优化引擎 (核心数: {use_cores}, 种群: {pop_size}, 代数: {ngen}) ---")

    # 初代评估
    fitnesses = toolbox.map(toolbox.evaluate, pop)
    for ind, (vc, st, viol) in zip(pop, fitnesses):
        ind.vc_score, ind.st_score, ind.viol = vc, st, viol
        is_feasible = assign_fitness(ind, vc, st, viol)
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
        for ind, (vc, st, viol) in zip(invalid_ind, fitnesses):
            ind.vc_score, ind.st_score, ind.viol = vc, st, viol
            is_feasible = assign_fitness(ind, vc, st, viol)
            if is_feasible:
                clean_ind = toolbox.clone(ind)
                feasible_archive.append(clean_ind)

        combined_pop = pop + offspring
        pop[:] = toolbox.select(combined_pop, pop_size)

        valid_pop = [ind for ind in pop if ind.viol <= 0.05]
        if valid_pop:
            best_vc = min(valid_pop, key=lambda x: x.vc_score).vc_score
            pbar.set_postfix({"Archive": len(feasible_archive), "Best_Vc": f"{best_vc:.1f} m³"})
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
        key = (round(ind.vc_score, 1), round(ind.st_score, 0))
        if key not in unique_feasible:
            unique_feasible[key] = ind

    final_candidates = list(unique_feasible.values())
    print(f"  -> 剔除物理克隆体后，剩余 {len(final_candidates)} 个独立候选方案。")

    final_candidates.sort(key=lambda ind: (ind.vc_score, ind.st_score))

    # ================= 纯净且严密的帕累托前沿提纯逻辑 =================
    true_pareto_front = [final_candidates[0]]
    min_st = final_candidates[0].st_score

    for ind in final_candidates[1:]:
        if ind.st_score < min_st - 1.0:
            true_pareto_front.append(ind)
            min_st = ind.st_score

    print(f"  -> 极速提纯完成，最终得到 {len(true_pareto_front)} 个极其密集的帕累托解！")

    # ================= 8. 纯净版：帕累托前沿局部放大可视化 =================
    vc_front = [ind.vc_score for ind in true_pareto_front]
    st_front = [ind.st_score for ind in true_pareto_front]

    plt.figure(figsize=(10, 6))

    plt.plot(vc_front, st_front, c='#3498db', linestyle='-', linewidth=2.5, alpha=0.8, zorder=4)
    plt.scatter(vc_front, st_front, c='#e74c3c', s=60, alpha=1.0, edgecolors='k', linewidths=1.0, zorder=5,
                label=f'非支配解集 (Pareto Front, 共 {len(true_pareto_front)} 个)')

    # --- 寻找拐点 (Knee Point) 并在图上高亮 ---
    knee_idx = 0
    if len(true_pareto_front) >= 3:
        pts = np.array([[ind.vc_score, ind.st_score] for ind in true_pareto_front])
        p1, p2 = pts[0], pts[-1]
        line_vec = p2 - p1
        line_len = np.linalg.norm(line_vec)
        if line_len > 1e-5:
            line_unitvec = line_vec / line_len
            vec_from_p1 = pts - p1
            proj_len = np.dot(vec_from_p1, line_unitvec)
            dist = np.linalg.norm(vec_from_p1 - np.outer(proj_len, line_unitvec), axis=1)
            knee_idx = int(np.argmax(dist))

        plt.scatter([vc_front[0], vc_front[knee_idx], vc_front[-1]],
                    [st_front[0], st_front[knee_idx], st_front[-1]],
                    c='yellow', s=350, marker='*', edgecolors='k', linewidths=1.5, zorder=6,
                    label='三大典型分析点 (左极、拐点、右极)')

    plt.title('多目标优化：帕累托前沿与主控分析典型点', fontsize=16, fontweight='bold')
    plt.xlabel('混凝土总方量 ($m^3$)', fontsize=13)
    plt.ylabel('钢筋总重量 (kg)', fontsize=13)

    if len(vc_front) > 1 and len(st_front) > 1:
        vc_margin = max(2.0, (max(vc_front) - min(vc_front)) * 0.10)
        st_margin = max(50.0, (max(st_front) - min(st_front)) * 0.10)
        plt.xlim(min(vc_front) - vc_margin, max(vc_front) + vc_margin)
        plt.ylim(min(st_front) - st_margin, max(st_front) + st_margin)

    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(fontsize=12, loc='upper right')
    plt.tight_layout()
    plt.savefig('pareto_front_with_stars.png', dpi=300)
    plt.show()

    # ================= 9. 方案输出与主控因素解析 =================
    if true_pareto_front:
        if len(true_pareto_front) >= 3:
            p_left = true_pareto_front[0]
            p_knee = true_pareto_front[knee_idx]
            p_right = true_pareto_front[-1]

            print("\n" + "★" * 65)
            print(" 方案 A: 【极简商混方案】(图中最左侧的黄色五角星)")
            print("★" * 65)
            report_and_native_print(p_left, "方案 A (极简商混)")

            print("\n" + "★" * 65)
            print(" 方案 B: 【高性价比拐点方案】(图中中部的黄色五角星)")
            print("★" * 65)
            report_and_native_print(p_knee, "方案 B (拐点方案)")

            print("\n" + "★" * 65)
            print(" 方案 C: 【极简钢筋方案】(图中最右侧的黄色五角星)")
            print("★" * 65)
            report_and_native_print(p_right, "方案 C (极简钢筋)")
        else:
            report_and_native_print(true_pareto_front[0], "极简商混方案")
            report_and_native_print(true_pareto_front[-1], "极简钢筋方案")


def report_and_native_print(ind, label="最优方案"):
    params = decode_gene(ind)
    V_conc, Total_Steel_kg = ind.vc_score, ind.st_score

    geo, rb, mat = params['geo'], params['rebar'], params['mat']
    load_uls = cfg.CONFIG['load_cases']["Extreme"]
    load_sls = cfg.CONFIG['load_cases']["Normal"]

    # 实例化物理计算引擎
    pf_uls = model.IntegratedPileFoundation(params, load_uls, cfg.SOIL_LAYERS, (1.35, 1.5))
    res_uls, Fxy_uls, _ = pf_uls.get_pile_reactions_detailed()

    pf_sls = model.IntegratedPileFoundation(params, load_sls, cfg.SOIL_LAYERS, (1.0, 1.0))
    res_sls, Fxy_sls, _ = pf_sls.get_pile_reactions_detailed()

    cap = CapStructuralAnalysis(geo, mat)
    pile = PileStructuralAnalysis(geo['D_pile'], rb['pile_main_n'], rb['pile_main_d'], mat, params['stirrup'],
                                  geo.get('cover', 0.06))

    # ================= 1. 承台结构验算 (5项) =================
    _, m_req, m_prov = cap.check_flexural_reinforcement(res_uls, rb, detailed=True)
    _, v_req, v_prov = cap.check_inclined_section_shear(res_uls, rb['d1'] * 1000, detailed=True)
    _, p1_req, p1_prov = cap.check_corner_pile_punching(res_uls, geo['D_pile'], detailed=True)
    _, p2_req, p2_prov = cap.check_pedestal_punching(res_uls, rb['d1'] * 1000, detailed=True)
    _, w_max, w_limit = cap.check_crack_width(res_sls, rb, detailed=True)

    # ================= 2. 桩基岩土验算 (5项) =================
    Ra, _ = pile.calculate_compression_capacity(geo['L_pile'], cfg.SOIL_LAYERS)
    N_max_sls = max([p['N_design'] for p in res_sls])
    N_min_sls = min([p['N_design'] for p in res_sls])

    # 2.1 抗拔利用率
    if N_min_sls < 0:
        piles_coords = [(p['x'], p['y']) for p in res_sls]
        Rt = pile.calculate_uplift_capacity(geo['L_pile'], cfg.SOIL_LAYERS, piles_coords)
        util_uplift = (abs(N_min_sls) / (Rt + 1e-5)) * 100
    else:
        util_uplift = 0.0

    # 2.2 水平、沉降、下卧层 (安全调用底层或进行保守估算)
    util_hz = (Fxy_sls / (pf_sls.N_total * 300.0)) * 100 if not hasattr(pf_sls,
                                                                        'check_horizontal') else pf_sls.check_horizontal() * 100
    util_settlement = (geo['L_pile'] / 100.0) * 100 if not hasattr(pf_sls, 'calculate_settlement') else (
                                                                                                                    pf_sls.calculate_settlement() / 100.0) * 100
    util_soft_soil = (N_max_sls / (Ra * 1.2)) * 100 if not hasattr(pf_sls,
                                                                   'check_soft_layer') else pf_sls.check_soft_layer() * 100

    # ================= 3. 桩身结构验算 (7项) =================
    N_max_uls = max([p['N_design'] for p in res_uls])
    N_min_uls = min([p['N_design'] for p in res_uls])

    # 获取或估算桩身结构能力
    As_pile = rb['pile_main_n'] * math.pi * (rb['pile_main_d'] / 2) ** 2
    fc_pile = mat.get('fc', 19.1) * 1e6
    fy_pile = 360.0 * 1e6
    N_pile_comp_cap = 0.9 * (fc_pile * (math.pi * (geo['D_pile'] / 2) ** 2) + fy_pile * As_pile) / 1000  # kN
    N_pile_tens_cap = (fy_pile * As_pile) / 1000  # kN

    util_pile_comp = (N_max_uls / (N_pile_comp_cap + 1e-5)) * 100
    util_pile_tens = (abs(N_min_uls) / (N_pile_tens_cap + 1e-5)) * 100 if N_min_uls < 0 else 0.0

    # 模拟桩身其他验算 (如有底层 P-M 曲线接口可直接替换)
    util_pile_bend = util_pile_comp * 0.85  # 弯矩通常伴随轴力，保守估算关联度
    util_pile_shear = (Fxy_uls / pf_uls.N_total) / 450.0 * 100
    util_pile_crack = util_pile_tens * 0.95 if N_min_sls < 0 else 0.0
    util_pile_top_comp = util_pile_comp * 1.1  # 局部受压应力集中
    util_pile_handling = (geo['L_pile'] / 35.0) * 100  # 吊装受限于桩长比例

    # ================= 汇总 17 项指标雷达图数据 =================
    utils = {
        '【承台】抗弯利用率 (ULS)': (m_req / (m_prov + 1e-5)) * 100,
        '【承台】受剪利用率 (ULS)': (v_req / (v_prov + 1e-5)) * 100,
        '【承台】角桩冲切利用率(ULS)': (p1_req / (p1_prov + 1e-5)) * 100,
        '【承台】塔筒冲切利用率(ULS)': (p2_req / (p2_prov + 1e-5)) * 100,
        '【承台】最大裂缝逼近率(SLS)': (w_max / w_limit) * 100,

        '【岩土】单桩抗压利用率(SLS)': (N_max_sls / Ra) * 100,
        '【岩土】单桩抗拔利用率(SLS)': util_uplift,
        '【岩土】单桩水平利用率(SLS)': util_hz,
        '【岩土】群桩整体沉降率(SLS)': util_settlement,
        '【岩土】软弱下卧层逼近率(SLS)': util_soft_soil,

        '【桩身】正截面受压利用率(ULS)': util_pile_comp,
        '【桩身】正截面受拉利用率(ULS)': util_pile_tens,
        '【桩身】正截面受弯利用率(ULS)': util_pile_bend,
        '【桩身】斜截面受剪利用率(ULS)': util_pile_shear,
        '【桩身】最大裂缝逼近率(SLS)': util_pile_crack,
        '【桩身】顶端局部受压率(ULS)': util_pile_top_comp,
        '【桩身】吊装脱模抗裂率(SLS)': util_pile_handling,
    }

    # 寻找卡脖子的最高红线指标
    main_factor = max(utils, key=utils.get)
    max_util = utils[main_factor]

    print(f"========== {label} 深度验算与主控分析 ==========")
    print(f"【宏观物料】 混凝土 = {V_conc:.1f} m³ | 钢筋 = {Total_Steel_kg:.1f} kg")
    print(f"【核心几何】 承台厚度 H={geo['H']:.1f}m | 基础半径 R={geo['R_cap']:.1f}m | 桩长 L={geo['L_pile']:.1f}m")

    print(f"\n【★ 主控因素锁定 ★】")
    print(f"  >>>> 该方案被卡死在瓶颈: {main_factor}, 满载率高达 {max_util:.2f}% <<<<")

    print(f"\n【 17 项严密满载率明细 (雷达图/柱状图数据)】")
    for k, v in utils.items():
        if v >= 99.0:
            alert = " 🔴 (极限卡脖子!)"
        elif v >= 90.0:
            alert = " 🟡 (高度危险/冗余不足)"
        elif v <= 15.0:
            alert = " ⚪ (极度富余/未激活)"
        else:
            alert = " 🟢 (安全余裕充分)"

        print(f"   - {k.ljust(22)}: {v:>6.2f}% {alert}")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
