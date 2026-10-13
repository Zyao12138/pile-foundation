# -*- coding: utf-8 -*-
"""
文件名: optimize_foundation_deb_rules.py (Deb可行性法则 + 完整业务逻辑 + 全景报告出图版)
核心特性:
 1. 采用 Deb's Feasibility Rules (无失真锦标赛选择，根除早熟与悬崖效应)
 2. 包含完整的基因解码、打捆穿刺校验与造价评估逻辑
 3. 恢复全景终端报告 (包含经济对比与 17 项验算明细)
 4. 完美修复手动限制桩长 (FIXED_PILE_LENGTH) 的全局配置
"""

import math
import random
import numpy as np
import matplotlib.pyplot as plt
from deap import base, creator, tools
from tqdm import tqdm
import multiprocessing
import json

import foundation_config as cfg
import analyze_integrated_pile_foundation as model

# 设置绘图中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# ================= 0. 全局控制中枢 (在此修改全局配置) =================
CURRENT_ITER = 0
ALLOW_TENSION = True  # 是否允许桩基受拉
FIXED_PILE_LENGTH = None  # 若需锁定桩长(如28.0)，在此填入数字；若需算法自动寻优，填 None
BASELINE_COST = 950.0  # 原图纸设计的等效造价基准值(m³)，用于计算节约了多少成本


def init_worker(allow_tension_flag, fixed_pile_length_val):
    global ALLOW_TENSION, FIXED_PILE_LENGTH
    ALLOW_TENSION = allow_tension_flag
    FIXED_PILE_LENGTH = fixed_pile_length_val


# ================= 1. 变量初始化与 LHS 抽样 =================
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
    """拉丁超立方抽样，确保初始种群的全局均匀性"""
    result = np.zeros((samples, n))
    for i in range(n):
        low, up = VAR_RANGES[i]
        d = 1.0 / samples
        col_uniform = np.random.uniform(0, d, samples) + np.arange(samples) * d
        np.random.shuffle(col_uniform)
        result[:, i] = low + col_uniform * (up - low)
    return result


# ================= 2. 基因解码引擎 (将基因数组转化为物理尺寸) =================
def decode_gene(ind):
    vals = []
    for i, val in enumerate(ind):
        if np.isnan(val) or np.isinf(val):
            if i in DISCRETE_MAP:
                vals.append(DISCRETE_MAP[i][0])
            else:
                vals.append(VAR_RANGES[i][0])
            continue

        if i in DISCRETE_MAP:
            idx = int(round(val))
            idx = max(0, min(len(DISCRETE_MAP[i]) - 1, idx))
            vals.append(DISCRETE_MAP[i][idx])
        else:
            vals.append(val)

    # 宏观尺寸解析
    R_cap_m = round((vals[0] / 100.0) * 10) / 10.0
    H_depth_m = round((vals[1] / 100.0) * 10) / 10.0
    h1_edge_m = round((vals[2] / 100.0) * 10) / 10.0
    h2_slope_m = round((vals[3] / 100.0) * 10) / 10.0
    h3_pedestal_m = round((vals[4] / 100.0) * 10) / 10.0
    r_ped_m = round((vals[5] / 100.0) * 10) / 10.0

    # 钢筋智能对齐函数
    def snap_rebar_count(R_m, raw_n, is_edge_doubled=False):
        n = int(round(raw_n / 10.0) * 10)
        n = max(10, n)
        perimeter_mm = 2 * math.pi * R_m * 1000
        eff_n = n * 2 if is_edge_doubled else n
        if perimeter_mm / eff_n > 300.0:
            req_eff_n = int(math.ceil((perimeter_mm / 300.0) / 10.0) * 10)
            n = req_eff_n // 2 if is_edge_doubled else req_eff_n
        elif perimeter_mm / eff_n < 100.0:
            req_eff_n = int(math.floor((perimeter_mm / 100.0) / 10.0) * 10)
            n = req_eff_n // 2 if is_edge_doubled else req_eff_n
        return max(10, n)

    # 底部配筋解析
    n1 = snap_rebar_count(R_cap_m, int(vals[6]), is_edge_doubled=True)
    d1 = vals[7] / 1000.0
    d2 = vals[8] / 1000.0
    D1_ring_1 = vals[9] / 1000.0
    D1_ring_2 = vals[10] / 1000.0
    s1_ring_1 = round(cfg.CONFIG['DESIGN_PARAMS'].get('s1', 150) / 50.0) * 50.0 / 1000.0
    s1_ring_2 = round(cfg.CONFIG['DESIGN_PARAMS'].get('s1_2', 200) / 50.0) * 50.0 / 1000.0
    n2 = n1
    n3 = n1
    d3 = d1

    # 顶部配筋解析
    n4 = snap_rebar_count(R_cap_m, int(vals[11]), is_edge_doubled=True)
    d4 = vals[12] / 1000.0
    D2_ring = vals[13] / 1000.0
    s2_ring = round(cfg.CONFIG['DESIGN_PARAMS'].get('s2', 150) / 50.0) * 50.0 / 1000.0
    n5 = n4
    d5 = d4

    # 桩基解析 (判断是否手动锁定桩长)
    if FIXED_PILE_LENGTH is not None:
        L_pile_m = float(FIXED_PILE_LENGTH)
    else:
        L_pile_m = float(round(vals[14]))

    D_pile_m = vals[15] / 100.0
    n_in = int(vals[16])
    n_out = int(vals[17])
    r_in = round(R_cap_m * (vals[18] / 100.0) * 10) / 10.0
    r_out = round(R_cap_m * (vals[19] / 100.0) * 10) / 10.0

    n_pile_main = int(vals[20])
    if n_pile_main % 2 != 0: n_pile_main += 1
    d_pile_main = vals[21] / 1000.0

    # 组装返回字典
    geo = {
        'R_cap': R_cap_m, 'H': H_depth_m,
        'h1': h1_edge_m, 'h2': h2_slope_m, 'h3': h3_pedestal_m,
        'r_col': r_ped_m, 'r_top': r_ped_m + 0.2,
        'D_pile': D_pile_m, 'L_pile': L_pile_m,
        'rings': [{'n': n_in, 'r': r_in}, {'n': n_out, 'r': r_out}]
    }

    rebar = {
        'n1': n1, 'd1': d1, 'n2': n2, 'd2': d2, 'n3': n3, 'd3': d3,
        'cap_bot_rad_n': n1, 'cap_bot_rad_d': d1,
        'ring_opt': {'d1': D1_ring_1, 'd2': D1_ring_2, 's1': s1_ring_1 * 1000.0, 's2': s1_ring_2 * 1000.0},
        'n4': n4, 'd4': d4, 'n5': n5, 'd5': d5,
        'top_ring_opt': {'d': D2_ring, 's': s2_ring * 1000.0},
        'pile_main_n': n_pile_main, 'pile_main_d': d_pile_main,
    }

    stirrup_consts = cfg.CONFIG['PILE_CONSTANTS']
    stirrup = {'d': stirrup_consts['STIRRUP_D'], 's': stirrup_consts['STIRRUP_S_DENSE'], 'fyv': 360.0}
    concrete_mat = cfg.CONFIG['CONCRETE_LIB'][1]

    return {'geo': geo, 'rebar': rebar, 'stirrup': stirrup, 'mat': concrete_mat}


# ================= 2.5 排布防碰撞检测模块 =================
def calculate_crowding_penalty(n_rebars, d_rebar, PCD_R, n_bolts, d_bolt, r_void=1.5):
    """计算钢筋穿过锚栓笼时的打捆与拥挤惩罚"""
    if n_rebars < 2: return 0.0
    min_clearance_req = 0.0  # 0.0 表示允许钢筋贴合(打捆)，但不能穿模重叠
    penalty_score = 0.0

    arc_length_void = (2 * math.pi * r_void) / n_rebars
    clear_dist_void = arc_length_void - d_rebar
    if clear_dist_void < min_clearance_req:
        penalty_score += (min_clearance_req - clear_dist_void) * 10000.0

    r_obstacle = (d_bolt / 2.0) + 0.010
    bolt_angles = np.linspace(0, 2 * math.pi, n_bolts, endpoint=False)
    step = 2 * math.pi / n_rebars
    actual_angles = []

    for i in range(n_rebars):
        theta_ideal = i * step
        diffs = np.abs(bolt_angles - theta_ideal)
        diffs = np.minimum(diffs, 2 * math.pi - diffs)
        min_idx = np.argmin(diffs)
        theta_bolt = bolt_angles[min_idx]
        min_diff = diffs[min_idx]
        d_perp = PCD_R * math.sin(min_diff)
        r_bar = d_rebar / 2.0

        if d_perp < (r_obstacle + r_bar):
            safe_val = min(1.0, (r_obstacle + r_bar + 0.005) / PCD_R)
            safe_angle = math.asin(safe_val)
            diff_signed = math.atan2(math.sin(theta_ideal - theta_bolt), math.cos(theta_ideal - theta_bolt))
            theta_actual = theta_bolt + safe_angle if diff_signed >= 0 else theta_bolt - safe_angle
        else:
            theta_actual = theta_ideal
        actual_angles.append(theta_actual % (2 * math.pi))

    actual_angles.sort()
    n = len(actual_angles)
    for i in range(n):
        theta1 = actual_angles[i]
        theta2 = actual_angles[(i + 1) % n]
        delta_theta = (theta2 - theta1) if (theta2 - theta1) >= 0 else (theta2 - theta1 + 2 * math.pi)
        clear_dist = delta_theta * PCD_R - d_rebar
        if clear_dist < min_clearance_req:
            penalty_score += (min_clearance_req - clear_dist) * 10000.0

    return penalty_score


# ================= 3. 核心评估函数 (返回造价与违规度) =================
def evaluate(ind):
    global ALLOW_TENSION
    if np.any(np.isnan(ind)):
        return (1e10, 1e10)

    try:
        safe_values = []
        for i, val in enumerate(ind):
            low, up = VAR_RANGES[i]
            safe_val = max(low, min(up, val))
            if i in INT_INDICES:
                safe_values.append(int(round(safe_val)))
            else:
                safe_values.append(safe_val)

        params = decode_gene(safe_values)
        geo = params['geo']
        rb = params['rebar']

        # 1. 调用底层结构验算
        total_viol, _ = model.assess_design_performance(params)

        # 2. 几何包络约束验算
        ach = cfg.CONFIG.get('ANCHOR_CAGE', {})
        L_tot, bg, te, bc, Do = ach.get('L_total', 4.2), ach.get('bottom_groove', 0.3), ach.get('top_expose',
                                                                                                0.3), ach.get(
            'bottom_cover', 0.2), ach.get('D_outer', 5.1)

        total_concrete_thickness = geo['h1'] + geo['h2'] + geo['h3'] - bg
        if total_concrete_thickness < (L_tot - te + bc):
            total_viol += ((L_tot - te + bc) - total_concrete_thickness) * 50.0

        min_r_col = (Do / 2.0) + 0.6
        if geo['r_col'] < min_r_col:
            total_viol += (min_r_col - geo['r_col']) * 50.0

        R_anchor_outer = Do / 2.0
        r_in = min([ring['r'] for ring in geo['rings']]) if geo['rings'] else 0.0
        pile_inner_edge = r_in - (geo['D_pile'] / 2.0)
        if pile_inner_edge < (R_anchor_outer + 0.6):
            total_viol += ((R_anchor_outer + 0.6) - pile_inner_edge) * 50.0

        # 3. 钢筋排布拥挤验算
        PCD_R = ach.get('PCD', 4.8) / 2.0
        n_bolts = ach.get('N_BOLTS', 160)
        d_bolt = ach.get('BOLT_D', 48) / 1000.0
        r_void = 1.5

        top_pen = calculate_crowding_penalty(rb['n4'], rb['d4'], PCD_R, n_bolts, d_bolt, r_void)
        d_max_bot = max(rb['d1'], rb.get('d2', rb['d1']))
        bot_pen = calculate_crowding_penalty(rb['n1'], d_max_bot, PCD_R, n_bolts, d_bolt, r_void)
        total_viol += top_pen + bot_pen

        # 4. 桩基受拉验算
        load_extreme = cfg.CONFIG['load_cases']['Extreme']
        if not ALLOW_TENSION:
            pf_check = model.IntegratedPileFoundation(params, load_extreme, cfg.SOIL_LAYERS, (1.35, 1.5))
            _, forces = pf_check.get_individual_pile_forces()
            if forces:
                min_force = min(forces)
                if min_force < 0: total_viol += abs(min_force) / 100.0

        # 5. 工程量与等效造价计算
        pf_basic = model.IntegratedPileFoundation(params, cfg.CONFIG['load_cases']["Normal"], cfg.SOIL_LAYERS)
        Vc, W_cap_steel, _, _ = pf_basic.get_quantities()
        W_pile = pf_basic.N_total * geo['L_pile'] * (rb['pile_main_n'] * math.pi * (rb['pile_main_d'] / 2) ** 2 * 7850)
        W_other = Vc * 40.0
        Total_Steel = W_cap_steel + W_pile + W_other

        price_ratio = (cfg.CONFIG.get('UNIT_PRICES', {}).get('STEEL', 5500.0) / 1000.0) / params['mat'].get('price',
                                                                                                            500.0)
        Cost_Eq_Volume = Vc + Total_Steel * price_ratio

    except Exception:
        return (1e10, 1e10)

    # 核心：解耦返回，不再做加法
    return (Cost_Eq_Volume, total_viol)


# ================= 4. Deb's Feasibility Rules (算法引擎核心) =================
def deb_sort_key(ind):
    """
    Deb 法则核心排序密钥：
    1. 合规解(0) 无条件击败 违规解(1)
    2. 如果都违规，违规度小的优先
    3. 如果都不违规，造价低的优先
    """
    if not hasattr(ind, 'mat_score') or not hasattr(ind, 'viol'):
        return (1, 1e10, 1e10)

    is_infeasible = 1 if ind.viol > 1e-4 else 0
    return (is_infeasible, ind.viol, ind.mat_score)


def selTournamentDeb(individuals, k, tournsize):
    """基于 Deb 法则的锦标赛选择"""
    chosen = []
    for _ in range(k):
        aspirants = random.sample(individuals, tournsize)
        best = min(aspirants, key=deb_sort_key)
        chosen.append(best)
    return chosen


# ================= 5. DEAP 架构注册 =================
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)

toolbox = base.Toolbox()
toolbox.register("population", lambda n: [creator.Individual(ind) for ind in lhs_sample(len(VAR_RANGES), n)])
toolbox.register("evaluate", evaluate)
toolbox.register("mate", tools.cxTwoPoint)


# ================= 6. 核心优化主流程 =================
def main():
    global CURRENT_ITER

    print("=" * 60)
    print(" 🌪️ 风机基础智能优化程序 (Deb法则引擎 + 完整功能版)")
    print("=" * 60)

    ga_params = cfg.CONFIG['GA_PARAMS']
    lows = [r[0] for r in VAR_RANGES]
    ups_raw = [r[1] for r in VAR_RANGES]
    ups_safe = [u + 1e-9 if u <= l else u for l, u in zip(lows, ups_raw)]

    toolbox.register("mutate", tools.mutPolynomialBounded, eta=ga_params.get('ETA', 12.0), low=lows, up=ups_safe,
                     indpb=0.15)
    toolbox.register("select", selTournamentDeb, tournsize=ga_params.get('TOURN_SIZE', 3))

    use_cores = min(8, multiprocessing.cpu_count())
    pool = multiprocessing.Pool(processes=use_cores, initializer=init_worker,
                                initargs=(ALLOW_TENSION, FIXED_PILE_LENGTH))
    toolbox.register("map", pool.map)

    pop = toolbox.population(ga_params['POP_SIZE'])
    history_fitness = []

    ngen = ga_params['N_GEN']
    pop_size = ga_params['POP_SIZE']
    n_elite = max(2, int(pop_size * 0.02))  # 精英保留比例
    n_mid = pop_size - n_elite

    pbar = tqdm(range(1, ngen + 1), desc="🚀 优化演化中 (Deb's Rules)", unit="gen", ncols=120)
    for gen in pbar:
        CURRENT_ITER = gen

        # 1. 评估适应度
        invalid_ind = [ind for ind in pop if not hasattr(ind, 'mat_score')]
        if invalid_ind:
            fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
            for ind, fit_tuple in zip(invalid_ind, fitnesses):
                ind.mat_score, ind.viol = fit_tuple
                ind.fitness.values = (ind.mat_score,)

                # 2. Deb 法则排序
        sorted_pop = sorted(pop, key=deb_sort_key)
        best_ind = sorted_pop[0]

        # 为了画图好看，如果尚未合规，加上部分违规度
        if best_ind.viol > 1e-4:
            display_cost = best_ind.mat_score + best_ind.viol * 10000.0
        else:
            display_cost = best_ind.mat_score

        if len(history_fitness) == 0:
            history_fitness.append(display_cost)
        else:
            history_fitness.append(min(history_fitness[-1], display_cost))

        pbar.set_postfix({
            "Best_Cost": f"{best_ind.mat_score:.1f}",
            "Viol": f"{best_ind.viol:.4f}",
            "Status": "✅Feasible" if best_ind.viol <= 1e-4 else "❌Infeasible"
        })

        # 3. 繁衍下一代 (彻底抛弃随机注入，依靠交叉变异)
        elites = [toolbox.clone(ind) for ind in sorted_pop[:n_elite]]
        mid_parents = toolbox.select(pop, n_mid)
        mids = [toolbox.clone(ind) for ind in mid_parents]

        for i in range(1, len(mids), 2):
            if random.random() < ga_params['CX_PROB']:
                toolbox.mate(mids[i - 1], mids[i])
                delattr(mids[i - 1], 'mat_score')
                delattr(mids[i], 'mat_score')

        for i in range(len(mids)):
            if random.random() < ga_params['MUT_PROB']:
                toolbox.mutate(mids[i])
                if hasattr(mids[i], 'mat_score'):
                    delattr(mids[i], 'mat_score')

        pop[:] = elites + mids

    pool.close()
    pool.join()

    # ================= 7. 最终结果处理与全景报告 =================
    final_pop = sorted(pop, key=deb_sort_key)
    best_overall = final_pop[0]
    best_params = decode_gene(best_overall)

    print("\n\n" + "=" * 60)
    print(" 🏆 智能优化完成: 最终设计方案全景报告 (Deb's Rules)")
    print("=" * 60)

    if best_overall.viol <= 1e-4:
        print(f" [结论] ✅ 成功找到完美合规的最优解！最终等效造价: {best_overall.mat_score:.2f} m³")
    else:
        print(f" [结论] ⚠️ 警告：输出违规最小妥协解。最小违规度: {best_overall.viol:.4f}")

    print("\n【一、 经济效益分析 (与原方案对比)】")
    cost_saved = BASELINE_COST - best_overall.mat_score
    save_ratio = (cost_saved / BASELINE_COST) * 100 if BASELINE_COST > 0 else 0

    print(f" - 优化前原方案造价 : {BASELINE_COST:.2f} m³ (参考基准)")
    print(f" - 优化后最优解造价 : {best_overall.mat_score:.2f} m³")
    if cost_saved > 0:
        print(f" - 🎉 共计节约造价 : {cost_saved:.2f} m³")
        print(f" - 📉 综合成本下降率 : {save_ratio:.2f} %")
    else:
        print(f" - ⚠️ 优化后成本未下降，可能是原方案已存在严重违规，当前为纠偏合规成本。")

    geo = best_params['geo']
    rb = best_params['rebar']
    print("\n【二、 核心设计参数提取】")
    print(f" - 承台底半径 R_cap  : {geo['R_cap']:.1f} m, 承台总厚度 H: {geo['H']:.1f} m")
    print(f" - 承台台柱半径 r_col: {geo['r_col']:.2f} m")
    if FIXED_PILE_LENGTH is not None:
        print(f" - 桩长 L_pile       : {geo['L_pile']:.1f} m (🔒已手动锁定), 桩径: {geo['D_pile']:.1f} m")
    else:
        print(f" - 桩长 L_pile       : {geo['L_pile']:.1f} m (🔄算法自动寻优), 桩径: {geo['D_pile']:.1f} m")

    print(f" - 底板通长主筋 (n1) : {rb['n1']} 根, 直径 Φ{rb['d1'] * 1000:.0f}")
    print(f" - 顶板通长主筋 (n4) : {rb['n4']} 根, 直径 Φ{rb['d4'] * 1000:.0f}")
    print(f" - 桩身纵向主筋      : {rb['pile_main_n']} 根, 直径 Φ{rb['pile_main_d'] * 1000:.0f}")

    print("\n【三、 规范结构与岩土安全验算明细】")
    try:
        print("-" * 50)
        viol, details = model.assess_design_performance(best_params, verbose=True)
        if isinstance(details, dict):
            for check_name, ratio in details.items():
                status = "✅ 满足" if ratio <= 1.0 else "❌ 违规"
                print(f"  {status} | {check_name:<20}: 利用率 {ratio * 100:>5.1f}%")
        print("-" * 50)
        print(f" [总体验算结果] 总体违规度 (Total Violation): {viol:.6f}")
    except Exception as e:
        print(f" -> ⚠️ 验算明细调用异常或底层 model 不支持 verbose 模式: {e}")

    with open("best_design_params.json", "w", encoding="utf-8") as f:
        json.dump(best_params, f, indent=4, ensure_ascii=False)
    print("\n📊 最佳设计参数已无损导出至 -> [ best_design_params.json ]")

    print("\n📈 正在生成算法收敛性能图线...")
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(history_fitness) + 1), history_fitness,
             color='#d62728', linewidth=2.5, marker='.', markersize=5,
             label="Best Objective (Deb's Feasibility Rules)")

    plt.title("Convergence History of Wind Turbine Foundation optimize_foundation_ga", fontsize=14, fontweight='bold')
    plt.xlabel("Generation", fontsize=12)
    plt.ylabel("Equivalent Cost Objective", fontsize=12)

    final_cost = history_fitness[-1]
    plt.ylim(final_cost * 0.9,
             max(history_fitness) * 1.05 if max(history_fitness) < final_cost * 3 else final_cost * 1.5)

    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
