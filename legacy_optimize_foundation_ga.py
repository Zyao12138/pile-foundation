# -*- coding: utf-8 -*-
"""
文件名: legacy_optimize_foundation_ga.py (历史版单目标优化器 + 动态视觉缩放 + 等效体积造价导向 + 允许打捆穿刺避让版)
修改记录:
 - 移除了 50mm 强制净距红线，允许钢筋在锚栓圈(PCD)和中心区(r_void)发生“并筋(接触)”
 - 仅当钢筋发生理论上的物理重叠 (clear_dist < 0) 时，才触发惩罚
"""

import math
import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from deap import base, creator, tools
from tqdm import tqdm

import multiprocessing
import os
import json  # <--- 新增 JSON 库用于导出

import foundation_config as cfg
import analyze_integrated_pile_foundation as model

# 尝试导入微观钢筋排布引擎
try:
    from render_bottom_rebar_layout import BottomRebarLayout
    from render_top_rebar_layout import TopRebarLayout

    HAS_LAYOUT_ENGINE = True
except ImportError:
    HAS_LAYOUT_ENGINE = False
    print("【提示】未检测到排布引擎(render_bottom_rebar_layout.py / render_top_rebar_layout.py)，将跳过可视化。")

# 尝试导入验算模块 (用于报告生成与刚度计算)
try:
    from analyze_cap_structure import CapStructuralAnalysis
    from analyze_pile_structure import PileStructuralAnalysis

    HAS_CHECK_MODULES = True
except ImportError:
    HAS_CHECK_MODULES = False
    print("【提示】未检测到验算模块(cap/analyze_pile_structure)，最终报告将跳过详细验算表。")

if not model.HAS_MODULES:
    print("【警告】optimize_foundation_ga: 未检测到物理计算模块，优化将基于几何约束运行。")

# 设置绘图字体，防止中文乱码
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# ================= 0. 全局设置 =================
CURRENT_ITER = 0
ALLOW_TENSION = True  # 默认允许受拉
FIXED_PILE_LENGTH = None
ABAQUS_OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "abaqus_postprocess")


def init_worker(allow_tension_flag, fixed_pile_length_val):
    """多进程子进程初始化函数：用于将主进程的用户选择同步给所有后台核心"""
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


def clip_gene(ind):
    """
    P4 fix: ??????????????
    """
    for i in range(len(ind)):
        low, up = VAR_RANGES[i]
        if ind[i] < low:
            ind[i] = low
        elif ind[i] > up:
            ind[i] = up



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

    R_cap_m = round((vals[0] / 100.0) * 10) / 10.0
    H_depth_m = round((vals[1] / 100.0) * 10) / 10.0
    h1_edge_m = round((vals[2] / 100.0) * 10) / 10.0
    h2_slope_m = round((vals[3] / 100.0) * 10) / 10.0
    h3_pedestal_m = round((vals[4] / 100.0) * 10) / 10.0
    r_ped_m = round((vals[5] / 100.0) * 10) / 10.0

    def snap_rebar_count(R_m, raw_n, is_edge_doubled=False):
        n = int(round(raw_n / 10.0) * 10)
        n = max(10, n)
        perimeter_mm = 2 * math.pi * R_m * 1000

        # 边缘实际钢筋根数（通长+外插交替时为 2n）
        eff_n = n * 2 if is_edge_doubled else n

        if perimeter_mm / eff_n > 300.0:
            req_eff_n = int(math.ceil((perimeter_mm / 300.0) / 10.0) * 10)
            n = req_eff_n // 2 if is_edge_doubled else req_eff_n
        elif perimeter_mm / eff_n < 100.0:
            req_eff_n = int(math.floor((perimeter_mm / 100.0) / 10.0) * 10)
            n = req_eff_n // 2 if is_edge_doubled else req_eff_n
        return max(10, n)

    # --- 底板配筋 (含真·双层环筋) ---
    n1 = snap_rebar_count(R_cap_m, int(vals[6]), is_edge_doubled=True)
    d1 = vals[7] / 1000.0
    d2 = vals[8] / 1000.0
    D1_ring_1 = vals[9] / 1000.0  # 底板: 底层环筋
    D1_ring_2 = vals[10] / 1000.0  # 底板: 附加层环筋

    s1_ring_1 = round(cfg.CONFIG['DESIGN_PARAMS'].get('s1', 150) / 50.0) * 50.0 / 1000.0
    s1_ring_2 = round(cfg.CONFIG['DESIGN_PARAMS'].get('s1_2', 200) / 50.0) * 50.0 / 1000.0
    n2 = n1
    n3 = n1
    d3 = d1

    # --- 顶板配筋 ---
    n4 = snap_rebar_count(R_cap_m, int(vals[11]), is_edge_doubled=True)
    d4 = vals[12] / 1000.0
    D2_ring = vals[13] / 1000.0
    s2_ring = round(cfg.CONFIG['DESIGN_PARAMS']['s2'] / 50.0) * 50.0 / 1000.0
    n5 = n4
    d5 = d4

    # --- 桩基部分 ---
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
    if n_pile_main % 2 != 0:
        n_pile_main += 1
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
        'cap_bot_rad_n': n1, 'cap_bot_rad_d': d1,
        'ring_opt': {'d1': D1_ring_1, 'd2': D1_ring_2, 's1': s1_ring_1 * 1000.0, 's2': s1_ring_2 * 1000.0,
                     's': s1_ring_1 * 1000.0},
        'n4': n4, 'd4': d4, 'n5': n5, 'd5': d5,
        'top_ring_opt': {'d': D2_ring, 's': s2_ring * 1000.0},
        'pile_main_n': n_pile_main, 'pile_main_d': d_pile_main,
    }

    stirrup_consts = cfg.CONFIG['PILE_CONSTANTS']
    stirrup = {'d': stirrup_consts['STIRRUP_D'], 's': stirrup_consts['STIRRUP_S_DENSE'], 'fyv': 360.0}
    concrete_mat = cfg.CONFIG['CONCRETE_LIB'][1]

    return {'geo': geo, 'rebar': rebar, 'stirrup': stirrup, 'mat': concrete_mat}


# ================= 2.5 拥挤踩踏校验引擎 (修改版：允许并筋穿刺) =================
def calculate_crowding_penalty(n_rebars, d_rebar, PCD_R, n_bolts, d_bolt, r_void=1.5):
    """
    内部极坐标避让计算引擎，用于在遗传算法演化过程中高速核对施工物理冲突。
    【修改说明】：移除了 50mm 的混凝土振捣净距红线。仅要求钢筋互相之间不发生“物理穿模(重叠)”。
    允许钢筋紧挨着（并筋打捆）穿过锚栓圈。
    """
    if n_rebars < 2:
        return 0.0

    # 核心修改：允许并筋接触，最低净距放宽至 0.0m
    min_clearance_req = 0.0
    penalty_score = 0.0

    # ====== 校验一：中心留空区 r_void 的物理重叠校验 ======
    arc_length_void = (2 * math.pi * r_void) / n_rebars
    clear_dist_void = arc_length_void - d_rebar

    if clear_dist_void < min_clearance_req:
        deficit = min_clearance_req - clear_dist_void
        penalty_score += deficit * 10000.0

    # ====== 校验二：锚栓圈 PCD_R 的推移避让及物理重叠校验 ======
    # 避让净距：锚栓套管半径 + 施工绑扎极小富余量 (10mm)
    r_obstacle = (d_bolt / 2.0) + 0.010
    bolt_angles = np.linspace(0, 2 * math.pi, n_bolts, endpoint=False)
    step = 2 * math.pi / n_rebars

    actual_angles = []

    # 1. 模拟所有钢筋的启发式推移避让 (只躲锚栓)
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
            # 发生碰撞，计算避让安全角
            safe_val = (r_obstacle + r_bar + 0.005) / PCD_R
            safe_val = min(1.0, safe_val)
            safe_angle = math.asin(safe_val)

            diff_signed = math.atan2(math.sin(theta_ideal - theta_bolt), math.cos(theta_ideal - theta_bolt))
            if diff_signed >= 0:
                theta_actual = theta_bolt + safe_angle
            else:
                theta_actual = theta_bolt - safe_angle
        else:
            theta_actual = theta_ideal

        actual_angles.append(theta_actual % (2 * math.pi))

    actual_angles.sort()

    n = len(actual_angles)

    # 2. 检查推移后的钢筋是否互相发生了物理重叠
    for i in range(n):
        theta1 = actual_angles[i]
        theta2 = actual_angles[(i + 1) % n]

        delta_theta = theta2 - theta1
        if delta_theta < 0:
            delta_theta += 2 * math.pi

        arc_length = delta_theta * PCD_R
        clear_dist = arc_length - d_rebar

        # 仅当物理干涉（重叠）时才施加惩罚；挤近一点不再惩罚
        if clear_dist < min_clearance_req:
            deficit = min_clearance_req - clear_dist
            penalty_score += deficit * 10000.0

    return penalty_score


# ================= 3. 核心评估函数 (物理计算 + 施工校验) =================
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

        total_viol, _ = model.assess_design_performance(params)

        ach = cfg.CONFIG.get('ANCHOR_CAGE', {})
        L_tot = ach.get('L_total', 4.2)
        bg = ach.get('bottom_groove', 0.3)
        te = ach.get('top_expose', 0.3)
        bc = ach.get('bottom_cover', 0.2)
        Do = ach.get('D_outer', 5.1)

        total_concrete_thickness = geo['h1'] + geo['h2'] + geo['h3'] - bg
        if total_concrete_thickness < (L_tot - te + bc):
            total_viol += ((L_tot - te + bc) - total_concrete_thickness) * 50.0

        min_r_col = (Do / 2.0) + 0.6
        if geo['r_col'] < min_r_col:
            total_viol += (min_r_col - geo['r_col']) * 50.0

        R_anchor_outer = Do / 2.0
        r_in = min([ring['r'] for ring in geo['rings']]) if geo['rings'] else 0.0
        pile_inner_edge = r_in - (geo['D_pile'] / 2.0)
        min_safe_clearance = 0.6
        if pile_inner_edge < (R_anchor_outer + min_safe_clearance):
            total_viol += ((R_anchor_outer + min_safe_clearance) - pile_inner_edge) * 50.0

        # ====== 微观三维钢筋排布“穿模”惩罚 ======
        PCD_R = ach.get('PCD', 4.8) / 2.0
        n_bolts = ach.get('N_BOLTS', 160)
        d_bolt = ach.get('BOLT_D', 48) / 1000.0
        r_void = 1.5  # 中心留空区半径 (与下文排布引擎统一)

        # 1. 顶板 n4 拥挤校验
        top_pen = calculate_crowding_penalty(rb['n4'], rb['d4'], PCD_R, n_bolts, d_bolt, r_void)
        # 2. 底板 n1/n2 捆绑避让拥挤校验 (取直径最大者)
        d_max_bot = max(rb['d1'], rb.get('d2', rb['d1']))
        bot_pen = calculate_crowding_penalty(rb['n1'], d_max_bot, PCD_R, n_bolts, d_bolt, r_void)

        # 叠加进入结构违规度
        total_viol += top_pen + bot_pen
        # =================================================

        load_extreme = cfg.CONFIG['load_cases']['Extreme']
        if not ALLOW_TENSION:
            pf_check = model.IntegratedPileFoundation(params, load_extreme, cfg.SOIL_LAYERS, (1.35, 1.5))
            _, forces = pf_check.get_individual_pile_forces()
            if forces:
                min_force = min(forces)
                if min_force < 0:
                    total_viol += abs(min_force) / 100.0

        if HAS_CHECK_MODULES:
            load_sls = cfg.CONFIG['load_cases']['Normal']
            pf_sls = model.IntegratedPileFoundation(params, load_sls, cfg.SOIL_LAYERS, (1.0, 1.0))
            pile_ana = PileStructuralAnalysis(
                geo['D_pile'], rb['pile_main_n'], rb['pile_main_d'], params['mat'], params['stirrup']
            )
            _, theta_sls = pile_ana.check_settlement_and_tilt(
                load_sls['FZ'], load_sls['MXY'], geo['L_pile'], cfg.SOIL_LAYERS, pf_sls.piles_coords
            )
            if theta_sls > 1e-6:
                K_theta = (load_sls['MXY'] / theta_sls) / 1e6
                MIN_K_THETA = 80.0
                if K_theta < MIN_K_THETA:
                    total_viol += (MIN_K_THETA - K_theta) / MIN_K_THETA * 20.0

        pf_basic = model.IntegratedPileFoundation(params, cfg.CONFIG['load_cases']["Normal"], cfg.SOIL_LAYERS)
        Vc, W_cap_steel, _, _ = pf_basic.get_quantities()

        if hasattr(model, 'calculate_pile_steel_detailed'):
            W_pile = model.calculate_pile_steel_detailed(
                pf_basic.N_total, geo['D_pile'], geo['L_pile'], rb['pile_main_n'], rb['pile_main_d']
            )
        else:
            W_pile = pf_basic.N_total * geo['L_pile'] * (
                    rb['pile_main_n'] * math.pi * (rb['pile_main_d'] / 2) ** 2 * 7850)

        W_other = Vc * 40.0
        Total_Steel = W_cap_steel + W_pile + W_other

        conc_price = params['mat'].get('price', 500.0)
        steel_price = cfg.CONFIG.get('UNIT_PRICES', {}).get('STEEL', 5500.0) / 1000.0
        price_ratio = steel_price / conc_price

        Cost_Eq_Volume = Vc + Total_Steel * price_ratio

    except Exception:
        tb_str = traceback.format_exc()
        try:
            with open(_EVAL_LOG_PATH, 'a', encoding='utf-8') as logf:
                logf.write(f"[evaluate error] ind={list(ind)[:5]}...\n{tb_str}\n")
        except Exception:
            pass
        print(f"\n[!] evaluate error -> {_EVAL_LOG_PATH}: {tb_str.splitlines()[-1]}", flush=True)
        return (1e10, 1e10)

    return (Cost_Eq_Volume, total_viol)


# ================= 4. 可视化绘图 =================
def plot_results(best_params, piles_coords, pile_forces, hist_mat_score, hist_vc, hist_ws, hist_viol):
    global ALLOW_TENSION
    plt.figure(figsize=(10, 6))

    legitimate_scores = [s for s, v in zip(hist_mat_score, hist_viol) if v < 1e-4]

    if not legitimate_scores:
        print("\n[!] 警告: 寻优全程未找到完全满足规范的合法解，收敛曲线将展示带惩罚的原始得分。")
        y_data = hist_mat_score
        start_gen = 0
        x_axis = range(start_gen, len(hist_mat_score))
    else:
        y_data = legitimate_scores
        start_gen = len(hist_mat_score) - len(legitimate_scores)
        x_axis = range(start_gen, start_gen + len(y_data))

    plt.plot(x_axis, y_data, color='#e74c3c', linewidth=2.5, label='最佳适应度 (Best Fitness)')
    if len(y_data) > 1:
        plt.fill_between(x_axis, y_data, min(y_data) * 0.95, color='#e74c3c', alpha=0.1)

    plt.title('遗传算法优化收敛曲线 (Convergence Curve)', fontsize=14, fontweight='bold')
    plt.xlabel('迭代代数 (Generation)', fontsize=12)
    plt.ylabel('最佳适应度得分', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.savefig('best_fitness_curve.png', dpi=300)
    plt.show()

    fig = plt.figure(figsize=(15, 8))
    ax1 = fig.add_subplot(2, 2, 1)
    ax1.plot(hist_mat_score, color='blue', linewidth=2)
    ax1.set_title('最佳适应度趋势')
    ax1.set_xlabel('迭代代数', fontsize=10)
    ax1.set_ylabel('适应度得分', fontsize=10)

    # P7 fix: adaptive y-axis using percentile
    if len(y_data) > 1:
        y_sorted = sorted(y_data)
        p95 = y_sorted[int(len(y_sorted) * 0.95)]
        p05 = y_sorted[int(len(y_sorted) * 0.05)]
        y_span = p95 - p05
        if y_span > 0:
            ax1.set_ylim(p05 - 0.1 * y_span, p95 + 0.3 * y_span)

    ax2 = fig.add_subplot(2, 2, 2)
    ax2.plot(hist_vc, color='tab:blue', linewidth=2)
    ax2.set_title('纯混凝土体积趋势')
    ax2.set_xlabel('迭代代数', fontsize=10)
    ax2.set_ylabel('真实混凝土体积 ($m^3$)', fontsize=10)

    ax3 = fig.add_subplot(2, 2, 4)
    ax3.plot(hist_ws, color='tab:blue', linewidth=2)
    ax3.set_title('纯钢筋用量趋势')
    ax3.set_xlabel('迭代代数', fontsize=10)
    ax3.set_ylabel('真实钢筋重量 (kg)', fontsize=10)

    plt.tight_layout()
    plt.savefig('optimization_metrics.png', dpi=300, bbox_inches='tight')
    plt.show()


# ================= 5. 详细报告输出函数 =================
def print_detailed_report(best_params, V_conc, Total_Steel_kg, best_cost):
    global ALLOW_TENSION, FIXED_PILE_LENGTH
    geo = best_params['geo']
    rb = best_params['rebar']
    mat = best_params['mat']

    load_uls = cfg.CONFIG['load_cases']["Extreme"]
    pf_uls = model.IntegratedPileFoundation(best_params, load_uls, cfg.SOIL_LAYERS, (1.35, 1.5))

    mode_str = " (允许受拉模式)" if ALLOW_TENSION else " (全抗压模式)"
    print("\n" + "=" * 80)
    print(f"                 风机基础优化设计最终报告{mode_str}")
    print("=" * 80)

    print(f"\n【1. 几何尺寸与材料总用量 (工程模数化最优解)】")
    print(f"  [承台整体] 埋深 H={geo['H']:.1f} m")
    print(f"  [承台底板] 底部半径 R={geo['R_cap']:.1f} m, 边缘厚度 h1={geo['h1']:.1f} m")
    print(f"  [承台斜面] 斜面高度 h2={geo['h2']:.1f} m, 斜面外缘(平台处)半径 r_top={geo['r_top']:.1f} m")
    print(f"  [承台台柱] 顶面半径 r_col={geo['r_col']:.1f} m, 台柱高度 h3={geo['h3']:.1f} m")

    total_h = geo['h1'] + geo['h2'] + geo['h3']
    print(f"            (承台总高度 = {total_h:.1f} m)")
    print(f"  [桩基规模] 共 {pf_uls.N_total} 根, 桩径 D={geo['D_pile']:.1f} m, 桩长 L={geo['L_pile']:.1f} m")
    print(f"  [真实混凝土总量] {V_conc:.1f} m³")
    print(f"  [真实钢筋总重量] {Total_Steel_kg:.1f} kg")
    print(f"  [优化等效体积] {best_cost:.1f} m³ (综合成本量纲)")

    R_b, r_col_b, r_top_b = 9.5, 3.4, 3.6
    h1_b, h2_b, h3_b = 1.0, 1.4, 1.2
    V_edge_b = math.pi * R_b ** 2 * h1_b
    V_slope_b = (1 / 3) * math.pi * h2_b * (R_b ** 2 + r_top_b ** 2 + R_b * r_top_b)
    V_stub_b = math.pi * r_col_b ** 2 * h3_b
    V_conc_baseline = V_edge_b + V_slope_b + V_stub_b

    unit_w_32 = 6.31
    unit_w_28 = 4.83
    unit_w_25 = 3.85
    L_long = 9.5 - 1.5
    L_short = L_long * 0.7

    W_bot_rad = (120 * L_long * unit_w_32) + (120 * L_short * unit_w_32) + (120 * L_short * unit_w_28)
    W_top_rad = (120 * L_long * unit_w_28) + (120 * L_short * unit_w_28)

    area_annulus = math.pi * ((9.5 - 0.05) ** 2 - 1.5 ** 2)
    W_bot_hoop = (area_annulus / 0.150 * unit_w_25) + (area_annulus / 0.200 * unit_w_25)
    W_top_hoop = (area_annulus / 0.120 * unit_w_25)

    W_cap_steel_b = W_bot_rad + W_top_rad + W_bot_hoop + W_top_hoop + (V_conc_baseline * 5.0)

    L_pile_baseline = float(FIXED_PILE_LENGTH) if FIXED_PILE_LENGTH is not None else 25.0
    if hasattr(model, 'calculate_pile_steel_detailed'):
        W_pile_steel_b = model.calculate_pile_steel_detailed(36, 0.8, L_pile_baseline, 18, 0.025)
    else:
        W_pile_steel_b = 36 * L_pile_baseline * (18 * math.pi * (0.025 / 2) ** 2 * 7850)

    W_other_b = V_conc_baseline * 40.0
    Total_Steel_baseline = W_cap_steel_b + W_pile_steel_b + W_other_b

    conc_price = mat.get('price', 500.0)
    steel_price = cfg.CONFIG.get('UNIT_PRICES', {}).get('STEEL', 5500.0) / 1000.0
    price_ratio = steel_price / conc_price
    cost_baseline_eq = V_conc_baseline + Total_Steel_baseline * price_ratio

    real_optimized_cost = V_conc + Total_Steel_kg * price_ratio

    print(f"\n【⭐ 算法寻优效益对标 (对比原施工图纸)】")
    if best_cost > real_optimized_cost + 1000:
        print(f"  [!] 提示: 当前最优解可能含有微小越界/不满足规范, 触发了核弹惩罚 (适应度得分:{best_cost:.1f})")

    print(f"  [等效造价对比] 原设计: {cost_baseline_eq:.1f} m³  ->  优化后: {real_optimized_cost:.1f} m³")
    cost_saving = (cost_baseline_eq - real_optimized_cost) / cost_baseline_eq * 100
    print(
        f"               => 综合造价下降率: ↓ {cost_saving:.1f}% (为您节省等效体积 {cost_baseline_eq - real_optimized_cost:.1f} m³)")
    print(f"  [真实混凝土对比] 原设计: {V_conc_baseline:.1f} m³  ->  优化后: {V_conc:.1f} m³")
    print(f"  [真实钢筋对比]   原设计: {Total_Steel_baseline:.1f} kg  ->  优化后: {Total_Steel_kg:.1f} kg")
    print(f"  [桩数对比]   原设计: 36 根  ->  优化后: {pf_uls.N_total} 根")

    s_bot_rad_outer = (2 * math.pi * geo['R_cap'] * 1000) / (rb['n1'] + rb.get('n3', rb['n1']))
    s_top_rad_outer = (2 * math.pi * geo['R_cap'] * 1000) / (rb['n4'] + rb.get('n5', rb['n4']))

    print(f"\n【2. 详细配筋方案 (构造与配置)】")
    print("--- 承台配筋 (Cap Rebar) ---")
    print(
        f"  [底板径向] ①通长筋: {rb['n1']}Φ{rb['d1'] * 1000:.0f} | ②外插筋: {rb.get('n3', rb['n1'])}Φ{rb.get('d3', rb['d1']) * 1000:.0f} | ③双层加强: {rb.get('n2', rb['n1'])}Φ{rb.get('d2', rb['d1']) * 1000:.0f}")
    print(f"             (外缘等效间距: {s_bot_rad_outer:.0f} mm)")

    d1_print = rb['ring_opt'].get('d1', rb['ring_opt'].get('d', 0.0))
    d2_print = rb['ring_opt'].get('d2', 0.0)
    s1_p = rb['ring_opt'].get('s1', rb['ring_opt'].get('s', 150))
    s2_p = rb['ring_opt'].get('s2', s1_p)
    print(f"  [底板环向] 底层 Φ{d1_print * 1000:.0f}@{s1_p:.0f} mm | 附加层 Φ{d2_print * 1000:.0f}@{s2_p:.0f} mm")

    print(
        f"  [顶板径向] ①通长筋: {rb['n4']}Φ{rb['d4'] * 1000:.0f} | ②外插筋: {rb.get('n5', rb['n4'])}Φ{rb.get('d5', rb['d4']) * 1000:.0f}")
    print(f"             (外缘等效间距: {s_top_rad_outer:.0f} mm)")
    print(f"  [顶板环向] Φ{rb['top_ring_opt']['d'] * 1000:.0f} @ {rb['top_ring_opt']['s']:.0f} mm")

    cover_mm = cfg.CONFIG.get('pile_geo', {}).get('cover', 0.06) * 1000.0
    D_core_mm = geo['D_pile'] * 1000.0 - 2 * cover_mm - (rb['pile_main_d'] * 1000.0)
    arc_spacing = (math.pi * D_core_mm) / rb['pile_main_n'] if D_core_mm > 0 else 0
    clear_spacing = arc_spacing - (rb['pile_main_d'] * 1000.0)

    dense_zone_len = 5.0 * geo['D_pile']
    stirrup_d_mm = best_params['stirrup']['d'] * 1000.0
    stirrup_s_norm = best_params['stirrup']['s'] * 1000.0
    stirrup_s_dense = cfg.CONFIG['PILE_CONSTANTS'].get('STIRRUP_S_DENSE', 0.1) * 1000.0

    print("--- 桩身配筋 (Pile Rebar) ---")
    print(f"  [主筋配置] {rb['pile_main_n']} 根 Φ{rb['pile_main_d'] * 1000:.0f} (沿截面圆周均匀布置)")
    print(f"  [主筋间距] 弧长间距: {arc_spacing:.1f} mm, 钢筋净距: {clear_spacing:.1f} mm")
    print(f"  [箍筋尺寸] 螺旋箍筋 Φ{stirrup_d_mm:.0f}")
    print(f"  [箍筋加密区] 桩顶以下 {dense_zone_len:.1f} m 范围内 (即 5d), 间距 {stirrup_s_dense:.0f} mm")
    print(f"  [箍筋非加密区] 间距 {stirrup_s_norm:.0f} mm")

    if not HAS_CHECK_MODULES: return

    try:
        from analyze_pile_structure import pm_manager
        HAS_PM = True
    except ImportError:
        HAS_PM = False

    print("\n【3. 结构力学验算结果】")
    print("-" * 80)
    print(f"{'No.':<4} | {'验算项目':<28} | {'设计值':<12} | {'限值/容许值':<12} | {'状态':<10}")
    print("-" * 80)

    def print_row(idx, name, val, limit, status=None, unit=""):
        if status is None:
            try:
                is_ok = float(val) <= float(limit)
                status = "[OK]" if is_ok else "[FAIL]"
            except:
                status = "INFO"

        if isinstance(val, (int, float)):
            val_str = f"{val:.5f}{unit}" if 0 < abs(val) < 0.01 else f"{val:.2f}{unit}"
        else:
            val_str = str(val)

        if isinstance(limit, (int, float)):
            lim_str = f"{limit:.5f}{unit}" if 0 < abs(limit) < 0.01 else f"{limit:.2f}{unit}"
        else:
            lim_str = str(limit)

        print(f"{idx:<4} | {name:<28} | {val_str:<12} | {lim_str:<12} | {status}")

    load_sls = cfg.CONFIG['load_cases']["Normal"]
    pf_sls = model.IntegratedPileFoundation(best_params, load_sls, cfg.SOIL_LAYERS, (1.0, 1.0))

    res_uls, Fxy_uls, _ = pf_uls.get_pile_reactions_detailed()
    res_sls, Fxy_sls, _ = pf_sls.get_pile_reactions_detailed()

    ring_dict_uls = {}
    for p in res_uls:
        rk = round(p['r'], 2)
        if rk not in ring_dict_uls: ring_dict_uls[rk] = {'r': p['r'], 'n': 0, 'N_comp': -1e9, 'N_tens': 1e9}
        ring_dict_uls[rk]['n'] += 1
        ring_dict_uls[rk]['N_comp'] = max(ring_dict_uls[rk]['N_comp'], p['N_design'])
        ring_dict_uls[rk]['N_tens'] = min(ring_dict_uls[rk]['N_tens'], p['N_design'])
    ring_env_uls = list(ring_dict_uls.values())

    N_max_uls = max([p['N_design'] for p in res_uls])
    N_min_uls = min([p['N_design'] for p in res_uls])

    ring_dict_sls = {}
    for p in res_sls:
        rk = round(p['r'], 2)
        if rk not in ring_dict_sls: ring_dict_sls[rk] = {'r': p['r'], 'n': 0, 'N_comp': -1e9, 'N_tens': 1e9}
        ring_dict_sls[rk]['n'] += 1
        ring_dict_sls[rk]['N_comp'] = max(ring_dict_sls[rk]['N_comp'], p['N_design'])
        ring_dict_sls[rk]['N_tens'] = min(ring_dict_sls[rk]['N_tens'], p['N_design'])
    ring_env_sls = list(ring_dict_sls.values())

    N_max_sls = max([p['N_design'] for p in res_sls])
    N_min_sls = min([p['N_design'] for p in res_sls])

    pile_ana = PileStructuralAnalysis(geo['D_pile'], rb['pile_main_n'], rb['pile_main_d'], mat, best_params['stirrup'],
                                      cover=cover_mm / 1000.0)
    cap_ana = CapStructuralAnalysis(geo, mat)
    fc_val = mat.get('fc', 19.1)

    _, theta_sls = pile_ana.check_settlement_and_tilt(load_sls['FZ'], load_sls['MXY'], geo['L_pile'], cfg.SOIL_LAYERS,
                                                      pf_sls.piles_coords)
    if theta_sls > 1e-6:
        K_theta = (load_sls['MXY'] / theta_sls) / 1e6
        print_row("Sys1", "系统旋转刚度", K_theta, 120.0, unit=" GN.m/rad",
                  status="[OK]" if K_theta >= 120.0 else "[FAIL]")

    _, as_req, as_prov = cap_ana.check_flexural_reinforcement(ring_env_uls, rb, detailed=True)
    print_row("C-1", "承台抗弯配筋 (As)", as_prov, as_req, unit=" mm2/m",
              status="[OK]" if as_prov >= as_req else "[FAIL]")

    _, v_des, v_cap = cap_ana.check_inclined_section_shear(ring_env_uls, rb['d1'] * 1000, detailed=True)
    print_row("C-2", "斜截面抗剪 (Shear)", v_des, v_cap, unit=" kN", status="[OK]" if v_des <= v_cap else "[FAIL]")

    _, n_punch, r_punch = cap_ana.check_corner_pile_punching(ring_env_uls, geo['D_pile'], detailed=True)
    print_row("C-3", "角桩冲切 (Punch)", n_punch, r_punch, unit=" kN",
              status="[OK]" if n_punch <= r_punch else "[FAIL]")

    crack_res = cap_ana.check_crack_width(ring_env_sls, rb, detailed=True)
    if len(crack_res) == 3:
        _, w_max, w_lim = crack_res
    else:
        _, w_max = crack_res
        w_lim = 0.20
    print_row("C-4", "承台裂缝宽度 (Crack)", w_max, w_lim, unit=" mm", status="[OK]" if w_max <= w_lim else "[FAIL]")

    print(f"--- 桩基结构 (Pile Structure) ---")
    Ra_norm, Ra_seis = pile_ana.calculate_compression_capacity(geo['L_pile'], cfg.SOIL_LAYERS)
    print_row("P-1", "单桩竖向抗压(SLS)", N_max_sls, 1.2 * Ra_norm, unit=" kN")

    if N_min_sls < 0:
        if not ALLOW_TENSION:
            print_row("P-2", "基桩抗拔(SLS)", N_min_sls, 0.0, unit=" kN", status="[FAIL] TENSION")
        else:
            Rt = pile_ana.calculate_uplift_capacity(geo['L_pile'], cfg.SOIL_LAYERS, pf_sls.piles_coords)
            print_row("P-2", "基桩抗拔(SLS)", abs(N_min_sls), Rt, unit=" kN")
    else:
        print_row("P-2", "基桩抗拔(SLS)", 0.0, "N/A", status="[NO PULL]")

    H_pile_sls = Fxy_sls / pf_sls.N_total
    Rh_norm, Rh_seis = pile_ana.calculate_horizontal_capacity(geo['L_pile'], cfg.SOIL_LAYERS)
    print_row("P-3", "水平承载力(SLS)", H_pile_sls, Rh_seis, unit=" kN")

    H_pile_uls = Fxy_uls / pf_uls.N_total
    util_str, cap_str = pile_ana.check_structural_strength(N_max_uls)
    print_row("P-4", "桩身轴压强度(ULS)", N_max_uls, cap_str, unit=" kN")

    r_outer = max([p['r'] for p in res_uls])
    ratio_local = pile_ana.check_pile_top_local_compression(N_max_uls, fc_val, geo['R_cap'], r_outer)
    print_row("P-5", "桩顶局部受压(ULS)", f"{ratio_local:.2f}", "1.0",
              status="[OK]" if ratio_local <= 1.0 else "[FAIL]")

    _, cap_buck, phi = pile_ana.check_buckling_stability(N_max_uls, geo['L_pile'], cfg.SOIL_LAYERS)
    print_row("P-6", "压屈稳定性(ULS)", N_max_uls, cap_buck, status=f"phi={phi:.2f}")

    if N_min_uls < 0:
        _, cap_tens = pile_ana.check_tensile_strength(N_min_uls)
        print_row("P-7", "桩身受拉强度(ULS)", abs(N_min_uls), cap_tens, unit=" kN",
                  status="[OK]" if cap_tens >= abs(N_min_uls) else "[FAIL]")
    else:
        print_row("P-7", "桩身受拉强度(ULS)", 0.0, "N/A", status="[NO PULL]")

    _, _, M_pile_uls, _ = pile_ana.calculate_horizontal_mechanics_fem_detailed(H_pile_uls, 0.0, geo['L_pile'],
                                                                               cfg.SOIL_LAYERS, pf_uls.piles_coords)

    Mu_c = pm_manager.get_capacity_usage(geo['D_pile'], rb['pile_main_n'], rb['pile_main_d'], N_max_uls, M_pile_uls,
                                         fc_val, cover_mm, True) if HAS_PM else pile_ana._calculate_circular_Mu(
        N_max_uls) * 1000.0
    u_sh_c, u_b_c, _ = pile_ana.check_bending_shear_capacity(M_pile_uls, H_pile_uls, N_max_uls, Mu_c)

    if N_min_uls < 0:
        Mu_t = pm_manager.get_capacity_usage(geo['D_pile'], rb['pile_main_n'], rb['pile_main_d'], N_min_uls, M_pile_uls,
                                             fc_val, cover_mm, True) if HAS_PM else pile_ana._calculate_circular_Mu(
            N_min_uls) * 1000.0
        u_sh_t, u_b_t, _ = pile_ana.check_bending_shear_capacity(M_pile_uls, H_pile_uls, N_min_uls, Mu_t)
        u_shear_max, u_bend_max = max(u_sh_c, u_sh_t), max(u_b_c, u_b_t)
    else:
        u_shear_max, u_bend_max = u_sh_c, u_b_c

    print_row("P-8", "桩身受弯剪承载(ULS)", f"V:{u_shear_max:.2f}", "1.00",
              status="[OK]" if max(u_shear_max, u_bend_max) <= 1.0 else "[FAIL]")

    _, _, M_pile_sls, _ = pile_ana.calculate_horizontal_mechanics_fem_detailed(H_pile_sls, 0.0, geo['L_pile'],
                                                                               cfg.SOIL_LAYERS, pf_sls.piles_coords)
    w_pile_comp = pile_ana.check_crack_width(M_pile_sls, N_max_sls)
    w_pile_comp = w_pile_comp if w_pile_comp is not None else 0.0

    if N_min_sls < 0:
        w_pile_tens = pile_ana.check_crack_width(M_pile_sls, N_min_sls)
        w_pile_tens = w_pile_tens if w_pile_tens is not None else 0.0
    else:
        w_pile_tens = 0.0

    w_pile = max(w_pile_comp, w_pile_tens)
    print_row("P-9", "桩身最大裂缝(SLS)", w_pile, 0.2, unit=" mm")

    Fz_tot_sls = load_sls['FZ'] + pf_sls.get_quantities()[0] * 25.0
    check_weak = pile_ana.check_weak_underlying_layer(Fz_tot_sls, 0.0, geo['L_pile'], cfg.SOIL_LAYERS,
                                                      pf_sls.piles_coords)
    print_row("P-10", "软弱下卧层承载力(SLS)", f"{check_weak:.2f}", "1.0",
              status="[OK]" if check_weak <= 1.0 else "[FAIL]")

    ln = sum([ly.thickness for ly in cfg.SOIL_LAYERS if getattr(ly, 'is_compressible', False)])
    ratio_neg = pile_ana.check_negative_friction(N_max_sls, Ra_norm, ln, cfg.SOIL_LAYERS, pf_uls.piles_coords)
    print_row("P-11", "负摩阻力影响(SLS)", f"{ratio_neg:.2f}", "1.0", status="[OK]" if ratio_neg <= 1.0 else "[FAIL]")

    s_final, theta = pile_ana.check_settlement_and_tilt(load_sls['FZ'], load_sls['MXY'], geo['L_pile'], cfg.SOIL_LAYERS,
                                                        pf_sls.piles_coords)
    print_row("P-12a", "基础总沉降(SLS)", s_final, "N/A", unit=" mm", status="INFO")
    print_row("P-12b", "基础总倾斜(SLS)", theta, 0.003, status="[OK]" if theta <= 0.003 else "[FAIL]")

    print("-" * 80)


# ================= 6. Abaqus 数据导出接口 =================
def export_abaqus_json(best_params, filename="abaqus_pile_model.json"):
    geo = best_params['geo']
    rb = best_params['rebar']
    mat = best_params['mat']

    soil_data = []
    for layer in cfg.SOIL_LAYERS:
        soil_data.append({
            "name": layer.name,
            "thickness_m": layer.thickness,
            "density_kN_m3": layer.gamma,
            "elastic_modulus_MPa": getattr(layer, 'Es', 15.0),
            "poisson_ratio": layer.v,
            "cohesion_kPa": layer.C,
            "friction_angle_deg": layer.angle,
            "qsik_kPa": layer.qsik,
            "qpk_kPa": layer.qpk
        })

    load_uls = cfg.CONFIG['load_cases']["Extreme"]
    pf_uls = model.IntegratedPileFoundation(best_params, load_uls, cfg.SOIL_LAYERS, (1.35, 1.5))
    res_uls, Fxy_uls, _ = pf_uls.get_pile_reactions_detailed()

    N_max_comp = max([p['N_design'] for p in res_uls])
    N_max_tens = min([p['N_design'] for p in res_uls])
    H_max = max([p.get('H_actual', Fxy_uls / pf_uls.N_total) for p in res_uls])

    critical_loads = {
        "max_compression_kN": round(N_max_comp, 2),
        "max_tension_kN": round(N_max_tens if N_max_tens < 0 else 0.0, 2),
        "max_shear_kN": round(H_max, 2)
    }

    stirrup_consts = cfg.CONFIG['PILE_CONSTANTS']
    pile_data = {
        "geometry": {
            "D_pile_m": geo['D_pile'],
            "L_pile_m": geo['L_pile'],
            "cover_thickness_m": cfg.CONFIG.get('pile_geo', {}).get('cover', 0.06)
        },
        "rebar": {
            "main_rebar_n": rb['pile_main_n'],
            "main_rebar_d_m": rb['pile_main_d'],
            "stirrup_d_m": best_params['stirrup']['d'],
            "stirrup_s_dense_m": stirrup_consts.get('STIRRUP_S_DENSE', 0.1),
            "stirrup_s_norm_m": stirrup_consts.get('STIRRUP_S_NORM', 0.2),
            "dense_zone_length_m": round(5.0 * geo['D_pile'], 2)
        },
        "concrete_material": {
            "fc_MPa": mat.get('fc', 19.1),
            "ft_MPa": mat.get('ft', 1.71),
            "Ec_MPa": mat.get('Ec', 3.25e4)
        }
    }

    export_data = {
        "pile_property": pile_data,
        "soil_layers": soil_data,
        "critical_loads_uls": critical_loads
    }

    os.makedirs(ABAQUS_OUTPUT_DIR, exist_ok=True)
    output_path = filename if os.path.isabs(filename) else os.path.join(ABAQUS_OUTPUT_DIR, filename)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(export_data, f, ensure_ascii=False, indent=4)
    print(f"\n[Abaqus 接口] 桩基建模参数已成功导出至: {output_path}")


# ================= 7. 主程序入口与 GA 流程 =================
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)

toolbox = base.Toolbox()
toolbox.register("population", lambda n: [creator.Individual(ind) for ind in lhs_sample(len(VAR_RANGES), n)])
toolbox.register("evaluate", evaluate)
toolbox.register("mate", tools.cxTwoPoint)


def main():
    global CURRENT_ITER, ALLOW_TENSION, FIXED_PILE_LENGTH

    print("=" * 60)
    print("   风机基础智能优化程序 (等效造价 + 允许打捆穿刺避让版)")
    print("=" * 60)

    while True:
        user_input = input(">> 是否允许出现受拉桩？(y/n) [默认:y] : ").strip().lower()
        if user_input in ['y', 'yes', '']:
            ALLOW_TENSION = True
            print("=> 【已选择】允许受拉模式")
            break
        elif user_input in ['n', 'no']:
            ALLOW_TENSION = False
            print("=> 【已选择】全抗压模式")
            break
        else:
            print("输入无效，请输入 'y' 或 'n'。")

    while True:
        l_input = input(">> 是否手动锁定桩长？(输入具体数字如 28，直接回车自动优化) : ").strip()
        if l_input == '':
            FIXED_PILE_LENGTH = None
            print("=> 【已选择】算法自动寻优最佳桩长")
            break
        else:
            try:
                FIXED_PILE_LENGTH = float(l_input)
                if FIXED_PILE_LENGTH <= 0: raise ValueError
                print(f"=> 【已选择】桩长被强制锁定为: {FIXED_PILE_LENGTH} m")
                break
            except ValueError:
                print("输入无效。")

    ga_params = cfg.CONFIG['GA_PARAMS']
    lows = [r[0] for r in VAR_RANGES]
    ups_raw = [r[1] for r in VAR_RANGES]
    ups_safe = [u + 1e-9 if u <= l else u for l, u in zip(lows, ups_raw)]

    toolbox.register("mutate", tools.mutPolynomialBounded, eta=ga_params.get('ETA', 12.0), low=lows, up=ups_safe,
                     indpb=0.15)
    toolbox.register("select", tools.selTournament, tournsize=ga_params.get('TOURN_SIZE', 3))

    total_cores = os.cpu_count()
    use_cores = 16
    print(f"\n--- 启动优化引擎 (核心数: {use_cores}) ---")

    pool = multiprocessing.Pool(processes=use_cores, initializer=init_worker,
                                initargs=(ALLOW_TENSION, FIXED_PILE_LENGTH))
    toolbox.register("map", pool.map)

    pop = toolbox.population(ga_params['POP_SIZE'])
    hof = tools.HallOfFame(1)

    # P5 fix: split into material score and violation tracking
    history_mat_score = []
    history_viol = []
    history_vc = []
    history_ws = []

    # ================= 初始代评估 =================
    invalid_ind = [ind for ind in pop if not ind.fitness.valid]
    fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)

    for ind, fit_tuple in zip(invalid_ind, fitnesses):
        ind.mat_score, ind.viol = fit_tuple
        if ind.viol > 1e-4:
            safe_viol = min(ind.viol, 1000.0)
            ind.fitness.values = (ind.mat_score + 4000.0 + safe_viol * 20000.0,)
        else:
            ind.fitness.values = (ind.mat_score,)

    hof.update(pop)

    ngen = ga_params['N_GEN']
    pop_size = ga_params['POP_SIZE']

    elite_ratio = min(0.015, ga_params.get('ELITE_RATIO', 0.05))
    n_elite = max(2, int(pop_size * elite_ratio))

    n_mid = int(pop_size * 0.7)
    n_bad = pop_size - n_elite - n_mid

    pbar = tqdm(range(1, ngen + 1), desc="🚀 并行优化中", unit="gen", ncols=120)
    for gen in pbar:
        CURRENT_ITER = gen

        sorted_pop = tools.selBest(pop, len(pop))
        elites = [toolbox.clone(ind) for ind in sorted_pop[:n_elite]]
        mid_parents = toolbox.select(sorted_pop[:(n_elite + n_mid)], n_mid)
        mids = [toolbox.clone(ind) for ind in mid_parents]

        for i in range(1, len(mids), 2):
            if random.random() < ga_params['CX_PROB']:
                toolbox.mate(mids[i - 1], mids[i])
                del mids[i - 1].fitness.values
                del mids[i].fitness.values

        for i in range(len(mids)):
            if random.random() < ga_params['MUT_PROB']:
                toolbox.mutate(mids[i])
                del mids[i].fitness.values

        # P4 fix: clip genes to bounds after crossover/mutation
        for ind in mids:
            clip_gene(ind)

        bads = [creator.Individual(ind) for ind in lhs_sample(len(VAR_RANGES), n_bad)]
        pop[:] = elites + mids + bads

        invalid_ind = [ind for ind in pop if not ind.fitness.valid]
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)

        for ind, fit_tuple in zip(invalid_ind, fitnesses):
            ind.mat_score, ind.viol = fit_tuple

        progress = gen / ngen
        base_penalty = 4000.0
        dynamic_factor = 20000.0 + 180000.0 * (progress ** 3)

        for ind in pop:
            if not hasattr(ind, 'mat_score'):
                ind.mat_score, ind.viol = 1e10, 1e10

            if ind.viol > 1e-4:
                safe_viol = min(ind.viol, 1000.0)
                final_fit = ind.mat_score + base_penalty + safe_viol * dynamic_factor
                ind.fitness.values = (final_fit,)
            else:
                ind.fitness.values = (ind.mat_score,)

        hof.clear()
        hof.update(pop)

        best_ind = tools.selBest(pop, 1)[0]
        params = decode_gene(best_ind)

        pf_basic = model.IntegratedPileFoundation(params, cfg.CONFIG['load_cases']["Normal"], cfg.SOIL_LAYERS)
        Vc, W_cap_steel, _, _ = pf_basic.get_quantities()

        if hasattr(model, 'calculate_pile_steel_detailed'):
            W_pile = model.calculate_pile_steel_detailed(
                pf_basic.N_total, params['geo']['D_pile'], params['geo']['L_pile'],
                params['rebar']['pile_main_n'], params['rebar']['pile_main_d']
            )
        else:
            W_pile = pf_basic.N_total * params['geo']['L_pile'] * (
                    params['rebar']['pile_main_n'] * math.pi * (params['rebar']['pile_main_d'] / 2) ** 2 * 7850)

        W_other = Vc * 40.0
        Total_Steel = W_cap_steel + W_pile + W_other

        # P5 fix: Pareto-style best tracking
        current_mat_score = best_ind.mat_score
        current_viol = best_ind.viol

        if len(history_mat_score) == 0:
            history_mat_score.append(current_mat_score)
            history_viol.append(current_viol)
        else:
            prev_best_viol = history_viol[-1]
            prev_best_score = history_mat_score[-1]
            if current_viol < prev_best_viol - 1e-8:
                history_mat_score.append(current_mat_score)
                history_viol.append(current_viol)
            elif current_viol <= prev_best_viol + 1e-8 and current_mat_score < prev_best_score:
                history_mat_score.append(current_mat_score)
                history_viol.append(current_viol)
            else:
                history_mat_score.append(prev_best_score)
                history_viol.append(prev_best_viol)

        history_vc.append(Vc)
        history_ws.append(Total_Steel)

        display_cost = best_ind.mat_score if best_ind.viol < 1e-4 else current_gen_best_fitness
        pbar.set_postfix({"Best_Score": f"{display_cost:.1f}"})

    pool.close()
    pool.join()

    if not np.any(np.isnan(hof[0])):
        best_params = decode_gene(hof[0])
        load_uls = cfg.CONFIG['load_cases']["Extreme"]
        pf_uls = model.IntegratedPileFoundation(best_params, load_uls, cfg.SOIL_LAYERS, (1.35, 1.5))
        coords, forces = pf_uls.get_individual_pile_forces()

        plot_results(best_params, coords, forces, history_mat_score, history_vc, history_ws, history_viol)
        print_detailed_report(best_params, history_vc[-1], history_ws[-1], history_mat_score[-1])

        export_abaqus_json(best_params, "abaqus_pile_model.json")

        if HAS_LAYOUT_ENGINE:
            print("\n【4. 启动微观钢筋排布与避让引擎 (底板)】")
            ach_config = cfg.CONFIG.get('ANCHOR_CAGE', {'PCD': 4.8, 'N_BOLTS': 160, 'BOLT_D': 48})

            geo_opt = best_params['geo']
            rb_opt = best_params['rebar']

            R_cap = geo_opt['R_cap']
            r_col = geo_opt['r_col']
            r_void = 1.5

            n1 = rb_opt['n1']
            d1 = rb_opt['d1']
            n2 = rb_opt.get('n2', n1)
            d2 = rb_opt.get('d2', d1)
            n3 = rb_opt.get('n3', n1)
            d3 = rb_opt.get('d3', d1)

            print(f"  -> 成功读取最优解配筋: 通长筋 n1={n1}, 加强筋 n2={n2}, 外插筋 n3={n3}")
            # =========================================================================
            # 【全新整合模块】自动打通数据链：将详细用量精准导出为标准化 JSON 文件
            # =========================================================================
            from datetime import datetime

            # 提取顶板径向钢筋参数
            n4 = rb_opt.get('n4', 110)
            d4 = rb_opt.get('d4', 25.0)
            n5 = rb_opt.get('n5', 110)
            d5 = rb_opt.get('d5', 25.0)

            # 获取极限荷载组合
            limit_m = cfg.LOAD_COMBINATIONS[0].get('Mz', 85000.0) if 'cfg' in locals() and hasattr(cfg, 'LOAD_COMBINATIONS') else 85000.0
            limit_n = cfg.LOAD_COMBINATIONS[0].get('Fz', 15000.0) if 'cfg' in locals() and hasattr(cfg, 'LOAD_COMBINATIONS') else 15000.0

            # 组装满足多层级径向配筋 + 构造环向筋要求的标准交底字典
            rebar_json_data = {
                "Metadata": {
                    "Project": "风机基础多目标宏观寻优输出报告",
                    "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                },
                "Boundary_Conditions": {
                    "Limit_M_kNm": float(limit_m),
                    "Limit_N_kN": float(limit_n)
                },
                "Foundation_Geometry": {
                    "Radius_m": float(R_cap),
                    "Thickness_m": float(geo_opt.get('H_cap', geo_opt.get('H', 2.8)))
                },
                "Bottom_Plate_Rebar": {
                    "Radial": {
                        "Through_Bars": {  # ① 通长筋
                            "Count": int(n1),
                            "Diameter_mm": float(d1)
                        },
                        "Insert_Bars": {  # ② 外插筋
                            "Count": int(n3),
                            "Diameter_mm": float(d3)
                        },
                        "Reinforced_Bars": {  # ③ 双层加强筋
                            "Count": int(n2),
                            "Diameter_mm": float(d2)
                        },
                        "Outer_Equivalent_Spacing_mm": float(geo_opt.get('outer_spacing_bot', 180.0))
                    },
                    "Circular_Constructive": {  # 环向筋构造
                        "Additional_Layer": "筋仅作构造，不参与排布寻优",
                        "Bottom_Layer": "Φ20@200mm"
                    }
                },
                "Top_Plate_Rebar": {
                    "Radial": {
                        "Through_Bars": {  # ① 通长筋
                            "Count": int(n4),
                            "Diameter_mm": float(d4)
                        },
                        "Insert_Bars": {  # ② 外插筋
                            "Count": int(n5),
                            "Diameter_mm": float(d5)
                        },
                        "Outer_Equivalent_Spacing_mm": float(geo_opt.get('outer_spacing_top', 294.0))
                    },
                    "Circular_Constructive": {  # 环向筋仅作构造
                        "Spec": "Φ20@100mm"
                    }
                },
                "Optimization_Summary": {
                    "Total_Cost_RMB": float(best_params.get('total_cost', 0.0))
                }
            }

            # 导出本地标准 JSON 数据桥接包
            json_output_path = "Optimized_Rebar_Data.json"
            try:
                with open(json_output_path, 'w', encoding='utf-8') as json_file:
                    json.dump(rebar_json_data, json_file, ensure_ascii=False, indent=4)
                print(f"  -> 🎉 [数据链路打通] 最优多层配筋清单已自动导出至本地: {json_output_path}")
            except Exception as json_err:
                print(f"  -> ❌ [警告] JSON 数据接口生成失败: {json_err}")
            # =========================================================================

            layout_engine = BottomRebarLayout(
                R_cap=R_cap, r_void=r_void, r_col=r_col,
                n1_count=n1, d1=d1,
                n2_count=n2, d2=d2,
                n3_count=n3, d3=d3,
                ach_config=ach_config
            )
            layout_engine.visualize()

            print("\n【5. 启动微观钢筋排布与避让引擎 (顶板)】")
            r_top = geo_opt.get('r_top', r_col + 0.2)
            n4 = rb_opt['n4']
            d4 = rb_opt['d4']
            n5 = rb_opt.get('n5', n4)
            d5 = rb_opt.get('d5', d4)

            print(f"  -> 成功读取最优解顶板配筋: 通长筋 n4={n4}, 外插筋 n5={n5}")

            top_layout_engine = TopRebarLayout(
                R_cap=R_cap, r_void=r_void, r_col=r_col, r_top=r_top,
                n4_count=n4, d4=d4,
                n5_count=n5, d5=d5,
                ach_config=ach_config
            )
            top_layout_engine.visualize()

    else:
        print("\n【错误】未能找到有效解。")

if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()