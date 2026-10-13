# -*- coding: utf-8 -*-
"""
文件名: optimize_foundation_ga.py (多进程加速 + 动态视觉缩放 + 等效体积造价导向 + 允许打捆穿刺避让版)
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
import time
from functools import partial

import foundation_config as cfg
import analyze_integrated_pile_foundation as model
from anchor_cage_solver import (solve_anchor_for_foundation, get_anchor_solver_stats,
                                reset_anchor_solver_stats, minimum_anchor_envelope)

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
ANCHOR_MODE = cfg.CONFIG.get("ANCHOR_OPT", {}).get("MODE", "fixed")
ANCHOR_FATIGUE_MOMENT_RATIO = cfg.CONFIG.get("ANCHOR_OPT", {}).get("FATIGUE_MOMENT_RATIO", 0.10)
ANCHOR_PCD_SAMPLES = cfg.CONFIG.get("ANCHOR_OPT", {}).get("SOLVER", {}).get("PCD_PAIRS_PER_COMBO", 8)
ANCHOR_FULL_PCD_GRID = cfg.CONFIG.get("ANCHOR_OPT", {}).get("SOLVER", {}).get("FULL_PCD_GRID", False)
ABAQUS_OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "abaqus_postprocess")

# ========== 改进1: 可行性控制初始化开关 ==========
USE_FEASIBLE_INIT = True
# ========== 改进2: 自适应交叉变异开关 ==========
USE_ADAPTIVE = False
# ========== 改进3: SA局部搜索开关 ==========
USE_SA = False


def init_worker(allow_tension_flag, fixed_pile_length_val, anchor_mode="fixed",
                fatigue_moment_ratio=None, anchor_pcd_samples=None,
                anchor_full_pcd_grid=False):
    """多进程子进程初始化函数：用于将主进程的用户选择同步给所有后台核心"""
    global ALLOW_TENSION, FIXED_PILE_LENGTH, ANCHOR_MODE, ANCHOR_FATIGUE_MOMENT_RATIO
    global ANCHOR_PCD_SAMPLES, ANCHOR_FULL_PCD_GRID
    ALLOW_TENSION = allow_tension_flag
    FIXED_PILE_LENGTH = fixed_pile_length_val
    ANCHOR_MODE = anchor_mode
    if fatigue_moment_ratio is not None:
        ANCHOR_FATIGUE_MOMENT_RATIO = float(fatigue_moment_ratio)
        cfg.CONFIG["ANCHOR_OPT"]["FATIGUE_MOMENT_RATIO"] = ANCHOR_FATIGUE_MOMENT_RATIO
    if anchor_pcd_samples is not None:
        ANCHOR_PCD_SAMPLES = int(anchor_pcd_samples)
        cfg.CONFIG["ANCHOR_OPT"]["SOLVER"]["PCD_PAIRS_PER_COMBO"] = ANCHOR_PCD_SAMPLES
    ANCHOR_FULL_PCD_GRID = bool(anchor_full_pcd_grid)
    cfg.CONFIG["ANCHOR_OPT"]["SOLVER"]["FULL_PCD_GRID"] = ANCHOR_FULL_PCD_GRID
    reset_anchor_solver_stats()


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


def repair_individual(ind):
    """将基因修复到合法边界，并把离散索引吸附到整数。"""
    for i, value in enumerate(ind):
        low, up = VAR_RANGES[i]
        fixed = max(low, min(up, float(value)))
        if i in DISCRETE_MAP:
            fixed = float(int(round(fixed)))
        ind[i] = fixed
    return ind


def _polynomial_delta(eta):
    """生成 [0, 1] 区间上的多项式变异增量。"""
    rand = random.random()
    if rand < 0.5:
        return (2.0 * rand) ** (1.0 / (eta + 1.0)) - 1.0
    return 1.0 - (2.0 * (1.0 - rand)) ** (1.0 / (eta + 1.0))


def mutate_mixed(ind, eta=10.0, indpb=0.15, discrete_step=2):
    """
    混合变量变异。

    连续变量使用多项式变异，离散工程变量跳到相邻档位。
    """
    for i in range(len(ind)):
        if random.random() >= indpb:
            continue

        low, up = VAR_RANGES[i]
        if i in DISCRETE_MAP:
            current_idx = int(round(float(ind[i])))
            max_idx = len(DISCRETE_MAP[i]) - 1
            max_step = max(1, int(discrete_step))
            step_options = tuple(sorted({
                -max_step, -1, 1, max_step
            }))
            step = random.choice(step_options)
            ind[i] = float(max(0, min(max_idx, current_idx + step)))
        else:
            delta = _polynomial_delta(float(eta))
            ind[i] = max(low, min(up, float(ind[i]) + delta * (up - low)))

    return ind,


def mate_mixed(ind1, ind2, eta=10.0):
    """
    混合变量交叉。

    连续变量使用模拟二进制交叉，离散变量采用档位交换。
    """
    for i in range(len(ind1)):
        low, up = VAR_RANGES[i]
        if i in DISCRETE_MAP:
            if random.random() < 0.5:
                ind1[i], ind2[i] = ind2[i], ind1[i]
            continue

        if random.random() >= 0.5:
            continue

        rand = random.random()
        if rand <= 0.5:
            beta = (2.0 * rand) ** (1.0 / (eta + 1.0))
        else:
            beta = (1.0 / (2.0 * (1.0 - rand))) ** (1.0 / (eta + 1.0))

        child1 = 0.5 * ((1.0 + beta) * ind1[i] + (1.0 - beta) * ind2[i])
        child2 = 0.5 * ((1.0 - beta) * ind1[i] + (1.0 + beta) * ind2[i])
        ind1[i] = max(low, min(up, child1))
        ind2[i] = max(low, min(up, child2))

    repair_individual(ind1)
    repair_individual(ind2)
    return ind1, ind2


# ================= 1.5 改进1: 可行性控制初始化 =================
def feasible_init_population(n, max_attempts_factor=200):
    """
    Enhanced pre-screening. Checks in order of computational cost:
      1-5: geometry (cheap, same as before)
      6:   pile spacing >= 3.0D (matches evaluate constraint)
      7:   min edge distance (outer ring + 1.5*D <= R_cap)
      8:   cap thickness accommodates anchor cage
      9:   inner ring clears anchor cage outer radius
      10:  quick single-pile compression capacity estimate
    """
    import foundation_config as _cfg
    soil = _cfg.SOIL_LAYERS
    anchor_opt = _cfg.CONFIG["ANCHOR_OPT"]
    ach_R_outer = minimum_anchor_envelope(anchor_opt, ANCHOR_MODE) / 2.0
    fixed = anchor_opt["FIXED"]
    ach_L_total = fixed["L_TOTAL_M"]
    ach_top_expose = fixed["TOP_EXPOSE_M"]
    ach_bottom_groove = fixed["BOTTOM_GROOVE_M"]
    ach_bottom_cover = fixed["BOTTOM_COVER_M"]

    # Rough loads for capacity check
    load_extreme = _cfg.CONFIG["load_cases"]["Extreme"]
    Fz_rough = load_extreme["FZ"] * 1.35
    conc_density = 25.0
    soil_density = 18.0
    SAFE_K = 2.0

    pop = []
    max_attempts = n * max_attempts_factor
    attempts = 0

    while len(pop) < n and attempts < max_attempts:
        ind = creator.Individual([
            random.uniform(VAR_RANGES[j][0], VAR_RANGES[j][1])
            for j in range(len(VAR_RANGES))
        ])
        safe = [max(VAR_RANGES[j][0], min(VAR_RANGES[j][1], ind[j]))
                for j in range(len(VAR_RANGES))]

        try:
            params = decode_gene(safe)
            geo = params["geo"]
            rb = params["rebar"]

            # 1. Pedestal not touching cap edge
            if geo["r_col"] >= geo["R_cap"] - 0.5:
                attempts += 1; continue
            # 2. Pedestal height OK
            if geo["h3"] < 0.2:
                attempts += 1; continue
            # 3. Pile rings within cap
            ok = True
            for ring in geo["rings"]:
                if ring["r"] > geo["R_cap"] - geo["D_pile"]:
                    ok = False; break
            if not ok:
                attempts += 1; continue
            # 4. Pile rebar ratio
            pile_area = math.pi * (geo["D_pile"] / 2) ** 2
            steel_area = rb["pile_main_n"] * math.pi * (rb["pile_main_d"] / 2) ** 2
            rho = steel_area / pile_area
            if rho < 0.004 or rho > 0.04:
                attempts += 1; continue

            # 5. Inter-ring pile spacing (both rings exist)
            r_in, r_out = geo["rings"][0]["r"], geo["rings"][1]["r"]
            n_in, n_out = geo["rings"][0]["n"], geo["rings"][1]["n"]
            Dp = geo["D_pile"]
            N_total = n_in + n_out

            # 6. Pile spacing >= 3.0D (stricter, matches evaluate)
            if n_in > 0:
                s_in = 2 * r_in * math.sin(math.pi / n_in)
                if s_in < 3.0 * Dp:
                    attempts += 1; continue
            if n_out > 0:
                s_out = 2 * r_out * math.sin(math.pi / n_out)
                if s_out < 3.0 * Dp:
                    attempts += 1; continue

            # 7. Edge distance: outer pile edge + 1.0*Dp <= R_cap
            if r_out + Dp / 2.0 + Dp > geo["R_cap"]:
                attempts += 1; continue

            # 8. Cap thickness accommodates anchor cage
            total_h = geo["h1"] + geo["h2"] + geo["h3"]
            req_ach_depth = ach_L_total - ach_top_expose + ach_bottom_cover
            if total_h - ach_bottom_groove < req_ach_depth:
                attempts += 1; continue

            # 9. Inner ring clears anchor cage (r_in - Dp/2 >= ach_R_outer + 0.6)
            if r_in - Dp / 2.0 < ach_R_outer + 0.6:
                attempts += 1; continue

            # 10. Quick single-pile compression capacity check
            u = math.pi * Dp
            Ap = math.pi * (Dp / 2) ** 2
            psi_s = (0.8 / Dp) ** (1.0 / 3.0) if Dp > 0.8 else 1.0
            psi_p = (0.8 / Dp) ** (1.0 / 3.0) if Dp > 0.8 else 1.0
            Q_sk, depth, qpk_end = 0.0, 0.0, 0.0
            for layer in soil:
                if depth >= geo["L_pile"]:
                    break
                t = min(layer.thickness, geo["L_pile"] - depth)
                if t > 0:
                    Q_sk += u * layer.qsik * t
                    qpk_end = layer.qpk
                depth += t
            Q_uk = Q_sk * psi_s + qpk_end * Ap * psi_p
            Ra_quick = Q_uk / SAFE_K

            # Rough load estimate
            Vc_rough = math.pi * geo["R_cap"] ** 2 * total_h * 0.85
            V_soil = math.pi * geo["R_cap"] ** 2 * geo["H"] - Vc_rough
            G_total = Vc_rough * conc_density + max(0, V_soil) * soil_density
            N_per_pile = (Fz_rough + G_total * 1.35) / N_total

            if N_per_pile > Ra_quick * 1.2:
                attempts += 1; continue

            # All checks passed
            pop.append(ind)

        except Exception:
            attempts += 1; continue
        attempts += 1

    # Fallback: pad with random individuals if not enough
    while len(pop) < n:
        pop.append(creator.Individual([random.uniform(*VAR_RANGES[j])
                                        for j in range(len(VAR_RANGES))]))
    print(f"  Enhanced feasible init: pop={len(pop)}  attempts={attempts}")
    return pop

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
def evaluate_detailed(ind, _anchor_result=None, _foundation_cache=None):
    """返回目标值、总违反度和逐项命名约束，供新实验路径使用。"""
    global ALLOW_TENSION
    if np.any(np.isnan(ind)):
        return {
            "objective": 1e10,
            "total_violation": 1e10,
            "constraints": {"invalid_individual": {"violation": 1e10}},
            "status": "invalid",
        }

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

        anchor_cfg = cfg.CONFIG["ANCHOR_OPT"]
        if _anchor_result is None:
            foundation_cache = {}
            anchor_result = solve_anchor_for_foundation(
                params, anchor_cfg, cfg.CONFIG["load_cases"], mode=ANCHOR_MODE,
                assess_candidate=lambda item: evaluate_detailed(safe_values, item, foundation_cache),
                selection_context={"allow_tension": ALLOW_TENSION,
                                   "config": cfg.CONFIG,
                                   "soil": [vars(layer) for layer in cfg.SOIL_LAYERS]},
            )
            return anchor_result["coupled_details"]
        anchor_result = _anchor_result
        params["anchor"] = anchor_result["anchor"]

        # Only the envelope affects the foundation structural assessment and
        # foundation quantities. Reuse it across count/type/inner-PCD variants.
        envelope = params["anchor"]["D_outer_m"]
        if _foundation_cache is not None and envelope in _foundation_cache:
            assessment = _foundation_cache[envelope]
        else:
            assessment = model.assess_design_performance(params, return_details=True)
            if _foundation_cache is not None:
                _foundation_cache[envelope] = assessment
        total_viol = float(assessment["total_violation"])
        constraints = dict(assessment["constraints"])

        def record_constraint(
            name,
            raw_violation,
            weight=1.0,
            hard=False,
            utilization=None,
            value=None,
            limit=None,
            unit="",
        ):
            raw = max(0.0, float(raw_violation))
            if not np.isfinite(raw):
                raw = 1e6
            weighted = raw * float(weight)
            current = constraints.get(name)
            if current is None or weighted > current.get("weighted_violation", -1.0):
                constraints[name] = {
                    "violation": raw,
                    "weight": float(weight),
                    "weighted_violation": weighted,
                    "hard": bool(hard),
                    "utilization": None if utilization is None else float(utilization),
                    "value": None if value is None else float(value),
                    "limit": None if limit is None else float(limit),
                    "unit": unit,
                }

        total_viol += float(anchor_result["total_violation"])
        for anchor_name, anchor_constraint in anchor_result["constraints"].items():
            item = dict(anchor_constraint)
            item.setdefault("weight", 1.0)
            item.setdefault("weighted_violation", item.get("violation", 0.0))
            item.setdefault("hard", True)
            constraints[f"anchor_{anchor_name}"] = item

        ach = params["anchor"]
        L_tot = ach.get('L_total_m', 3.4)
        bg = ach.get('bottom_groove_m', 0.3)
        te = ach.get('top_expose_m', 0.3)
        bc = ach.get('bottom_cover_m', 0.2)
        Do = ach.get('D_outer_m', 5.1)

        total_concrete_thickness = geo['h1'] + geo['h2'] + geo['h3'] - bg
        required_depth = L_tot - te + bc
        if total_concrete_thickness < required_depth:
            raw = required_depth - total_concrete_thickness
            total_viol += raw * 50.0
            record_constraint(
                "anchor_cage_embedment_depth",
                raw,
                weight=50.0,
                hard=True,
                utilization=required_depth / max(total_concrete_thickness, 1e-9),
                value=total_concrete_thickness,
                limit=required_depth,
                unit="m",
            )

        min_r_col = (Do / 2.0) + 0.6
        if geo['r_col'] < min_r_col:
            raw = min_r_col - geo['r_col']
            total_viol += raw * 50.0
            record_constraint(
                "pedestal_anchor_clearance",
                raw,
                weight=50.0,
                hard=True,
                utilization=min_r_col / max(geo['r_col'], 1e-9),
                value=geo['r_col'],
                limit=min_r_col,
                unit="m",
            )

        R_anchor_outer = Do / 2.0
        r_in = min([ring['r'] for ring in geo['rings']]) if geo['rings'] else 0.0
        pile_inner_edge = r_in - (geo['D_pile'] / 2.0)
        min_safe_clearance = 0.6
        required_inner_edge = R_anchor_outer + min_safe_clearance
        if pile_inner_edge < required_inner_edge:
            raw = required_inner_edge - pile_inner_edge
            total_viol += raw * 50.0
            record_constraint(
                "inner_pile_anchor_clearance",
                raw,
                weight=50.0,
                hard=True,
                utilization=required_inner_edge / max(pile_inner_edge, 1e-9),
                value=pile_inner_edge,
                limit=required_inner_edge,
                unit="m",
            )

        # ====== 微观三维钢筋排布“穿模”惩罚 ======
        PCD_R_out = ach.get('PCD_out_m', 4.8) / 2.0
        PCD_R_in = ach.get('PCD_in_m', 4.8) / 2.0
        n_bolts = ach.get('N_per_ring', 80)
        d_bolt = ach.get('d_sleeve_mm', 65) / 1000.0
        r_void = anchor_cfg['CONSTRAINTS']['R_VOID_M']

        # 1. 顶板 n4 拥挤校验
        top_pen = max(
            calculate_crowding_penalty(rb['n4'], rb['d4'], PCD_R_in, n_bolts, d_bolt, r_void),
            calculate_crowding_penalty(rb['n4'], rb['d4'], PCD_R_out, n_bolts, d_bolt, r_void),
        )
        # 2. 底板 n1/n2 捆绑避让拥挤校验 (取直径最大者)
        d_max_bot = max(rb['d1'], rb.get('d2', rb['d1']))
        bot_pen = max(
            calculate_crowding_penalty(rb['n1'], d_max_bot, PCD_R_in, n_bolts, d_bolt, r_void),
            calculate_crowding_penalty(rb['n1'], d_max_bot, PCD_R_out, n_bolts, d_bolt, r_void),
        )

        # 叠加进入结构违规度
        total_viol += top_pen + bot_pen
        record_constraint(
            "top_rebar_crowding_penalty",
            top_pen,
            hard=True,
        )
        record_constraint(
            "bottom_rebar_crowding_penalty",
            bot_pen,
            hard=True,
        )
        # =================================================

        load_extreme = cfg.CONFIG['load_cases']['Extreme']
        if not ALLOW_TENSION:
            pf_check = model.IntegratedPileFoundation(params, load_extreme, cfg.SOIL_LAYERS, (1.35, 1.5))
            _, forces = pf_check.get_individual_pile_forces()
            min_force = min(forces) if forces else 0.0
            raw = max(0.0, -min_force) / 100.0
            if forces:
                if raw > 0.0:
                    total_viol += raw
            record_constraint(
                "tension_disallowed_extreme",
                raw,
                hard=True,
                value=max(0.0, -min_force),
                unit="kN",
            )

        if HAS_CHECK_MODULES:
            load_sls = cfg.CONFIG['load_cases']['Normal']
            pf_sls = model.IntegratedPileFoundation(params, load_sls, cfg.SOIL_LAYERS, (1.0, 1.0))
            pile_ana = PileStructuralAnalysis(
                geo['D_pile'], rb['pile_main_n'], rb['pile_main_d'], params['mat'], params['stirrup']
            )
            _, theta_sls = pile_ana.check_settlement_and_tilt(
                load_sls['FZ'], load_sls['MXY'], geo['L_pile'], cfg.SOIL_LAYERS, pf_sls.piles_coords
            )
            MIN_K_THETA = 80.0
            K_theta = (load_sls['MXY'] / theta_sls) / 1e6 if theta_sls > 1e-6 else 1e9
            raw = max(0.0, (MIN_K_THETA - K_theta) / MIN_K_THETA)
            if theta_sls > 1e-6:
                if raw > 0.0:
                    total_viol += raw * 20.0
            record_constraint(
                "rotational_stiffness_minimum",
                raw,
                weight=20.0,
                utilization=MIN_K_THETA / max(K_theta, 1e-9),
                value=K_theta,
                limit=MIN_K_THETA,
            )

        foundation_cost_rmb = assessment["costs"]["foundation_cost_rmb"]
        anchor_cost_rmb = float(anchor_result["cost_rmb"])
        total_cost_rmb = foundation_cost_rmb + anchor_cost_rmb
        Cost_Eq_Volume = total_cost_rmb / params["mat"].get("price", 500.0)

    except Exception as e:
        return {
            "objective": 1e10,
            "total_violation": 1e10,
            "constraints": {
                "evaluation_error": {
                    "violation": 1e10,
                    "weight": 1.0,
                    "weighted_violation": 1e10,
                    "hard": True,
                    "message": str(e),
                }
            },
            "status": "error",
        }

    return {
        "objective": float(Cost_Eq_Volume),
        "total_violation": float(total_viol),
        "constraints": constraints,
        "status": "ok",
        "anchor": anchor_result["anchor"],
        "anchor_feasible": bool(anchor_result["feasible"]),
        "foundation_cost_rmb": float(foundation_cost_rmb),
        "anchor_cost_rmb": float(anchor_cost_rmb),
        "total_cost_rmb": float(total_cost_rmb),
    }


def evaluate(ind):
    """兼容原有 DEAP 调用的二元组接口。"""
    details = evaluate_detailed(ind)
    return details["objective"], details["total_violation"]


def compute_epsilon(
    gen,
    max_gen,
    epsilon0=0.10,
    power=2.0,
    decay_fraction=1.0,
):
    """计算当前代允许的归一化约束违反阈值。"""
    decay_generations = max(1.0, float(max_gen) * float(decay_fraction))
    progress = min(1.0, max(0.0, float(gen) / decay_generations))
    return max(0.0, float(epsilon0) * (1.0 - progress) ** float(power))


def evaluate_with_epsilon(ind, epsilon):
    """
    自适应 epsilon 专用评估。

    返回目标值、原始总违反度、最大约束违反度，以及超过 epsilon 的最大超额。
    """
    details = evaluate_detailed(ind)
    max_violation = 0.0
    max_excess = 0.0

    for constraint in details["constraints"].values():
        utilization = constraint.get("utilization")
        if utilization is not None:
            normalized = max(0.0, float(utilization) - 1.0)
        else:
            normalized = max(0.0, float(constraint.get("violation", 0.0)))

        max_violation = max(max_violation, normalized)
        max_excess = max(max_excess, max(0.0, normalized - epsilon))

    return (
        float(details["objective"]),
        float(details["total_violation"]),
        float(max_violation),
        float(max_excess),
    )


def adaptive_fitness_value(
    objective,
    total_violation,
    max_violation,
    max_excess,
    epsilon,
    progress,
):
    """将 epsilon 约束违反度转换为与基线兼容的单目标选择值。"""
    if total_violation <= 1e-4 or max_violation <= epsilon:
        return float(objective)

    effective_violation = (
        total_violation if epsilon <= 0.0 else total_violation + max_excess
    )
    dynamic_factor = 20000.0 + 180000.0 * min(1.0, progress) ** 3
    return float(
        objective + 4000.0 + min(effective_violation, 1000.0) * dynamic_factor
    )


def update_feasible_archive(archive, population, limit=30):
    """保存每代严格可行的优质解，确保最终结果不会退回不可行区域。"""
    for ind in population:
        if getattr(ind, "viol", 1e10) <= 1e-4:
            archive.append(toolbox.clone(ind))

    unique = {}
    for ind in archive:
        key = tuple(round(float(value), 8) for value in ind)
        previous = unique.get(key)
        if previous is None or ind.mat_score < previous.mat_score:
            unique[key] = ind

    archive[:] = sorted(unique.values(), key=lambda ind: ind.mat_score)[:limit]


def get_local_search_steps(variable_index):
    """返回单个设计变量的工程化局部搜索步长。"""
    if variable_index in DISCRETE_MAP:
        return (1.0,)
    if variable_index in (0, 1, 2, 3, 4, 5):
        return (10.0, 5.0, 2.0, 1.0)
    if variable_index == 13:
        return (1.0, 0.5)
    if variable_index in (17, 18):
        return (2.0, 1.0)
    return (1.0,)


def strict_feasible_coordinate_search(
    start_ind,
    max_passes=6,
    max_evaluations=1200,
):
    """
    严格可行坐标局部搜索。

    每次只改变一个变量。候选解必须严格可行且目标值更低，否则拒绝。
    """
    current = toolbox.clone(start_ind)
    current_details = evaluate_detailed(current)
    current_cost = float(current_details["objective"])
    current_violation = float(current_details["total_violation"])

    if current_violation > 1e-4:
        return current, [], 0

    history = []
    evaluations = 0

    for pass_index in range(1, int(max_passes) + 1):
        pass_improved = False

        for variable_index in range(len(VAR_RANGES)):
            if FIXED_PILE_LENGTH is not None and variable_index == 13:
                continue

            best_candidate = None
            best_cost = current_cost
            best_details = None
            low, up = VAR_RANGES[variable_index]

            for step in get_local_search_steps(variable_index):
                for signed_step in (-step, step):
                    if evaluations >= int(max_evaluations):
                        break

                    candidate = toolbox.clone(current)
                    candidate_value = float(candidate[variable_index]) + signed_step
                    candidate_value = max(low, min(up, candidate_value))
                    if variable_index in DISCRETE_MAP:
                        candidate_value = float(int(round(candidate_value)))

                    if abs(candidate_value - float(candidate[variable_index])) < 1e-12:
                        continue

                    candidate[variable_index] = candidate_value
                    repair_individual(candidate)
                    details = evaluate_detailed(candidate)
                    evaluations += 1

                    candidate_violation = float(details["total_violation"])
                    candidate_cost = float(details["objective"])
                    if candidate_violation > 1e-4:
                        continue
                    if candidate_cost < best_cost - 1e-7:
                        best_candidate = candidate
                        best_cost = candidate_cost
                        best_details = details

                if evaluations >= int(max_evaluations):
                    break

            if best_candidate is not None:
                history.append({
                    "pass": pass_index,
                    "variable_index": variable_index,
                    "before_cost": current_cost,
                    "after_cost": best_cost,
                    "improvement": current_cost - best_cost,
                    "evaluations": evaluations,
                })
                current = best_candidate
                current_cost = best_cost
                current_details = best_details
                pass_improved = True

            if evaluations >= int(max_evaluations):
                break

        if not pass_improved or evaluations >= int(max_evaluations):
            break

    return current, history, evaluations


# ================= 4. 可视化绘图 =================
def plot_results(best_params, piles_coords, pile_forces, hist_fit, hist_vc, hist_ws):
    global ALLOW_TENSION
    plt.figure(figsize=(10, 6))

    valid_fits = [v for v in hist_fit if v < 50000]

    if not valid_fits:
        print("\n[!] 警告: 寻优全程未找到完全满足规范的合法解，收敛曲线将展示带惩罚的原始得分。")
        valid_fits = hist_fit
        start_gen = 0
        x_axis = range(start_gen, len(hist_fit))
    else:
        start_gen = len(hist_fit) - len(valid_fits)
        x_axis = range(start_gen, len(hist_fit))

    plt.plot(x_axis, valid_fits, color='#e74c3c', linewidth=2.5, label='最佳适应度 (Best Fitness)')
    plt.fill_between(x_axis, valid_fits, min(valid_fits) * 0.95, color='#e74c3c', alpha=0.1)

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
    ax1.plot(hist_fit, color='blue', linewidth=2)
    ax1.set_title('最佳适应度趋势')
    ax1.set_xlabel('迭代代数', fontsize=10)
    ax1.set_ylabel('适应度得分', fontsize=10)

    if valid_fits and np.max(hist_fit) > 5000:
        y_max = max(valid_fits) * 1.1
        y_min = min(valid_fits) * 0.9
        ax1.set_ylim(y_min, y_max)
    else:
        formatter = ticker.ScalarFormatter(useMathText=True)
        formatter.set_scientific(True)
        formatter.set_powerlimits((-1, 1))
        ax1.yaxis.set_major_formatter(formatter)

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


# ================= 4.5 桩位2D排布图 =================
def plot_pile_layout(best_params, piles_coords, pile_forces):
    geo = best_params['geo']
    R_cap = geo['R_cap']
    coords = np.array(piles_coords)
    forces = np.array(pile_forces)

    fig, ax = plt.subplots(figsize=(8, 8))

    # 承台边缘
    cap_circle = plt.Circle((0, 0), R_cap, fill=False, color='black',
                            linewidth=2, linestyle='--',
                            label=f'承台边缘 R={R_cap:.1f} m')
    ax.add_patch(cap_circle)

    # 桩圈辅助线
    for ring in geo.get('rings', []):
        rc = plt.Circle((0, 0), ring['r'], fill=False, color='gray',
                        linewidth=1, linestyle=':', alpha=0.6)
        ax.add_patch(rc)

    # 桩位散点 (颜色表示轴力)
    sc = ax.scatter(coords[:, 0], coords[:, 1], c=forces, cmap='coolwarm',
                    s=300, edgecolors='black', linewidths=1, zorder=5)
    plt.colorbar(sc, ax=ax, label='桩轴力 (kN)', shrink=0.8)

    # 标注轴力值
    for (x, y), f in zip(coords, forces):
        ax.annotate(f'{f:.0f}', (x, y), fontsize=7, ha='center', va='center',
                    color='black', fontweight='bold', zorder=6)

    ax.set_xlabel('X (m)', fontsize=12)
    ax.set_ylabel('Y (m)', fontsize=12)
    ax.set_title(f'桩位排布图 (共{len(coords)}根桩, 桩径 D={geo["D_pile"]:.1f} m)',
                 fontsize=14, fontweight='bold')
    ax.set_aspect('equal')
    ax.grid(True, linestyle='--', alpha=0.3)
    ax.legend(loc='upper right')

    lim = R_cap * 1.15
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)

    plt.tight_layout()
    plt.savefig('pile_layout.png', dpi=300, bbox_inches='tight')
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
    for _i, _ring in enumerate(geo.get('rings', []), 1):
        print(f"  [桩圈{_i}] {_ring['n']} 根, 半径 r={_ring['r']:.1f} m")
    print(f"  [真实混凝土总量] {V_conc:.1f} m³")
    print(f"  [真实钢筋总重量] {Total_Steel_kg:.1f} kg")
    print(f"  [优化等效体积] {best_cost:.1f} m³ (综合成本量纲)")

    costs = model.calculate_design_cost(best_params)
    print(f"  [基础材料成本] {costs['foundation_cost_rmb']:.2f} RMB")
    print(f"  [锚栓笼成本] {costs['anchor_cost_rmb']:.2f} RMB")
    print(f"  [总材料成本] {costs['total_cost_rmb']:.2f} RMB")
    print("  [研究参数] 临时双圈基线、0.3M疲劳荷载、35d、50MPa及构造限值待施工图校准。")

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



# ================= Improvement 2: adaptive crossover/mutation probs =================

# ================= Improvement 3: SA local search =================
def is_sa_better(cost_a, viol_a, cost_b, viol_b):
    """Lexicographic comparison: feasible always beats infeasible."""
    if viol_a < 1e-4 and viol_b < 1e-4:
        return cost_a < cost_b
    if viol_a < 1e-4: return True
    if viol_b < 1e-4: return False
    return viol_a < viol_b


def sa_local_search(ind, evaluate_fn, T, steps=6):
    """Simulated annealing on one individual."""
    cur = creator.Individual(ind[:])
    del cur.fitness.values
    cc, cv = evaluate_fn(cur)
    bc, bv, best = cc, cv, creator.Individual(cur[:])

    for _ in range(steps):
        nb = creator.Individual(cur[:])
        for j in range(len(VAR_RANGES)):
            if random.random() < 0.35:
                lo, up = VAR_RANGES[j]
                if j in DISCRETE_MAP:
                    opts = DISCRETE_MAP[j]
                    try: ci = opts.index(int(round(nb[j])))
                    except: ci = len(opts) // 2
                    ci = max(0, min(len(opts)-1, ci + random.choice([-1, 0, 1])))
                    nb[j] = float(ci)
                else:
                    nb[j] = max(lo, min(up, nb[j] + random.gauss(0, 0.015*(up-lo))))
        del nb.fitness.values
        nc, nv = evaluate_fn(nb)

        if is_sa_better(nc, nv, cc, cv):
            cur, cc, cv = creator.Individual(nb[:]), nc, nv
            if is_sa_better(cc, cv, bc, bv):
                best, bc, bv = creator.Individual(nb[:]), nc, nv
        else:
            de_abs = nc - cc
            if random.random() < float(np.exp(np.clip(-de_abs / (T + 1e-9), -50, 50))):
                cur, cc, cv = creator.Individual(nb[:]), nc, nv
    return best, bc, bv


def compute_adaptive_probs(pop):
    """
    Returns (Pc, Pm_base) based on Srinivas & Patnaik (1994).
    Pc is per-generation baseline for crossovers.
    The key correction: when population converges (diversity low),
    we increase Pc to encourage exploration via recombination.
    """
    f_vals = []
    for ind in pop:
        if ind.fitness.valid and hasattr(ind, "mat_score"):
            viol = getattr(ind, "viol", 1e10)
            if viol < 1e-4:
                f_vals.append(ind.mat_score)
    if len(f_vals) < 2:
        return 0.7, 0.3
    f_vals = np.array(f_vals)
    f_max, f_min, f_avg = f_vals.max(), f_vals.min(), f_vals.mean()
    delta = f_max - f_min + 1e-9
    # Diversity ratio: 1.0 = diverse, 0.0 = converged
    diversity = (f_max - f_avg) / delta
    # When converged (diversity near 0): Pc high (0.85) to escape
    # When diverse (diversity near 1): Pc moderate (0.5)
    Pc = 0.85 - 0.35 * diversity
    Pc = max(0.5, min(0.9, Pc))
    # Pm_base: moderate base rate, per-individual adjustment in main loop
    Pm_base = 0.05 + 0.20 * (1.0 - diversity)
    return Pc, Pm_base


def summarize_population(pop, generation, elapsed_sec, ga_evaluations, epsilon=None):
    """生成一代种群的可比较统计信息，供批量实验记录。"""
    valid = [
        ind for ind in pop
        if hasattr(ind, "mat_score") and hasattr(ind, "viol")
    ]
    feasible = [ind for ind in valid if ind.viol <= 1e-4]

    best_any = min(valid, key=lambda ind: ind.mat_score) if valid else None
    best_feasible = min(feasible, key=lambda ind: ind.mat_score) if feasible else None
    violations = [float(ind.viol) for ind in valid]

    return {
        "generation": int(generation),
        "elapsed_sec": float(elapsed_sec),
        "ga_evaluations": int(ga_evaluations),
        "best_objective": float(best_any.mat_score) if best_any is not None else None,
        "best_feasible_objective": (
            float(best_feasible.mat_score) if best_feasible is not None else None
        ),
        "best_violation": min(violations) if violations else None,
        "mean_violation": float(np.mean(violations)) if violations else None,
        "feasible_count": len(feasible),
        "feasible_ratio": float(len(feasible) / len(valid)) if valid else 0.0,
        "epsilon": None if epsilon is None else float(epsilon),
    }


def run_optimization(
    allow_tension=True,
    fixed_pile_length=None,
    seed=None,
    pop_size=None,
    n_gen=None,
    cores=16,
    verbose=True,
    save_artifacts=True,
    use_adaptive_epsilon=False,
    epsilon0=0.10,
    epsilon_power=2.0,
    epsilon_decay_fraction=1.0,
    use_mixed_variables=False,
    mixed_crossover=True,
    mixed_mutation_indpb=0.15,
    mixed_mutation_step=2,
    use_feasible_local_search=False,
    local_search_passes=6,
    local_search_max_evaluations=1200,
    anchor_mode="fixed",
    fatigue_moment_ratio=None,
    anchor_pcd_samples=None,
    anchor_full_pcd_grid=False,
):
    """
    可重复调用的优化入口。

    原有 main() 保留交互方式；批量实验直接调用本函数，避免 input() 阻塞。
    """
    global CURRENT_ITER, ALLOW_TENSION, FIXED_PILE_LENGTH, ANCHOR_MODE, ANCHOR_FATIGUE_MOMENT_RATIO
    global ANCHOR_PCD_SAMPLES, ANCHOR_FULL_PCD_GRID

    if seed is not None:
        random.seed(seed)
        np.random.seed(int(seed))

    ALLOW_TENSION = bool(allow_tension)
    FIXED_PILE_LENGTH = fixed_pile_length
    if anchor_mode not in {"fixed", "nested_2", "nested"}:
        raise ValueError("anchor_mode must be 'fixed', 'nested_2' or 'nested'")
    ANCHOR_MODE = anchor_mode
    if fatigue_moment_ratio is None:
        fatigue_moment_ratio = cfg.CONFIG["ANCHOR_OPT"].get("FATIGUE_MOMENT_RATIO", 0.10)
    if not 0.0 < float(fatigue_moment_ratio) <= 1.0:
        raise ValueError("fatigue_moment_ratio must be in (0, 1]")
    ANCHOR_FATIGUE_MOMENT_RATIO = float(fatigue_moment_ratio)
    cfg.CONFIG["ANCHOR_OPT"]["FATIGUE_MOMENT_RATIO"] = ANCHOR_FATIGUE_MOMENT_RATIO
    if anchor_pcd_samples is None:
        anchor_pcd_samples = cfg.CONFIG["ANCHOR_OPT"]["SOLVER"].get("PCD_PAIRS_PER_COMBO", 8)
    if int(anchor_pcd_samples) < 1:
        raise ValueError("anchor_pcd_samples must be positive")
    ANCHOR_PCD_SAMPLES = int(anchor_pcd_samples)
    ANCHOR_FULL_PCD_GRID = bool(anchor_full_pcd_grid)
    cfg.CONFIG["ANCHOR_OPT"]["SOLVER"]["PCD_PAIRS_PER_COMBO"] = ANCHOR_PCD_SAMPLES
    cfg.CONFIG["ANCHOR_OPT"]["SOLVER"]["FULL_PCD_GRID"] = ANCHOR_FULL_PCD_GRID
    reset_anchor_solver_stats()
    start_time = time.perf_counter()
    history = []
    ga_evaluations = 0

    if verbose:
        print("=" * 60)
        print("   风机基础智能优化程序 (等效造价 + 允许打捆穿刺避让版)")
        print("=" * 60)
        if seed is not None:
            print(f"=> 随机种子: {seed}")

    ga_params = dict(cfg.CONFIG['GA_PARAMS'])
    if pop_size is not None:
        ga_params['POP_SIZE'] = int(pop_size)
    if n_gen is not None:
        ga_params['N_GEN'] = int(n_gen)
    lows = [r[0] for r in VAR_RANGES]
    ups_raw = [r[1] for r in VAR_RANGES]
    ups_safe = [u + 1e-9 if u <= l else u for l, u in zip(lows, ups_raw)]

    if use_mixed_variables:
        if mixed_crossover:
            toolbox.register("mate", mate_mixed, eta=ga_params.get('ETA', 10.0))
        else:
            toolbox.register("mate", tools.cxTwoPoint)
        toolbox.register(
            "mutate",
            mutate_mixed,
            eta=ga_params.get('ETA', 10.0),
            indpb=float(mixed_mutation_indpb),
            discrete_step=int(mixed_mutation_step),
        )
    else:
        toolbox.register("mate", tools.cxTwoPoint)
        toolbox.register(
            "mutate",
            tools.mutPolynomialBounded,
            eta=ga_params.get('ETA', 12.0),
            low=lows,
            up=ups_safe,
            indpb=0.15,
        )
    toolbox.register("select", tools.selTournament, tournsize=ga_params.get('TOURN_SIZE', 3))

    total_cores = os.cpu_count() or 1
    use_cores = max(1, min(int(cores), total_cores))
    if verbose:
        print(f"\n--- 启动优化引擎 (核心数: {use_cores}) ---")

    if use_cores == 1:
        init_worker(ALLOW_TENSION, FIXED_PILE_LENGTH, ANCHOR_MODE,
                    ANCHOR_FATIGUE_MOMENT_RATIO, ANCHOR_PCD_SAMPLES,
                    ANCHOR_FULL_PCD_GRID)
        pool = None
        parallel_map = map
        toolbox.register("map", map)
    else:
        pool = multiprocessing.Pool(processes=use_cores, initializer=init_worker,
                                    initargs=(ALLOW_TENSION, FIXED_PILE_LENGTH, ANCHOR_MODE,
                                              ANCHOR_FATIGUE_MOMENT_RATIO,
                                              ANCHOR_PCD_SAMPLES,
                                              ANCHOR_FULL_PCD_GRID))
        parallel_map = pool.map
        toolbox.register("map", pool.map)

    # ====== 改进1: 可行性控制初始化 ======
    if USE_FEASIBLE_INIT:
        if verbose:
            print("\n>>> [改进1] 正在执行可行性控制初始化...")
        pop = feasible_init_population(ga_params['POP_SIZE'])
    else:
        pop = toolbox.population(ga_params['POP_SIZE'])
    # =====================================
    hof = tools.HallOfFame(1)

    history_fitness = []
    history_vc = []
    history_ws = []
    ngen = ga_params['N_GEN']
    initial_epsilon = (
        compute_epsilon(
            0,
            ngen,
            epsilon0,
            epsilon_power,
            epsilon_decay_fraction,
        )
        if use_adaptive_epsilon else None
    )
    feasible_archive = []

    # ================= 初始代评估 =================
    invalid_ind = [ind for ind in pop if not ind.fitness.valid]
    if use_adaptive_epsilon:
        fitnesses = parallel_map(
            partial(evaluate_with_epsilon, epsilon=initial_epsilon),
            invalid_ind,
        )
    else:
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
    ga_evaluations += len(invalid_ind)

    for ind, fit_tuple in zip(invalid_ind, fitnesses):
        if use_adaptive_epsilon:
            ind.mat_score, ind.viol, ind.hard_violation, ind.soft_violation = fit_tuple
            ind.epsilon_violation = ind.soft_violation
            ind.fitness.values = (
                adaptive_fitness_value(
                    ind.mat_score,
                    ind.viol,
                    ind.hard_violation,
                    ind.soft_violation,
                    initial_epsilon,
                    progress=0.0,
                ),
            )
        else:
            ind.mat_score, ind.viol = fit_tuple
            if ind.viol > 1e-4:
                safe_viol = min(ind.viol, 1000.0)
                ind.fitness.values = (ind.mat_score + 4000.0 + safe_viol * 20000.0,)
            else:
                ind.fitness.values = (ind.mat_score,)

    update_feasible_archive(feasible_archive, pop)
    hof.update(pop)
    history.append(
        summarize_population(
            pop,
            0,
            time.perf_counter() - start_time,
            ga_evaluations,
            epsilon=initial_epsilon,
        )
    )

    pop_size = ga_params['POP_SIZE']

    elite_ratio = min(0.015, ga_params.get('ELITE_RATIO', 0.05))
    n_elite = max(2, int(pop_size * elite_ratio))

    n_mid = int(pop_size * 0.7)
    n_bad = pop_size - n_elite - n_mid

    pbar = tqdm(
        range(1, ngen + 1),
        desc="🚀 并行优化中",
        unit="gen",
        ncols=120,
        disable=not verbose,
    )
    for gen in pbar:
        CURRENT_ITER = gen

        progress = gen / ngen
        current_epsilon = (
            compute_epsilon(
                gen,
                ngen,
                epsilon0,
                epsilon_power,
                epsilon_decay_fraction,
            )
            if use_adaptive_epsilon else None
        )
        sorted_pop = tools.selBest(pop, len(pop))
        if (
            use_adaptive_epsilon
            and current_epsilon is not None
            and current_epsilon > 0.0
            and feasible_archive
        ):
            n_archive_elite = min(len(feasible_archive), max(1, n_elite // 2))
            archive_elites = [
                toolbox.clone(ind) for ind in feasible_archive[:n_archive_elite]
            ]
            archive_keys = {
                tuple(round(float(value), 8) for value in ind)
                for ind in archive_elites
            }
            other_elites = [
                toolbox.clone(ind)
                for ind in sorted_pop
                if tuple(round(float(value), 8) for value in ind) not in archive_keys
            ][:max(0, n_elite - len(archive_elites))]
            elites = archive_elites + other_elites
        else:
            elites = [toolbox.clone(ind) for ind in sorted_pop[:n_elite]]
        mid_parents = toolbox.select(sorted_pop[:(n_elite + n_mid)], n_mid)
        mids = [toolbox.clone(ind) for ind in mid_parents]

        # ====== Improvement 2: adaptive crossover/mutation probs ======
        if USE_ADAPTIVE:
            Pc, Pm_base = compute_adaptive_probs(pop)
        else:
            Pc = ga_params['CX_PROB']
            Pm_base = ga_params['MUT_PROB']
        # =============================================================

        for i in range(1, len(mids), 2):
            # Per-pair adaptive crossover: good parents -> preserve (lower Pc)
            if USE_ADAPTIVE:
                f_vals3 = np.array([ind.fitness.values[0] for ind in sorted_pop if ind.fitness.valid])
                if len(f_vals3) > 2:
                    f_max3, f_avg3 = f_vals3.max(), f_vals3.mean()
                    f_pair = max(mids[i-1].fitness.values[0] if mids[i-1].fitness.valid else f_max3,
                                 mids[i].fitness.values[0] if mids[i].fitness.valid else f_max3)
                    Pc_pair = 0.9 if f_pair < f_avg3 else max(0.5, 0.9 - 0.4 * (f_pair - f_avg3) / (f_max3 - f_avg3 + 1e-9))
                else:
                    Pc_pair = Pc
            else:
                Pc_pair = Pc

            if random.random() < Pc_pair:
                toolbox.mate(mids[i - 1], mids[i])
                del mids[i - 1].fitness.values
                del mids[i].fitness.values

        for i in range(len(mids)):
            if USE_ADAPTIVE:
                f_vals2 = np.array([ind.fitness.values[0] for ind in sorted_pop if ind.fitness.valid])
                if len(f_vals2) > 1:
                    fM, fm = f_vals2.max(), f_vals2.min()
                    fi = mids[i].fitness.values[0] if mids[i].fitness.valid else fM
                    Pm_i = max(0.05, min(0.4, 0.05 + 0.35 * (fi - fm) / (fM - fm + 1e-9)))
                else:
                    Pm_i = Pm_base
            else:
                Pm_i = Pm_base
            if random.random() < Pm_i:
                toolbox.mutate(mids[i])
                del mids[i].fitness.values

        bads = [creator.Individual(ind) for ind in lhs_sample(len(VAR_RANGES), n_bad)]
        if (
            use_adaptive_epsilon
            and current_epsilon is not None
            and current_epsilon > 0.0
            and feasible_archive
            and n_bad > 0
        ):
            n_migrate = min(len(feasible_archive), max(1, n_bad // 2))
            migrated = [toolbox.clone(ind) for ind in feasible_archive[:n_migrate]]
            bads = migrated + bads[:max(0, n_bad - len(migrated))]
        pop[:] = elites + mids + bads

        invalid_ind = [ind for ind in pop if not ind.fitness.valid]
        if use_adaptive_epsilon:
            fitnesses = parallel_map(
                partial(evaluate_with_epsilon, epsilon=current_epsilon),
                invalid_ind,
            )
        else:
            fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
        ga_evaluations += len(invalid_ind)

        for ind, fit_tuple in zip(invalid_ind, fitnesses):
            if use_adaptive_epsilon:
                ind.mat_score, ind.viol, ind.hard_violation, ind.soft_violation = fit_tuple
                ind.epsilon_violation = ind.soft_violation
            else:
                ind.mat_score, ind.viol = fit_tuple

        base_penalty = 4000.0
        dynamic_factor = 20000.0 + 180000.0 * (progress ** 3)

        for ind in pop:
            if not hasattr(ind, 'mat_score'):
                ind.mat_score, ind.viol = 1e10, 1e10

            if use_adaptive_epsilon:
                if not hasattr(ind, "hard_violation"):
                    ind.hard_violation = 0.0
                if not hasattr(ind, "soft_violation"):
                    ind.soft_violation = 0.0
                ind.fitness.values = (
                    adaptive_fitness_value(
                        ind.mat_score,
                        ind.viol,
                        ind.hard_violation,
                        ind.soft_violation,
                        current_epsilon,
                        progress=progress,
                    ),
                )
            else:
                if ind.viol > 1e-4:
                    safe_viol = min(ind.viol, 1000.0)
                    final_fit = ind.mat_score + base_penalty + safe_viol * dynamic_factor
                    ind.fitness.values = (final_fit,)
                else:
                    ind.fitness.values = (ind.mat_score,)

        hof.clear()

        # ====== Improvement 3: SA local search on elites ======
        if USE_SA:
            T0 = 80.0  # absolute temp scale
            T = max(1e-6, T0 * (0.94 ** gen))
            for i in range(min(len(elites), 50)):
                improved, bc2, bv2 = sa_local_search(elites[i], evaluate, T, steps=6)
                improved.mat_score = bc2
                improved.viol = bv2
                sv2 = min(bv2, 1000.0)
                improved.fitness.values = (bc2 if bv2 < 1e-4 else bc2 + 4000.0 + sv2 * dynamic_factor,)
                if improved.fitness.values[0] < elites[i].fitness.values[0]:
                    elites[i] = improved
        # ========================================================

        hof.clear()
        hof.update(pop)
        update_feasible_archive(feasible_archive, pop)

        best_ind = tools.selBest(pop, 1)[0]
        params = decode_gene(best_ind)
        params["anchor"] = evaluate_detailed(best_ind)["anchor"]

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

        current_gen_best_fitness = best_ind.fitness.values[0]

        if len(history_fitness) == 0:
            history_fitness.append(current_gen_best_fitness)
        else:
            history_fitness.append(min(history_fitness[-1], current_gen_best_fitness))

        history_vc.append(Vc + pf_basic.N_total * math.pi * (params["geo"]["D_pile"] / 2) ** 2 * params["geo"]["L_pile"])
        history_ws.append(Total_Steel)
        history.append(
            summarize_population(
                pop,
                gen,
                time.perf_counter() - start_time,
                ga_evaluations,
                epsilon=current_epsilon,
            )
        )

        display_cost = best_ind.mat_score if best_ind.viol < 1e-4 else current_gen_best_fitness
        pbar.set_postfix({"Best_Score": f"{display_cost:.1f}"})

    if pool is not None:
        pool.close()
        pool.join()

    if save_artifacts:
        active_plot_results = plot_results
        active_plot_pile_layout = plot_pile_layout
        active_print_detailed_report = print_detailed_report
        active_export_abaqus_json = export_abaqus_json
    else:
        def _skip_artifact(*_args, **_kwargs):
            return None

        active_plot_results = _skip_artifact
        active_plot_pile_layout = _skip_artifact
        active_print_detailed_report = _skip_artifact
        active_export_abaqus_json = _skip_artifact

    selected_ind = (
        min(feasible_archive, key=lambda ind: ind.mat_score)
        if feasible_archive
        else (hof[0] if len(hof) > 0 else None)
    )
    has_valid_solution = selected_ind is not None and not np.any(np.isnan(selected_ind))
    result = {
        "seed": seed,
        "status": "no_valid_solution",
        "best_fitness": None,
        "best_objective": None,
        "best_violation": None,
        "best_feasible": False,
        "best_params": None,
        "runtime_sec": float(time.perf_counter() - start_time),
        "ga_evaluations": int(ga_evaluations),
        "history": history,
        "adaptive_epsilon": bool(use_adaptive_epsilon),
        "mixed_variables": bool(use_mixed_variables),
        "mixed_crossover": bool(mixed_crossover),
        "mixed_mutation_indpb": float(mixed_mutation_indpb),
        "mixed_mutation_step": int(mixed_mutation_step),
        "feasible_local_search": bool(use_feasible_local_search),
        "local_search_passes": int(local_search_passes),
        "local_search_max_evaluations": int(local_search_max_evaluations),
        "epsilon0": float(epsilon0),
        "epsilon_power": float(epsilon_power),
        "epsilon_decay_fraction": float(epsilon_decay_fraction),
        "final_epsilon": (
            float(
                compute_epsilon(
                    ngen,
                    ngen,
                    epsilon0,
                    epsilon_power,
                    epsilon_decay_fraction,
                )
            )
            if use_adaptive_epsilon else None
        ),
    }

    if has_valid_solution:
        best_ind = selected_ind
        local_search_history = []
        local_search_evaluations = 0
        if use_feasible_local_search:
            best_ind, local_search_history, local_search_evaluations = (
                strict_feasible_coordinate_search(
                    best_ind,
                    max_passes=local_search_passes,
                    max_evaluations=local_search_max_evaluations,
                )
            )
            ga_evaluations += local_search_evaluations
            best_ind.fitness.values = (evaluate_detailed(best_ind)["objective"],)
        best_params = decode_gene(best_ind)
        best_details = evaluate_detailed(best_ind)
        best_params["anchor"] = best_details["anchor"]
        best_params["total_cost"] = best_details["total_cost_rmb"]
        best_quantities = model.calculate_design_cost(best_params)
        load_uls = cfg.CONFIG['load_cases']["Extreme"]
        pf_uls = model.IntegratedPileFoundation(best_params, load_uls, cfg.SOIL_LAYERS, (1.35, 1.5))
        coords, forces = pf_uls.get_individual_pile_forces()

        active_plot_results(best_params, coords, forces, history_fitness, history_vc, history_ws)
        active_plot_pile_layout(best_params, coords, forces)
        active_print_detailed_report(best_params, best_quantities["cap_concrete_m3"] + best_quantities["pile_concrete_m3"], best_quantities["steel_kg"], best_details["objective"])

        active_export_abaqus_json(best_params, "abaqus_pile_model.json")

        if HAS_LAYOUT_ENGINE and save_artifacts:
            print("\n【4. 启动微观钢筋排布与避让引擎 (底板)】")
            anchor = best_params["anchor"]
            ach_config = {"PCD_in": anchor["PCD_in_m"], "PCD_out": anchor["PCD_out_m"],
                          "PCD": anchor["PCD_out_m"], "N_per_ring": anchor["N_per_ring"],
                          "N_BOLTS": anchor["N_total"], "BOLT_D": anchor["Bolt_D_mm"],
                          "d_sleeve": anchor["d_sleeve_mm"], "D_outer": anchor["D_outer_m"]}

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
            bolt_d_mm = float(ach_config.get('BOLT_D', 48.0))
            sleeve_d_mm = float(ach_config.get('d_sleeve', bolt_d_mm + 17.0))
            pcd_in_m = float(ach_config.get('PCD_in', ach_config.get('PCD', 4.8)))
            pcd_out_m = float(
                ach_config.get(
                    'PCD_out',
                    max(pcd_in_m + 0.4, ach_config.get('D_outer', 5.1) - sleeve_d_mm / 1000.0),
                )
            )
            n_per_ring = int(
                round(
                    ach_config.get(
                        'N_per_ring',
                        ach_config.get('N_BOLTS', 180) / 2.0,
                    )
                )
            )

            # 组装满足多层级径向配筋 + 构造环向筋要求的标准交底字典
            rebar_json_data = {
                "Metadata": {
                    "Project": "风机基础多目标宏观寻优输出报告",
                    "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                },
                "Boundary_Conditions": {
                    "Limit_M_kNm": float(cfg.CONFIG["load_cases"]["Extreme"]["MXY"]),
                    "Limit_N_kN": float(cfg.CONFIG["load_cases"]["Extreme"]["FZ"])
                },
                "Foundation_Geometry": {
                    "Radius_m": float(R_cap),
                    "Thickness_m": float(geo_opt.get('H_cap', geo_opt.get('H', 2.8))),
                    "Pedestal_Radius_m": float(r_col),
                    "Pedestal_Top_Radius_m": float(geo_opt.get('r_top', r_col + 0.2))
                },
                "Bottom_Plate_Rebar": {
                    "Radial": {
                        "Through_Bars": {  # ① 通长筋
                            "Count": int(n1),
                            "Diameter_mm": float(d1) * 1000.0
                        },
                        "Insert_Bars": {  # ② 外插筋
                            "Count": int(n3),
                            "Diameter_mm": float(d3) * 1000.0
                        },
                        "Reinforced_Bars": {  # ③ 双层加强筋
                            "Count": int(n2),
                            "Diameter_mm": float(d2) * 1000.0
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
                            "Diameter_mm": float(d4) * 1000.0
                        },
                        "Insert_Bars": {  # ② 外插筋
                            "Count": int(n5),
                            "Diameter_mm": float(d5) * 1000.0
                        },
                        "Outer_Equivalent_Spacing_mm": float(geo_opt.get('outer_spacing_top', 294.0))
                    },
                    "Circular_Constructive": {  # 环向筋仅作构造
                        "Spec": "Φ20@100mm"
                    }
                },
                "Anchor_Cage": {
                    "PCD_in_m": pcd_in_m,
                    "PCD_out_m": pcd_out_m,
                    "N_per_ring": n_per_ring,
                    "N_total": int(2 * n_per_ring),
                    "Bolt_Type": f"M{int(round(bolt_d_mm))}",
                    "Bolt_D_mm": bolt_d_mm,
                    "d_sleeve_mm": sleeve_d_mm
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

        result.update({
            "status": "feasible" if best_ind.viol <= 1e-4 else "infeasible_best",
            "best_fitness": float(best_ind.fitness.values[0]),
            "best_objective": float(best_ind.mat_score),
            "best_violation": float(best_ind.viol),
            "best_feasible": bool(best_ind.viol <= 1e-4),
            "best_gene": [float(value) for value in best_ind],
            "best_params": best_params,
            "constraint_total_violation": float(best_details["total_violation"]),
            "constraint_details": best_details["constraints"],
            "anchor_mode": ANCHOR_MODE,
            "fatigue_moment_ratio": ANCHOR_FATIGUE_MOMENT_RATIO,
            "anchor_pcd_samples": ANCHOR_PCD_SAMPLES,
            "anchor_full_pcd_grid": ANCHOR_FULL_PCD_GRID,
            "best_anchor": best_details["anchor"],
            "foundation_cost_rmb": best_details["foundation_cost_rmb"],
            "anchor_cost_rmb": best_details["anchor_cost_rmb"],
            "total_cost_rmb": best_details["total_cost_rmb"],
            "anchor_solver_stats_main_process": get_anchor_solver_stats(),
            "cost_breakdown": best_quantities,
            "research_parameters_pending_calibration": cfg.CONFIG["ANCHOR_OPT"]["RESEARCH_PARAMETERS_PENDING_CALIBRATION"],
            "local_search_history": local_search_history,
            "local_search_evaluations": int(local_search_evaluations),
        })
    else:
        if verbose:
            print("\n【错误】未能找到有效解。")

    result["runtime_sec"] = float(time.perf_counter() - start_time)
    return result


def main():
    """保留原有命令行交互入口。"""
    print("=" * 60)
    print("   风机基础智能优化程序 (等效造价 + 允许打捆穿刺避让版)")
    print("=" * 60)

    while True:
        user_input = input(">> 是否允许出现受拉桩？(y/n) [默认:y] : ").strip().lower()
        if user_input in ['y', 'yes', '']:
            allow_tension = True
            print("=> 【已选择】允许受拉模式")
            break
        elif user_input in ['n', 'no']:
            allow_tension = False
            print("=> 【已选择】全抗压模式")
            break
        else:
            print("输入无效，请输入 'y' 或 'n'。")

    while True:
        l_input = input(">> 是否手动锁定桩长？(输入具体数字如 28，直接回车自动优化) : ").strip()
        if l_input == '':
            fixed_pile_length = None
            print("=> 【已选择】算法自动寻优最佳桩长")
            break
        else:
            try:
                fixed_pile_length = float(l_input)
                if fixed_pile_length <= 0:
                    raise ValueError
                print(f"=> 【已选择】桩长被强制锁定为: {fixed_pile_length} m")
                break
            except ValueError:
                print("输入无效。")

    return run_optimization(
        allow_tension=allow_tension,
        fixed_pile_length=fixed_pile_length,
        verbose=True,
        save_artifacts=True,
    )


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
