# -*- coding: utf-8 -*-
"""
文件名: optimize_anchor_cage.py
功能:
  1. 【双圈拓扑升维】内外双圈锚栓阵列协同寻优
  2. 极限承载力(ULS) + 疲劳(FLS) + 台柱三维空间防干涉
  3. 【高级绘图】智能剥离致死惩罚掩蔽，还原真实收敛曲线
  4. 自动化生成顶底板双圈无干涉排布图
"""

import sys
import json
import math
import random
import numpy as np
from deap import base, creator, tools, algorithms
import matplotlib.pyplot as plt

from layout_geometry import evaluate_rebar_groups

try:
    from render_bottom_rebar_layout import BottomRebarLayout
    from render_top_rebar_layout import TopRebarLayout

    HAS_LAYOUT = True
except ImportError:
    HAS_LAYOUT = False
    print("❌ 警告：找不到 render_bottom_rebar_layout.py 或 render_top_rebar_layout.py，将跳过绘图环节！")


# ==========================================
# 1. 深度数据解析
# ==========================================
def load_foundation_data(json_path="Optimized_Rebar_Data.json"):
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        bc = data.get("Boundary_Conditions", {})
        M = float(bc.get("Limit_M_kNm", 185957.0))
        N = float(bc.get("Limit_N_kN", 7918.5))
        delta_M = float(bc.get("Fatigue_Delta_M_kNm", M * 0.3))

        geom = data.get("Foundation_Geometry", {})
        R_cap = float(geom.get("Radius_m", 10.5))
        H_cap = float(geom.get("Thickness_m", 2.8))
        r_col = float(geom.get("Pedestal_Radius_m", 3.0))
        r_top = float(geom.get("Pedestal_Top_Radius_m", r_col + 0.2))
        r_void = 1.5

        bot_radial = data.get("Bottom_Plate_Rebar", {}).get("Radial", {})
        n1 = int(bot_radial.get("Through_Bars", {}).get("Count", 180))
        d1 = float(bot_radial.get("Through_Bars", {}).get("Diameter_mm", 22.0))
        n3 = int(bot_radial.get("Insert_Bars", {}).get("Count", 180))
        d3 = float(bot_radial.get("Insert_Bars", {}).get("Diameter_mm", 22.0))
        n2 = int(bot_radial.get("Reinforced_Bars", {}).get("Count", 180))
        d2 = float(bot_radial.get("Reinforced_Bars", {}).get("Diameter_mm", 32.0))

        top_radial = data.get("Top_Plate_Rebar", {}).get("Radial", {})
        n4 = int(top_radial.get("Through_Bars", {}).get("Count", 110))
        d4 = float(top_radial.get("Through_Bars", {}).get("Diameter_mm", 25.0))
        n5 = int(top_radial.get("Insert_Bars", {}).get("Count", 110))
        d5 = float(top_radial.get("Insert_Bars", {}).get("Diameter_mm", 25.0))

        ach = data.get("Anchor_Cage", {})

        d_bot_mm = d2 * 1000.0 if d2 < 1.0 else d2
        d_top_mm = d4 * 1000.0 if d4 < 1.0 else d4
        max_rebar_dia = max(d_bot_mm, d_top_mm)

        layout_params = {
            'R_cap': R_cap, 'H_cap': H_cap, 'r_void': r_void, 'r_col': r_col, 'r_top': r_top,
            'n1': n1, 'd1': d1, 'n2': n2, 'd2': d2, 'n3': n3, 'd3': d3,
            'n4': n4, 'd4': d4, 'n5': n5, 'd5': d5,
            'PCD_in_m': float(ach.get('PCD_in_m', 4.8)),
            'PCD_out_m': float(ach.get('PCD_out_m', 5.2)),
            'N_per_ring': int(ach.get('N_per_ring', ach.get('N_total', 180) // 2)),
            'Bolt_D_mm': float(ach.get('Bolt_D_mm', 48.0)),
            'd_sleeve_mm': float(ach.get('d_sleeve_mm', 65.0)),
        }
        return M, N, delta_M, max_rebar_dia, layout_params
    except Exception as e:
        print(f"\n❌ 读取 JSON 发生致命异常: {e}")
        sys.exit(1)


# ==========================================
# 2. 材料数据库
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
    d_eff = b['d'] - 0.9382 * b['P']
    Ae = (math.pi / 4.0) * (d_eff ** 2)
    T_allow = (Ae * F_T_DESIGN) / 1000.0
    return {
        'type': f"M{int(b['d'])}", 'd': b['d'], 'Ae': Ae,
        'T_allow': T_allow, 'd_sleeve': b['d_sleeve'], 'unit_cost': b['unit_cost']
    }


creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)
toolbox = base.Toolbox()


# ==========================================
# 3. 双圈架构评估函数
# ==========================================
def evaluate_anchor(
    individual,
    M,
    N_load,
    delta_M,
    max_rebar_dia,
    r_col,
    H_cap,
    layout_params=None,
):
    PCD_A = round(individual[0] / 0.01) * 0.01
    PCD_B = round(individual[1] / 0.01) * 0.01
    PCD_out = max(PCD_A, PCD_B)
    PCD_in = min(PCD_A, PCD_B)

    N_ring = int(round(individual[2] / 4.0) * 4)
    bolt_idx = max(0, min(3, int(round(individual[3]))))

    bolt_specs = get_bolt_specs(bolt_idx)

    # 双圈法兰较宽厚，造价系数取 3.0
    PCD_avg = (PCD_out + PCD_in) / 2.0
    cost_flanges = 3.0 * (math.pi * PCD_avg * FLANGE_COST_PER_M)
    cost_bolts = (2 * N_ring) * bolt_specs['unit_cost']
    total_cost = cost_bolts + cost_flanges

    penalty = 0.0

    # A. 极限拉断校验
    T_max_out = (4 * M * PCD_out) / (N_ring * (PCD_out ** 2 + PCD_in ** 2)) - (N_load / (2 * N_ring))
    if T_max_out > bolt_specs['T_allow']:
        penalty += (T_max_out - bolt_specs['T_allow']) * 50000.0

    # B. 疲劳应力校验
    delta_T_out = (4 * delta_M * PCD_out) / (N_ring * (PCD_out ** 2 + PCD_in ** 2))
    delta_sigma = (delta_T_out * 1000.0) / bolt_specs['Ae']
    if delta_sigma > ALLOWABLE_DELTA_SIGMA:
        penalty += (delta_sigma - ALLOWABLE_DELTA_SIGMA) * 100000.0

    # C1. 钢筋穿插净距 (内外双圈均检查)
    clear_gap_circ_in = (math.pi * PCD_in * 1000.0 / N_ring) - bolt_specs['d_sleeve']
    clear_gap_circ_out = (math.pi * PCD_out * 1000.0 / N_ring) - bolt_specs['d_sleeve']
    clear_gap_circ = min(clear_gap_circ_in, clear_gap_circ_out)
    required_gap_circ = max_rebar_dia + 50.0
    if clear_gap_circ < required_gap_circ:
        penalty += (required_gap_circ - clear_gap_circ) * 100000.0

    if layout_params is not None:
        R_cap = float(layout_params['R_cap'])
        r_void = float(layout_params['r_void'])
        r_col_value = float(layout_params['r_col'])
        groups = [
            {
                'name': 'bottom_through',
                'count': layout_params['n1'],
                'diameter': layout_params['d1'],
                'r_start': r_void,
                'r_end': R_cap,
            },
            {
                'name': 'bottom_reinforced',
                'count': layout_params['n2'],
                'diameter': layout_params['d2'],
                'r_start': r_void,
                'r_end': 0.7 * R_cap,
                'offset_angle': math.pi / max(1, layout_params['n2']),
            },
            {
                'name': 'bottom_insert',
                'count': layout_params['n3'],
                'diameter': layout_params['d3'],
                'r_start': r_col_value,
                'r_end': R_cap,
                'offset_angle': math.pi / max(1, layout_params['n3']),
            },
            {
                'name': 'top_through',
                'count': layout_params['n4'],
                'diameter': layout_params['d4'],
                'r_start': r_void,
                'r_end': R_cap,
            },
            {
                'name': 'top_insert',
                'count': layout_params['n5'],
                'diameter': layout_params['d5'],
                'r_start': r_col_value,
                'r_end': R_cap,
                'offset_angle': math.pi / max(1, layout_params['n4']),
            },
        ]

        # 快速几何代理：分别计算底板、顶板对每个锚栓圈的环向通道占用率。
        # 精确线段碰撞只用于最终最优解复核，避免 GA 每次评估过慢。
        for ring_radius in (PCD_in / 2.0, PCD_out / 2.0):
            available_arc = (
                2.0 * math.pi * ring_radius
                - N_ring * bolt_specs['d_sleeve'] / 1000.0
            )
            if available_arc <= 1e-6:
                penalty += 1e6
                continue

            plate_demand = {
                'bottom': 0.0,
                'top': 0.0,
            }
            for group in groups:
                if not (
                    float(group['r_start']) - 1e-9
                    <= ring_radius
                    <= float(group['r_end']) + 1e-9
                ):
                    continue

                family = (
                    'bottom'
                    if str(group['name']).startswith('bottom')
                    else 'top'
                )
                plate_demand[family] += (
                    int(group['count'])
                    * float(group['diameter'])
                    / 1000.0
                )

            allowable_demand = 0.85 * available_arc
            for demand in plate_demand.values():
                if demand > allowable_demand:
                    penalty += (
                        demand - allowable_demand
                    ) / available_arc * 500000.0

    # C2. 内外圈径向间距 (液压张拉空间)
    radial_spacing = (PCD_out - PCD_in) / 2.0 * 1000.0
    required_radial = bolt_specs['d_sleeve'] + 100.0
    if radial_spacing < required_radial:
        penalty += (required_radial - radial_spacing) * 50000.0

    # D. 台柱外缘防露筋约束
    min_pedestal_cover = max(0.25, 4.0 * bolt_specs['d'] / 1000.0)
    actual_cover = r_col - (PCD_out / 2.0) - (bolt_specs['d_sleeve'] / 2000.0)
    if actual_cover < min_pedestal_cover:
        penalty += (min_pedestal_cover - actual_cover) * 1000000.0

        # E. Z 轴三维锚固深度约束
    required_anchorage_length = 35.0 * (bolt_specs['d'] / 1000.0)
    max_allowable_cage_height = H_cap - 0.2
    if required_anchorage_length > max_allowable_cage_height:
        penalty += (required_anchorage_length - max_allowable_cage_height) * 500000.0

    # F. 中心区防冲切下限
    if PCD_in < r_col * 0.6:
        penalty += (r_col * 0.6 - PCD_in) * 100000.0

    return (total_cost + 1e7 + penalty,) if penalty > 0 else (total_cost,)


# ==========================================
# 4. 主程序运行
# ==========================================
if __name__ == "__main__":
    print("\n>>> 1. 正在读取主程序 JSON 结构数据...")
    M, N_load, delta_M, max_dia, lp = load_foundation_data("Optimized_Rebar_Data.json")
    r_col_actual = lp['r_col']
    H_cap_actual = lp['H_cap']
    print(f"✅ 数据加载成功！(探测极限弯矩 M={M} kNm, 台柱半径 r_col={r_col_actual}m)")

    max_pcd_allowed = (r_col_actual - 0.25) * 2.0
    min_pcd_allowed = r_col_actual * 0.6

    toolbox.register("attr_pcd_1", random.uniform, min_pcd_allowed, max_pcd_allowed)
    toolbox.register("attr_pcd_2", random.uniform, min_pcd_allowed, max_pcd_allowed)
    toolbox.register("attr_nring", random.randint, 60, 160)
    toolbox.register("attr_type", random.randint, 0, 3)

    toolbox.register("individual", tools.initCycle, creator.Individual,
                     (toolbox.attr_pcd_1, toolbox.attr_pcd_2, toolbox.attr_nring, toolbox.attr_type), n=1)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    toolbox.register("evaluate", evaluate_anchor, M=M, N_load=N_load, delta_M=delta_M,
                     max_rebar_dia=max_dia, r_col=r_col_actual, H_cap=H_cap_actual,
                     layout_params=lp)
    toolbox.register("mate", tools.cxBlend, alpha=0.5)
    toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=1.0, indpb=0.2)
    toolbox.register("select", tools.selTournament, tournsize=3)

    print(f">>> 2. 启动【高算力双圈阵列】高保真锚栓笼寻优...")
    pop = toolbox.population(n=100)  # 提升算力
    hof = tools.HallOfFame(1)

    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("min", np.min)

    pop, logbook = algorithms.eaSimple(pop, toolbox, cxpb=0.7, mutpb=0.2, ngen=80,  # 提升代数
                                       stats=stats, halloffame=hof, verbose=False)

    best_ind = hof[0]
    PCD_A = round(best_ind[0] / 0.01) * 0.01
    PCD_B = round(best_ind[1] / 0.01) * 0.01
    best_PCD_out = max(PCD_A, PCD_B)
    best_PCD_in = min(PCD_A, PCD_B)
    best_N_ring = int(round(best_ind[2] / 4.0) * 4)
    best_bolt = get_bolt_specs(max(0, min(3, int(round(best_ind[3])))))

    delta_T_out = (4 * delta_M * best_PCD_out) / (best_N_ring * (best_PCD_out ** 2 + best_PCD_in ** 2))
    best_delta_sigma = (delta_T_out * 1000.0) / best_bolt['Ae']

    # 动态判定是否带着惩罚
    raw_best_cost = evaluate_anchor(best_ind, M, N_load, delta_M, max_dia, r_col_actual, H_cap_actual)[0]
    is_compromised = raw_best_cost > 1e7

    print(f"\n========== 🎉 【内外双圈】锚栓笼最优解 ==========")
    if is_compromised:
        print("⚠️ 注意：受限于 3.0m 的严苛台柱尺寸，此方案为带有轻微违规的妥协解！")
    print(f"外圈直径 (PCD_out): {best_PCD_out:.2f} m")
    print(f"内圈直径 (PCD_in) : {best_PCD_in:.2f} m (环间距: {(best_PCD_out - best_PCD_in) / 2 * 1000:.1f} mm)")
    print(f"锚栓总数量        : {best_N_ring * 2} 根 (每圈 {best_N_ring} 根)")
    print(f"优化后锚栓规格    : {best_bolt['type']}")
    print(f"外圈混凝土保护层  : {(r_col_actual - best_PCD_out / 2.0) * 1000.0:.1f} mm")
    print(f"最恶劣疲劳应力幅  : {best_delta_sigma:.1f} MPa (限值: 50.0 MPa)")
    print("===========================================\n")

    clear_gap_circ = (
        math.pi * best_PCD_in * 1000.0 / best_N_ring
    ) - best_bolt['d_sleeve']
    required_circ_gap = max_dia + 50.0
    radial_spacing = (best_PCD_out - best_PCD_in) / 2.0 * 1000.0
    required_radial_spacing = best_bolt['d_sleeve'] + 100.0
    required_anchorage_length = 35.0 * (best_bolt['d'] / 1000.0)
    allowable_cage_height = H_cap_actual - 0.2
    final_ach_config = {
        'PCD_in': best_PCD_in,
        'PCD_out': best_PCD_out,
        'N_per_ring': best_N_ring,
        'd_sleeve': best_bolt['d_sleeve'],
        'clear_gap_mm': 50.0,
    }
    final_layout_metrics = evaluate_rebar_groups(
        [
            {
                'name': 'bottom_through',
                'count': lp['n1'],
                'diameter': lp['d1'],
                'r_start': lp['r_void'],
                'r_end': lp['R_cap'],
            },
            {
                'name': 'bottom_reinforced',
                'count': lp['n2'],
                'diameter': lp['d2'],
                'r_start': lp['r_void'],
                'r_end': 0.7 * lp['R_cap'],
                'offset_angle': math.pi / max(1, lp['n2']),
            },
            {
                'name': 'bottom_insert',
                'count': lp['n3'],
                'diameter': lp['d3'],
                'r_start': lp['r_col'],
                'r_end': lp['R_cap'],
                'offset_angle': math.pi / max(1, lp['n3']),
            },
            {
                'name': 'top_through',
                'count': lp['n4'],
                'diameter': lp['d4'],
                'r_start': lp['r_void'],
                'r_end': lp['R_cap'],
            },
            {
                'name': 'top_insert',
                'count': lp['n5'],
                'diameter': lp['d5'],
                'r_start': lp['r_col'],
                'r_end': lp['R_cap'],
                'offset_angle': math.pi / max(1, lp['n4']),
            },
        ],
        final_ach_config,
    )

    anchor_result = {
        "anchor_cage": {
            "PCD_in_m": best_PCD_in,
            "PCD_out_m": best_PCD_out,
            "N_per_ring": best_N_ring,
            "N_total": 2 * best_N_ring,
            "Bolt_Type": best_bolt['type'],
            "Bolt_D_mm": best_bolt['d'],
            "d_sleeve_mm": best_bolt['d_sleeve'],
        },
        "cost": {
            "total_anchor_cost_rmb": raw_best_cost,
            "is_compromised": is_compromised,
        },
        "metrics": {
            "max_tension_kN": T_max_out,
            "allowable_tension_kN": best_bolt['T_allow'],
            "fatigue_stress_MPa": best_delta_sigma,
            "allowable_fatigue_stress_MPa": ALLOWABLE_DELTA_SIGMA,
            "clear_gap_circ_mm": clear_gap_circ,
            "required_circ_gap_mm": required_circ_gap,
            "radial_spacing_mm": radial_spacing,
            "required_radial_spacing_mm": required_radial_spacing,
            "required_anchorage_length_m": required_anchorage_length,
            "allowable_cage_height_m": allowable_cage_height,
            "minimum_rebar_clearance_m": final_layout_metrics['minimum_clearance_m'],
            "rebar_collision_count": final_layout_metrics['total_collisions'],
        },
        "source": {
            "r_col_m": r_col_actual,
            "H_cap_m": H_cap_actual,
            "max_rebar_diameter_mm": max_dia,
            "R_cap_m": lp['R_cap'],
            "r_top_m": lp['r_top'],
        },
    }
    with open("anchor_cage_result.json", "w", encoding="utf-8") as result_file:
        json.dump(anchor_result, result_file, ensure_ascii=False, indent=4)

    # ==========================================
    # 5. 绘制科研级 GA 收敛迭代图 (智能剥离惩罚层)
    # ==========================================
    gen = logbook.select("gen")
    fit_min = logbook.select("min")

    # 剥离致死惩罚，还原真实的波动
    real_costs = []
    has_penalty = False
    for val in fit_min:
        if val > 1e7:
            real_costs.append(val - 1e7)
            has_penalty = True
        else:
            real_costs.append(val)

    fig_iter, ax_iter = plt.subplots(figsize=(8, 5), facecolor='white')

    # 根据是否包含违规解动态调整线型和颜色
    line_style = '--' if has_penalty else '-'
    line_color = '#e67e22' if has_penalty else '#e74c3c'

    ax_iter.plot(gen, real_costs, color=line_color, marker='o', linestyle=line_style, linewidth=2, markersize=5)

    title_suffix = "\n(Sub-optimal Solution: Constraint Violated)" if has_penalty else ""
    ax_iter.set_title(f"GA Convergence for Double-Ring Search{title_suffix}", fontsize=14, fontweight='bold')
    ax_iter.set_xlabel("Generations", fontsize=12)
    ax_iter.set_ylabel("Optimal Cost + Penalty Margin (RMB)", fontsize=12)

    ax_iter.grid(True, linestyle=':', color='#BDC3C7', alpha=0.8)

    y_min, y_max = min(real_costs), max(real_costs)
    if y_max > y_min:
        ax_iter.set_ylim(bottom=y_min * 0.95, top=y_max * 1.05)

    plt.tight_layout()

    # ==========================================
    # 6. 自动化启动微观智能排布与出图
    # ==========================================
    if HAS_LAYOUT:
        print(">>> 3. 数据融合完毕，正在驱动底层【双圈排布】引擎...")
        ach_config = {
            'PCD_out': best_PCD_out,
            'PCD_in': best_PCD_in,
            'N_per_ring': best_N_ring,
            'd_sleeve': best_bolt['d_sleeve']
        }

        bottom_engine = BottomRebarLayout(
            R_cap=lp['R_cap'], r_void=lp['r_void'], r_col=lp['r_col'],
            n1_count=lp['n1'], d1=lp['d1'], n2_count=lp['n2'], d2=lp['d2'], n3_count=lp['n3'], d3=lp['d3'],
            ach_config=ach_config
        )
        top_engine = TopRebarLayout(
            R_cap=lp['R_cap'], r_void=lp['r_void'], r_col=lp['r_col'], r_top=lp['r_top'],
            n4_count=lp['n4'], d4=lp['d4'], n5_count=lp['n5'], d5=lp['d5'],
            ach_config=ach_config
        )
        bottom_engine.visualize()
        top_engine.visualize()

    print(">>> ✅ 全部流程就绪！")
    plt.show()
