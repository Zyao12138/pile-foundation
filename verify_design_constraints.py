# -*- coding: utf-8 -*-
"""
模块名称: verify_design_constraints.py (真包络极限 + 偏航扭矩 Mz 完整版)
功能: 承台与桩基 17 项完整安全验算独立运行器
修改记录:
  - [致命安全修复] 引入风机偏航扭矩 (Mz)，计算扭矩分配到单桩的最不利切向剪力，并与直接水平剪力进行【矢量叠加】，获得单桩真实水平合力。
  - [致命物理修复] 废弃原先将“全局最大水平力”与“全局最大轴力”强行缝合的“幽灵桩”验算方法。
  - [核心重构] P-8(压弯剪) 和 P-9(裂缝) 改为遍历提取实体单桩的独立 (N_i, H_i) 状态进行真实双向核算，消除了过度保守的假阳性。
  - [核心修复] 重构了群桩内力提取逻辑，按圈(Ring)同时提取最大压力和最大拉力。
  - [接口修复] 适配底层物理模型返回的“双重承载力(常规/抗震)”，并精准分发至 P-1 抗压和 P-11 负摩阻力验算。
  - [展示修复] 补充打印了被遗漏的 P-12a 基础总沉降数值，并确保倾斜显示足够的精度。
"""
import math
import copy
import foundation_config
from analyze_cap_structure import CapStructuralAnalysis
from analyze_pile_structure import PileStructuralAnalysis
from analyze_pile_structure import pm_manager  # 直接引入 P-M 曲线求解器


# ================= 辅助函数：生成圆周分布坐标 =================
def generate_ring_coords(n, r, start_angle_deg=0.0):
    """
    生成群桩圆周均匀分布的平面坐标
    :param n: 桩数量
    :param r: 桩圈半径 (m)
    :param start_angle_deg: 起始角度
    :return: [(x1, y1), (x2, y2), ...]
    """
    coords = []
    if n <= 0:
        return coords
    step = 360.0 / n
    for i in range(n):
        theta = math.radians(start_angle_deg + i * step)
        x = r * math.cos(theta)
        y = r * math.sin(theta)
        coords.append((x, y))
    return coords


# ================= 主运行函数 =================
def run_full_verification(data):
    """
    执行 17 项风机基础安全验算
    :param data: 包含几何、材料、配筋和荷载的综合数据字典
    """
    print("=" * 80)
    print("                 风电基础结构完整验算报告 (真包络极限版)")
    print("=" * 80)

    # --- 1. 基础数据准备 ---
    soil_layers = data['soil_layers']
    piles_coords = data['piles_coords']
    n_piles = len(piles_coords)

    # 构造有效土层: 剔除承台埋深H以上的土, 与优化器口径一致
    H_cap_actual = data['cap_geo'].get('H', 3.6)
    eff_soil_layers = []
    d_acc = 0.0
    for lyr in soil_layers:
        if d_acc + lyr.thickness <= H_cap_actual:
            d_acc += lyr.thickness
            continue
        new_lyr = copy.copy(lyr)
        if d_acc < H_cap_actual:
            new_lyr.thickness = (d_acc + lyr.thickness) - H_cap_actual
        eff_soil_layers.append(new_lyr)
        d_acc += lyr.thickness

    # 计算群桩抗弯惯性矩和极惯性矩
    sum_x2 = sum([x ** 2 for x, y in piles_coords])
    sum_r2 = sum([x ** 2 + y ** 2 for x, y in piles_coords])

    cap_analyzer = CapStructuralAnalysis(data['cap_geo'], data['cap_material'])
    pile_analyzer = PileStructuralAnalysis(
        data['pile_geo']['D'],
        data['pile_rebar']['n_bars'],
        data['pile_rebar']['d_bar'],
        data['pile_material'],
        data['pile_rebar']['stirrup'],
        data['pile_geo']['cover']
    )

    # 读取分离后的 ULS (极限) 与 SLS (正常使用) 荷载，包含偏航扭矩 Mz
    F_z_uls, M_xy_uls, H_k_uls = data['loads_uls']['F_z'], data['loads_uls']['M_xy'], data['loads_uls']['H_k']
    M_z_uls = data['loads_uls'].get('M_z', 0.0)

    F_z_sls, M_xy_sls, H_k_sls = data['loads_sls']['F_z'], data['loads_sls']['M_xy'], data['loads_sls']['H_k']
    M_z_sls = data['loads_sls'].get('M_z', 0.0)

    # =========================================================================
    # [核弹修复区]: 提取单桩真实 (N_i, H_i) 矢量状态与圈层极值包络
    # 原理：直接水平剪力与扭矩切向剪力必须正交矢量叠加，绝不可标量相加。
    # =========================================================================

    # 假定基础总水平力沿 X 轴正向
    H_dir_x_uls = H_k_uls / n_piles if n_piles > 0 else 0
    H_dir_x_sls = H_k_sls / n_piles if n_piles > 0 else 0

    # --- A. ULS 工况受力计算 ---
    piles_res_uls = []
    ring_dict_uls = {}

    for x, y in piles_coords:
        r = math.sqrt(x ** 2 + y ** 2)
        # 1. 竖向轴力 (均分 + 弯矩引起的拉压增量)
        N_pile = F_z_uls / n_piles + (M_xy_uls * x / sum_x2 if sum_x2 > 0 else 0)

        # 2. 偏航扭矩 Mz 引起的切向剪力向量 (垂直于半径方向)
        H_tor_x = -(M_z_uls * y) / sum_r2 if sum_r2 > 0 else 0
        H_tor_y = (M_z_uls * x) / sum_r2 if sum_r2 > 0 else 0

        # 3. 矢量叠加得到该桩真实的水平剪力合力大小
        H_actual = math.sqrt((H_dir_x_uls + H_tor_x) ** 2 + H_tor_y ** 2)

        piles_res_uls.append({
            'r': r, 'x': x, 'y': y,
            'N_design': N_pile,
            'N_comp': max(0.0, N_pile),  # <--- 【新增】：补充给承台计算的属性
            'N_tens': min(0.0, N_pile),  # <--- 【新增】：补充给承台计算的属性
            'H_actual': H_actual
        })

        rk = round(r, 2)
        if rk not in ring_dict_uls:
            ring_dict_uls[rk] = {'r': r, 'n': 0, 'N_comp': -1e9, 'N_tens': 1e9}

        ring_dict_uls[rk]['n'] += 1
        ring_dict_uls[rk]['N_comp'] = max(ring_dict_uls[rk]['N_comp'], N_pile)
        ring_dict_uls[rk]['N_tens'] = min(ring_dict_uls[rk]['N_tens'], N_pile)

    ring_env_uls = list(ring_dict_uls.values())
    max_N_comp_uls = max(p['N_design'] for p in piles_res_uls)
    min_N_uls = min(p['N_design'] for p in piles_res_uls)
    max_N_tens_uls = abs(min_N_uls) if min_N_uls < 0 else 0

    # --- B. SLS 工况受力计算 ---
    piles_res_sls = []
    ring_dict_sls = {}

    for x, y in piles_coords:
        r = math.sqrt(x ** 2 + y ** 2)
        N_pile = F_z_sls / n_piles + (M_xy_sls * x / sum_x2 if sum_x2 > 0 else 0)

        H_tor_x = -(M_z_sls * y) / sum_r2 if sum_r2 > 0 else 0
        H_tor_y = (M_z_sls * x) / sum_r2 if sum_r2 > 0 else 0
        H_actual = math.sqrt((H_dir_x_sls + H_tor_x) ** 2 + H_tor_y ** 2)

        piles_res_sls.append({
            'r': r, 'x': x, 'y': y,
            'N_design': N_pile,
            'N_comp': max(0.0, N_pile),
            'N_tens': min(0.0, N_pile),
            'H_actual': H_actual
        })

        rk = round(r, 2)
        if rk not in ring_dict_sls:
            ring_dict_sls[rk] = {'r': r, 'n': 0, 'N_comp': -1e9, 'N_tens': 1e9}
        ring_dict_sls[rk]['n'] += 1
        ring_dict_sls[rk]['N_comp'] = max(ring_dict_sls[rk]['N_comp'], N_pile)
        ring_dict_sls[rk]['N_tens'] = min(ring_dict_sls[rk]['N_tens'], N_pile)

    ring_env_sls = list(ring_dict_sls.values())
    max_N_comp_sls = max(p['N_design'] for p in piles_res_sls)
    min_N_sls = min(p['N_design'] for p in piles_res_sls)
    max_N_tens_sls = abs(min_N_sls) if min_N_sls < 0 else 0

    # --- 3. FEM 有限元计算"单位水平力"下的桩身弯矩分布系数 ---
    _, _, M_per_H_uls, _ = pile_analyzer.calculate_horizontal_mechanics_fem_detailed(
        1.0, 0, data['pile_geo']['L'], eff_soil_layers, piles_coords)

    _, _, M_per_H_sls, _ = pile_analyzer.calculate_horizontal_mechanics_fem_detailed(
        1.0, 0, data['pile_geo']['L'], eff_soil_layers, piles_coords)

    # =========================================================================
    # Part 1: 桩基结构验算 (12项)
    # =========================================================================
    print("--- Part 1: 桩基结构验算 (12项) ---")

    # 【接口修复】：正确接收常规和抗震液化双重承载力，独立工况取用 Ra_norm
    Ra_norm, Ra_seis = pile_analyzer.calculate_compression_capacity(data['pile_geo']['L'], eff_soil_layers)
    print(
        f"P-1  [单桩抗压承载力(SLS)] Ra_limit={1.2 * Ra_norm:.1f} kN, N_max={max_N_comp_sls:.1f} kN -> {'[OK]' if max_N_comp_sls <= 1.2 * Ra_norm else '[FAIL]'}")

    Rt = pile_analyzer.calculate_uplift_capacity(data['pile_geo']['L'], eff_soil_layers, piles_coords)
    print(
        f"P-2  [单桩抗拔承载力(SLS)] Rt={Rt:.1f} kN, T_max={max_N_tens_sls:.1f} kN -> {'[OK]' if max_N_tens_sls <= Rt else '[FAIL]'}")

    _, Rha_seis = pile_analyzer.calculate_horizontal_capacity(data['pile_geo']['L'], eff_soil_layers)
    max_H_sls = max(p['H_actual'] for p in piles_res_sls)
    print(
        f"P-3  [单桩水平承载力(SLS)] Rha={Rha_seis:.1f} kN, H_max_real={max_H_sls:.1f} kN -> {'[OK]' if max_H_sls <= Rha_seis else '[FAIL]'}")

    u4, cap_str = pile_analyzer.check_structural_strength(max_N_comp_uls)
    print(f"P-4  [桩身轴压强度(ULS)] 利用率: {u4:.2f} (极限: {cap_str:.1f} kN) -> {'[OK]' if u4 <= 1.0 else '[FAIL]'}")

    u5 = pile_analyzer.check_pile_top_local_compression(max_N_comp_uls, data['cap_material']['FC_DESIGN'],
                                                        data['cap_geo']['R_cap'], max(p['r'] for p in ring_env_uls))
    print(f"P-5  [桩顶局部受压(ULS)] 利用率: {u5:.2f} -> {'[OK]' if u5 <= 1.0 else '[FAIL]'}")

    u6, _, _ = pile_analyzer.check_buckling_stability(max_N_comp_uls, data['pile_geo']['L'], eff_soil_layers)
    print(f"P-6  [桩身压屈稳定性(ULS)] 利用率: {u6:.2f} -> {'[OK]' if u6 <= 1.0 else '[FAIL]'}")

    if max_N_tens_uls > 0:
        u7, _ = pile_analyzer.check_tensile_strength(max_N_tens_uls)
    else:
        u7 = 0.0
    print(f"P-7  [桩身受拉强度(ULS)] 利用率: {u7:.2f} -> {'[OK]' if u7 <= 1.0 else '[FAIL]'}")

    # =====================================================================
    # P-8: 压弯剪与拉弯剪双向真实核算 (遍历实体桩)
    # =====================================================================
    u_sh_max = 0.0
    u_bd_max = 0.0

    for p in piles_res_uls:
        N_i = p['N_design']
        H_i = p['H_actual']
        M_pile_i = M_per_H_uls * H_i

        Mu_cap = pm_manager.get_capacity_usage(
            data['pile_geo']['D'], data['pile_rebar']['n_bars'],
            data['pile_rebar']['d_bar'], N_i, M_pile_i,
            data['pile_material']['fc'], data['pile_geo']['cover'] * 1000.0, return_capacity=True
        )
        u_sh_i, u_bd_i, _ = pile_analyzer.check_bending_shear_capacity(M_pile_i, H_i, N_i, Mu_cap)

        u_sh_max = max(u_sh_max, u_sh_i)
        u_bd_max = max(u_bd_max, u_bd_i)

    print(
        f"P-8  [桩身受弯剪承载(ULS)] 最大剪力率: {u_sh_max:.2f}, 最大弯矩率: {u_bd_max:.2f} -> {'[OK]' if max(u_sh_max, u_bd_max) <= 1.0 else '[FAIL]'}")

    # =====================================================================
    # P-9: SLS 裂缝双侧真实核算 (遍历实体桩)
    # =====================================================================
    w_max_crack = 0.0
    for p in piles_res_sls:
        N_i = p['N_design']
        H_i = p['H_actual']
        M_pile_i = M_per_H_sls * H_i
        w_i = pile_analyzer.check_crack_width(M_pile_i, N_i)
        if w_i is not None:
            w_max_crack = max(w_max_crack, w_i)

    print(f"P-9  [桩身裂缝宽度(SLS)] w_max={w_max_crack:.3f} mm -> {'[OK]' if w_max_crack <= 0.2 else '[FAIL]'}")

    res_10 = pile_analyzer.check_weak_underlying_layer(F_z_sls, 0.0, data['pile_geo']['L'], soil_layers, piles_coords, H_cap=H_cap_actual)
    print(f"P-10 [软弱下卧层验算(SLS)] 利用率: {res_10:.2f} -> {'[OK]' if res_10 <= 1.0 else '[FAIL]'}")

    # 【接口修复】：负摩阻力传入未打折的常规承载力 Ra_norm，供内部正确扣减正摩擦
    res_11 = pile_analyzer.check_negative_friction(max_N_comp_uls, Ra_norm, sum(
        l.thickness for l in eff_soil_layers if getattr(l, 'is_compressible', False)), eff_soil_layers, piles_coords)
    print(f"P-11 [负摩阻力验算(ULS)] 利用率: {res_11:.2f} -> {'[OK]' if res_11 <= 1.0 else '[FAIL]'}")

    s_m, t_m = pile_analyzer.check_settlement_and_tilt(F_z_sls, M_xy_sls, data['pile_geo']['L'], soil_layers,
                                                       piles_coords, H_cap=H_cap_actual)
    # 【打印补充区】
    print(f"P-12a [基础总沉降(SLS)] 沉降={s_m:.2f} mm -> INFO")
    print(f"P-12b [基础总倾斜(SLS)] 倾斜={t_m:.5f} rad -> {'[OK]' if t_m <= 0.003 else '[FAIL]'}")

    # =========================================================================
    # Part 2: 承台结构验算 (5项)
    # =========================================================================
    print("\n--- Part 2: 承台结构验算 (5项) ---")
    cap_rebar = data['cap_rebar']

    # 【重要修复】：把 ring_env 改为全量 piles_res 传入，恢复静力平衡
    v13, r13, p13 = cap_analyzer.check_flexural_reinforcement(piles_res_uls, cap_rebar, True)
    print(f"C-1  [承台抗弯配筋(ULS)] Req={r13:.0f}, Prov={p13:.0f} mm2/m -> {'[OK]' if v13 < 1e-3 else '[FAIL]'}")

    v14, vd14, vc14 = cap_analyzer.check_inclined_section_shear(piles_res_uls, cap_rebar['d1'] * 1000, True)
    print(f"C-2  [斜截面抗剪切(ULS)] V_des={vd14:.1f}, V_cap={vc14:.1f} kN -> {'[OK]' if v14 < 1e-3 else '[FAIL]'}")

    v15, np15, rp15 = cap_analyzer.check_corner_pile_punching(piles_res_uls, data['pile_geo']['D'], True)
    print(f"C-3  [承台角桩抗冲切(ULS)] N_max={np15:.1f}, Cap={rp15:.1f} kN -> {'[OK]' if v15 < 1e-3 else '[FAIL]'}")

    v16, w16, wl16 = cap_analyzer.check_crack_width(piles_res_sls, cap_rebar, True)
    print(f"C-4  [承台裂缝宽度(SLS)] w_max={w16:.3f} mm -> {'[OK]' if v16 < 1e-3 else '[FAIL]'}")

    v17, fp17, fc17 = cap_analyzer.check_pedestal_punching(piles_res_uls, cap_rebar['d1'] * 1000, True)
    print(f"C-5  [台柱抗冲切验算(ULS)] F_des={fp17:.1f}, F_cap={fc17:.1f} kN -> {'[OK]' if v17 < 1e-3 else '[FAIL]'}")


if __name__ == "__main__":
    import math
    from analyze_integrated_pile_foundation import calculate_pile_steel_detailed

    print("\n" + "=" * 80)
    print("           【人工施工图纸(360根底筋+双层环筋) vs 优化算法 对标验证】")
    print("=" * 80)

    L_pile_real = 25.0  # 桩长

    # -------------------------------------------------------------
    # 1. 精准计算 CAD 图纸的真实材料工程量
    # -------------------------------------------------------------
    R, r_col, r_top = 9.5, 3.4, 3.6
    h1, h2, h3 = 1.0, 1.4, 1.2
    V_edge = math.pi * R ** 2 * h1
    V_slope = (1 / 3) * math.pi * h2 * (R ** 2 + r_top ** 2 + R * r_top)
    V_stub = math.pi * r_col ** 2 * h3
    V_conc_real = V_edge + V_slope + V_stub

    unit_w_32 = 6.31
    unit_w_28 = 4.83
    unit_w_25 = 3.85

    L_long = 9.5 - 1.5
    L_short = L_long * 0.7

    W_bot_rad = (120 * L_long * unit_w_32) + (120 * L_short * unit_w_32) + (120 * L_short * unit_w_28)
    W_top_rad = (240 * L_long * unit_w_28)

    area_annulus = math.pi * ((9.5 - 0.05) ** 2 - 1.5 ** 2)
    W_bot_hoop = (area_annulus / 0.150 * unit_w_25) + (area_annulus / 0.200 * unit_w_25)
    W_top_hoop = (area_annulus / 0.120 * unit_w_25)

    W_cap_steel = W_bot_rad + W_top_rad + W_bot_hoop + W_top_hoop + (V_conc_real * 5.0)
    W_pile_steel = calculate_pile_steel_detailed(36, 0.8, L_pile_real, 18, 0.025)
    W_other = V_conc_real * 40.0
    Total_Steel_real = W_cap_steel + W_pile_steel + W_other

    # -------------------------------------------------------------
    # 2. 将图纸等效为验算器可读的参数 (Equivalent Area Method)
    # -------------------------------------------------------------
    eq_bot_rad_n = 332
    eq_bot_rad_d = 0.032
    eq_bot_hoop_s = 85.7

    # 增加覆土计算
    V_total_cylinder_real = math.pi * R ** 2 * 3.6  # H=3.6
    V_soil_real = max(0.0, V_total_cylinder_real - V_conc_real)
    G_soil_real = V_soil_real * 18.0
    # 【新增修复】：考虑水位 2.0~5.6m 产生的浮力
    water_top, water_bot = 2.0, 5.6
    submerged_depth = max(0.0, min(3.6, water_bot) - water_top)
    buoyancy = math.pi * (R ** 2) * submerged_depth * 10.0

    CAP_WEIGHT = V_conc_real * 25.0 + G_soil_real - buoyancy
    load_uls_raw = foundation_config.CONFIG['load_cases']["Extreme"]
    load_sls_raw = foundation_config.CONFIG['load_cases']["Normal"]

    H_arm = h1 + h2 + h3

    test_data = {
        'cap_geo': {
            'R_cap': R, 'r_col': r_col, 'h1': h1, 'h2': h2, 'h3': h3, 'D_pile': 0.8, 'H': 3.6
        },
        'cap_material': {'FT_DESIGN': 1.71, 'FC_DESIGN': 19.1, 'FT_STD': 2.39, 'FY_STEEL': 360.0, 'ES_STEEL': 2.0e5,
                         'COVER': 50.0},
        'cap_rebar': {
            'n1': eq_bot_rad_n, 'd1': eq_bot_rad_d,
            'n4': 240, 'd4': 0.028,
            # 直接输入真实图纸配置：底层25，附加层25，标准间距150
            'ring_opt': {'d1': 0.025, 's1': 150.0, 'd2': 0.025, 's2': 200.0, 's': 150.0},
            'top_ring_opt': {'d': 0.025, 's': 120.0}
        },
        'pile_geo': {'D': 0.8, 'L': L_pile_real, 'cover': 0.06},
        'pile_rebar': {'n_bars': 18, 'd_bar': 0.025, 'stirrup': {'d': 0.012, 's': 0.100, 'fyv': 360.0}},
        'pile_material': {'fc': 19.1, 'ft': 1.71, 'ftk': 2.39, 'fy': 360.0, 'Es': 2.0e5, 'Ec': 3.25e4},

        'loads_uls': {
            'F_z': load_uls_raw['FZ'] * 1.35 + CAP_WEIGHT * 1.35,
            'M_xy': load_uls_raw['MXY'] * 1.5 + load_uls_raw['FXY'] * 1.5 * H_arm,
            'H_k': load_uls_raw['FXY'] * 1.5,
            'M_z': load_uls_raw.get('Mz', 0.0) * 1.5  # 【关键补丁】传入极限偏航扭矩
        },
        'loads_sls': {
            'F_z': load_sls_raw['FZ'] * 1.0 + CAP_WEIGHT * 1.0,
            'M_xy': load_sls_raw['MXY'] * 1.0 + load_sls_raw['FXY'] * 1.0 * H_arm,
            'H_k': load_sls_raw['FXY'] * 1.0,
            'M_z': load_sls_raw.get('Mz', 0.0) * 1.0  # 【关键补丁】传入正常使用偏航扭矩
        },
        'piles_coords': generate_ring_coords(14, 6.3) + generate_ring_coords(22, 8.5),
        'soil_layers': foundation_config.SOIL_LAYERS
    }

    # -------------------------------------------------------------
    # 3. 打印量耗对比并运行验算
    # -------------------------------------------------------------
    print(f"【施工图原始工程量统计 (精确计算)】")
    print(f"  👉 混凝土总体积: {V_conc_real:.1f} m³")
    print(f"  👉 钢筋总重量:   {Total_Steel_real:.1f} kg (约 {Total_Steel_real / 1000:.1f} 吨)")
    print(f"  👉 桩基总数量:   36 根")
    print("-" * 80)
    print(" ⬇ 开始对这套『堆料版』方案进行受力验算 (注意看利用率有多低) ⬇\n")

    run_full_verification(test_data)
