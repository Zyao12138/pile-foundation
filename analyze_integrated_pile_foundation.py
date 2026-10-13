# -*- coding: utf-8 -*-
"""
模块名称: analyze_integrated_pile_foundation.py
功能: 风机基础物理建模、结构验算与工程量评估核心模块 (全物理真解 + 宏观排布包络版)
修改记录:
    - [新增约束] 引入锚栓笼瓶颈圈的极端拥挤度惩罚，保证算法在迭代中筛选出的钢筋根数具备可施工绑扎性。
    - [新增约束] 算量扣减区半径 r1 动态跟随锚栓笼外径 (D_outer)，消除硬编码导致的算量虚假缩水。
    - [核心修复] 彻底舍弃“圆环极值包络”，直接传递 40 根桩的真实 2D 坐标与受力，还原真实静力平衡。
    - [造价修复] 修正 Cost 计算逻辑，将钢筋用量折算为等效混凝土体积，杜绝“薄皮纯钢”的怪物设计。
    - [物理修复] 引入 2.0m ~ 5.6m 局部地下水位，精准计算承台/覆土在此区间的真实浮力，防止抗拔力虚假缩水。
    - [物理修复] 在评估算法中补全了偏航扭矩 Mz 的切向剪力。
    - [规范修复] 严格对标 JGJ 94-2008，SLS 增加抗震提高系数 k_ax，并消除 SLS 状态下的裂缝“幽灵桩”计算。
    - [致命修复] 接收 `calculate_compression_capacity` 返回的常规和抗震双重承载力，在 SLS 验算中正确分发液化折减。
    - [土力学修复] 引入 `eff_soil_layers` 截除承台埋深 H_cap 以上土层，彻底解决水平刚度误判与沉降双重扣减问题。
"""
import math
import numpy as np
import copy
import foundation_config as cfg

# 尝试导入具体的结构计算模块
try:
    from analyze_pile_structure import PileStructuralAnalysis
    from analyze_cap_structure import CapStructuralAnalysis
    from analyze_pile_structure import pm_manager

    HAS_MODULES = True
    HAS_PM = True
except ImportError as e:
    print(f"【模型警告】结构计算模块导入失败: {e} (将跳过详细力学验算)")
    HAS_MODULES = False
    HAS_PM = False


# ================= 1. 辅助算量函数 =================
def calculate_pile_steel_detailed(n_pile, D_pile_m, L_pile_m, n_main, d_main_m):
    """
    计算桩基系统整体钢筋用量 (kg)
    包含主筋与螺旋箍筋的估算。
    """
    # 1. 主筋长度 (单根桩长 + 伸入承台的锚固长度，按规范约 35d 估算)
    len_main = L_pile_m + max(1.0, 35 * d_main_m)
    w_main = n_main * len_main * (0.25 * math.pi * d_main_m ** 2 * 7850)

    # 2. 箍筋 (螺旋箍)
    D_cage = D_pile_m - 0.1  # 扣除两边保护层后的钢筋笼直径
    # 简化估算: 假设全长平均间距 150mm
    vol_stir = (L_pile_m / 0.15) * (math.pi * D_cage)
    w_stir = vol_stir * (0.25 * math.pi * 0.010 ** 2 * 7850)  # 假设箍筋直径 10mm

    return (w_main + w_stir) * n_pile


# ================= 2. 基础物理模型类 =================
class IntegratedPileFoundation:
    """
    风电桩基综合物理模型
    负责计算基础几何体积、配筋量，以及在特定工况下计算每根单桩的受力分配。
    """

    def __init__(self, full_params, load_case, soil_layers, factors=(1.0, 1.0)):
        self.geo = full_params['geo']
        self.load = load_case
        self.rebar = full_params['rebar']
        self.anchor = full_params.get('anchor', {})
        self.gamma_G = factors[0]  # 恒载分项系数
        self.gamma_Q = factors[1]  # 活载分项系数

        self.R_cap = self.geo['R_cap']
        self.h1 = self.geo['h1']
        self.h2 = self.geo['h2']
        self.h3 = self.geo['h3']

        self.soil_layers = soil_layers

        # 生成所有单桩的平面坐标 (x, y)
        self.piles_coords = []
        for ring in self.geo['rings']:
            n_in_ring = ring['n']
            r_ring = ring['r']
            for i in range(n_in_ring):
                theta = 2 * math.pi * i / n_in_ring
                self.piles_coords.append((r_ring * math.cos(theta), r_ring * math.sin(theta)))
        self.N_total = len(self.piles_coords)

    def get_quantities(self):
        """计算基础的物理工程量 (混凝土体积, 钢筋总重)"""
        # --- 1. 混凝土体积 (m3) ---
        V_edge = math.pi * self.R_cap ** 2 * self.h1
        V_slope = (1.0 / 3.0) * math.pi * self.h2 * (
                self.R_cap ** 2 + self.geo['r_top'] ** 2 + self.R_cap * self.geo['r_top'])
        V_stub = math.pi * self.geo['r_col'] ** 2 * self.h3
        Vc = V_edge + V_slope + V_stub

        # --- 2. 承台钢筋量估算 (kg) ---
        # ⭐ 【宏观排布包络修复】：动态读取锚栓笼外缘半径，再加 100mm 施工避让空间作为中心留空区
        ach_D_outer = self.anchor.get(
            'D_outer_m',
            cfg.CONFIG.get('ANCHOR_CAGE', {}).get('D_outer', 5.1),
        )
        r1 = (ach_D_outer / 2.0) + 0.1  # 中心不布筋的真实半径 (例如 5.1/2 + 0.1 = 2.65m)

        R_eff = self.R_cap - 0.05

        unit_w = 0.25 * math.pi * 7850

        L_1 = max(0.0, R_eff - r1)  # ① 通长筋长度 (真实起点为 r1)
        L_2 = max(0.0, 0.7 * self.R_cap - r1)  # ③ 加强筋长度
        L_3 = max(0.0, R_eff - self.geo['r_col'])  # ② 外插筋长度

        W_bot_rad = (self.rebar['n1'] * self.rebar['d1'] ** 2 * L_1 +
                     self.rebar.get('n2', self.rebar['n1']) * self.rebar.get('d2', self.rebar['d1']) ** 2 * L_2 +
                     self.rebar.get('n3', self.rebar['n1']) * self.rebar.get('d3',
                                                                             self.rebar['d1']) ** 2 * L_3) * unit_w

        W_top_rad = (self.rebar['n4'] * self.rebar['d4'] ** 2 * L_1 +
                     self.rebar.get('n5', self.rebar['n4']) * self.rebar.get('d5',
                                                                             self.rebar['d4']) ** 2 * L_3) * unit_w

        L_range = max(0.0, R_eff - r1)
        avg_perimeter = 2 * math.pi * (r1 + L_range / 2.0)

        d1_hoop = self.rebar['ring_opt'].get('d1', self.rebar['ring_opt'].get('d', 0.0))
        d2_hoop = self.rebar['ring_opt'].get('d2', 0.0)
        s1_1 = self.rebar['ring_opt'].get('s1', self.rebar['ring_opt'].get('s', 150.0)) / 1000.0
        s1_2 = self.rebar['ring_opt'].get('s2', s1_1 * 1000.0) / 1000.0

        w_ring_1 = (L_range / s1_1 * avg_perimeter * (d1_hoop ** 2) * unit_w) if s1_1 > 0 else 0
        w_ring_2 = (L_range / s1_2 * avg_perimeter * (d2_hoop ** 2) * unit_w) if (s1_2 > 0 and d2_hoop > 0) else 0
        W_bot_ring = w_ring_1 + w_ring_2

        s2 = self.rebar['top_ring_opt']['s'] / 1000.0
        W_top_ring = (L_range / s2 * avg_perimeter * self.rebar['top_ring_opt']['d'] ** 2 * unit_w) if s2 > 0 else 0

        W_cap_total = W_bot_rad + W_top_rad + W_bot_ring + W_top_ring

        area_cap = math.pi * self.R_cap ** 2
        n_ties = area_cap / 0.36
        len_tie = self.h1 + self.h2 * 0.5
        W_ties = n_ties * len_tie * (0.25 * math.pi * 0.016 ** 2 * 7850)
        W_cap_total += W_ties

        return Vc, W_cap_total, 0, 0

    def get_pile_reactions_detailed(self):
        """计算每根单桩的真实竖向反力与矢量叠加后的真实水平剪力"""
        if self.N_total == 0:
            return [], 0, 0

        V_conc = self.get_quantities()[0]
        G_cap = V_conc * 25.0

        V_total_cylinder = math.pi * (self.geo['R_cap'] ** 2) * self.geo['H']
        V_soil = max(0.0, V_total_cylinder - V_conc)
        G_soil = V_soil * 18.0

        # 【物理修复：精准的区间水位浮力】
        water_top = 2.0
        water_bot = 5.6
        submerged_depth = max(0.0, min(self.geo['H'], water_bot) - water_top)
        Buoyancy_cap_soil = math.pi * (self.geo['R_cap'] ** 2) * submerged_depth * 10.0

        G_k_unfav = G_cap + G_soil
        G_k_fav = max(0.0, G_cap + G_soil - Buoyancy_cap_soil)

        gamma_G_unfav = self.gamma_G
        gamma_G_fav = 1.0 if self.gamma_G > 1.0 else self.gamma_G

        Fz_unfav = self.load['FZ'] * gamma_G_unfav + G_k_unfav * gamma_G_unfav
        Fz_fav = self.load['FZ'] * gamma_G_fav + G_k_fav * gamma_G_fav
        H_arm = self.h1 + self.h2 + self.h3
        M_total = self.load['MXY'] * self.gamma_Q + self.load['FXY'] * self.gamma_Q * H_arm

        # 提取基础剪力与偏航扭矩
        Fxy_design = self.load['FXY'] * self.gamma_Q
        Mz_design = self.load.get('Mz', 0.0) * self.gamma_Q

        N_avg_unfav = Fz_unfav / self.N_total
        N_avg_fav = Fz_fav / self.N_total
        sum_x2 = sum([x ** 2 for x, y in self.piles_coords])
        sum_r2 = sum([x ** 2 + y ** 2 for x, y in self.piles_coords])

        # 假定风向沿 X 轴正方向，计算基础剪力在单桩上的 X/Y 分量
        H_dir_x = Fxy_design / self.N_total
        H_dir_y = 0.0

        # 【包络修复】：计算 360度风向下的绝对最大理论水平剪力 (标量相加)
        R_max = max([math.sqrt(x ** 2 + y ** 2) for x, y in self.piles_coords]) if self.piles_coords else 0.0
        H_max_envelope = (Fxy_design / self.N_total) + (Mz_design * R_max / sum_r2 if sum_r2 > 0 else 0)

        res = []
        for (x, y) in self.piles_coords:
            r_dist = math.sqrt(x ** 2 + y ** 2)

            # 1. 轴力增量
            delta_x = (M_total * x) / sum_x2 if sum_x2 > 0 else 0
            N_c = N_avg_unfav + delta_x
            N_t = N_avg_fav + delta_x

            # 2. 偏航扭矩 Mz 产生的切向剪力向量分量
            H_tor_x = - (Mz_design * y) / sum_r2 if sum_r2 > 0 else 0
            H_tor_y = (Mz_design * x) / sum_r2 if sum_r2 > 0 else 0

            # 3. 矢量叠加 (针对当前假定 X 风向的真实合力)
            H_actual_vector = math.sqrt((H_dir_x + H_tor_x) ** 2 + (H_dir_y + H_tor_y) ** 2)

            # 【包络兜底】：对于最外圈的桩，采用理论包络极大值，防止漏算最恶劣风向
            if r_dist >= R_max - 0.1:
                H_actual = max(H_actual_vector, H_max_envelope)
            else:
                H_actual = H_actual_vector

            res.append({
                'r': r_dist, 'x': x, 'y': y, 'n': 1,
                'N_comp': N_c,
                'N_tens': N_t,
                'N_design': N_c if abs(N_c) > abs(N_t) else N_t,
                'H_actual': H_actual
            })
        return res, Fxy_design, M_total

    def get_individual_pile_forces(self):
        res, _, _ = self.get_pile_reactions_detailed()
        forces = [p['N_design'] for p in res]
        return self.piles_coords, forces


# ================= 3. 核心物理评估逻辑 =================
def assess_design_performance(params, return_details=False):
    geo = params['geo']
    rb = params['rebar']
    mat = params['mat']
    total_viol = 0.0
    constraint_records = {}

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
        """
        记录命名约束。

        total_viol 仍沿用原有加权累加逻辑；constraint_records 保存可供后续
        epsilon 约束处理的独立约束信息。同名约束保留最不利的一次。
        """
        try:
            raw = max(0.0, float(raw_violation))
        except (TypeError, ValueError):
            raw = 1e6
        if not np.isfinite(raw):
            raw = 1e6

        weighted = raw * float(weight)
        current = constraint_records.get(name)
        if current is None or weighted > current["weighted_violation"]:
            constraint_records[name] = {
                "violation": raw,
                "weight": float(weight),
                "weighted_violation": weighted,
                "hard": bool(hard),
                "utilization": None if utilization is None else float(utilization),
                "value": None if value is None else float(value),
                "limit": None if limit is None else float(limit),
                "unit": unit,
            }

    # --- 1. 基础几何及排布约束 ---
    raw = max(0.0, 0.2 - geo['h3'])
    if raw > 0.0:
        total_viol += raw
    record_constraint(
        "h3_minimum",
        raw,
        hard=True,
        utilization=0.2 / max(geo['h3'], 1e-9),
        value=geo['h3'],
        limit=0.2,
        unit="m",
    )
    raw = 1.0 if geo['r_col'] >= geo['R_cap'] - 1.0 else 0.0
    if raw > 0.0:
        total_viol += 1.0
    record_constraint(
        "pedestal_edge_clearance",
        raw,
        hard=True,
        value=geo['R_cap'] - geo['r_col'],
        limit=1.0,
        unit="m",
    )

    # 1.1 最外边缘的净距校验
    perimeter_bot = 2 * math.pi * (geo['R_cap'] - 0.05)
    s_bot = (perimeter_bot / rb['n1']) - max(rb['d1'], rb['d2'])
    raw = max(0.0, 0.05 - s_bot)
    if raw > 0.0:
        total_viol += raw * 10.0
    record_constraint(
        "bottom_rebar_edge_spacing",
        raw,
        weight=10.0,
        hard=True,
        utilization=0.05 / max(s_bot, 1e-9),
        value=s_bot,
        limit=0.05,
        unit="m",
    )

    # ⭐ 【新增宏观排布约束：锚栓笼瓶颈圈净距严苛校验】 ⭐
    # 在锚栓笼外边缘处，几百根通长径向筋交汇，是现场最容易绑扎打架的地方
    anchor = params.get('anchor', {})
    ach_D_outer = anchor.get(
        'D_outer_m',
        cfg.CONFIG.get('ANCHOR_CAGE', {}).get('D_outer', 5.1),
    )
    r_bottleneck = (ach_D_outer / 2.0) + 0.1  # 钢筋抵达锚栓笼外侧的位置
    perimeter_bottleneck = 2 * math.pi * r_bottleneck

    # 计算瓶颈处的理论钢筋净距 (通长筋 n1 全部聚集在此)
    s_bottleneck = (perimeter_bottleneck / rb['n1']) - rb['d1']

    # 强制要求瓶颈净距 >= 45mm (保证混凝土骨料通过，且留有后期局部避让偏移的余量)
    raw = max(0.0, 0.045 - s_bottleneck)
    if raw > 0.0:
        total_viol += raw * 50.0  # 给予重度惩罚，逼迫算法减少根数改用大直径
    record_constraint(
        "bottom_rebar_anchor_bottleneck_spacing",
        raw,
        weight=50.0,
        hard=True,
        utilization=0.045 / max(s_bottleneck, 1e-9),
        value=s_bottleneck,
        limit=0.045,
        unit="m",
    )

    # 1.2 桩基与承台边缘的安全距离
    r_pile_max = max([r['r'] for r in geo['rings']]) if geo['rings'] else 0.0
    dist_edge_actual = geo['R_cap'] - r_pile_max
    raw = max(0.0, (geo['D_pile'] - dist_edge_actual) / geo['D_pile'])
    if raw > 0.0:
        total_viol += raw * 10.0
    record_constraint(
        "pile_edge_clearance",
        raw,
        weight=10.0,
        hard=True,
        utilization=geo['D_pile'] / max(dist_edge_actual, 1e-9),
        value=dist_edge_actual,
        limit=geo['D_pile'],
        unit="m",
    )

    # --- 2. 结构与力学验算 ---
    normal_load = cfg.CONFIG['load_cases']["Normal"]
    try:
        if HAS_MODULES:
            cap_ana = CapStructuralAnalysis(geo, mat)
            pile_ana = PileStructuralAnalysis(geo['D_pile'], rb['pile_main_n'], rb['pile_main_d'], mat,
                                              params['stirrup'], cover=cfg.CONFIG['pile_geo'].get('cover', 0.06))

            viol_const, _, _ = pile_ana.check_rebar_construction_rules(min_ratio=0.004, max_ratio=0.04,
                                                                       min_clear_mm=50.0, max_spacing_mm=300.0)
            total_viol += viol_const * 10.0
            record_constraint(
                "pile_rebar_construction",
                viol_const,
                weight=10.0,
                hard=True,
            )

            pf_temp = IntegratedPileFoundation(params, normal_load, cfg.SOIL_LAYERS)
            _, s_avg = pile_ana._analyze_group_geometry(pf_temp.piles_coords)

            raw = max(0.0, (3.0 * geo['D_pile'] - s_avg) / geo['D_pile'])
            if raw > 0.0:
                total_viol += raw * 10.0
            record_constraint(
                "pile_spacing_minimum",
                raw,
                weight=10.0,
                hard=True,
                utilization=(3.0 * geo['D_pile']) / max(s_avg, 1e-9),
                value=s_avg,
                limit=3.0 * geo['D_pile'],
                unit="m",
            )
            raw = max(0.0, (s_avg - 6.0 * geo['D_pile']) / geo['D_pile'])
            if raw > 0.0:
                total_viol += raw * 10.0
            record_constraint(
                "pile_spacing_maximum",
                raw,
                weight=10.0,
                hard=False,
                utilization=s_avg / max(6.0 * geo['D_pile'], 1e-9),
                value=s_avg,
                limit=6.0 * geo['D_pile'],
                unit="m",
            )

            # =========================================================================
            # ⭐ 【致命物理修复 1】：地层深度坐标系对齐
            # =========================================================================
            H_cap = geo['H']
            eff_soil_layers = []
            d_acc = 0.0
            for lyr in cfg.SOIL_LAYERS:
                if d_acc + lyr.thickness <= H_cap:
                    d_acc += lyr.thickness
                    continue
                new_lyr = copy.copy(lyr)
                if d_acc < H_cap:
                    new_lyr.thickness = (d_acc + lyr.thickness) - H_cap
                eff_soil_layers.append(new_lyr)
                d_acc += lyr.thickness

            # 【双重承载力读取】
            Ra_char_norm, Ra_char_seis = pile_ana.calculate_compression_capacity(geo['L_pile'], eff_soil_layers)
            Ra_char = Ra_char_norm

            Rt_char = pile_ana.calculate_uplift_capacity(geo['L_pile'], eff_soil_layers, pf_temp.piles_coords)
            Rh_norm, Rh_seis = pile_ana.calculate_horizontal_capacity(geo['L_pile'], eff_soil_layers)

            lims = cfg.CONFIG.get('DESIGN_PARAMS', {}).get('LIMITS',
                                                           {'MIN_Ra': 2200.0, 'MIN_Rt': 1000.0, 'MIN_Rh': 70.0})
            raw = max(0.0, (lims['MIN_Ra'] - Ra_char) / lims['MIN_Ra'])
            if raw > 0.0:
                total_viol += raw * 5.0
            record_constraint(
                "minimum_compression_capacity",
                raw,
                weight=5.0,
                utilization=lims['MIN_Ra'] / max(Ra_char, 1e-9),
                value=Ra_char,
                limit=lims['MIN_Ra'],
                unit="kN",
            )
            raw = max(0.0, (lims['MIN_Rt'] - Rt_char) / lims['MIN_Rt'])
            if raw > 0.0:
                total_viol += raw * 5.0
            record_constraint(
                "minimum_uplift_capacity",
                raw,
                weight=5.0,
                utilization=lims['MIN_Rt'] / max(Rt_char, 1e-9),
                value=Rt_char,
                limit=lims['MIN_Rt'],
                unit="kN",
            )
            raw = max(0.0, (lims['MIN_Rh'] - Rh_norm) / lims['MIN_Rh'])
            if raw > 0.0:
                total_viol += raw * 5.0
            record_constraint(
                "minimum_horizontal_capacity",
                raw,
                weight=5.0,
                utilization=lims['MIN_Rh'] / max(Rh_norm, 1e-9),
                value=Rh_norm,
                limit=lims['MIN_Rh'],
                unit="kN",
            )

            # === 遍历所有荷载组合 ===
            for combo in cfg.CONFIG['LOAD_COMBINATIONS']:
                load_data = cfg.CONFIG['load_cases'][combo['case_ref']]
                pf = IntegratedPileFoundation(params, load_data, cfg.SOIL_LAYERS, (combo['gamma_G'], combo['gamma_Q']))
                piles_res, Fxy, _ = pf.get_pile_reactions_detailed()
                if not piles_res:
                    continue

                if combo['type'] == 'ULS':
                    N_max = max(p['N_design'] for p in piles_res)
                    N_min = min(p['N_design'] for p in piles_res)

                    # --- 1. 轴向压/拉与局部稳定验算 ---
                    u_comp, _ = pile_ana.check_structural_strength(N_max)
                    if u_comp > 1.0:
                        total_viol += (u_comp - 1.0) * 20.0
                    record_constraint(
                        "pile_compression_strength_uls",
                        max(0.0, u_comp - 1.0),
                        weight=20.0,
                        utilization=u_comp,
                        value=N_max,
                    )

                    u_tens = 0.0
                    ratio_neg = 0.0
                    if N_min < 0:
                        u_tens, _ = pile_ana.check_tensile_strength(N_min)
                        if u_tens > 1.0:
                            total_viol += (u_tens - 1.0) * 50.0

                        ratio_neg = pile_ana.check_negative_friction(N_max, Ra_char_norm, sum(
                            l.thickness for l in eff_soil_layers if getattr(l, 'is_compressible', False)),
                                                                     eff_soil_layers, pf.piles_coords)
                        if ratio_neg > 1.0:
                            total_viol += (ratio_neg - 1.0) * 10.0
                    record_constraint(
                        "pile_tensile_strength_uls",
                        max(0.0, u_tens - 1.0),
                        weight=50.0,
                        utilization=u_tens,
                        value=abs(N_min) if N_min < 0 else 0.0,
                    )
                    record_constraint(
                        "pile_negative_friction_uls",
                        max(0.0, ratio_neg - 1.0),
                        weight=10.0,
                        utilization=ratio_neg,
                    )

                    # =========================================================
                    # ⭐ [核弹修复] P-8 压弯剪/拉弯剪的独立实体桩遍历验算
                    # =========================================================
                    fc_val, cvr = mat.get('fc', 19.1), cfg.CONFIG['pile_geo'].get('cover', 0.06) * 1000.0

                    _, _, M_per_H_uls, _ = pile_ana.calculate_horizontal_mechanics_fem_detailed(
                        1.0, 0.0, geo['L_pile'], eff_soil_layers, pf.piles_coords
                    )

                    max_u_shear = 0.0
                    max_u_bend = 0.0

                    for p_pile in piles_res:
                        N_i = p_pile['N_design']
                        H_i = p_pile.get('H_actual', Fxy / pf.N_total)
                        M_pile_i = M_per_H_uls * H_i

                        if HAS_PM:
                            Mu_cap = pm_manager.get_capacity_usage(
                                geo['D_pile'], rb['pile_main_n'], rb['pile_main_d'],
                                N_i, M_pile_i, fc_val, cvr, True
                            )
                        else:
                            Mu_cap = pile_ana._calculate_circular_Mu(N_i) * 1000.0

                        u_sh_i, u_b_i, _ = pile_ana.check_bending_shear_capacity(M_pile_i, H_i, N_i, Mu_cap)

                        max_u_shear = max(max_u_shear, u_sh_i)
                        max_u_bend = max(max_u_bend, u_b_i)

                    if max_u_shear > 1.0: total_viol += (max_u_shear - 1.0) * 20.0
                    if max_u_bend > 1.0: total_viol += (max_u_bend - 1.0) * 20.0
                    record_constraint(
                        "pile_shear_uls",
                        max(0.0, max_u_shear - 1.0),
                        weight=20.0,
                        utilization=max_u_shear,
                    )
                    record_constraint(
                        "pile_bending_uls",
                        max(0.0, max_u_bend - 1.0),
                        weight=20.0,
                        utilization=max_u_bend,
                    )

                    # --- 2. 承台 C-1 ~ C-5 结构验算 ---
                    d1_mm = rb['d1'] * 1000.0
                    cap_flexure = cap_ana.check_flexural_reinforcement(piles_res, rb)
                    cap_shear = cap_ana.check_inclined_section_shear(piles_res, d1_mm)
                    cap_corner = cap_ana.check_corner_pile_punching(piles_res, geo['D_pile'])
                    cap_pedestal = cap_ana.check_pedestal_punching(piles_res, d1_mm)
                    total_viol += cap_flexure * 10.0
                    total_viol += cap_shear * 10.0
                    total_viol += cap_corner * 10.0
                    total_viol += cap_pedestal * 10.0
                    record_constraint("cap_flexure_uls", cap_flexure, weight=10.0)
                    record_constraint("cap_shear_uls", cap_shear, weight=10.0)
                    record_constraint("cap_corner_punching_uls", cap_corner, weight=10.0)
                    record_constraint("cap_pedestal_punching_uls", cap_pedestal, weight=10.0)

                # =========================================================
                # ⭐ [终极修复] 彻底重构 SLS 正常使用极限状态验算
                # =========================================================
                elif combo['type'] in ['SLS', 'SLS_Normal', 'SLS_Crack', 'SLS_Extreme']:
                    N_max = max(p['N_design'] for p in piles_res)
                    N_min = min(p['N_design'] for p in piles_res)
                    N_avg = sum(p['N_design'] for p in piles_res) / pf.N_total
                    k_ax = 1.25 if "Seismic" in combo['name'] else 1.0
                    reserve = 0.95

                    curr_Ra = Ra_char_seis if "Seismic" in combo['name'] else Ra_char_norm

                    if N_avg > curr_Ra * k_ax * reserve:
                        ratio_avg = N_avg / (curr_Ra * k_ax * reserve)
                        total_viol += (ratio_avg - 1) * 10.0
                    else:
                        ratio_avg = N_avg / (curr_Ra * k_ax * reserve)
                    record_constraint(
                        "pile_compression_average_sls",
                        max(0.0, ratio_avg - 1.0),
                        weight=10.0,
                        utilization=ratio_avg,
                        value=N_avg,
                    )
                    if N_max > curr_Ra * 1.2 * k_ax * reserve:
                        ratio_max = N_max / (curr_Ra * 1.2 * k_ax * reserve)
                        total_viol += (ratio_max - 1) * 10.0
                    else:
                        ratio_max = N_max / (curr_Ra * 1.2 * k_ax * reserve)
                    record_constraint(
                        "pile_compression_maximum_sls",
                        max(0.0, ratio_max - 1.0),
                        weight=10.0,
                        utilization=ratio_max,
                        value=N_max,
                    )

                    if N_min < 0:
                        if abs(N_min) > Rt_char * k_ax * reserve:
                            ratio_uplift = abs(N_min) / (Rt_char * k_ax * reserve)
                            total_viol += (ratio_uplift - 1.0) * 10.0
                        else:
                            ratio_uplift = abs(N_min) / (Rt_char * k_ax * reserve)
                        record_constraint(
                            "pile_uplift_sls",
                            max(0.0, ratio_uplift - 1.0),
                            weight=10.0,
                            utilization=ratio_uplift,
                            value=abs(N_min),
                        )

                    H_pile_sls_max = max(p.get('H_actual', Fxy / pf.N_total) for p in piles_res)

                    if H_pile_sls_max > Rh_seis * reserve:
                        ratio_horizontal = H_pile_sls_max / (Rh_seis * reserve)
                        total_viol += (ratio_horizontal - 1.0) * 10.0
                    else:
                        ratio_horizontal = H_pile_sls_max / (Rh_seis * reserve)
                    record_constraint(
                        "pile_horizontal_sls",
                        max(0.0, ratio_horizontal - 1.0),
                        weight=10.0,
                        utilization=ratio_horizontal,
                        value=H_pile_sls_max,
                    )

                    V_conc_tmp = pf.get_quantities()[0]
                    V_cyl_tmp = math.pi * geo['R_cap'] ** 2 * geo['H']
                    G_soil_tmp = max(0.0, V_cyl_tmp - V_conc_tmp) * 18.0

                    water_top, water_bot = 2.0, 5.6
                    submerged_depth = max(0.0, min(geo['H'], water_bot) - water_top)
                    buoyancy = math.pi * (geo['R_cap'] ** 2) * submerged_depth * 10.0

                    Fz_tot_sls_real = load_data['FZ'] + V_conc_tmp * 25.0 + G_soil_tmp - buoyancy

                    chk_weak = pile_ana.check_weak_underlying_layer(Fz_tot_sls_real, 0.0, geo['L_pile'],
                                                                    cfg.SOIL_LAYERS, pf.piles_coords, H_cap=geo['H'])
                    if chk_weak > 1.0:
                        total_viol += (chk_weak - 1.0) * 10.0
                    record_constraint(
                        "weak_underlying_layer_sls",
                        max(0.0, chk_weak - 1.0),
                        weight=10.0,
                        utilization=chk_weak,
                    )

                    s_val, t_val = pile_ana.check_settlement_and_tilt(Fz_tot_sls_real, load_data['MXY'], geo['L_pile'],
                                                                      cfg.SOIL_LAYERS, pf.piles_coords, H_cap=geo['H'])
                    if t_val > 0.003:
                        ratio_tilt = t_val / 0.003
                        total_viol += (ratio_tilt - 1.0) * 10.0
                    else:
                        ratio_tilt = t_val / 0.003
                    record_constraint(
                        "settlement_tilt_sls",
                        max(0.0, ratio_tilt - 1.0),
                        weight=10.0,
                        utilization=ratio_tilt,
                        value=t_val,
                        limit=0.003,
                    )

                    cap_crack = cap_ana.check_crack_width(piles_res, rb)
                    total_viol += cap_crack * 10.0
                    record_constraint("cap_crack_width_sls", cap_crack, weight=10.0)

                    # =========================================================
                    # ⭐ [核弹修复] P-9 裂缝宽度的独立实体桩遍历验算
                    # =========================================================
                    _, _, M_per_H_sls, _ = pile_ana.calculate_horizontal_mechanics_fem_detailed(
                        1.0, 0.0, geo['L_pile'], eff_soil_layers, pf.piles_coords)

                    max_w_pile = 0.0
                    for p_pile in piles_res:
                        N_i = p_pile['N_design']
                        H_i = p_pile.get('H_actual', Fxy / pf.N_total)
                        M_pile_i = M_per_H_sls * H_i

                        w_i = pile_ana.check_crack_width(M_pile_i, N_i)
                        if w_i is not None:
                            max_w_pile = max(max_w_pile, w_i)

                    if max_w_pile > 0.2:
                        ratio_crack = max_w_pile / 0.2
                        total_viol += (ratio_crack - 1.0) * 10.0
                    else:
                        ratio_crack = max_w_pile / 0.2
                    record_constraint(
                        "pile_crack_width_sls",
                        max(0.0, ratio_crack - 1.0),
                        weight=10.0,
                        utilization=ratio_crack,
                        value=max_w_pile,
                        limit=0.2,
                        unit="mm",
                    )

        if np.isnan(total_viol):
            total_viol = 100.0

    except Exception as e:
        total_viol += 100.0

    # Same physical cost formula as the main evaluator (no penalties in cost).
    costs = calculate_design_cost(params)
    Cost = costs["total_cost_rmb"] / params["mat"].get("price", 500.0)

    if return_details:
        return {
            "total_violation": float(total_viol),
            "constraints": constraint_records,
            "objectives": {"cost": float(Cost)},
            "costs": costs,
        }
    return total_viol, Cost


def calculate_design_cost(params):
    """Research direct-material estimate shared by fixed/nested and analysis.

    Includes cap/pile concrete, detailed cap/pile steel, cap auxiliary steel,
    bolts and flange estimate. Excludes installation, excavation and tax.
    """
    pf = IntegratedPileFoundation(params, cfg.CONFIG["load_cases"]["Normal"], cfg.SOIL_LAYERS)
    geo, rb = params["geo"], params["rebar"]
    cap_volume, cap_steel, _, _ = pf.get_quantities()
    pile_volume = pf.N_total * math.pi * (geo["D_pile"] / 2) ** 2 * geo["L_pile"]
    pile_steel = calculate_pile_steel_detailed(pf.N_total, geo["D_pile"], geo["L_pile"],
                                             rb["pile_main_n"], rb["pile_main_d"])
    steel_kg = cap_steel + pile_steel + cap_volume * 40.0
    concrete_cost = (cap_volume + pile_volume) * params["mat"].get("price", 500.0)
    steel_cost = steel_kg * cfg.CONFIG.get("UNIT_PRICES", {}).get("STEEL", 5500.0) / 1000
    anchor_cost = 0.0
    if params.get("anchor"):
        a, c = params["anchor"], cfg.CONFIG["ANCHOR_OPT"]
        reference_length = float(c["COST"].get("UNIT_COST_REFERENCE_LENGTH_M", 4.0))
        length_factor = float(a.get("L_total_m", c["FIXED"]["L_TOTAL_M"])) / reference_length
        anchor_cost = (2 * a["N_per_ring"] * c["BOLT_CATALOG"][a["bolt_type"]]["unit_cost_rmb"]
                       * length_factor
                       + c["COST"]["FLANGE_FACTOR"] * math.pi
                       * (a["PCD_in_m"] + a["PCD_out_m"]) / 2
                       * c["COST"]["FLANGE_COST_PER_M_RMB"])
    return {"foundation_cost_rmb": concrete_cost + steel_cost,
            "anchor_cost_rmb": anchor_cost,
            "total_cost_rmb": concrete_cost + steel_cost + anchor_cost,
            "concrete_cost_rmb": concrete_cost, "rebar_cost_rmb": steel_cost,
            "cap_concrete_m3": cap_volume, "pile_concrete_m3": pile_volume,
            "steel_kg": steel_kg}

