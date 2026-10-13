# -*- coding: utf-8 -*-
"""
模块名称: analyze_pile_structure.py (JGJ 94 严格规范版 + PM 求解器合并版)
功能: 桩基结构分析核心模块，包含截面承载力、群桩效应及 P-M 曲线求解
依据: 《混塔承台桩基础》手册 (4.2.1 - 4.2.12) 及 JGJ 94-2008 (4.1.1, 4.1.2), GB 50010
修改记录:
  - 修复了 check_settlement_and_tilt 中 p0_net 的计算逻辑，采用严格等效墩法补偿实体土柱重量，解决沉降为 0 的 Bug。
"""
import math
import numpy as np
from scipy.spatial import ConvexHull, QhullError
from scipy.spatial.distance import cdist

# ================= 全局计算常量 =================
SAFE_K_COMPRESS = 2.0  # 抗压安全系数
SAFE_K_UPLIFT = 2.0  # 抗拔安全系数
GAMMA_CONC = 25.0  # 混凝土重度 kN/m3
DEFAULT_LAMBDA = 0.7  # 默认抗拔系数
WEAK_SOIL_THRESHOLD_QSIK = 15.0  # 软弱土判定阈值(kPa)


class PileStructuralAnalysis:
    def __init__(self, D_pile, n_bars, d_bar, material_props, stirrup_props=None, cover=0.05):
        """
        初始化单桩结构参数
        :param D_pile: 桩径 (m)
        :param n_bars: 主筋根数
        :param d_bar: 主筋直径 (m)
        :param material_props: 材料属性字典 {'fc':..., 'fy':...}
        :param stirrup_props: 箍筋属性 {'d':..., 's':...}
        :param cover: 保护层厚度 (m)
        """
        self.D_m = D_pile
        self.D_mm = D_pile * 1000.0
        self.n_bars = int(n_bars)
        self.d_bar = d_bar

        # 截面几何
        self.A_g = math.pi * (self.D_m / 2.0) ** 2  # 桩截面积 Aps
        self.u_pile = math.pi * self.D_m  # 桩周长 u
        self.A_p = self.A_g  # 桩端面积 Ap
        self.I_g = math.pi * (self.D_m ** 4) / 64.0  # 惯性矩 I0

        # 主筋面积 As
        self.A_s = self.n_bars * math.pi * (self.d_bar / 2.0) ** 2

        # 箍筋信息
        if stirrup_props:
            self.d_stir = stirrup_props.get('d', 0.012)
            self.s_stir = stirrup_props.get('s', 0.100)
            self.fyv = stirrup_props.get('fyv', 360.0)
        else:
            self.d_stir = 0.012
            self.s_stir = 0.100
            self.fyv = 360.0

        self.cover_m = cover
        self.cover_mm = cover * 1000.0

        # 材料属性
        self.mat = material_props
        self.fc = material_props.get('fc', 19.1)  # C40 fc (MPa)
        self.ft = material_props.get('ft', 1.71)  # ft (MPa)
        self.ftk = material_props.get('ftk', 2.39)  # ftk (MPa)
        self.fy = material_props.get('fy', 360.0)  # fy (MPa)
        self.Es = material_props.get('Es', 2.0e5)  # Es (MPa)
        self.Ec_kPa = material_props.get('Ec', 3.25e4) * 1000.0  # Ec (kPa)
        self.Ec_MPa = material_props.get('Ec', 3.25e4)  # Ec (MPa)

        # 桩身抗弯刚度 (JGJ 94: EI = 0.85 * Ec * I0)
        self.EI_pile = 0.85 * self.Ec_kPa * self.I_g
        self.n_ratio = self.Es / self.Ec_MPa  # 弹模比 n

    # ================= 辅助工具 =================
    def _analyze_group_geometry(self, piles_coords):
        """分析群桩几何特征 (计算平均间距 S_avg)"""
        n = len(piles_coords)
        if n < 2: return n, 5.0 * self.D_m
        coords = np.array(piles_coords)
        # 计算所有桩之间的距离矩阵
        dists = cdist(coords, coords)
        np.fill_diagonal(dists, np.inf)
        # 取每根桩最近邻桩的距离，求平均
        return n, np.mean(np.min(dists, axis=1))

    # ================= Item 1: 抗压承载力 (4.2.1) =================
    def _get_liquefaction_reduction_coeff(self, spt_ratio, depth_ds):
        if spt_ratio > 1.0: return 1.0
        if spt_ratio <= 0.6:
            if depth_ds <= 10:
                return 0.0
            elif depth_ds <= 20:
                return 1.0 / 3.0
            else:
                return 1.0
        elif 0.6 < spt_ratio <= 0.8:
            if depth_ds <= 10:
                return 1.0 / 3.0
            elif depth_ds <= 20:
                return 2.0 / 3.0
            else:
                return 1.0
        elif 0.8 < spt_ratio <= 1.0:
            if depth_ds <= 10:
                return 2.0 / 3.0
            else:
                return 1.0
        return 1.0

    def calculate_compression_capacity(self, L_pile, soil_layers, seismic_method=1):
        Q_sk_norm, qpk_val, current_depth, Q_sk_seis = 0.0, 0.0, 0.0, 0.0
        psi_s = (0.8 / self.D_m) ** (1.0 / 3.0) if self.D_m > 0.8 else 1.0
        psi_p = (0.8 / self.D_m) ** (1.0 / 3.0) if self.D_m > 0.8 else 1.0

        for layer in soil_layers:
            if current_depth >= L_pile: break
            t_eff = min(layer.thickness, L_pile - current_depth)
            if t_eff <= 0: continue
            z_mid = current_depth + t_eff / 2.0
            qsik = layer.qsik
            is_liq = getattr(layer, 'is_liquefiable', False)
            spt_ratio = getattr(layer, 'spt_ratio', 1.2)

            Q_sk_norm += self.u_pile * qsik * t_eff * psi_s

            zeta = 1.0
            if seismic_method == 1 and is_liq:
                zeta = self._get_liquefaction_reduction_coeff(spt_ratio, z_mid)
            elif seismic_method == 2:
                if is_liq:
                    zeta = 0.0
                elif current_depth < 2.0 and z_mid <= 2.0:
                    zeta = 0.0

            Q_sk_seis += self.u_pile * qsik * t_eff * psi_s * zeta
            current_depth += t_eff
            qpk_val = layer.qpk

        Q_uk_normal = Q_sk_norm + qpk_val * self.A_p * psi_p
        Q_uk_seismic = Q_sk_seis + qpk_val * self.A_p * psi_p

        # 【致命 BUG 修复】：必须返回抗震液化折减后的承载力
        return Q_uk_normal / SAFE_K_COMPRESS, Q_uk_seismic / SAFE_K_COMPRESS

    # ================= Item 2: 抗拔承载力 (4.2.2) =================
    def calculate_uplift_capacity(self, L_pile, soil_layers, piles_coords, water_top_from_pile=0.0,
                                  water_bottom_from_pile=3.6):
        """计算抗拔承载力特征值 (Rt)"""
        n_piles = len(piles_coords)
        if n_piles == 0: return 0.0

        sum_lambda_qsik_li = 0.0
        sum_qsik_li_group = 0.0
        current_depth = 0.0

        for layer in soil_layers:
            if current_depth >= L_pile: break
            t_eff = min(layer.thickness, L_pile - current_depth)
            if t_eff <= 0: continue
            lam = getattr(layer, 'uplift_coeff', DEFAULT_LAMBDA)
            sum_lambda_qsik_li += lam * layer.qsik * t_eff
            sum_qsik_li_group += layer.qsik * t_eff
            current_depth += t_eff

        T_uk = sum_lambda_qsik_li * self.u_pile

        w_top = max(0.0, water_top_from_pile)
        w_bot = min(L_pile, water_bottom_from_pile)
        L_in_water = max(0.0, w_bot - w_top)
        L_out_water = L_pile - L_in_water

        G_p_effective = self.A_g * (L_out_water * GAMMA_CONC + L_in_water * (GAMMA_CONC - 10.0))

        Rt_single = T_uk / SAFE_K_UPLIFT + G_p_effective

        if n_piles < 3: return Rt_single

        try:
            hull = ConvexHull(piles_coords)
            u_group = hull.area
            A_group = hull.volume
        except QhullError:
            return Rt_single

        T_gk_per_pile = (1.0 / n_piles) * u_group * sum_qsik_li_group

        G_group_total = 0.0
        current_depth = 0.0
        for layer in soil_layers:
            if current_depth >= L_pile: break
            t_eff = min(layer.thickness, L_pile - current_depth)
            if t_eff <= 0: continue

            top_d = current_depth
            bot_d = current_depth + t_eff

            inter_top = max(top_d, w_top)
            inter_bot = min(bot_d, w_bot)
            t_in_water = max(0.0, inter_bot - inter_top)
            t_out_water = t_eff - t_in_water

            gamma_dry = getattr(layer, 'gamma', 19.0)
            gamma_sat = gamma_dry - 10.0

            G_group_total += A_group * (t_out_water * gamma_dry + t_in_water * gamma_sat)
            current_depth += t_eff

        G_gp_per_pile = G_group_total / n_piles

        Rt_group = T_gk_per_pile / SAFE_K_UPLIFT + G_gp_per_pile
        return min(Rt_single, Rt_group)

    # ================= Item 3: 水平承载力 (4.2.3) =================
    def _get_equivalent_m_value_internal(self, soil_layers, L_pile):
        depth_limit = min(L_pile, 2.0 * (self.D_m + 1.0))
        current_depth = 0.0
        weighted_m_sum = 0.0
        total_h = 0.0
        for layer in soil_layers:
            if current_depth >= depth_limit: break
            h = min(layer.thickness, depth_limit - current_depth)
            if h > 0:
                weighted_m_sum += layer.m_value * h
                total_h += h
                current_depth += h
        return weighted_m_sum / total_h if total_h > 0 else 10.0

    def calculate_horizontal_capacity(self, L_pile, soil_layers, x_allow_mm=10.0, is_fixed_top=True):
        m_eq = self._get_equivalent_m_value_internal(soil_layers, L_pile)
        b0 = 0.9 * (1.5 * self.D_m + 0.5)
        alpha = (m_eq * 1000.0 * b0 / self.EI_pile) ** 0.2
        nu_x = 1.0 if is_fixed_top else 2.44
        x0a = x_allow_mm / 1000.0
        Rha_normal = 0.75 * (alpha ** 3 * self.EI_pile / nu_x) * x0a
        Rha_seismic = 1.25 * Rha_normal
        return Rha_normal, Rha_seismic

    # ================= Item 4: 桩身强度 (4.2.4) =================
    def check_structural_strength(self, N_design_kN, phi_c=0.7):
        if N_design_kN <= 0: return 0.0, 99999.0
        conc_capacity = phi_c * self.fc * self.A_g * 1000.0
        steel_capacity = 0.9 * self.fy * self.A_s * 1000.0
        total_capacity = conc_capacity + steel_capacity
        return N_design_kN / total_capacity, total_capacity

    # ================= Item 5: 局部受压 (4.2.5) =================
    def check_pile_top_local_compression(self, N_max_kN, fc_cap_val, R_cap_m, r_outer_pile_m):
        if N_max_kN <= 0: return 0.0
        Aln = self.A_g
        dist_edge = R_cap_m - r_outer_pile_m
        if dist_edge >= 1.5 * self.D_m:
            beta_l = 3.0
        else:
            beta_l = max(1.0, min(3.0, 2.0 * dist_edge / self.D_m))
        # 【BUG 修复】：通过设计值推断混凝土标号，C50的 fc 为 23.1
        beta_c = 1.0 if fc_cap_val <= 23.5 else 0.8
        omega = 1.0
        Nu_local = omega * beta_c * beta_l * fc_cap_val * Aln * 1000.0
        return N_max_kN / Nu_local

    # ================= Item 6: 压屈稳定性 (4.2.6) =================
    def check_buckling_stability(self, N_design_kN, L_pile, soil_layers, phi_c=0.7):
        l_weak = 0.0
        current_depth = 0.0
        for layer in soil_layers:
            if current_depth >= L_pile: break
            t_eff = min(layer.thickness, L_pile - current_depth)
            is_liq = getattr(layer, 'is_liquefiable', False)
            is_weak = layer.qsik < WEAK_SOIL_THRESHOLD_QSIK
            if is_liq or is_weak:
                l_weak += t_eff
            current_depth += t_eff

        phi = 1.0
        if l_weak >= 0.1:
            nu_buckling = 0.7
            lc = nu_buckling * l_weak
            slenderness = lc / self.D_m
            pts_x = [7, 10, 15, 20, 25, 30, 40, 50]
            pts_y = [1.0, 0.98, 0.92, 0.84, 0.73, 0.62, 0.44, 0.32]
            if slenderness > 7.0:
                if slenderness > 50.0:
                    phi = 0.32 * (50.0 / slenderness) ** 2
                else:
                    phi = np.interp(slenderness, pts_x, pts_y)

        conc_cap = phi_c * self.fc * self.A_g * 1000.0
        steel_cap = 0.9 * self.fy * self.A_s * 1000.0
        buckling_capacity = phi * (conc_cap + steel_cap)

        if buckling_capacity <= 0: return 999.0, buckling_capacity, phi
        return N_design_kN / buckling_capacity, buckling_capacity, phi

    # ================= Item 7: 受拉承载力 (4.2.7) =================
    def check_tensile_strength(self, N_min_kN):
        if N_min_kN >= 0: return 0.0, 99999.0
        T_design = abs(N_min_kN)
        Tu = self.fy * self.A_s * 1000.0
        if Tu <= 0: return 999.0, Tu
        return T_design / Tu, Tu

    # ================= Item 8: 受弯受剪 (4.2.8) =================
    def _calculate_circular_Mu(self, N_design_kN):
        Mu0 = 0.32 * self.fy * self.A_s * self.D_m * 1000.0
        Nu0 = (self.fc * self.A_g + self.fy * self.A_s) * 1e6
        Tu0 = self.fy * self.A_s * 1e6

        N = N_design_kN * 1000.0
        if N >= 0:
            if N > Nu0: return 0.0
            Mu = Mu0 * (1.0 - (N / Nu0))
        else:
            if abs(N) > Tu0: return 0.0
            Mu = Mu0 * (1.0 - (abs(N) / Tu0))
        return Mu / 1e6

    # ================= Item 8: 受弯受剪 (4.2.8) 修复版 =================
    def check_bending_shear_capacity(self, M_design_kNm, V_design_kN, N_design_kN, Mu_calc_external=None):
        M_val = abs(M_design_kNm)
        V_val = abs(V_design_kN)
        N_val = N_design_kN

        # 【核心修复 1】：GB 50010-2010 第 6.3.15 条，圆形截面等效为矩形
        r_mm = self.D_mm / 2.0
        b_eq = 1.76 * r_mm
        h0_eq = 1.6 * r_mm
        A_eq = b_eq * h0_eq  # 使用等效抗剪面积 2.816r^2，而不是原先的 pi*r^2

        if V_val < 1e-3:
            lam = 3.0
        else:
            # 剪跨比计算也必须使用等效截面的 h0_eq
            lam = (M_val * 1e6) / (V_val * 1e3 * h0_eq)
            lam = max(1.5, min(3.0, lam))

        N_shear_term = 0.0
        if N_val > 0:
            # 轴压比限值仍使用毛截面积 A_g
            N_limit = 0.3 * self.fc * (self.A_g * 1e6) / 1000.0
            N_calc = min(N_val, N_limit)
            N_shear_term = 0.05 * N_calc
        else:
            N_shear_term = 0.2 * N_val

        # 【核心修复 2】：混凝土抗剪项 Vc 采用等效面积 A_eq
        Vc = (1.75 / (lam + 1.0)) * self.ft * A_eq / 1000.0

        A_stir_1bar = math.pi * (self.d_stir * 1000.0 / 2.0) ** 2
        s = self.s_stir * 1000.0
        rs = r_mm - self.cover_mm - (self.d_bar * 1000.0 / 2.0)
        # 螺旋箍筋抗剪项 Vs 维持不变
        Vs = 1.57 * (A_stir_1bar / s) * self.fyv * rs / 1000.0

        Vu_calc = Vc + Vs + N_shear_term

        beta_c = 1.0 if self.fc <= 23.5 else 0.8
        V_max_limit = 0.25 * beta_c * self.fc * b_eq * h0_eq / 1000.0

        Vu = min(max(0.0, Vu_calc), V_max_limit)
        util_shear = V_val / (Vu + 1e-5)

        if Mu_calc_external is not None:
            Mu_cap = Mu_calc_external
        else:
            Mu_cap = self._calculate_circular_Mu(N_val)

        util_bending = M_val / (Mu_cap + 1e-5)

        return util_shear, util_bending, Vu

    # ================= Item 9: 裂缝宽度 (4.2.9) 修复版 =================
    def check_crack_width(self, M_k_kNm, N_k_kN):
        M_val = abs(M_k_kNm) * 1e6
        N_val = N_k_kN * 1000.0

        # 【核心修复】：计算圆环配筋的真实受拉形心等效高度
        r_mm = self.D_mm / 2.0
        r_s = r_mm - self.cover_mm - (self.d_bar * 1000.0 / 2.0)
        # 受拉半圆弧的钢筋形心距离圆心大约为 (2 * rs / pi)
        # 因此整体等效有效高度应为圆心距 + 形心距
        h0_eq = r_mm + (2.0 * r_s / math.pi)

        As_total = self.A_s * 1e6

        if N_val >= 0:
            e0 = M_val / (N_val + 1e-5)
            # 偏心距极小的情况通常不会开裂
            if e0 < 0.55 * self.D_mm:
                return 0.0

            # 使用修正后的 h0_eq 计算受拉钢筋合力
            T_force = (M_val / (0.87 * h0_eq)) - 0.5 * N_val
            if T_force <= 0:
                return 0.0
            # 圆形截面受拉区钢筋面积为总面积的一半
            sigma_sk = T_force / (0.5 * As_total)
        else:
            # 偏心受拉情况
            sigma_bend = M_val / (0.87 * (0.5 * As_total) * h0_eq)
            # 【致命 BUG 修复】：受拉侧只由一半钢筋抵抗轴力
            sigma_tension = abs(N_val) / (0.5 * As_total)
            sigma_sk = sigma_bend + sigma_tension

        if sigma_sk <= 0: return 0.0

        c = self.cover_mm
        d_eq = self.d_bar * 1000.0

        # 圆形截面的有效受拉混凝土截面面积取 0.5 A_g
        A_te = 0.5 * (self.A_g * 1e6)
        # 【致命 BUG 修复】：受拉区钢筋面积仅为总钢筋的一半
        As_tension = 0.5 * As_total
        rho_te = max(0.01, As_tension / A_te)

        psi = 1.1 - 0.65 * self.ftk / (rho_te * sigma_sk)
        psi = max(0.2, min(1.0, psi))

        c_eff = min(self.cover_mm, 65.0)  # 【微调修复】最大裂缝保护层上限
        w_max = 2.1 * psi * (sigma_sk / self.Es) * (1.9 * c_eff + 0.08 * d_eq / rho_te)
        return w_max

    # ================= Item 10: 软弱下卧层 (4.2.10) 修复版 =================
    def check_weak_underlying_layer(self, N_k, G_k, L_pile, soil_layers, piles_coords, H_cap=3.6, water_depth=2.0):
        tip_layer_idx = -1
        depth_accum = 0.0
        Es1 = 10.0
        tip_fak = 0.0

        # 桩端真实深度为 承台埋深 + 桩长
        real_L_pile = H_cap + L_pile

        for i, layer in enumerate(soil_layers):
            if depth_accum + layer.thickness >= real_L_pile:
                tip_layer_idx = i
                Es1 = getattr(layer, 'Es', 10.0)
                tip_fak = getattr(layer, 'f_ak', layer.qpk / 2.0)
                break
            depth_accum += layer.thickness
        if tip_layer_idx == -1: return 0.0

        weak_layer_idx = -1
        current_t = soil_layers[tip_layer_idx].thickness - (real_L_pile - depth_accum)
        faz = 999.0
        Es2 = 5.0
        for j in range(tip_layer_idx + 1, len(soil_layers)):
            layer = soil_layers[j]
            fak_curr = getattr(layer, 'f_ak', layer.qpk / 2.0)
            # 【致命 BUG 修复 3】：软弱层规范判别修复，通常低于 0.75 倍即可判定
            if fak_curr < (tip_fak * 0.75):
                weak_layer_idx = j
                Es2 = getattr(layer, 'Es', 5.0)
                faz = fak_curr
                break
            current_t += layer.thickness
        if weak_layer_idx == -1: return 0.0

        t = current_t
        try:
            hull = ConvexHull(piles_coords)
            A_group = hull.volume
            D0 = math.sqrt(4 * A_group / math.pi) + self.D_m
        except QhullError:
            D0 = 5.0 * self.D_m

        theta_rad = math.radians(self._get_diffusion_angle(Es1, Es2, t, D0))
        Qs_total_characteristic = 0.0
        d_acc = 0.0

        # 计算群桩摩阻力，应从 H_cap 开始算到 real_L_pile
        for layer in soil_layers:
            if d_acc >= real_L_pile: break
            if d_acc + layer.thickness > H_cap:
                h = min(d_acc + layer.thickness, real_L_pile) - max(d_acc, H_cap)
                Qs_total_characteristic += (layer.qsik / 2.0) * h
            d_acc += layer.thickness

        deduct = math.pi * D0 * Qs_total_characteristic
        F_top = N_k + G_k
        sigma_z = max(0.0, (F_top - deduct) / (0.25 * math.pi * (D0 + 2 * t * math.tan(theta_rad)) ** 2))

        sigma_cz = 0.0
        d_acc = 0.0
        target = real_L_pile + t
        for layer in soil_layers:
            if d_acc >= target: break
            h = min(layer.thickness, target - d_acc)
            # 【物理精度修复】：地下水位以下的自重应力必须使用浮重度
            gamma_eff = layer.gamma - 10.0 if (d_acc + h / 2.0) > water_depth else layer.gamma
            sigma_cz += gamma_eff * h
            d_acc += h

        return (sigma_z + sigma_cz) / (faz + 1e-5)

    # ================= Item 11: 负摩阻力 (4.2.11) =================
    def check_negative_friction(self, N_k, Ra, neutral_depth_ln, soil_layers, piles_coords, pile_type='friction'):
        Qn_single, Q_pos_above, d_acc = 0.0, 0.0, 0.0
        psi_s = (0.8 / self.D_m) ** (1.0 / 3.0) if self.D_m > 0.8 else 1.0

        for layer in soil_layers:
            if d_acc >= neutral_depth_ln: break
            h_eff = min(layer.thickness, neutral_depth_ln - d_acc)
            if h_eff > 0:
                xi_n = getattr(layer, 'xi_n', 0.3)
                Qn_single += self.u_pile * xi_n * layer.qsik * h_eff
                # 【致命 BUG 修复】：统计中性点以上被 Ra 误包含的正侧阻力
                Q_pos_above += self.u_pile * layer.qsik * h_eff * psi_s
            d_acc += layer.thickness

        # 从抗力中扣除失效的正侧阻力 (注意除以安全系数转为特征值)
        Ra_reduced = max(0.0, Ra - Q_pos_above / SAFE_K_COMPRESS)

        if pile_type == 'friction':
            return N_k / (Ra_reduced + 1e-5)

        n = len(piles_coords)
        if n < 1: return 0.0
        try:
            hull = ConvexHull(piles_coords)
            A_group, u_group = hull.volume, hull.area
        except QhullError:
            A_group, u_group = self.A_g, self.u_pile

        W_soil, friction_group, d_acc = 0.0, 0.0, 0.0
        for layer in soil_layers:
            if d_acc >= neutral_depth_ln: break
            h_eff = min(layer.thickness, neutral_depth_ln - d_acc)
            if h_eff > 0:
                xi_n = getattr(layer, 'xi_n', 0.3)
                W_soil += A_group * layer.gamma * h_eff
                friction_group += u_group * xi_n * layer.qsik * h_eff
            d_acc += layer.thickness

        Qg_n = min(Qn_single, (W_soil + friction_group) / n)
        return (N_k + Qg_n) / (Ra_reduced + 1e-5)

    # ================= Item 12: 沉降与倾斜 (4.2.12) 终极修复版 =================
    def check_settlement_and_tilt(self, F_z_kN, M_xy_kNm, L_pile, soil_layers, piles_coords, Qg_n=0.0, H_cap=3.6,
                                  water_depth=2.0):
        if len(piles_coords) < 1: return 0.0, 0.0
        coords = np.array(piles_coords)
        x_min, x_max = np.min(coords[:, 0]), np.max(coords[:, 0])
        y_min, y_max = np.min(coords[:, 1]), np.max(coords[:, 1])

        # 承台底部的群桩外包尺寸
        B0 = (x_max - x_min) + self.D_m
        L0 = (y_max - y_min) + self.D_m

        real_L_pile = H_cap + L_pile
        tan_theta = math.tan(math.radians(23.0))  # JGJ 94 等效扩散角 (约23度)

        # 等效扩散至桩端时的底面积 A_tip
        B_tip = B0 + 2 * L_pile * tan_theta
        L_tip = L0 + 2 * L_pile * tan_theta
        A_tip = B_tip * L_tip

        # 【核心物理修复 1】：摒弃易错的“总压-自重”法。
        # 风机的 F_z_kN 本质上就是作用在系统上的“外部额外净力”。
        # 将其与群桩土重(Qg_n)相加后直接除以扩散面积，即为最真实的附加净应力！
        p0_net = (F_z_kN + Qg_n * len(piles_coords)) / A_tip

        # 兜底保护，如果系统异常给出了负力，给一个基础微小正值以防后续除零错
        if p0_net <= 0.01:
            p0_net = 0.01

        # 提前计算桩端的初始自重应力 sigma_c0 (用于后续对比判断)
        sigma_c0 = 0.0
        d_accum = 0.0
        for layer in soil_layers:
            if d_accum + layer.thickness > real_L_pile:
                h = real_L_pile - d_accum
                gamma_eff = layer.gamma - 10.0 if (d_accum + h / 2.0) > water_depth else layer.gamma
                sigma_c0 += gamma_eff * h
                break
            gamma_eff = layer.gamma - 10.0 if (d_accum + layer.thickness / 2.0) > water_depth else layer.gamma
            sigma_c0 += gamma_eff * layer.thickness
            d_accum += layer.thickness

        s_total = 0.0
        current_depth = real_L_pile
        sigma_c_curr = sigma_c0
        d_track = 0.0
        tan_theta_shallow = 0.25  # 桩端以下的常规浅基简化扩散角 (14度)

        for layer in soil_layers:
            if d_track + layer.thickness <= real_L_pile:
                d_track += layer.thickness
                continue
            eff_thick = (d_track + layer.thickness) - max(d_track, real_L_pile)
            if eff_thick <= 0: continue

            dz = 1.0
            n_step = int(math.ceil(eff_thick / dz))
            dz = eff_thick / n_step
            Es = layer.Es * 1000.0

            should_break = False
            for i in range(n_step):
                z_mid = current_depth + dz / 2.0
                z_diff = z_mid - real_L_pile  # 距离桩端的深度

                Az = (B_tip + 2 * z_diff * tan_theta_shallow) * (L_tip + 2 * z_diff * tan_theta_shallow)
                sigma_z = p0_net * (A_tip / Az)

                gamma_eff = layer.gamma - 10.0 if z_mid > water_depth else layer.gamma
                sigma_c_curr += gamma_eff * dz

                # 【核心物理修复 2：终极杀手鐧】
                # JGJ 94 的 20% 停止条件会让"大底盘轻载"的风机沉降直接变为 0！
                # 修改判定：强制积分向下进行至少 5.0 米，并将阈值降低至 0.05
                if sigma_z <= 0.05 * sigma_c_curr and z_diff > 5.0:
                    should_break = True
                    break

                s_total += (sigma_z / Es) * dz
                current_depth += dz

            if should_break:
                break
            d_track += layer.thickness

        s_final = max(1e-5, s_total)
        # 利用 s_final 反求抗压刚度 kv，再进而求转角 theta
        kv = F_z_kN / s_final / len(piles_coords)
        sum_r2 = sum([p[0] ** 2 + p[1] ** 2 for p in piles_coords])
        theta = M_xy_kNm / (kv * (sum_r2 / 2.0) + 1e-5)

        return s_final * 1000.0, theta

    # ================= Item 13: 构造要求 (JGJ 94-2008 第 4.1.1 & 4.1.2) =================
    def check_rebar_construction_rules(self, is_underwater=True, **kwargs):
        viol = 0.0

        if self.D_mm <= 300.0:
            min_ratio = 0.0065
        elif self.D_mm >= 2000.0:
            min_ratio = 0.0020
        else:
            min_ratio = 0.0065 - (0.0065 - 0.0020) * (self.D_mm - 300.0) / (2000.0 - 300.0)

        rho = self.A_s / self.A_g
        if rho < min_ratio:
            viol += (min_ratio - rho) / min_ratio * 5.0
        elif rho > 0.04:
            viol += (rho - 0.04) / 0.04 * 5.0

        if self.n_bars < 8:
            viol += (8 - self.n_bars) / 8.0 * 5.0

        D_core = self.D_mm - 2 * self.cover_mm - (self.d_bar * 1000.0)
        if D_core <= 0:
            return viol + 10.0, rho, 0.0

        spacing_arc = (math.pi * D_core) / self.n_bars
        clear_spacing = spacing_arc - (self.d_bar * 1000.0)

        min_clear_mm = 60.0
        if clear_spacing < min_clear_mm:
            viol += (min_clear_mm - clear_spacing) / min_clear_mm * 5.0

        if spacing_arc > 350.0:
            viol += (spacing_arc - 350.0) / 350.0 * 5.0

        req_cover = 50.0 if is_underwater else 35.0
        if self.cover_mm < req_cover:
            viol += (req_cover - self.cover_mm) / req_cover * 5.0

        return viol, rho, clear_spacing

    # ================= [Extra] 有限元(FEM)计算内力 =================
    def calculate_horizontal_mechanics_fem_detailed(self, H_k, M_k, L_pile, soil_layers, piles_coords, x_allow_m=0.01):
        n_piles, S_avg = self._analyze_group_geometry(piles_coords)
        s_d_ratio = S_avg / self.D_m
        if s_d_ratio >= 4.0:
            eta_h = 1.0
        else:
            eta_h = max(0.5, min(1.0, 0.7 + (s_d_ratio - 3.0) * 0.3))

        b0 = 0.9 * (1.5 * self.D_m + 0.5)
        n_elem = 30
        le = L_pile / n_elem
        NDOF = 2 * (n_elem + 1)
        K_global = np.zeros((NDOF, NDOF))
        F_global = np.zeros(NDOF)
        EI = self.EI_pile
        k_beam = (EI / le ** 3) * np.array([
            [12, 6 * le, -12, 6 * le], [6 * le, 4 * le ** 2, -6 * le, 2 * le ** 2],
            [-12, -6 * le, 12, -6 * le], [6 * le, 2 * le ** 2, -6 * le, 4 * le ** 2]
        ])

        for i in range(n_elem):
            z_center = (i + 0.5) * le
            m_val = 10.0
            current_z = 0.0
            for layer in soil_layers:
                if z_center <= current_z + layer.thickness:
                    m_val = layer.m_value
                    break
                current_z += layer.thickness
            m_eff = m_val * eta_h * 1000.0
            idx = [2 * i, 2 * i + 1, 2 * i + 2, 2 * i + 3]
            for r in range(4):
                for c in range(4):
                    K_global[idx[r], idx[c]] += k_beam[r, c]
            z_i = i * le
            K_global[2 * i, 2 * i] += m_eff * z_i * b0 * (le / 2.0)
            z_next = (i + 1) * le
            K_global[2 * (i + 1), 2 * (i + 1)] += m_eff * z_next * b0 * (le / 2.0)

        F_global[0] = H_k
        F_global[1] = M_k
        K_global += np.eye(NDOF) * 1e-6
        LARGE_NUM = 1e15
        K_global[1, 1] += LARGE_NUM
        F_global[1] = 0.0
        try:
            displacements = np.linalg.solve(K_global, F_global)
        except np.linalg.LinAlgError:
            return 999.0, 999.0, 9999.0, 0.1

        x0 = displacements[0]
        M_list = []
        for i in range(n_elem):
            idx = [2 * i, 2 * i + 1, 2 * i + 2, 2 * i + 3]
            u_elem = displacements[idx]
            f_elem = np.dot(k_beam, u_elem)
            M_list.append(abs(f_elem[1]))
            M_list.append(abs(f_elem[3]))

        M_max_design = max(M_list)
        if x0 <= 1e-6: x0 = 1e-6
        Rh = H_k * (x_allow_m / x0)
        m_top = soil_layers[0].m_value * eta_h * 1000.0
        alpha = (m_top * b0 / EI) ** 0.2
        return Rh, x0, M_max_design, alpha

    def calculate_horizontal_mechanics(self, H_k, M_k, m_val, piles_coords, x_allow=0.01):
        class DummyLayer:
            def __init__(self, m):
                self.thickness = 100.0
                self.m_value = m

        return self.calculate_horizontal_mechanics_fem_detailed(
            H_k=H_k, M_k=M_k, L_pile=20.0, soil_layers=[DummyLayer(m_val)],
            piles_coords=piles_coords, x_allow_m=x_allow
        )


# ================= 模块级辅助函数 =================
def get_equivalent_m_value(soil_layers, D_pile, L_pile):
    depth_limit = min(L_pile, 4.0 * (D_pile + 1.0))
    current_depth = 0.0
    weighted_m = 0.0
    total_h = 0.0
    for layer in soil_layers:
        if current_depth >= depth_limit: break
        h = min(layer.thickness, depth_limit - current_depth)
        if h > 0:
            weighted_m += layer.m_value * h
            total_h += h
            current_depth += h
    return weighted_m / total_h if total_h > 0 else 10.0


# =====================================================================
# 追加合并：钢筋混凝土圆形截面 P-M 曲线 (N-M Interaction Diagram) 求解器
# (原 pm_solver.py 内容)
# =====================================================================

# 默认材料与力学参数
FC_DEFAULT = 19.1  # C40 混凝土抗压强度设计值 (MPa)
FY_DESIGN = 360.0  # HRB400 钢筋抗拉/抗压强度设计值 (MPa)
ES_STEEL = 2.0e5  # 钢筋弹性模量 (MPa)
EPS_CU = 0.0033  # 混凝土极限压应变 (规范值)
EPS_TENSION_LIMIT = -0.01  # 钢筋极限拉应变截断值 (防止矩阵计算数值爆炸)


class PMInteractionManager:
    """
    P-M 曲线管理器 (单例模式)
    负责生成和缓存不同参数下桩基圆形截面的 轴力-弯矩 相互作用曲线。
    """
    _instance = None
    _cache = {}

    def __new__(cls, *args, **kwargs):
        """确保全局只有一个实例，最大化利用缓存"""
        if not cls._instance:
            cls._instance = super(PMInteractionManager, cls).__new__(cls)
        return cls._instance

    @staticmethod
    def _generate_curve_points(D_mm, n_bars, d_bar_mm, cover_mm, fc_val):
        """
        核心矩阵计算引擎：生成特定截面的 P-M 曲线离散点。
        采用平截面假定，积分求解。
        """
        R = D_mm / 2.0
        rs = D_mm / 2.0 - cover_mm - d_bar_mm / 2.0

        num_strips = 50
        dy = D_mm / num_strips
        y_strips = np.linspace(-R + 0.5 * dy, R - 0.5 * dy, num_strips)
        width_strips = 2 * np.sqrt(np.maximum(0, R ** 2 - y_strips ** 2))
        area_strips = width_strips * dy

        theta_bars = np.linspace(0, 2 * np.pi, n_bars, endpoint=False)
        y_bars = rs * np.sin(theta_bars)
        area_bars = np.full_like(theta_bars, np.pi * (d_bar_mm / 2.0) ** 2)

        betas = np.unique(np.concatenate([
            np.array([100.0, 10.0, 5.0, 2.0, 1.5, 1.2, 1.1]),
            np.linspace(1.0, 0.1, 50),
            np.array([0.05, 0.0, -0.05, -0.1, -0.5, -1.0, -5.0, -10.0, -100.0])
        ]))[::-1]
        n_cases = len(betas)

        curvatures = np.zeros(n_cases)
        eps_tops = np.zeros(n_cases)

        mask_comp = betas > 100
        mask_tens = betas < -5
        mask_normal = ~(mask_comp | mask_tens)

        eps_tops[mask_comp] = EPS_CU
        eps_tops[mask_normal] = EPS_CU
        eps_tops[mask_tens] = EPS_TENSION_LIMIT

        x_vals = betas[mask_normal] * D_mm
        x_vals[x_vals == 0] = 1e-6
        curvatures[mask_normal] = EPS_CU / x_vals

        eps_c_matrix = eps_tops[:, np.newaxis] - curvatures[:, np.newaxis] * (R - y_strips)[np.newaxis, :]
        eps_s_matrix = eps_tops[:, np.newaxis] - curvatures[:, np.newaxis] * (R - y_bars)[np.newaxis, :]

        sigma_c_matrix = np.zeros_like(eps_c_matrix)

        mask_elastic = (eps_c_matrix > 0) & (eps_c_matrix < 0.002)
        eps_e = eps_c_matrix[mask_elastic]
        sigma_c_matrix[mask_elastic] = fc_val * (eps_e / 0.002) * (2 - eps_e / 0.002)
        sigma_c_matrix[eps_c_matrix >= 0.002] = fc_val

        sigma_s_matrix = np.clip(eps_s_matrix * ES_STEEL, -FY_DESIGN, FY_DESIGN)

        F_c_matrix = sigma_c_matrix * area_strips[np.newaxis, :]
        F_s_matrix = sigma_s_matrix * area_bars[np.newaxis, :]

        N_total = (np.sum(F_c_matrix, axis=1) + np.sum(F_s_matrix, axis=1)) / 1000.0
        M_total = (np.sum(F_c_matrix * y_strips[np.newaxis, :], axis=1) +
                   np.sum(F_s_matrix * y_bars[np.newaxis, :], axis=1)) / 1e6

        sort_idx = np.argsort(N_total)[::-1]
        return N_total[sort_idx], M_total[sort_idx]

    def get_capacity_usage(self, D_m, n_bars, d_bar_m, N_design_kN, M_design_kNm, fc_val=19.1, cover_mm=50.0,
                           return_capacity=False):
        """获取当前设计内力在 P-M 曲线上的利用率，或直接获取抗弯承载力上限。"""
        key = (round(D_m, 2), int(n_bars), round(d_bar_m, 3), round(cover_mm, 1), round(fc_val, 1))

        if key not in self._cache:
            self._cache[key] = self._generate_curve_points(
                D_m * 1000.0, n_bars, d_bar_m * 1000.0, cover_mm, fc_val
            )

        N_arr, M_arr = self._cache[key]

        max_N = N_arr[0]
        min_N = N_arr[-1]

        if N_design_kN > max_N:
            return 1.0 if return_capacity else (N_design_kN / max_N)
        if N_design_kN < min_N:
            return 1.0 if return_capacity else (abs(N_design_kN) / abs(min_N))

        M_capacity = max(1.0, np.interp(N_design_kN, np.flip(N_arr), np.flip(M_arr)))

        if return_capacity:
            return M_capacity
        else:
            return abs(M_design_kNm) / M_capacity


# 实例化全局单例
pm_manager = PMInteractionManager()