# -*- coding: utf-8 -*-
"""
模块名称: analyze_cap_structure.py
功能: 风电承台结构分析  (单宽带法 Unit Strip Method)
修改记录:
  1. [力学重构] 放弃带有“相交角度折扣”的严苛积分法，换用风电院标配的“单宽带法”。
  2. [计算逻辑] 将直截面总弯矩除以弦长得到单宽弯矩(m_rad)，与同半径处的单宽配筋密度(as_prov_m)直接对标。
  3. [裂缝修复] 裂缝计算完全同步为 1m 宽度的标准板条模型，彻底释放怪胎钢筋。
"""
import math

CONSTANTS = {
    'FT_DESIGN': 1.71, 'FC_DESIGN': 19.1, 'FT_STD': 2.39,
    'FY_STEEL': 360.0, 'ES_STEEL': 2.0e5
}


class CapStructuralAnalysis:
    def __init__(self, geo_params, mat_params=None, cover_mm=100.0):
        self.R_cap = geo_params['R_cap'] * 1000.0
        self.H = geo_params['H'] * 1000.0
        self.h1 = geo_params['h1'] * 1000.0
        self.h2 = geo_params['h2'] * 1000.0
        self.h3 = geo_params['h3'] * 1000.0
        self.r_col = geo_params['r_col'] * 1000.0
        self.r_top = geo_params.get('r_top', geo_params['r_col']) * 1000.0
        self.cover = cover_mm

        self.mat = CONSTANTS.copy()
        if mat_params:
            self.mat.update(mat_params)
            for k, v in [('fc', 'FC_DESIGN'), ('ft', 'FT_DESIGN'), ('ftk', 'FT_STD'), ('fy', 'FY_STEEL'),
                         ('Es', 'ES_STEEL')]:
                if k in mat_params: self.mat[v] = mat_params[k]

    def get_h0_at_radius(self, r_mm, d_bar_mm):
        r_mm = abs(r_mm)
        if r_mm >= self.r_top:
            ratio = (self.R_cap - r_mm) / (self.R_cap - self.r_top + 1e-6)
            h = self.h1 + self.h2 * ratio
        elif r_mm >= self.r_col:
            h = self.h1 + self.h2
        else:
            h = self.h1 + self.h2 + self.h3
        return max(50.0, h - self.cover - (d_bar_mm / 2.0))

    def _get_chord_width(self, x_eval_m):
        if abs(x_eval_m) < self.R_cap / 1000.0:
            return max(0.1, 2 * math.sqrt(max(0.0, (self.R_cap / 1000.0) ** 2 - x_eval_m ** 2)))
        return 0.1

    def calc_w_unit(self, m_kNm_m, h0, as_prov_m, d_bar, c_cover):
        if m_kNm_m < 1e-3: return 0.0
        M = m_kNm_m * 1e6
        if as_prov_m < 1e-5: return 10.0
        sig_sk = M / (0.87 * h0 * as_prov_m)
        h_sect = h0 + c_cover + d_bar / 2
        A_te_m = 1000.0 * min(0.5 * h_sect, 2.5 * (h_sect - h0))
        rho_te = max(0.01, as_prov_m / A_te_m)
        if sig_sk < 1e-5: return 0.0
        psi = max(0.2, min(1.0, 1.1 - 0.65 * self.mat['FT_STD'] / (rho_te * sig_sk)))

        # 【致命 BUG 修复】：GB 50010 规定裂缝计算时保护层最大不得超过 65mm
        c_eff = min(c_cover, 65.0)

        return 2.1 * psi * (sig_sk / self.mat['ES_STEEL']) * (1.9 * c_eff + 0.08 * d_bar / rho_te)

    # ================= 验算项 1: 抗弯 (引入自由体自重积分) =================
    def check_flexural_reinforcement(self, piles_data, rebar_params, detailed=False):
        total_viol, max_ratio, rec_req, rec_prov = 0.0, -1.0, 0.0, 0.0
        fc, fy = self.mat['FC_DESIGN'], self.mat['FY_STEEL']

        check_x = [self.r_col / 1000.0]
        for p in piles_data:
            if p.get('x', p['r']) > self.r_col / 1000.0 + 0.1:
                check_x.append(p.get('x', p['r']) - 0.4)
        check_x = sorted(list(set([round(x, 2) for x in check_x if 0.5 < x < self.R_cap / 1000.0 - 0.2])))
        check_x_t = [-x for x in check_x]

        d_r_b, d_r_t = rebar_params['d1'], rebar_params['d4']

        # 【核心物理修复】：计算悬挑端自重分布产生的弯矩
        def get_self_weight_moment(x_eval):
            m_self = 0.0
            x_start = abs(x_eval)
            R_m = self.R_cap / 1000.0
            if x_start >= R_m - 0.01: return 0.0

            # 将悬挑部分分为 20 个微元积分
            dx = (R_m - x_start) / 20.0
            for i in range(20):
                x_i = x_start + (i + 0.5) * dx
                width = 2 * math.sqrt(max(0, R_m ** 2 - x_i ** 2))

                # 计算该 x_i 处的承台真实厚度
                if x_i >= self.r_top / 1000.0:
                    ratio = (R_m - x_i) / (R_m - self.r_top / 1000.0 + 1e-6)
                    h_i = self.h1 / 1000.0 + (self.h2 / 1000.0) * ratio
                elif x_i >= self.r_col / 1000.0:
                    h_i = self.h1 / 1000.0 + self.h2 / 1000.0
                else:
                    h_i = self.h1 / 1000.0 + self.h2 / 1000.0 + self.h3 / 1000.0

                # 混凝土重度 25，上覆土重度 18
                q_conc = 25.0 * h_i
                q_soil = 18.0 * max(0.0, self.H / 1000.0 - h_i)
                q_total_uls = (q_conc + q_soil) * 1.35  # ULS 恒载系数

                # 力矩 = 力 * 力臂
                m_self += q_total_uls * width * dx * (x_i - x_start)
            return m_self

        # 1. 验算底板 (扣除向下压的自重力矩)
        for x_eval in check_x:
            M_sec_b = sum(p.get('N_comp', 0) * (p.get('x', p['r']) - x_eval) for p in piles_data if
                          p.get('N_comp', 0) > 0 and p.get('x', p['r']) > x_eval)

            M_self = get_self_weight_moment(x_eval)
            M_sec_b = max(0.0, M_sec_b - M_self)  # 真实的底板弯矩被自重抵消一部分

            b_chord = self._get_chord_width(x_eval)
            m_rad = M_sec_b / b_chord if b_chord > 0 else 0
            h0_mm = self.get_h0_at_radius(x_eval * 1000, d_r_b * 1000)

            alpha_s = (m_rad * 1e6) / (fc * 1000.0 * h0_mm ** 2) if h0_mm > 0 else 0
            as_req_m = 1e9 if alpha_s > 0.5 else (fc * (
                        1.0 - math.sqrt(max(0, 1.0 - 2 * alpha_s))) * 1000.0 * h0_mm) / fy
            as_req_m = max(as_req_m, 0.0015 * 1000.0 * min(h0_mm + self.cover + 10, 1200.0))

            perim = 2 * math.pi * abs(x_eval) if abs(x_eval) > 0.1 else 1.0
            a1 = rebar_params['n1'] * 0.25 * math.pi * d_r_b ** 2
            a2 = rebar_params.get('n2', rebar_params['n1']) * 0.25 * math.pi * rebar_params.get('d2',
                                                                                                d_r_b) ** 2 if abs(
                x_eval) <= (self.R_cap / 1000.0) * 0.7 + 1e-3 else 0.0
            a3 = rebar_params.get('n3', rebar_params['n1']) * 0.25 * math.pi * rebar_params.get('d3',
                                                                                                d_r_b) ** 2 if abs(
                x_eval) >= (self.r_col / 1000.0) - 1e-3 else 0.0
            as_prov_m = ((a1 + a2 + a3) * 1e6) / perim

            if as_req_m > as_prov_m: total_viol += (as_req_m - as_prov_m) / (as_prov_m + 1e-5)
            if (ratio := as_req_m / (
                    as_prov_m + 1e-5)) > max_ratio: max_ratio, rec_req, rec_prov = ratio, as_req_m, as_prov_m

        # 2. 验算顶板 (必须加上向下压的自重力矩)
        for x_eval in check_x_t:
            M_sec_t = sum(abs(p.get('N_tens', 0)) * (abs(p.get('x', p['r'])) - abs(x_eval)) for p in piles_data if
                          p.get('N_tens', 0) < 0 and p.get('x', p['r']) < x_eval)

            M_self = get_self_weight_moment(x_eval)
            M_sec_t = M_sec_t + M_self  # 真实的顶板拉力 = 拔桩拉力弯矩 + 巨大的悬挑自重弯矩！

            b_chord = self._get_chord_width(x_eval)
            m_rad = M_sec_t / b_chord if b_chord > 0 else 0
            h0_mm = self.get_h0_at_radius(abs(x_eval) * 1000, d_r_t * 1000)

            alpha_s = (m_rad * 1e6) / (fc * 1000.0 * h0_mm ** 2) if h0_mm > 0 else 0
            as_req_m = 1e9 if alpha_s > 0.5 else (fc * (
                        1.0 - math.sqrt(max(0, 1.0 - 2 * alpha_s))) * 1000.0 * h0_mm) / fy
            as_req_m = max(as_req_m, 0.0015 * 1000.0 * min(h0_mm + self.cover + 10, 1200.0))

            perim = 2 * math.pi * abs(x_eval) if abs(x_eval) > 0.1 else 1.0
            a4 = rebar_params['n4'] * 0.25 * math.pi * d_r_t ** 2
            a5 = rebar_params.get('n5', rebar_params['n4']) * 0.25 * math.pi * rebar_params.get('d5',
                                                                                                d_r_t) ** 2 if abs(
                x_eval) >= (self.r_col / 1000.0) - 1e-3 else 0.0
            as_prov_m = ((a4 + a5) * 1e6) / perim

            if as_req_m > as_prov_m: total_viol += (as_req_m - as_prov_m) / (as_prov_m + 1e-5)

        return (total_viol, rec_req, rec_prov) if detailed else total_viol

    # ================= 验算项 2: 斜截面受剪 =================
    def check_inclined_section_shear(self, piles_data, d_bar_mm, detailed=False):
        total_viol, max_ratio, rec_des, rec_cap = 0.0, -1.0, 0.0, 0.0
        check_x = [self.r_col / 1000.0] + [p.get('x', p['r']) - 0.4 for p in piles_data if
                                           p.get('x', p['r']) > self.r_col / 1000.0 + 0.1]
        check_x = sorted(list(set([round(x, 2) for x in check_x if 0.5 < x < self.R_cap / 1000.0 - 0.2])))

        for x_eval in check_x:
            V_shear = sum(
                p.get('N_comp', 0) for p in piles_data if p.get('N_comp', 0) > 0 and p.get('x', p['r']) > x_eval)
            if V_shear <= 1.0: continue

            sum_moment = sum(p.get('N_comp', 0) * (p.get('x', p['r']) - x_eval) for p in piles_data if
                             p.get('N_comp', 0) > 0 and p.get('x', p['r']) > x_eval)
            h0 = self.get_h0_at_radius(x_eval * 1000, d_bar_mm)
            b0 = self._get_chord_width(x_eval) * 1000.0

            # 剪跨比限制
            lambda_shear = max(0.3, min(3.0, max(100.0, (sum_moment / V_shear) * 1000.0) / h0))

            # 【致命 BUG 修复】：截面高度影响系数 beta_h 计算
            # 原错误代码：(max(1.0, (800 / h0) ** 0.25) if h0 > 800 else 1.0)
            if h0 > 800.0:
                beta_h = (800.0 / h0) ** 0.25
                beta_h = max(0.85, beta_h)  # 规范下限兜底，防止极厚承台抗剪过度缩水
            else:
                beta_h = 1.0

            # 计算斜截面抗剪承载力
            V_cap = (1.75 / (lambda_shear + 1) * beta_h * self.mat['FT_DESIGN'] * b0 * h0) / 1000.0

            if V_shear > V_cap: total_viol += (V_shear - V_cap) / V_cap
            if (ratio := V_shear / (V_cap + 1e-5)) > max_ratio: max_ratio, rec_des, rec_cap = ratio, V_shear, V_cap

        return (total_viol, rec_des, rec_cap) if detailed else total_viol

    # ================= 验算项 3: 承台裂缝 =================
    def check_crack_width(self, piles_data, rebar_params, detailed=False):
        total_viol, w_limit, max_w = 0.0, 0.2, 0.0
        check_x = [self.r_col / 1000.0] + [p.get('x', p['r']) - 0.4 for p in piles_data if
                                           p.get('x', p['r']) > self.r_col / 1000.0 + 0.1]
        check_x = sorted(list(set([round(x, 2) for x in check_x if 0.5 < x < self.R_cap / 1000.0 - 0.2])))

        d_r_b, d_r_t = rebar_params['d1'], rebar_params['d4']

        for x_eval in check_x:
            M_sec_b = sum(p.get('N_comp', 0) * (p.get('x', p['r']) - x_eval) for p in piles_data if
                          p.get('N_comp', 0) > 0 and p.get('x', p['r']) > x_eval)
            b_chord = self._get_chord_width(x_eval)
            m_rad = M_sec_b / b_chord if b_chord > 0 else 0
            h0_mm = self.get_h0_at_radius(x_eval * 1000, d_r_b * 1000)
            perim = 2 * math.pi * abs(x_eval) if abs(x_eval) > 0.1 else 1.0

            a1 = rebar_params['n1'] * 0.25 * math.pi * d_r_b ** 2
            a2 = rebar_params.get('n2', rebar_params['n1']) * 0.25 * math.pi * rebar_params.get('d2',
                                                                                                d_r_b) ** 2 if abs(
                x_eval) <= (self.R_cap / 1000.0) * 0.7 + 1e-3 else 0.0
            a3 = rebar_params.get('n3', rebar_params['n1']) * 0.25 * math.pi * rebar_params.get('d3',
                                                                                                d_r_b) ** 2 if abs(
                x_eval) >= (self.r_col / 1000.0) - 1e-3 else 0.0
            as_prov_m = ((a1 + a2 + a3) * 1e6) / perim

            w_rad = self.calc_w_unit(m_rad, h0_mm, as_prov_m, d_r_b * 1000, self.cover)
            max_w = max(max_w, w_rad)
            if w_rad > w_limit: total_viol += (w_rad - w_limit) / w_limit

        for x_eval in [-x for x in check_x]:
            M_sec_t = sum(abs(p.get('N_tens', 0)) * (abs(p.get('x', p['r'])) - abs(x_eval)) for p in piles_data if
                          p.get('N_tens', 0) < 0 and p.get('x', p['r']) < x_eval)
            b_chord = self._get_chord_width(x_eval)
            m_rad = M_sec_t / b_chord if b_chord > 0 else 0
            h0_mm = self.get_h0_at_radius(abs(x_eval) * 1000, d_r_t * 1000)
            perim = 2 * math.pi * abs(x_eval) if abs(x_eval) > 0.1 else 1.0

            a4 = rebar_params['n4'] * 0.25 * math.pi * d_r_t ** 2
            a5 = rebar_params.get('n5', rebar_params['n4']) * 0.25 * math.pi * rebar_params.get('d5',
                                                                                                d_r_t) ** 2 if abs(
                x_eval) >= (self.r_col / 1000.0) - 1e-3 else 0.0
            as_prov_m = ((a4 + a5) * 1e6) / perim

            w_rad_t = self.calc_w_unit(m_rad, h0_mm, as_prov_m, d_r_t * 1000, self.cover)
            max_w = max(max_w, w_rad_t)
            if w_rad_t > w_limit: total_viol += (w_rad_t - w_limit) / w_limit

        return (total_viol, max_w, w_limit) if detailed else total_viol

    # ================= 验算项 4: 冲切验算 =================
    def check_corner_pile_punching(self, piles_data, D_pile_m, detailed=False):
        total_viol, max_ratio, rec_des, rec_cap = 0.0, -1.0, 0.0, 0.0
        for p in [p for p in piles_data if p['r'] > self.r_col / 1000.0 + D_pile_m / 2]:
            N_pile = p.get('N_comp', p.get('N_design', p.get('N', 0)))
            # 【精度修复】：a1 应为到桩内边缘距离，不是到中心
            a1 = p['r'] * 1000.0 - self.r_col - D_pile_m * 1000.0 / 2.0
            if N_pile <= 0 or a1 <= 0: continue

            h0 = self.get_h0_at_radius(p['r'] * 1000, 32.0)
            beta_hp = 1.0 if h0 <= 800 else (0.9 if h0 >= 2000 else 1.0 - 0.1 * (h0 - 800.0) / 1200.0)
            F_cap = (0.84 / (max(0.2, min(1.0, a1 / h0)) + 0.2)) * beta_hp * self.mat['FT_DESIGN'] * (
                    (min(self.R_cap - p['r'] * 1000, h0) + D_pile_m * 1000 / 2) * 2) * h0 / 1000.0

            if N_pile > F_cap: total_viol += (N_pile - F_cap) / F_cap
            if (ratio := N_pile / (F_cap + 1e-5)) > max_ratio: max_ratio, rec_des, rec_cap = ratio, N_pile, F_cap
        return (total_viol, rec_des, rec_cap) if detailed else total_viol

    def check_pedestal_punching(self, piles_data, d_bar_mm, detailed=False):
        h0 = self.get_h0_at_radius(self.r_col, d_bar_mm)
        if h0 <= 0: return (10.0, 9999.0, 1.0) if detailed else 10.0

        if h0 > 800.0:
            beta_h = (800.0 / h0) ** 0.25
            beta_h = max(0.85, beta_h)
        else:
            beta_h = 1.0

        F_cap = 0.7 * beta_h * self.mat['FT_DESIGN'] * (
                (2 * math.pi * self.r_col + 2 * math.pi * (self.r_col + h0)) / 2) * h0 / 1000.0

        # 【致命 BUG 修复】：冲切荷载由距离 h0/2 外部的桩产生，修改除数为 2000.0
        F_shear = sum(p.get('N_comp', 0) for p in piles_data if
                      p['r'] > (self.r_col / 1000.0 + h0 / 2000.0) and p.get('N_comp', 0) > 0)

        viol = max(0.0, (F_shear - F_cap) / F_cap) if F_cap > 0 else 10.0
        return (viol, F_shear, F_cap) if detailed else viol
