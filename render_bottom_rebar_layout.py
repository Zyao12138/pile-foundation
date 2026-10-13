# -*- coding: utf-8 -*-
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from layout_geometry import evaluate_rebar_groups, get_nudged_angles


def safe_flt(v): return float(v[0]) if isinstance(v, (list, tuple, np.ndarray)) else float(v)


class BottomRebarLayout:
    def __init__(self, R_cap, r_void, r_col, n1_count, d1, n2_count, d2, n3_count, d3, ach_config=None):
        # 【严格保留你的原始参数】
        self.R_cap, self.r_void, self.r_col = safe_flt(R_cap), safe_flt(r_void), safe_flt(r_col)
        self.r_n2_end = 0.7 * self.R_cap
        self.n1, self.n2, self.n3 = int(safe_flt(n1_count)), int(safe_flt(n2_count)), int(safe_flt(n3_count))
        self.d1, self.d2, self.d3 = d1, d2, d3

        if ach_config is None: ach_config = {'PCD_out': 5.2, 'PCD_in': 4.8, 'N_per_ring': 120, 'd_sleeve': 65.0}

        self.PCD_out = safe_flt(ach_config.get('PCD_out', 5.2))
        self.PCD_in = safe_flt(ach_config.get('PCD_in', 4.8))
        self.n_ring = int(safe_flt(ach_config.get('N_per_ring', 120)))
        self.d_sleeve_m = safe_flt(ach_config.get('d_sleeve', 65.0)) / 1000.0
        self.ach_config = ach_config

    def visualize(self):
        fig, ax = plt.subplots(figsize=(8, 8))

        ax.add_patch(patches.Circle((0, 0), self.R_cap, fill=False, edgecolor='black', linewidth=1.5))

        # 画双圈锚栓
        bolt_angles = [i * (2 * math.pi / self.n_ring) for i in range(self.n_ring)]
        for beta in bolt_angles:
            r_out, r_in = self.PCD_out / 2.0, self.PCD_in / 2.0
            ax.add_patch(patches.Circle((r_out * math.cos(beta), r_out * math.sin(beta)), self.d_sleeve_m / 2.0,
                                        facecolor='#E74C3C', edgecolor='#900C3F', linewidth=0.5, alpha=0.85, zorder=6))
            ax.add_patch(patches.Circle((r_in * math.cos(beta), r_in * math.sin(beta)), self.d_sleeve_m / 2.0,
                                        facecolor='#E74C3C', edgecolor='#900C3F', linewidth=0.5, alpha=0.85, zorder=6))

        # 【引入避让引擎，替换了原来的 [] 列表推导式】
        n1_angles = get_nudged_angles(
            self.n1, self.d1, self.ach_config,
            r_start=self.r_void, r_end=self.R_cap
        )
        n2_angles = get_nudged_angles(self.n2, self.d2, self.ach_config,
                                      offset_angle=(math.pi / self.n2) if self.n2 > 0 else 0,
                                      r_start=self.r_void, r_end=self.r_n2_end)
        n3_angles = get_nudged_angles(self.n3, self.d3, self.ach_config,
                                      offset_angle=(math.pi / self.n3) if self.n3 > 0 else 0,
                                      r_start=self.r_col, r_end=self.R_cap)
        layout_metrics = evaluate_rebar_groups(
            [
                {"name": "n1", "count": self.n1, "diameter": self.d1,
                 "r_start": self.r_void, "r_end": self.R_cap},
                {"name": "n2", "count": self.n2, "diameter": self.d2,
                 "r_start": self.r_void, "r_end": self.r_n2_end,
                 "offset_angle": (math.pi / self.n2) if self.n2 > 0 else 0.0},
                {"name": "n3", "count": self.n3, "diameter": self.d3,
                 "r_start": self.r_col, "r_end": self.R_cap,
                 "offset_angle": (math.pi / self.n3) if self.n3 > 0 else 0.0},
            ],
            self.ach_config,
        )

        # 严格保留你的原始笛卡尔连线绘制方式
        for tr in n1_angles:
            ax.plot([self.r_void * math.cos(tr), self.R_cap * math.cos(tr)],
                    [self.r_void * math.sin(tr), self.R_cap * math.sin(tr)], color='#1F618D', linewidth=2.0, alpha=0.85,
                    zorder=4)
        for tr in n2_angles:
            ax.plot([self.r_void * math.cos(tr), self.r_n2_end * math.cos(tr)],
                    [self.r_void * math.sin(tr), self.r_n2_end * math.sin(tr)], color='#D35400', linewidth=2.0,
                    linestyle='--', alpha=0.85, zorder=4)
        for tr in n3_angles:
            ax.plot([self.r_col * math.cos(tr), self.R_cap * math.cos(tr)],
                    [self.r_col * math.sin(tr), self.R_cap * math.sin(tr)], color='#27AE60', linewidth=1.2, alpha=0.9,
                    zorder=4)

        plt.title(
            f"Optimized Bottom Rebar with Double-Ring Anchors\n"
            f"({self.n_ring * 2} Bolts | Min Clearance: "
            f"{layout_metrics['minimum_clearance_m'] * 1000.0:.1f} mm)"
        )
        ax.set_aspect('equal')
        plt.xlim(-self.R_cap * 1.1, self.R_cap * 1.1)
        plt.ylim(-self.R_cap * 1.1, self.R_cap * 1.1)
        plt.show()
