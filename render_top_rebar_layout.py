# -*- coding: utf-8 -*-
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from layout_geometry import evaluate_rebar_groups, get_nudged_angles


def safe_flt(v): return float(v[0]) if isinstance(v, (list, tuple, np.ndarray)) else float(v)


class TopRebarLayout:
    def __init__(self, R_cap, r_void, r_col, r_top, n4_count, d4, n5_count, d5, ach_config=None):
        # 【严格保留你的原始参数】
        self.R_cap, self.r_void, self.r_col, self.r_top = safe_flt(R_cap), safe_flt(r_void), safe_flt(r_col), safe_flt(
            r_top)
        self.n4, self.n5 = int(safe_flt(n4_count)), int(safe_flt(n5_count))
        self.d4, self.d5 = d4, d5

        if ach_config is None: ach_config = {'PCD_out': 5.2, 'PCD_in': 4.8, 'N_per_ring': 120, 'd_sleeve': 65.0}

        self.PCD_out = safe_flt(ach_config.get('PCD_out', 5.2))
        self.PCD_in = safe_flt(ach_config.get('PCD_in', 4.8))
        self.n_ring = int(safe_flt(ach_config.get('N_per_ring', 120)))
        self.d_sleeve_m = safe_flt(ach_config.get('d_sleeve', 65.0)) / 1000.0
        self.ach_config = ach_config

    def visualize(self):
        fig, ax = plt.subplots(figsize=(8, 8))

        # 画承台和预留孔
        ax.add_patch(patches.Circle((0, 0), self.R_cap, fill=False, edgecolor='black', linewidth=1.5))
        ax.add_patch(patches.Circle((0, 0), self.r_col, fill=False, edgecolor='blue', linestyle='--', linewidth=1.0))

        # 画锚栓
        bolt_angles = [i * (2 * math.pi / self.n_ring) for i in range(self.n_ring)]
        for beta in bolt_angles:
            r_out, r_in = self.PCD_out / 2.0, self.PCD_in / 2.0
            ax.add_patch(patches.Circle((r_out * math.cos(beta), r_out * math.sin(beta)), self.d_sleeve_m / 2.0,
                                        facecolor='#E74C3C', edgecolor='#900C3F', linewidth=0.5, alpha=0.85, zorder=6))
            ax.add_patch(patches.Circle((r_in * math.cos(beta), r_in * math.sin(beta)), self.d_sleeve_m / 2.0,
                                        facecolor='#E74C3C', edgecolor='#900C3F', linewidth=0.5, alpha=0.85, zorder=6))

        # 【仅替换此处的生成角度，引入避让算法】
        n4_angles = get_nudged_angles(
            self.n4, self.d4, self.ach_config,
            r_start=self.r_void, r_end=self.R_cap
        )
        n5_angles = get_nudged_angles(self.n5, self.d5, self.ach_config,
                                      offset_angle=(math.pi / self.n4) if self.n4 > 0 else 0,
                                      r_start=self.r_col, r_end=self.R_cap)
        layout_metrics = evaluate_rebar_groups(
            [
                {"name": "n4", "count": self.n4, "diameter": self.d4,
                 "r_start": self.r_void, "r_end": self.R_cap},
                {"name": "n5", "count": self.n5, "diameter": self.d5,
                 "r_start": self.r_col, "r_end": self.R_cap,
                 "offset_angle": (math.pi / self.n4) if self.n4 > 0 else 0.0},
            ],
            self.ach_config,
        )

        # 严格保留你的直角坐标系绘制方法
        for tr in n4_angles:
            ax.plot([self.r_void * math.cos(tr), self.R_cap * math.cos(tr)],
                    [self.r_void * math.sin(tr), self.R_cap * math.sin(tr)], color='#76448A', linewidth=1.8, alpha=0.9,
                    zorder=5)
        for tr in n5_angles:
            ax.plot([self.r_col * math.cos(tr), self.R_cap * math.cos(tr)],
                    [self.r_col * math.sin(tr), self.R_cap * math.sin(tr)], color='#2874A6', linewidth=1.5, alpha=0.85,
                    zorder=5)

        plt.title(
            f"Optimized Top Rebar with Double-Ring Anchors\n"
            f"({self.n_ring * 2} Bolts | Min Clearance: "
            f"{layout_metrics['minimum_clearance_m'] * 1000.0:.1f} mm)"
        )
        ax.set_aspect('equal')
        plt.xlim(-self.R_cap * 1.1, self.R_cap * 1.1)
        plt.ylim(-self.R_cap * 1.1, self.R_cap * 1.1)
        plt.show()
