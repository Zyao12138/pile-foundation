# -*- coding: utf-8 -*-
"""将最优方案喂入 run_full_verification, 输出17项约束利用率"""
import sys, math
sys.path.insert(0, r"C:\Users\18445\PyCharmMiscProject\pile foundation")

import foundation_config
from verify_design_constraints import run_full_verification, generate_ring_coords

# ============ 最优方案参数 (第二次优化结果) ============
R_cap = 11.0
H = 2.4
h1 = 0.9
h2 = 1.6
h3 = 1.1
r_col = 3.6
D_pile = 0.9
L_pile = 25.0

# 桩圈配置 (来自优化输出: 桩圈1 12根@6.5m, 桩圈2 26根@10.0m)
N_INNER, R_INNER = 12, 6.5
N_OUTER, R_OUTER = 26, 10.0

# 承台配筋
cap_rebar = {
    'n1': 180, 'd1': 0.022,      # 底板通长筋 180Φ22
    'n2': 180, 'd2': 0.032,      # 底板加强筋 180Φ32
    'n3': 180, 'd3': 0.022,      # 底板外插筋 180Φ22
    'n4': 120, 'd4': 0.025,      # 顶板通长筋 120Φ25
    'n5': 120, 'd5': 0.025,      # 顶板外插筋 120Φ25
    'ring_opt': {'d1': 0.020, 's1': 150.0, 'd2': 0.020, 's2': 200.0, 's': 150.0},
    'top_ring_opt': {'d': 0.020, 's': 100.0},
}

# 桩身配筋
pile_rebar = {
    'n_bars': 16, 'd_bar': 0.018,   # 16Φ18
    'stirrup': {'d': 0.012, 's': 0.100, 'fyv': 360.0},  # Φ12@100
}

# ============ 荷载与材料 ============
load_uls_raw = foundation_config.CONFIG['load_cases']["Extreme"]
load_sls_raw = foundation_config.CONFIG['load_cases']["Normal"]

V_edge = math.pi * R_cap**2 * h1
V_slope = (1/3) * math.pi * h2 * (R_cap**2 + (r_col+0.2)**2 + R_cap*(r_col+0.2))
V_stub = math.pi * r_col**2 * h3
V_conc = V_edge + V_slope + V_stub

V_cyl = math.pi * R_cap**2 * H
V_soil = max(0.0, V_cyl - V_conc)
G_soil = V_soil * 18.0
water_top, water_bot = 2.0, 5.6
submerged = max(0.0, min(H, water_bot) - water_top)
buoyancy = math.pi * R_cap**2 * submerged * 10.0
CAP_WEIGHT = V_conc * 25.0 + G_soil - buoyancy
H_arm = h1 + h2 + h3

test_data = {
    'cap_geo': {
        'R_cap': R_cap, 'r_col': r_col, 'h1': h1, 'h2': h2, 'h3': h3,
        'D_pile': D_pile, 'H': H,
    },
    'cap_material': {'FT_DESIGN': 1.71, 'FC_DESIGN': 19.1, 'FT_STD': 2.39,
                     'FY_STEEL': 360.0, 'ES_STEEL': 2.0e5, 'COVER': 50.0},
    'cap_rebar': cap_rebar,
    'pile_geo': {'D': D_pile, 'L': L_pile, 'cover': 0.06},
    'pile_rebar': pile_rebar,
    'pile_material': {'fc': 19.1, 'ft': 1.71, 'ftk': 2.39, 'fy': 360.0,
                      'Es': 2.0e5, 'Ec': 3.25e4},
    'loads_uls': {
        'F_z': load_uls_raw['FZ'] * 1.35 + CAP_WEIGHT * 1.35,
        'M_xy': load_uls_raw['MXY'] * 1.5 + load_uls_raw['FXY'] * 1.5 * H_arm,
        'H_k': load_uls_raw['FXY'] * 1.5,
        'M_z': load_uls_raw.get('Mz', 0.0) * 1.5,
    },
    'loads_sls': {
        'F_z': load_sls_raw['FZ'] * 1.0 + CAP_WEIGHT * 1.0,
        'M_xy': load_sls_raw['MXY'] * 1.0 + load_sls_raw['FXY'] * 1.0 * H_arm,
        'H_k': load_sls_raw['FXY'] * 1.0,
        'M_z': load_sls_raw.get('Mz', 0.0) * 1.0,
    },
    'piles_coords': generate_ring_coords(N_INNER, R_INNER) + generate_ring_coords(N_OUTER, R_OUTER),
    'soil_layers': foundation_config.SOIL_LAYERS,
}

print("=" * 70)
print(f"最优方案验算: R={R_cap}m, H={H}m, {N_INNER+N_OUTER}根D={D_pile}m桩")
print(f"桩圈: 内圈{N_INNER}根@R={R_INNER}m, 外圈{N_OUTER}根@R={R_OUTER}m")
print(f"混凝土 Vc={V_conc:.1f} m3")
print("=" * 70)

run_full_verification(test_data)
