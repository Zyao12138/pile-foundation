# -*- coding: utf-8 -*-
"""读取已保存的trial JSON, 补默认环筋值后验算最优解(无需重跑GA)"""
import sys, os, json, math, glob
sys.path.insert(0, r"C:\Users\18445\PyCharmMiscProject\pile foundation")

import foundation_config as cfg
from verify_design_constraints import run_full_verification

RESULT_DIR = os.path.join(r"C:\Users\18445\PyCharmMiscProject\pile foundation", "multi_run_results")

# 默认环筋值 (只影响造价, 不影响17项结构验算)
DEFAULT_RING_OPT = {'d1': 0.020, 's1': 150.0, 'd2': 0.020, 's2': 200.0, 's': 150.0}
DEFAULT_TOP_RING_OPT = {'d': 0.020, 's': 100.0}


def load_best_trial():
    files = sorted(glob.glob(os.path.join(RESULT_DIR, "trial_*.json")))
    trials = []
    for f in files:
        with open(f, encoding="utf-8") as fp:
            d = json.load(fp)
        trials.append(d)
    feasible = [t for t in trials if t.get("viol", 1e10) < 1e-4]
    if not feasible:
        feasible = trials
    return min(feasible, key=lambda t: t.get("cost", 1e10))


def convert(params):
    geo = params["geo"]
    rb = params["rebar"]
    R_cap = geo["R_cap"]; H = geo["H"]
    h1 = geo["h1"]; h2 = geo["h2"]; h3 = geo["h3"]; r_col = geo["r_col"]
    D = geo["D_pile"]; L = geo["L_pile"]

    coords = []
    for ring in geo["rings"]:
        n = ring["n"]; r = ring["r"]
        for k in range(n):
            th = 2*math.pi*k/n
            coords.append((r*math.cos(th), r*math.sin(th)))

    cap_rebar = {
        'n1': rb['n1'], 'd1': rb['d1'],
        'n2': rb.get('n2', rb['n1']), 'd2': rb.get('d2', rb['d1']),
        'n3': rb.get('n3', rb['n1']), 'd3': rb.get('d3', rb['d1']),
        'n4': rb['n4'], 'd4': rb['d4'],
        'n5': rb.get('n5', rb['n4']), 'd5': rb.get('d5', rb['d4']),
        'ring_opt': rb.get('ring_opt', DEFAULT_RING_OPT),
        'top_ring_opt': rb.get('top_ring_opt', DEFAULT_TOP_RING_OPT),
    }

    load_uls_raw = cfg.CONFIG['load_cases']["Extreme"]
    load_sls_raw = cfg.CONFIG['load_cases']["Normal"]
    V_edge = math.pi*R_cap**2*h1
    V_slope = (1/3)*math.pi*h2*(R_cap**2 + (r_col+0.2)**2 + R_cap*(r_col+0.2))
    V_stub = math.pi*r_col**2*h3
    V_conc = V_edge + V_slope + V_stub
    V_cyl = math.pi*R_cap**2*H
    G_soil = max(0, V_cyl - V_conc)*18.0
    submerged = max(0, min(H, 5.6) - 2.0)
    buoyancy = math.pi*R_cap**2*submerged*10.0
    CAP_WEIGHT = V_conc*25.0 + G_soil - buoyancy
    H_arm = h1 + h2 + h3

    return {
        'cap_geo': {'R_cap': R_cap, 'r_col': r_col, 'h1': h1, 'h2': h2, 'h3': h3,
                    'D_pile': D, 'H': H},
        'cap_material': {'FT_DESIGN': 1.71, 'FC_DESIGN': 19.1, 'FT_STD': 2.39,
                         'FY_STEEL': 360.0, 'ES_STEEL': 2.0e5, 'COVER': 50.0},
        'cap_rebar': cap_rebar,
        'pile_geo': {'D': D, 'L': L, 'cover': 0.06},
        'pile_rebar': {'n_bars': rb['pile_main_n'], 'd_bar': rb['pile_main_d'],
                       'stirrup': params.get('stirrup', {'d': 0.012, 's': 0.100, 'fyv': 360.0})},
        'pile_material': {'fc': 19.1, 'ft': 1.71, 'ftk': 2.39, 'fy': 360.0,
                          'Es': 2.0e5, 'Ec': 3.25e4},
        'loads_uls': {'F_z': load_uls_raw['FZ']*1.35 + CAP_WEIGHT*1.35,
                      'M_xy': load_uls_raw['MXY']*1.5 + load_uls_raw['FXY']*1.5*H_arm,
                      'H_k': load_uls_raw['FXY']*1.5,
                      'M_z': load_uls_raw.get('Mz', 0.0)*1.5},
        'loads_sls': {'F_z': load_sls_raw['FZ']*1.0 + CAP_WEIGHT*1.0,
                      'M_xy': load_sls_raw['MXY']*1.0 + load_sls_raw['FXY']*1.0*H_arm,
                      'H_k': load_sls_raw['FXY']*1.0,
                      'M_z': load_sls_raw.get('Mz', 0.0)*1.0},
        'piles_coords': coords,
        'soil_layers': cfg.SOIL_LAYERS,
    }


if __name__ == "__main__":
    best = load_best_trial()
    print("="*70)
    print(f"读取最优可行解: seed={best['seed']}, cost={best['cost']:.1f}, viol={best['viol']:.4f}")
    geo = best["params"]["geo"]
    print(f"几何: R={geo['R_cap']:.1f}m, H={geo['H']:.1f}m, "
          f"{sum(r['n'] for r in geo['rings'])}根D={geo['D_pile']:.1f}m桩")
    for i, ring in enumerate(geo["rings"], 1):
        print(f"  桩圈{i}: {ring['n']}根 @ R={ring['r']:.1f}m")
    print("="*70)

    test_data = convert(best["params"])
    run_full_verification(test_data)
