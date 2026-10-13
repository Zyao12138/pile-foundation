# -*- coding: utf-8 -*-
"""多次运行GA取最优解, 并自动验算 + 存档"""
import sys, os, json, math, random, time
import numpy as np
import multiprocessing

PROJECT_DIR = r"C:\Users\18445\PyCharmMiscProject\pile foundation"
sys.path.insert(0, PROJECT_DIR)

import foundation_config as cfg
import optimize_foundation_ga as opt
from deap import base, creator, tools

GA_PARAMS = cfg.CONFIG["GA_PARAMS"]
N_VARS = len(opt.VAR_RANGES)

# ============ 实验参数 ============
N_TRIALS = 10          # 运行次数(建议10次, 每次约5-10分钟)
OUTPUT_DIR = os.path.join(PROJECT_DIR, "multi_run_results")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def run_one_trial(seed):
    """运行一次GA, 返回完整最优参数和成本"""
    random.seed(seed)
    np.random.seed(int(seed))
    opt.ALLOW_TENSION = True
    opt.FIXED_PILE_LENGTH = None

    nc = min(16, os.cpu_count())
    pool = multiprocessing.Pool(nc, initializer=opt.init_worker, initargs=(True, None))
    tb = base.Toolbox()
    tb.register("population", lambda n: opt.feasible_init_population(n))
    tb.register("evaluate", opt.evaluate)
    tb.register("mate", tools.cxTwoPoint)
    tb.register("map", pool.map)
    los = [r[0] for r in opt.VAR_RANGES]
    ups = [r[1]+1e-9 if r[1] <= r[0] else r[1] for r in opt.VAR_RANGES]
    tb.register("mutate", tools.mutPolynomialBounded, eta=GA_PARAMS.get("ETA", 10.0),
                low=los, up=ups, indpb=0.15)
    tb.register("select", tools.selTournament, tournsize=GA_PARAMS.get("TOURN_SIZE", 3))

    pop = tb.population(GA_PARAMS["POP_SIZE"])
    inv = [i for i in pop if not i.fitness.valid]
    fits = tb.map(tb.evaluate, inv)
    for i, ft in zip(inv, fits):
        i.mat_score, i.viol = ft
    for i in pop:
        if not hasattr(i, "mat_score"):
            i.mat_score = 1e10; i.viol = 1e10
        sv = min(i.viol, 1000.0)
        i.fitness.values = (i.mat_score if i.viol < 1e-4 else i.mat_score + 4000.0 + sv * 20000.0,)

    hof = tools.HallOfFame(1); hof.update(pop)
    ng, ps = GA_PARAMS["N_GEN"], GA_PARAMS["POP_SIZE"]
    er = min(0.015, GA_PARAMS.get("ELITE_RATIO", 0.05))
    ne = max(2, int(ps*er)); nm = int(ps*0.7); nb = ps - ne - nm
    hfit = []

    for g in range(1, ng+1):
        sp = tools.selBest(pop, len(pop))
        el = [tb.clone(i) for i in sp[:ne]]
        mp = tb.select(sp[:(ne+nm)], nm)
        md = [tb.clone(i) for i in mp]
        for i in range(1, len(md), 2):
            if random.random() < GA_PARAMS["CX_PROB"]:
                tb.mate(md[i-1], md[i]); del md[i-1].fitness.values; del md[i].fitness.values
        for i in range(len(md)):
            if random.random() < GA_PARAMS["MUT_PROB"]:
                tb.mutate(md[i]); del md[i].fitness.values
        bd = [creator.Individual(ind) for ind in opt.lhs_sample(N_VARS, nb)]
        pop = el + md + bd
        inv = [i for i in pop if not i.fitness.valid]
        fits = tb.map(tb.evaluate, inv)
        for i, ft in zip(inv, fits):
            i.mat_score, i.viol = ft
        prog = g/ng; df = 20000.0 + 180000.0*prog**3
        for i in pop:
            if not hasattr(i, "mat_score"):
                i.mat_score = 1e10; i.viol = 1e10
            sv = min(i.viol, 1000.0)
            i.fitness.values = (i.mat_score if i.viol < 1e-4 else i.mat_score + 4000.0 + sv*df,)
        hof.clear(); hof.update(pop)
        bi = tools.selBest(pop, 1)[0]
        cbf = min(hfit[-1], bi.fitness.values[0]) if hfit else bi.fitness.values[0]
        hfit.append(cbf)

    pool.close(); pool.join()

    best_ind = hof[0]
    params = opt.decode_gene(best_ind)
    cost = getattr(best_ind, "mat_score", 1e10)
    viol = getattr(best_ind, "viol", 1e10)

    # 简化参数以便JSON保存
    simple = {
        "geo": {k: v for k, v in params["geo"].items() if isinstance(v, (int, float, bool, str))},
        "rebar": {k: v for k, v in params["rebar"].items()
                   if isinstance(v, (int, float, bool, str))},
        "stirrup": params["stirrup"],
        "mat": {k: v for k, v in params["mat"].items() if isinstance(v, (int, float, bool, str))},
    }
    # rings 单独处理(list of dict)
    simple["geo"]["rings"] = [{"n": r["n"], "r": r["r"]} for r in params["geo"]["rings"]]
    # 嵌套dict单独补上 (之前被类型过滤掉了)
    simple["rebar"]["ring_opt"] = params["rebar"]["ring_opt"]
    simple["rebar"]["top_ring_opt"] = params["rebar"]["top_ring_opt"]

    return {"seed": seed, "cost": cost, "viol": viol, "params": simple, "history": hfit}


def convert_to_test_data(params):
    """将最优参数转为 verify_design_constraints 的 test_data 格式"""
    geo = params["geo"]
    rb = params["rebar"]
    R_cap = geo["R_cap"]; H = geo["H"]
    h1 = geo["h1"]; h2 = geo["h2"]; h3 = geo["h3"]; r_col = geo["r_col"]
    D = geo["D_pile"]; L = geo["L_pile"]

    # 桩位坐标
    coords = []
    for ring in geo["rings"]:
        n = ring["n"]; r = ring["r"]
        for k in range(n):
            theta = 2*math.pi*k/n
            coords.append((r*math.cos(theta), r*math.sin(theta)))

    # 承台配筋
    cap_rebar = {
        'n1': rb['n1'], 'd1': rb['d1'],
        'n2': rb.get('n2', rb['n1']), 'd2': rb.get('d2', rb['d1']),
        'n3': rb.get('n3', rb['n1']), 'd3': rb.get('d3', rb['d1']),
        'n4': rb['n4'], 'd4': rb['d4'],
        'n5': rb.get('n5', rb['n4']), 'd5': rb.get('d5', rb['d4']),
        'ring_opt': {'d1': rb['ring_opt']['d1'], 's1': rb['ring_opt']['s1'],
                     'd2': rb['ring_opt'].get('d2', 0), 's2': rb['ring_opt'].get('s2', 200),
                     's': rb['ring_opt'].get('s', 150)},
        'top_ring_opt': {'d': rb['top_ring_opt']['d'], 's': rb['top_ring_opt']['s']},
    }

    # 荷载
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
                       'stirrup': params['stirrup']},
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
        'V_conc': V_conc,
    }


if __name__ == "__main__":
    from verify_design_constraints import run_full_verification

    print("="*70)
    print(f"  多次优化实验: {N_TRIALS} 次运行, 取最优解验算")
    print("="*70)

    results = []
    t0 = time.time()
    for trial in range(N_TRIALS):
        seed = trial
        print(f"\n[{trial+1}/{N_TRIALS}] seed={seed}")
        tt = time.time()
        try:
            r = run_one_trial(seed)
            results.append(r)
            with open(os.path.join(OUTPUT_DIR, f"trial_{trial:02d}.json"), "w", encoding="utf-8") as f:
                json.dump(r, f, ensure_ascii=False, indent=2, default=str)
            print(f"  cost={r['cost']:.1f}  viol={r['viol']:.4f}  ({time.time()-tt:.0f}s)")
        except Exception as e:
            print(f"  [FAIL] {e}")
            import traceback; traceback.print_exc()

    # 挑选最优可行解
    feasible = [r for r in results if r["viol"] < 1e-4]
    if feasible:
        best = min(feasible, key=lambda r: r["cost"])
        print(f"\n{'='*70}")
        print(f"最优可行解: seed={best['seed']}, cost={best['cost']:.1f}")
    else:
        print("\n[警告] 未找到完全可行解, 取viol最小的")
        best = min(results, key=lambda r: r["viol"])

    # 保存汇总
    costs = [r["cost"] for r in feasible] if feasible else [r["cost"] for r in results]
    summary = {
        "n_trials": N_TRIALS,
        "n_feasible": len(feasible),
        "best_cost": best["cost"],
        "best_seed": best["seed"],
        "mean_cost": float(np.mean(costs)),
        "std_cost": float(np.std(costs)),
        "min_cost": float(np.min(costs)),
        "max_cost": float(np.max(costs)),
    }
    with open(os.path.join(OUTPUT_DIR, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"可行解 {len(feasible)}/{N_TRIALS}, 最优成本 {best['cost']:.1f}")
    print(f"成本统计: mean={summary['mean_cost']:.1f}, std={summary['std_cost']:.1f}, "
          f"min={summary['min_cost']:.1f}, max={summary['max_cost']:.1f}")

    # 验算最优解
    print(f"\n{'='*70}")
    print("最优方案验算:")
    test_data = convert_to_test_data(best["params"])
    print(f"  混凝土 Vc={test_data['V_conc']:.1f} m3")
    geo = best["params"]["geo"]
    print(f"  R={geo['R_cap']:.1f}m, H={geo['H']:.1f}m, "
          f"{sum(r['n'] for r in geo['rings'])}根D={geo['D_pile']:.1f}m桩")
    for i, ring in enumerate(geo["rings"], 1):
        print(f"  桩圈{i}: {ring['n']}根 @ R={ring['r']:.1f}m")
    print("="*70)
    run_full_verification(test_data)

    print(f"\n总耗时: {(time.time()-t0)/60:.1f} 分钟")
    print(f"结果目录: {OUTPUT_DIR}")
