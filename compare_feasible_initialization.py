# -*- coding: utf-8 -*-
"""改进1对比实验: 自动跑两轮并输出对比图"""
import sys, os, random, time
import matplotlib.pyplot as plt
import multiprocessing

PROJECT_DIR = r"C:\Users\18445\PyCharmMiscProject\pile foundation"
sys.path.insert(0, PROJECT_DIR)

import foundation_config as cfg
import analyze_integrated_pile_foundation as model
import optimize_foundation_ga as opt
from deap import base, creator, tools

plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False

GA_PARAMS = cfg.CONFIG["GA_PARAMS"]
N_VARS = len(opt.VAR_RANGES)


def run_one(use_feasible):
    label = "feasible_init" if use_feasible else "original_lhs"
    print(f"\n>>> Running: {label}")

    opt.ALLOW_TENSION = True
    opt.FIXED_PILE_LENGTH = None

    nc = min(16, os.cpu_count())
    pool = multiprocessing.Pool(nc, initializer=opt.init_worker, initargs=(True, None))
    tb = base.Toolbox()
    if use_feasible:
        tb.register("population", lambda n: opt.feasible_init_population(n))
    else:
        tb.register("population", lambda n: [creator.Individual(ind)
                     for ind in opt.lhs_sample(N_VARS, n)])
    tb.register("evaluate", opt.evaluate)
    tb.register("mate", tools.cxTwoPoint)
    tb.register("map", pool.map)
    los = [r[0] for r in opt.VAR_RANGES]
    ups = [r[1] for r in opt.VAR_RANGES]
    ups_safe = [u + 1e-9 if u <= l else u for l, u in zip(los, ups)]
    tb.register("mutate", tools.mutPolynomialBounded, eta=GA_PARAMS.get("ETA", 10.0),
                low=los, up=ups_safe, indpb=0.15)
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

    init_feasible = sum(1 for i in pop if hasattr(i, "viol") and i.viol < 1e-4)
    print(f"  init feasible: {init_feasible}/{GA_PARAMS['POP_SIZE']}")

    hof = tools.HallOfFame(1); hof.update(pop)
    ng, ps = GA_PARAMS["N_GEN"], GA_PARAMS["POP_SIZE"]
    er = min(0.015, GA_PARAMS.get("ELITE_RATIO", 0.05))
    ne = max(2, int(ps * er)); nm = int(ps * 0.7); nb = ps - ne - nm

    hfit, hvc, hws = [], [], []

    for g in range(1, ng + 1):
        sp = tools.selBest(pop, len(pop))
        el = [tb.clone(i) for i in sp[:ne]]
        mp = tb.select(sp[:(ne + nm)], nm)
        md = [tb.clone(i) for i in mp]
        for i in range(1, len(md), 2):
            if random.random() < GA_PARAMS["CX_PROB"]:
                tb.mate(md[i - 1], md[i])
                del md[i - 1].fitness.values; del md[i].fitness.values
        for i in range(len(md)):
            if random.random() < GA_PARAMS["MUT_PROB"]:
                tb.mutate(md[i]); del md[i].fitness.values
        bd = [creator.Individual(ind) for ind in opt.lhs_sample(N_VARS, nb)]
        pop = el + md + bd
        inv = [i for i in pop if not i.fitness.valid]
        fits = tb.map(tb.evaluate, inv)
        for i, ft in zip(inv, fits):
            i.mat_score, i.viol = ft
        prog = g / ng; df = 20000.0 + 180000.0 * prog ** 3
        for i in pop:
            if not hasattr(i, "mat_score"):
                i.mat_score = 1e10; i.viol = 1e10
            sv = min(i.viol, 1000.0)
            i.fitness.values = (i.mat_score if i.viol < 1e-4 else i.mat_score + 4000.0 + sv * df,)
        hof.clear(); hof.update(pop)
        bi = tools.selBest(pop, 1)[0]
        cbf = min(hfit[-1], bi.fitness.values[0]) if hfit else bi.fitness.values[0]
        hfit.append(cbf)
        try:
            params = opt.decode_gene(bi)
            pf = model.IntegratedPileFoundation(params, cfg.CONFIG["load_cases"]["Normal"], cfg.SOIL_LAYERS)
            Vc, Wc, _, _ = pf.get_quantities()
            Wp = model.calculate_pile_steel_detailed(pf.N_total, params["geo"]["D_pile"],
                    params["geo"]["L_pile"], params["rebar"]["pile_main_n"],
                    params["rebar"]["pile_main_d"]) if hasattr(model, "calculate_pile_steel_detailed") else 0
            hvc.append(Vc); hws.append(Wc + Wp + Vc * 40.0)
        except:
            hvc.append(0); hws.append(0)

    pool.close(); pool.join()
    return {"label": label, "fitness": hfit, "vc": hvc, "ws": hws, "init_feasible": init_feasible}


# ========== MAIN ==========
if __name__ == "__main__":
    print("=" * 60)
    print("  Ablation: original LHS vs feasible init")
    print("=" * 60)
    t0 = time.time()
    r1 = run_one(use_feasible=False)
    r2 = run_one(use_feasible=True)
    print(f"\nTotal time: {(time.time()-t0)/60:.1f} min")

    # ---- Plot ----
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Panel 1: full convergence
    ax = axes[0]
    ax.plot(range(1, len(r1["fitness"]) + 1), r1["fitness"], "#e74c3c", ls="--", lw=2, label="Original LHS")
    ax.plot(range(1, len(r2["fitness"]) + 1), r2["fitness"], "#2ecc71", ls="-", lw=2, label="Feasible Init")
    ax.set_xlabel("Generation"); ax.set_ylabel("Best Fitness"); ax.set_title("Convergence")
    ax.legend(); ax.grid(True, ls="--", alpha=0.5)

    # Panel 2: first 50 gens
    ax = axes[1]
    n50 = min(50, len(r1["fitness"]))
    ax.plot(range(1, n50 + 1), r1["fitness"][:n50], "#e74c3c", ls="--", lw=2, label="Original LHS")
    ax.plot(range(1, n50 + 1), r2["fitness"][:n50], "#2ecc71", ls="-", lw=2, label="Feasible Init")
    ax.set_xlabel("Generation"); ax.set_ylabel("Best Fitness"); ax.set_title("First 50 Generations")
    ax.legend(); ax.grid(True, ls="--", alpha=0.5)

    # Panel 3: bar chart
    ax = axes[2]
    x = [0, 1]
    w = 0.35
    f1 = r1["init_feasible"] / GA_PARAMS["POP_SIZE"] * 100
    f2 = r2["init_feasible"] / GA_PARAMS["POP_SIZE"] * 100
    ax.bar(x[0] - w/2, f1, w, color="#e74c3c", label="Init Feasible %")
    ax.bar(x[1] - w/2, f2, w, color="#2ecc71")
    # final fitness
    ff = [r1["fitness"][-1], r2["fitness"][-1]]
    ff_norm = [v / max(ff) * 100 for v in ff]
    ax.bar(x[0] + w/2, ff_norm[0], w, color="#e67e22", alpha=0.7, label="Final Fitness (norm)")
    ax.bar(x[1] + w/2, ff_norm[1], w, color="#e67e22", alpha=0.7)
    ax.set_xticks(x); ax.set_xticklabels(["Original LHS", "Feasible Init"])
    ax.set_title("Key Metrics"); ax.legend(fontsize=8)
    ax.text(x[0] - w/2, f1 + 1, f"{f1:.1f}%", ha="center", fontweight="bold")
    ax.text(x[1] - w/2, f2 + 1, f"{f2:.1f}%", ha="center", fontweight="bold")

    plt.tight_layout()
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "improvement_1_compare.png")
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.show()
    print(f"\nSaved: {out}")
    print(f"Original: init_feas={r1['init_feasible']}/{GA_PARAMS['POP_SIZE']}  final={r1['fitness'][-1]:.1f}")
    print(f"Improved: init_feas={r2['init_feasible']}/{GA_PARAMS['POP_SIZE']}  final={r2['fitness'][-1]:.1f}")
    print(f"Improvement: {(1 - r2['fitness'][-1] / r1['fitness'][-1]) * 100:.1f}%")
