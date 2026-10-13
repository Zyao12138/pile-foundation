# -*- coding: utf-8 -*-
"""Improvement 2 comparison: feasible_init vs feasible_init + adaptive"""
import sys, os, random, time
import numpy as np
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


def compute_adaptive_probs(pop):
    f_vals = []
    for ind in pop:
        if ind.fitness.valid and hasattr(ind, "mat_score"):
            viol = getattr(ind, "viol", 1e10)
            if viol < 1e-4:
                f_vals.append(ind.mat_score)
    if len(f_vals) < 2:
        return 0.7, 0.3
    f_vals = np.array(f_vals)
    f_max, f_min, f_avg = f_vals.max(), f_vals.min(), f_vals.mean()
    delta = f_max - f_min + 1e-9
    Pc = max(0.5, min(0.9, 0.85 - 0.35 * (f_max - f_avg) / delta))
    Pm_base = 0.05 + 0.25 * (1.0 - (f_avg - f_min) / delta)
    return Pc, Pm_base


def run_one(use_adaptive):
    label = "GA-II (feasible+adaptive)" if use_adaptive else "GA-I (feasible only)"
    print(f"\n>>> Running: {label}")

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

        # ---- adaptive probabilities ----
        if use_adaptive:
            Pc, Pm_base = compute_adaptive_probs(pop)
        else:
            Pc = GA_PARAMS["CX_PROB"]
            Pm_base = GA_PARAMS["MUT_PROB"]

        for i in range(1, len(md), 2):
            # Per-pair adaptive crossover
            if use_adaptive:
                fv3 = np.array([ind.fitness.values[0] for ind in sp if ind.fitness.valid])
                if len(fv3) > 2:
                    fM3, fa3 = fv3.max(), fv3.mean()
                    fp = max(md[i-1].fitness.values[0] if md[i-1].fitness.valid else fM3,
                             md[i].fitness.values[0] if md[i].fitness.valid else fM3)
                    Pc_pair = 0.9 if fp < fa3 else max(0.5, 0.9 - 0.4 * (fp - fa3) / (fM3 - fa3 + 1e-9))
                else:
                    Pc_pair = Pc
            else:
                Pc_pair = Pc

            if random.random() < Pc_pair:
                tb.mate(md[i - 1], md[i])
                del md[i - 1].fitness.values; del md[i].fitness.values

        for i in range(len(md)):
            if use_adaptive:
                fv = np.array([ind.fitness.values[0] for ind in sp if ind.fitness.valid])
                if len(fv) > 1:
                    fM, fm = fv.max(), fv.min()
                    fi = md[i].fitness.values[0] if md[i].fitness.valid else fM
                    Pm_i = max(0.05, min(0.4, 0.05 + 0.35 * (fi - fm) / (fM - fm + 1e-9)))
                else:
                    Pm_i = Pm_base
            else:
                Pm_i = Pm_base
            if random.random() < Pm_i:
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
    return {"label": label, "fitness": hfit, "vc": hvc, "ws": hws}


# ========== MAIN ==========
if __name__ == "__main__":
    print("=" * 60)
    print("  Ablation 2: GA-I (feasible only) vs GA-II (feasible + adaptive)")
    print("=" * 60)
    t0 = time.time()
    r1 = run_one(use_adaptive=False)   # GA-I
    r2 = run_one(use_adaptive=True)    # GA-II
    print(f"\nTotal time: {(time.time()-t0)/60:.1f} min")

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Panel 1: full convergence
    ax = axes[0]
    ax.plot(range(1, len(r1["fitness"]) + 1), r1["fitness"], "#3498db", ls="--", lw=2, label="GA-I (feasible only)")
    ax.plot(range(1, len(r2["fitness"]) + 1), r2["fitness"], "#e74c3c", ls="-", lw=2, label="GA-II (feasible+adaptive)")
    ax.set_xlabel("Generation"); ax.set_ylabel("Best Fitness"); ax.set_title("Convergence")
    ax.legend(); ax.grid(True, ls="--", alpha=0.5)

    # Panel 2: first 50 gens
    ax = axes[1]
    n50 = min(50, len(r1["fitness"]))
    ax.plot(range(1, n50 + 1), r1["fitness"][:n50], "#3498db", ls="--", lw=2, label="GA-I")
    ax.plot(range(1, n50 + 1), r2["fitness"][:n50], "#e74c3c", ls="-", lw=2, label="GA-II")
    ax.set_xlabel("Generation"); ax.set_ylabel("Best Fitness"); ax.set_title("First 50 Generations")
    ax.legend(); ax.grid(True, ls="--", alpha=0.5)

    # Panel 3: final cost bar
    ax = axes[2]
    x = [0, 1]; w = 0.4
    f1, f2 = r1["fitness"][-1], r2["fitness"][-1]
    ax.bar(x, [f1, f2], w, color=["#3498db", "#e74c3c"])
    ax.set_xticks(x); ax.set_xticklabels(["GA-I\n(feasible only)", "GA-II\n(+ adaptive)"])
    ax.set_ylabel("Final Best Fitness"); ax.set_title("Final Cost Comparison")
    for i, v in enumerate([f1, f2]):
        ax.text(i, v + max(f1, f2) * 0.01, f"{v:.1f}", ha="center", fontweight="bold")

    plt.tight_layout()
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "improvement_2_compare.png")
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.show()

    imp = (1 - f2 / f1) * 100
    print(f"\nGA-I  (feasible only):     final={f1:.1f}")
    print(f"GA-II (feasible+adaptive): final={f2:.1f}")
    print(f"Improvement: {imp:.1f}%")
    print(f"Saved: {out}")
