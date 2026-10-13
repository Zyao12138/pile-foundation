# -*- coding: utf-8 -*-
"""Improvement 3: GA-I (feasible init) vs GA-I+SA (feasible init + SA local search)"""
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


def is_sa_better(ca, va, cb, vb):
    if va < 1e-4 and vb < 1e-4: return ca < cb
    if va < 1e-4: return True
    if vb < 1e-4: return False
    return va < vb


def sa_local_search(ind, evaluate_fn, T, steps=6):
    cur = creator.Individual(ind[:])
    del cur.fitness.values
    cc, cv = evaluate_fn(cur)
    bc, bv, best = cc, cv, creator.Individual(cur[:])
    for _ in range(steps):
        nb = creator.Individual(cur[:])
        for j in range(N_VARS):
            if random.random() < 0.35:
                lo, up = opt.VAR_RANGES[j]
                if j in opt.DISCRETE_MAP:
                    opts = opt.DISCRETE_MAP[j]
                    try: ci = opts.index(int(round(nb[j])))
                    except: ci = len(opts)//2
                    ci = max(0, min(len(opts)-1, ci + random.choice([-1,0,1])))
                    nb[j] = float(ci)
                else:
                    nb[j] = max(lo, min(up, nb[j] + random.gauss(0, 0.015*(up-lo))))
        del nb.fitness.values
        nc, nv = evaluate_fn(nb)
        if is_sa_better(nc, nv, cc, cv):
            cur, cc, cv = creator.Individual(nb[:]), nc, nv
            if is_sa_better(cc, cv, bc, bv):
                best, bc, bv = creator.Individual(nb[:]), nc, nv
        else:
            de_abs = nc - cc
            if random.random() < float(np.exp(np.clip(-de_abs/(T+1e-9), -50, 50))):
                cur, cc, cv = creator.Individual(nb[:]), nc, nv
    return best, bc, bv


def run_one(use_sa):
    label = "GA-I + SA" if use_sa else "GA-I (baseline)"
    print(f"\n>>> Running: {label}")
    opt.ALLOW_TENSION = True; opt.FIXED_PILE_LENGTH = None
    nc = min(16, os.cpu_count())
    pool = multiprocessing.Pool(nc, initializer=opt.init_worker, initargs=(True, None))
    tb = base.Toolbox()
    tb.register("population", lambda n: opt.feasible_init_population(n))
    tb.register("evaluate", opt.evaluate)
    tb.register("mate", tools.cxTwoPoint)
    tb.register("map", pool.map)
    los = [r[0] for r in opt.VAR_RANGES]
    ups_s = [r[1]+1e-9 if r[1]<=r[0] else r[1] for r in opt.VAR_RANGES]
    tb.register("mutate", tools.mutPolynomialBounded, eta=GA_PARAMS.get("ETA",10.0),
                low=los, up=ups_s, indpb=0.15)
    tb.register("select", tools.selTournament, tournsize=GA_PARAMS.get("TOURN_SIZE",3))

    pop = tb.population(GA_PARAMS["POP_SIZE"])
    inv = [i for i in pop if not i.fitness.valid]
    fits = tb.map(tb.evaluate, inv)
    for i,ft in zip(inv,fits): i.mat_score,i.viol = ft
    for i in pop:
        if not hasattr(i,"mat_score"): i.mat_score=1e10; i.viol=1e10
        sv = min(i.viol,1000.0)
        i.fitness.values = (i.mat_score if i.viol<1e-4 else i.mat_score+4000.0+sv*20000.0,)

    hof = tools.HallOfFame(1); hof.update(pop)
    ng,ps = GA_PARAMS["N_GEN"], GA_PARAMS["POP_SIZE"]
    er = min(0.015, GA_PARAMS.get("ELITE_RATIO",0.05))
    ne = max(2,int(ps*er)); nm = int(ps*0.7); nb2 = ps-ne-nm
    hfit,hvc,hws = [],[],[]

    for g in range(1,ng+1):
        sp = tools.selBest(pop,len(pop))
        el = [tb.clone(i) for i in sp[:ne]]
        mp = tb.select(sp[:(ne+nm)],nm)
        md = [tb.clone(i) for i in mp]
        for i in range(1,len(md),2):
            if random.random()<GA_PARAMS["CX_PROB"]:
                tb.mate(md[i-1],md[i]); del md[i-1].fitness.values; del md[i].fitness.values
        for i in range(len(md)):
            if random.random()<GA_PARAMS["MUT_PROB"]: tb.mutate(md[i]); del md[i].fitness.values
        bd = [creator.Individual(ind) for ind in opt.lhs_sample(N_VARS,nb2)]
        pop = el+md+bd
        inv = [i for i in pop if not i.fitness.valid]
        fits = tb.map(tb.evaluate, inv)
        for i,ft in zip(inv,fits): i.mat_score,i.viol = ft
        prog = g/ng; df = 20000.0+180000.0*prog**3
        for i in pop:
            if not hasattr(i,"mat_score"): i.mat_score=1e10; i.viol=1e10
            sv = min(i.viol,1000.0)
            i.fitness.values = (i.mat_score if i.viol<1e-4 else i.mat_score+4000.0+sv*df,)

        # ---- SA local search ----
        if use_sa:
            T0 = 80.0  # absolute temp, matches cost diff scale
            T = max(1e-6, T0*0.94**g)
            for i in range(min(len(el),50)):
                improved, bc2, bv2 = sa_local_search(el[i], opt.evaluate, T, 6)
                improved.mat_score, improved.viol = bc2, bv2
                sv2 = min(bv2,1000.0)
                improved.fitness.values = (bc2 if bv2<1e-4 else bc2+4000.0+sv2*df,)
                if improved.fitness.values[0] < el[i].fitness.values[0]:
                    el[i] = improved

        hof.clear(); hof.update(pop)
        bi = tools.selBest(pop,1)[0]
        cbf = min(hfit[-1], bi.fitness.values[0]) if hfit else bi.fitness.values[0]
        hfit.append(cbf)
        try:
            params = opt.decode_gene(bi)
            pf = model.IntegratedPileFoundation(params, cfg.CONFIG["load_cases"]["Normal"], cfg.SOIL_LAYERS)
            Vc,Wc,_,_ = pf.get_quantities()
            Wp = model.calculate_pile_steel_detailed(pf.N_total, params["geo"]["D_pile"],
                    params["geo"]["L_pile"], params["rebar"]["pile_main_n"],
                    params["rebar"]["pile_main_d"]) if hasattr(model,"calculate_pile_steel_detailed") else 0
            hvc.append(Vc); hws.append(Wc+Wp+Vc*40.0)
        except: hvc.append(0); hws.append(0)
    pool.close(); pool.join()
    return {"label":label, "fitness":hfit, "vc":hvc, "ws":hws}


if __name__ == "__main__":
    print("="*60)
    print("  Ablation 3: GA-I (feasible init) vs GA-I + SA")
    print("="*60)
    t0 = time.time()
    r1 = run_one(use_sa=False)
    r2 = run_one(use_sa=True)
    print(f"\nTotal time: {(time.time()-t0)/60:.1f} min")

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    ax = axes[0]
    ax.plot(range(1,len(r1["fitness"])+1), r1["fitness"], "#3498db", ls="--", lw=2, label=r1["label"])
    ax.plot(range(1,len(r2["fitness"])+1), r2["fitness"], "#e74c3c", ls="-", lw=2, label=r2["label"])
    ax.set_xlabel("Generation"); ax.set_ylabel("Best Fitness"); ax.set_title("Convergence")
    ax.legend(); ax.grid(True, ls="--", alpha=0.5)

    ax = axes[1]
    n50 = min(50, len(r1["fitness"]))
    ax.plot(range(1,n50+1), r1["fitness"][:n50], "#3498db", ls="--", lw=2, label=r1["label"])
    ax.plot(range(1,n50+1), r2["fitness"][:n50], "#e74c3c", ls="-", lw=2, label=r2["label"])
    ax.set_xlabel("Generation"); ax.set_ylabel("Best Fitness"); ax.set_title("First 50 Generations")
    ax.legend(); ax.grid(True, ls="--", alpha=0.5)

    ax = axes[2]
    # Show last 100 generations in detail (where SA matters most)
    n100 = min(100, len(r1["fitness"]))
    start = len(r1["fitness"]) - n100
    ax.plot(range(start+1, len(r1["fitness"])+1), r1["fitness"][start:], "#3498db", ls="--", lw=2)
    ax.plot(range(start+1, len(r2["fitness"])+1), r2["fitness"][start:], "#e74c3c", ls="-", lw=2)
    ax.set_xlabel("Generation"); ax.set_ylabel("Best Fitness"); ax.set_title("Last 100 Generations (SA matters)")
    ax.grid(True, ls="--", alpha=0.5)

    plt.tight_layout()
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "improvement_3_compare.png")
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.show()

    f1, f2 = r1["fitness"][-1], r2["fitness"][-1]
    imp = (1 - f2/f1)*100
    print(f"\nGA-I  (feasible init):    final={f1:.1f}")
    print(f"GA-I + SA:               final={f2:.1f}")
    print(f"Improvement: {imp:.1f}%")
    print(f"Saved: {out}")
