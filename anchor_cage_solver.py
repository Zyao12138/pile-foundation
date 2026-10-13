# -*- coding: utf-8 -*-
"""Deterministic inner solver for the 22+4 foundation/anchor optimization."""

import math
import json
from copy import deepcopy


_CACHE = {}
_STATS = {"calls": 0, "cache_hits": 0, "candidate_evaluations": 0}


def reset_anchor_solver_stats():
    _CACHE.clear()
    _STATS.update(calls=0, cache_hits=0, candidate_evaluations=0)


def get_anchor_solver_stats():
    return dict(_STATS)


def _bolt_spec(name, config):
    raw = config["BOLT_CATALOG"][name]
    d_mm = float(raw["d_mm"])
    pitch_mm = float(raw["pitch_mm"])
    d_eff_mm = d_mm - 0.9382 * pitch_mm
    area_mm2 = math.pi * d_eff_mm ** 2 / 4.0
    tensile_allow_kN = area_mm2 * float(config["MATERIAL"]["F_T_DESIGN_MPA"]) / 1000.0
    return {
        **raw,
        "name": name,
        "area_mm2": area_mm2,
        "tensile_allow_kN": tensile_allow_kN,
    }


def _context(params, config, load_cases):
    geo = params["geo"]
    rb = params["rebar"]
    load_extreme = load_cases["Extreme"]
    return {
        "M_kNm": float(load_extreme["MXY"]),
        "N_kN": float(load_extreme["FZ"]),
        "delta_M_kNm": float(config.get(
            "FATIGUE_DELTA_M_KNM",
            load_extreme["MXY"] * float(config.get("FATIGUE_MOMENT_RATIO", 0.10)),
        )),
        "R_cap_m": float(geo["R_cap"]),
        "r_col_m": float(geo["r_col"]),
        "cap_thickness_m": float(geo["h1"] + geo["h2"] + geo["h3"]),
        "r_void_m": float(config["CONSTRAINTS"].get("R_VOID_M", 1.5)),
        "max_rebar_d_m": max(float(rb["d1"]), float(rb.get("d2", rb["d1"])), float(rb["d4"])),
        "bottom_rebar_count": int(rb["n1"]),
        "top_rebar_count": int(rb["n4"]),
    }


def _cache_key(params, config, load_cases, mode, selection_context):
    # Exact inputs: rounding or partial keys can reuse an invalid candidate.
    return json.dumps([params, config, load_cases, mode, selection_context],
                      sort_keys=True, separators=(",", ":"))


def minimum_anchor_envelope(config, mode):
    """Optimistic existence bound, never an assumed fixed cage in nested mode."""
    if mode == "fixed":
        b = config["BASELINE"]
        return build_anchor_design(b["PCD_in_m"], b["PCD_out_m"],
                                   b["N_per_ring"], b["bolt_type"], config)["D_outer_m"]
    if mode == "nested_2":
        b = config["BASELINE"]
        allowance = 2 * config["CONSTRAINTS"].get("ENVELOPE_ALLOWANCE_M", 0)
        return min(b["PCD_out_m"] + config["BOLT_CATALOG"][name]["sleeve_d_mm"] / 1000
                   + allowance for name in config["VARIABLES"]["BOLT_TYPES"])
    c = config["CONSTRAINTS"]
    return min(max(config["VARIABLES"]["PCD_IN_MIN_M"],
                   2 * c["R_VOID_M"] + config["BOLT_CATALOG"][name]["sleeve_d_mm"] / 1000)
               + 3 * config["BOLT_CATALOG"][name]["sleeve_d_mm"] / 1000
               + 2 * c["EXTRA_RADIAL_CLEARANCE_M"]
               + 2 * c.get("ENVELOPE_ALLOWANCE_M", 0)
               for name in config["VARIABLES"]["BOLT_TYPES"])


def build_anchor_design(pcd_in_m, pcd_out_m, n_per_ring, bolt_name, config,
                        l_total_m=None):
    spec = _bolt_spec(bolt_name, config)
    envelope_allowance = float(config["CONSTRAINTS"].get("ENVELOPE_ALLOWANCE_M", 0.0))
    if l_total_m is None:
        l_total_m = config["FIXED"]["L_TOTAL_M"]
    return {
        "PCD_in_m": round(float(pcd_in_m), 4),
        "PCD_out_m": round(float(pcd_out_m), 4),
        "N_per_ring": int(n_per_ring),
        "N_total": int(2 * n_per_ring),
        "bolt_type": bolt_name,
        "Bolt_D_mm": float(spec["d_mm"]),
        "d_sleeve_mm": float(spec["sleeve_d_mm"]),
        "D_outer_m": float(pcd_out_m) + float(spec["sleeve_d_mm"]) / 1000.0 + 2.0 * envelope_allowance,
        "L_total_m": round(float(l_total_m), 4),
        "top_expose_m": float(config["FIXED"]["TOP_EXPOSE_M"]),
        "bottom_groove_m": float(config["FIXED"]["BOTTOM_GROOVE_M"]),
        "bottom_cover_m": float(config["FIXED"]["BOTTOM_COVER_M"]),
    }


def evaluate_anchor_design(anchor, ctx, config):
    """Return cost and normalized violations separately; never bury penalties in cost."""
    spec = _bolt_spec(anchor["bolt_type"], config)
    pcd_in = float(anchor["PCD_in_m"])
    pcd_out = float(anchor["PCD_out_m"])
    n_ring = int(anchor["N_per_ring"])
    constraints = {}

    def add(name, violation, value=None, limit=None, unit=""):
        violation = max(0.0, float(violation))
        constraints[name] = {
            "violation": violation,
            "weighted_violation": violation,
            "hard": True,
            "value": value,
            "limit": limit,
            "unit": unit,
        }
        return violation

    denominator = max(1e-9, n_ring * (pcd_out ** 2 + pcd_in ** 2))
    tensile_kN = 4.0 * ctx["M_kNm"] * pcd_out / denominator - ctx["N_kN"] / (2.0 * n_ring)
    add("uls_tension", (tensile_kN - spec["tensile_allow_kN"]) / max(spec["tensile_allow_kN"], 1e-9), tensile_kN, spec["tensile_allow_kN"], "kN")

    delta_t_kN = 4.0 * ctx["delta_M_kNm"] * pcd_out / denominator
    delta_sigma = delta_t_kN * 1000.0 / spec["area_mm2"]
    fatigue_limit = float(config["CONSTRAINTS"]["ALLOWABLE_FATIGUE_STRESS_MPA"])
    add("fatigue_stress", (delta_sigma - fatigue_limit) / fatigue_limit, delta_sigma, fatigue_limit, "MPa")

    clear_in_mm = math.pi * pcd_in * 1000.0 / n_ring - spec["sleeve_d_mm"]
    clear_out_mm = math.pi * pcd_out * 1000.0 / n_ring - spec["sleeve_d_mm"]
    required_circ_mm = ctx["max_rebar_d_m"] * 1000.0 + float(config["CONSTRAINTS"]["EXTRA_CIRC_CLEARANCE_MM"])
    add("circumferential_clearance", (required_circ_mm - min(clear_in_mm, clear_out_mm)) / max(required_circ_mm, 1e-9), min(clear_in_mm, clear_out_mm), required_circ_mm, "mm")

    radial_gap_m = (pcd_out - pcd_in) / 2.0
    required_radial_m = spec["sleeve_d_mm"] / 1000.0 + float(config["CONSTRAINTS"]["EXTRA_RADIAL_CLEARANCE_M"])
    add("radial_clearance", (required_radial_m - radial_gap_m) / required_radial_m, radial_gap_m, required_radial_m, "m")

    min_cover_m = max(float(config["CONSTRAINTS"]["MIN_EDGE_COVER_M"]), 4.0 * spec["d_mm"] / 1000.0)
    actual_cover_m = ctx["r_col_m"] - pcd_out / 2.0 - spec["sleeve_d_mm"] / 2000.0
    add("pedestal_cover", (min_cover_m - actual_cover_m) / min_cover_m, actual_cover_m, min_cover_m, "m")

    required_anchor_m = float(config["FIXED"]["ANCHORAGE_FACTOR_D"]) * spec["d_mm"] / 1000.0
    effective_anchor_m = (float(anchor["L_total_m"])
                          - float(anchor["top_expose_m"])
                          - float(anchor["bottom_groove_m"]))
    add("anchorage_depth", (required_anchor_m - effective_anchor_m) / max(required_anchor_m, 1e-9),
        effective_anchor_m, required_anchor_m, "m")

    required_depth_m = (float(anchor["L_total_m"])
                        - float(anchor["top_expose_m"])
                        + float(anchor["bottom_cover_m"]))
    available_depth_m = ctx["cap_thickness_m"] - float(anchor["bottom_groove_m"])
    add("foundation_depth_fit", (required_depth_m - available_depth_m) / max(required_depth_m, 1e-9),
        available_depth_m, required_depth_m, "m")

    center_clear_m = pcd_in / 2.0 - spec["sleeve_d_mm"] / 2000.0
    add("center_clearance", (ctx["r_void_m"] - center_clear_m) / max(ctx["r_void_m"], 1e-9), center_clear_m, ctx["r_void_m"], "m")

    pcd_average = (pcd_in + pcd_out) / 2.0
    flange_cost = float(config["COST"]["FLANGE_FACTOR"]) * math.pi * pcd_average * float(config["COST"]["FLANGE_COST_PER_M_RMB"])
    reference_length = float(config["COST"].get("UNIT_COST_REFERENCE_LENGTH_M", 4.0))
    if reference_length <= 0:
        raise ValueError("UNIT_COST_REFERENCE_LENGTH_M must be positive")
    length_factor = float(anchor["L_total_m"]) / reference_length
    bolt_cost = 2.0 * n_ring * float(spec["unit_cost_rmb"]) * length_factor
    total_violation = sum(item["weighted_violation"] for item in constraints.values())
    return {
        "feasible": total_violation <= 1e-9,
        "cost_rmb": flange_cost + bolt_cost,
        "total_violation": total_violation,
        "constraints": constraints,
        "anchor": dict(anchor),
        "uls_tension_kN": tensile_kN,
        "fatigue_stress_MPa": delta_sigma,
    }


def _pcd_candidates(ctx, spec, config):
    """Independent inner/outer grid, pruned only by necessary geometry bounds."""
    v, c = config["VARIABLES"], config["CONSTRAINTS"]
    step = float(v["PCD_STEP_M"])
    if step <= 0:
        raise ValueError("PCD_STEP_M must be positive")
    sleeve = spec["sleeve_d_mm"] / 1000
    min_inner = max(v["PCD_IN_MIN_M"], 2 * ctx["r_void_m"] + sleeve)
    min_cover = max(c["MIN_EDGE_COVER_M"], 4 * spec["d_mm"] / 1000)
    max_outer = min(v["PCD_OUT_MAX_M"], 2 * (ctx["r_col_m"] - min_cover) - sleeve)
    gap = 2 * (sleeve + c["EXTRA_RADIAL_CLEARANCE_M"])
    start = math.ceil((min_inner - 1e-10) / step)
    stop = math.floor((max_outer + 1e-10) / step)
    gap_steps = math.ceil((gap - 1e-10) / step)
    return [(round(i * step, 8), round(j * step, 8))
            for i in range(start, stop + 1)
            for j in range(i + gap_steps, stop + 1)]


def _nested_uniform_sample(values, sample_count):
    """Return a deterministic nested subset that progressively fills gaps.

    Unlike independent linspace rounding, every larger sample budget contains
    every point selected by a smaller budget, so the best result cannot worsen
    merely because the sampling budget increased.
    """
    count = len(values)
    if sample_count >= count:
        return list(values)
    if sample_count <= 0 or count == 0:
        return []
    selected = [0]
    if sample_count > 1:
        selected.append(count - 1)
    while len(selected) < sample_count:
        chosen = set(selected)
        index = max(
            (i for i in range(count) if i not in chosen),
            key=lambda i: (min(abs(i - j) for j in selected), -i),
        )
        selected.append(index)
    return [values[i] for i in sorted(selected)]


def solve_anchor_for_foundation(params, config, load_cases, mode="nested",
                                assess_candidate=None, selection_context=None):
    """Bounded deterministic discrete search; callback ranks coupled total cost.

    Includes baseline, samples the independent 2-D PCD grid, and uses anchor
    violation as a lower bound to prune full foundation evaluations. No inner GA.
    Cache is bounded and local to each process; cached objects never escape.
    """
    if mode not in {"fixed", "nested_2", "nested"}:
        raise ValueError("anchor mode must be fixed, nested_2 or nested")
    _STATS["calls"] += 1
    key = _cache_key(params, config, load_cases, mode,
                     [assess_candidate is not None, selection_context])
    cache_enabled = config["SOLVER"].get("CACHE_ENABLED", True)
    if cache_enabled and key in _CACHE:
        _STATS["cache_hits"] += 1
        return deepcopy(_CACHE[key])
    ctx = _context(params, config, load_cases)
    base = config["BASELINE"]
    candidates = [build_anchor_design(base["PCD_in_m"], base["PCD_out_m"],
                                      base["N_per_ring"], base["bolt_type"], config)]
    if mode == "nested_2":
        for name in config["VARIABLES"]["BOLT_TYPES"]:
            for count in config["VARIABLES"]["N_PER_RING_VALUES"]:
                candidates.append(build_anchor_design(
                    base["PCD_in_m"], base["PCD_out_m"], count, name, config))
    elif mode == "nested":
        types = config["VARIABLES"]["BOLT_TYPES"]
        counts = config["VARIABLES"]["N_PER_RING_VALUES"]
        lengths = config["VARIABLES"].get(
            "L_TOTAL_VALUES_M", [config["FIXED"]["L_TOTAL_M"]])
        if not types or not counts or not lengths:
            raise ValueError("bolt types, counts and lengths cannot be empty")
        if any(float(length) <= 0 for length in lengths):
            raise ValueError("anchor lengths must be positive")
        per_combo = int(config["SOLVER"].get("PCD_PAIRS_PER_COMBO", 8))
        full_grid = bool(config["SOLVER"].get("FULL_PCD_GRID", False))
        if per_combo < 1 and not full_grid:
            raise ValueError("PCD_PAIRS_PER_COMBO must be positive")
        for name in types:
            pairs = _pcd_candidates(ctx, _bolt_spec(name, config), config)
            if not full_grid and len(pairs) > per_combo:
                pairs = _nested_uniform_sample(pairs, per_combo)
            for count in counts:
                candidates.extend(build_anchor_design(pi, po, count, name, config, length)
                                  for length in lengths for pi, po in pairs)
    items = []
    seen = set()
    for anchor in candidates:
        identity = (anchor["PCD_in_m"], anchor["PCD_out_m"], anchor["N_per_ring"],
                    anchor["bolt_type"], anchor["L_total_m"])
        if identity in seen:
            continue
        seen.add(identity)
        _STATS["candidate_evaluations"] += 1
        items.append(evaluate_anchor_design(anchor, ctx, config))
    items.sort(key=lambda x: (x["total_violation"], x["cost_rmb"]))
    best, best_rank = None, None
    for item in items:
        if best_rank is not None and item["total_violation"] > best_rank[1] + 1e-9:
            continue
        if assess_candidate is not None:
            details = assess_candidate(item)
            if details["status"] != "ok":
                raise RuntimeError(str(details["constraints"]))
            violation, cost = details["total_violation"], details["total_cost_rmb"]
            item["coupled_details"] = details
        else:
            violation, cost = item["total_violation"], item["cost_rmb"]
        rank = (violation > 1e-9, violation if violation > 1e-9 else 0.0, cost)
        if best_rank is None or rank < best_rank:
            best, best_rank = item, rank
    if cache_enabled:
        if len(_CACHE) >= int(config["SOLVER"].get("CACHE_MAX_ENTRIES", 2048)):
            _CACHE.pop(next(iter(_CACHE)))
        _CACHE[key] = deepcopy(best)
    return best
