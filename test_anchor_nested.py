"""Regression checks for the 22+4 integration; no engineering certification."""
import ast
import copy
import json
from pathlib import Path
import unittest

import foundation_config as cfg
import optimize_foundation_ga as opt
import analyze_integrated_pile_foundation as model
import anchor_cage_solver as solver


class AnchorIntegrationTests(unittest.TestCase):
    def setUp(self):
        solver.reset_anchor_solver_stats()
        self.config = copy.deepcopy(cfg.CONFIG["ANCHOR_OPT"])
        self.gene = [(lo + hi) / 2 for lo, hi in opt.VAR_RANGES]
        self.params = opt.decode_gene(self.gene)

    def test_outer_ranges_unchanged(self):
        backup = sorted(Path("backups").glob("anchor_22_plus_4_*/foundation_config.py"))[0]
        def ranges(path):
            tree = ast.parse(Path(path).read_text(encoding="utf-8-sig"))
            for node in ast.walk(tree):
                if isinstance(node, ast.Dict):
                    for key, value in zip(node.keys, node.values):
                        if isinstance(key, ast.Constant) and key.value == "var_ranges":
                            return ast.dump(value)
        self.assertEqual(ranges(backup), ranges("foundation_config.py"))
        self.assertEqual(len(opt.VAR_RANGES), 22)

    def test_pure_evaluation_and_real_cost(self):
        b = self.config["BASELINE"]
        anchor = solver.build_anchor_design(b["PCD_in_m"], b["PCD_out_m"],
                                           b["N_per_ring"], b["bolt_type"], self.config)
        ctx = solver._context(self.params, self.config, cfg.CONFIG["load_cases"])
        before = copy.deepcopy((anchor, ctx, self.config, solver.get_anchor_solver_stats()))
        a = solver.evaluate_anchor_design(anchor, ctx, self.config)
        c = solver.evaluate_anchor_design(anchor, {**ctx, "delta_M_kNm": 0}, self.config)
        self.assertEqual(a["cost_rmb"], c["cost_rmb"])
        self.assertGreater(a["fatigue_stress_MPa"], c["fatigue_stress_MPa"])
        self.assertEqual(c["fatigue_stress_MPa"], 0.0)
        self.assertEqual(before, (anchor, ctx, self.config, solver.get_anchor_solver_stats()))
        self.assertAlmostEqual(a["total_violation"], sum(x["weighted_violation"] for x in a["constraints"].values()))

    def test_cache_configuration_and_return_isolation(self):
        args = (self.params, self.config, cfg.CONFIG["load_cases"])
        a = solver.solve_anchor_for_foundation(*args, mode="fixed")
        a["anchor"]["PCD_out_m"] = -1
        b = solver.solve_anchor_for_foundation(*args, mode="fixed")
        self.assertEqual(b["anchor"]["PCD_out_m"], 6.3)
        self.assertEqual(solver.get_anchor_solver_stats()["cache_hits"], 1)
        self.config["COST"]["FLANGE_COST_PER_M_RMB"] *= 2
        c = solver.solve_anchor_for_foundation(*args, mode="fixed")
        self.assertGreater(c["cost_rmb"], b["cost_rmb"])

    def test_two_independent_pcds_and_existence_bound(self):
        ctx = solver._context(self.params, self.config, cfg.CONFIG["load_cases"])
        pairs = solver._pcd_candidates(ctx, solver._bolt_spec("M48", self.config), self.config)
        self.assertGreater(len({round(po-pi, 6) for pi, po in pairs}), 1)
        lower = solver.minimum_anchor_envelope(self.config, "nested")
        self.assertLess(lower, solver.minimum_anchor_envelope(self.config, "fixed"))
        self.assertTrue(all(po + .065 >= lower for pi, po in pairs))

    def test_nested_2_locks_pcd_and_varies_two_dimensions(self):
        result = solver.solve_anchor_for_foundation(
            self.params, self.config, cfg.CONFIG["load_cases"], mode="nested_2")
        base = self.config["BASELINE"]
        self.assertEqual(result["anchor"]["PCD_in_m"], base["PCD_in_m"])
        self.assertEqual(result["anchor"]["PCD_out_m"], base["PCD_out_m"])
        self.assertIn(result["anchor"]["bolt_type"], self.config["VARIABLES"]["BOLT_TYPES"])
        self.assertIn(result["anchor"]["N_per_ring"], self.config["VARIABLES"]["N_PER_RING_VALUES"])
        self.assertEqual(result["anchor"]["L_total_m"], self.config["FIXED"]["L_TOTAL_M"])

    def test_nested_five_variables_include_length_and_length_cost(self):
        self.config["VARIABLES"]["BOLT_TYPES"] = ["M48"]
        self.config["VARIABLES"]["N_PER_RING_VALUES"] = [96]
        self.config["VARIABLES"]["L_TOTAL_VALUES_M"] = [3.4, 4.0]
        short = solver.build_anchor_design(5.25, 5.60, 96, "M48", self.config, 3.4)
        base = solver.build_anchor_design(5.25, 5.60, 96, "M48", self.config, 4.0)
        ctx = solver._context(self.params, self.config, cfg.CONFIG["load_cases"])
        short_eval = solver.evaluate_anchor_design(short, ctx, self.config)
        base_eval = solver.evaluate_anchor_design(base, ctx, self.config)
        self.assertLess(short_eval["cost_rmb"], base_eval["cost_rmb"])
        result = solver.solve_anchor_for_foundation(
            self.params, self.config, cfg.CONFIG["load_cases"], mode="nested")
        self.assertIn(result["anchor"]["L_total_m"], [3.4, 4.0])

    def test_pcd_sampling_is_nested(self):
        values = list(range(101))
        previous = set()
        for count in (4, 8, 16, 32, 64):
            current = set(solver._nested_uniform_sample(values, count))
            self.assertEqual(len(current), count)
            self.assertTrue(previous.issubset(current))
            previous = current

    def test_coupled_selection_uses_total_cost(self):
        self.config["VARIABLES"]["BOLT_TYPES"] = ["M48"]
        self.config["VARIABLES"]["N_PER_RING_VALUES"] = [60]
        self.config["SOLVER"]["MAX_CANDIDATES"] = 9
        loads = {"Extreme": {"MXY": 0.0, "FZ": 0.0}}
        evaluated = []
        def assess(item):
            # Synthetic foundation saving reverses the anchor-only cost order.
            total = 1e7 - item["anchor"]["PCD_out_m"] * 1e6
            details = {"status": "ok", "total_violation": item["total_violation"],
                       "total_cost_rmb": total}
            evaluated.append((item["anchor"], details))
            return details
        answer = solver.solve_anchor_for_foundation(
            self.params, self.config, loads, assess_candidate=assess,
            selection_context="synthetic-total-cost-test")
        self.assertEqual(answer["coupled_details"]["total_cost_rmb"],
                         min(d["total_cost_rmb"] for _, d in evaluated if d["total_violation"] <= 1e-9))
        standalone = solver.solve_anchor_for_foundation(self.params, self.config, loads)
        self.assertLess(standalone["cost_rmb"], answer["cost_rmb"])

    def test_single_design_modes_coupling_and_budget(self):
        original_config = copy.deepcopy(cfg.CONFIG)
        results = {}
        for mode in ("fixed", "nested"):
            opt.init_worker(True, None, mode)
            d = opt.evaluate_detailed(self.gene)
            self.assertEqual(d["status"], "ok", d)
            self.assertAlmostEqual(d["total_cost_rmb"], d["foundation_cost_rmb"] + d["anchor_cost_rmb"])
            params = {**self.params, "anchor": d["anchor"]}
            costs = model.calculate_design_cost(params)
            self.assertAlmostEqual(costs["total_cost_rmb"], d["total_cost_rmb"])
            assessment = model.assess_design_performance(params, return_details=True)
            self.assertAlmostEqual(assessment["objectives"]["cost"], d["objective"])
            self.assertLessEqual(solver.get_anchor_solver_stats()["candidate_evaluations"], 1921)
            results[mode] = d
        self.assertEqual(cfg.CONFIG, original_config)
        self.assertLessEqual(results["nested"]["total_violation"], results["fixed"]["total_violation"])
        p1 = {**self.params, "anchor": results["fixed"]["anchor"]}
        p2 = {**self.params, "anchor": {**p1["anchor"], "D_outer_m": 6.0}}
        self.assertNotEqual(model.calculate_design_cost(p1)["steel_kg"], model.calculate_design_cost(p2)["steel_kg"])
        out = Path("experiments_anchor_22_plus_4")
        out.mkdir(exist_ok=True)
        (out / "single_design.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    unittest.main()
