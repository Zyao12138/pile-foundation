import unittest

import optimize_foundation_ga as legacy
from bilevel_foundation_optimizer import (
    OUTER_INDICES, INNER_REBAR_INDICES, VARIABLE_PARTITION,
    compose_full_gene, split_full_gene, DeterministicInnerSolver,
)


class BilevelPartitionTests(unittest.TestCase):
    def test_partition_is_complete_and_disjoint(self):
        self.assertEqual(len(OUTER_INDICES), 12)
        self.assertEqual(len(INNER_REBAR_INDICES), 10)
        self.assertFalse(set(OUTER_INDICES) & set(INNER_REBAR_INDICES))
        self.assertEqual(set(OUTER_INDICES) | set(INNER_REBAR_INDICES), set(range(22)))
        self.assertEqual(len(VARIABLE_PARTITION["inner_anchor"]["names"]), 5)

    def test_split_compose_round_trip(self):
        full = [(lo + hi) / 2 for lo, hi in legacy.VAR_RANGES]
        outer, inner = split_full_gene(full)
        self.assertEqual(compose_full_gene(outer, inner), full)

    def test_inner_solver_is_deterministic_and_cached(self):
        full = [(lo + hi) / 2 for lo, hi in legacy.VAR_RANGES]
        outer, _ = split_full_gene(full)
        solver = DeterministicInnerSolver(
            fatigue_moment_ratio=0.10, pcd_samples=2,
            coordinate_passes=1, seed_fractions=(0.5,),
        )
        first = solver.solve(outer)
        second = solver.solve(outer)
        self.assertEqual(first, second)
        self.assertEqual(solver.stats["outer_cache_hits"], 1)
        self.assertEqual(len(first["full_gene"]), 22)
        self.assertIn("anchor", first["details"])


if __name__ == "__main__":
    unittest.main()
