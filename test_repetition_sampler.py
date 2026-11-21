"""Numerical fixtures for the real Transformers sampler and decoder controls."""
import unittest

from repetition_sampler import BOUND_TOL, Boundary, match_marker, temperature_scores
from repetition_sampler_reach import bounded_cast, choose

try:
    import torch
    import transformers
except ImportError:
    torch = None


class CandidateTest(unittest.TestCase):
    def test_positive_control_and_gaussian_are_ineligible_for_replay(self):
        def entry(receiver, q, norm=2.):
            return {"receiver": receiver, "q": q, "reference_q": 0., "in_p": q > 0, "norm": norm}
        records = [{"patches": {"3:left_to_blank": entry("blank", 1.),
                                 "3:old_gaussian": entry("native", 1.),
                                 "3:left_to_native": entry("native", .25)}} for _ in range(3)]
        picked, stats = choose(records)
        self.assertEqual(picked, ["3:left_to_native"])
        self.assertEqual(stats[picked[0]]["new_support"], 3)


@unittest.skipIf(torch is None, "requires the pinned Torch/Transformers runtime")
class BoundaryTest(unittest.TestCase):
    def test_exact_topk_topp_bisection(self):
        boundary = Boundary(torch, transformers, marker=2, k=2, p=.6)
        raw = torch.tensor([[4., 3., 1., 0.], [4., 1., 3., 0.]])
        dropped_by_k, dropped_by_p = boundary.measures(raw, raw)
        self.assertFalse(dropped_by_k["in_k"])
        self.assertTrue(dropped_by_p["in_k"])
        self.assertFalse(dropped_by_p["in_p"])
        for r in (dropped_by_k, dropped_by_p):
            lo, hi, q_lo, q_hi = r["b_support"]
            self.assertLessEqual(hi - lo, BOUND_TOL)
            self.assertEqual(q_lo, 0.)
            self.assertGreater(q_hi, 0.)
            lo, hi, q_lo, q_hi = r["b_half"]
            self.assertLessEqual(hi - lo, BOUND_TOL)
            self.assertLess(q_lo, .5)
            self.assertGreaterEqual(q_hi, .5)
        self.assertGreater(dropped_by_k["b_support"][1], dropped_by_p["b_support"][1])

    def test_already_supported_and_nonfinite_guard(self):
        b = Boundary(torch, transformers, marker=0, k=2, p=.6)
        raw = torch.tensor([[4., 1., 0.]])
        r = b.measures(raw, raw)[0]
        self.assertEqual(r["b_half"][:2], [0., 0.])
        with self.assertRaises(ValueError):
            b.measures(raw, torch.tensor([[float("nan"), 1., 0.]]))

    def test_match_mass_and_non_marker_conditionals(self):
        filtered = torch.tensor([[.2, .5, .3], [0., 1., 0.]])
        base = torch.tensor([[1., 2., 3.], [5., 9., 2.]])
        p, fallback = match_marker(filtered, base, 1, [.75, .2], torch)
        torch.testing.assert_close(p, torch.tensor([[.1, .75, .15], [.8, .2, 0.]]))
        self.assertEqual(fallback, [False, True])

    def test_presence_once_per_prior_id_before_temperature(self):
        raw = torch.tensor([[0., 1., 4.], [4., 1., 0.]])
        adjusted = temperature_scores(raw, [{"completion_prefix": [0, 0]}, {"completion_prefix": []}], torch)
        torch.testing.assert_close(adjusted, torch.tensor([[-1.5, 1., 4.], [4., 1., 0.]]) / .6)

    def test_actual_bf16_norm_does_not_exceed_budget(self):
        original = torch.tensor([[1., .75, 2.]], dtype=torch.bfloat16)
        delta = torch.tensor([[.002, -.002, .009]])
        changed = bounded_cast(original, delta, .0095, torch)
        self.assertLessEqual(float((changed.float() - original.float()).norm()), .0095)


if __name__ == "__main__":
    unittest.main()
