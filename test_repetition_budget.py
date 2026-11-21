"""Regression cases for nested limits and common-prefix verification."""
import unittest

from repetition_budget_run import common_trunk
from repetition_budget_score import cut
from repetition_state_data import THINK_END


def replay(tokens, eos=None, close=None):
    return {"tokens": tokens, "eos_token": eos, "think_step": close}


class CutTest(unittest.TestCase):
    def test_total_budget_counts_marker_and_eos(self):
        record = replay([2, THINK_END, 3], 9, 1)
        self.assertEqual(cut(record, "total", 1), ([2], False))
        self.assertEqual(cut(record, "total", 2), ([2, THINK_END], False))
        self.assertEqual(cut(record, "total", 3), ([2, THINK_END, 3], False))
        self.assertEqual(cut(record, "total", 4), ([2, THINK_END, 3], True))

    def test_post_budget_has_its_own_eos_boundary(self):
        record = replay([2, THINK_END, 3, 4], 9, 1)
        self.assertEqual(cut(record, "post", 2), ([2, THINK_END, 3, 4], False))
        self.assertEqual(cut(record, "post", 3), ([2, THINK_END, 3, 4], True))

    def test_eos_before_tag_is_unaffected_by_post_allowance(self):
        self.assertEqual(cut(replay([2], 9), "post", 256), ([2], True))


class TrunkTest(unittest.TestCase):
    def test_allow_different_forced_marker_after_the_fork(self):
        trunk = [1] * 32
        short = replay(trunk+[THINK_END, 3], 9, 32)
        long = replay(trunk+[2]*96+[THINK_END, 2], 9, 128)
        native = replay(trunk+[2]*96+[3])
        checks = common_trunk({"e32": short, "e128": long, "native": native}, 1536)
        self.assertTrue(all(v["before_fork"] for v in checks.values()))
        self.assertIsNone(checks["e32:e128"]["matched_post"])

    def test_natural_close_reuses_the_post_trace(self):
        short = replay([1, THINK_END, 2, 3], 9, 1)
        native = replay([1, THINK_END, 2], close=1)
        checks = common_trunk({"e32": short, "e128": short, "native": native}, 3)
        self.assertTrue(all(v["before_fork"] and v["matched_post"] for v in checks.values()))


if __name__ == "__main__":
    unittest.main()
