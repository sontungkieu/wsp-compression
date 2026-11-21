"""Regression checks for charging the abandoned trunk and matching restart cuts."""
import unittest

from repetition_baseline import astra_instruction
from repetition_restart import nested_restart, output_row, reaches_switch, score_policy
from repetition_restart_cohort import canonical_problem
from repetition_state_data import THINK_END


def result(ids, eos=None):
    return {"id": "math500:example", "draw": 0, "tokens": ids, "eos_token": eos,
            "seed": 10, "post_seed": 20}


class TestRestartBudget(unittest.TestCase):
    def test_staged_dataset_can_already_have_the_canonical_instruction(self):
        instruction = astra_instruction("canonical")
        expected = "Find the ratio.\n"+instruction
        self.assertEqual(canonical_problem("Find the ratio."), expected)
        self.assertEqual(canonical_problem(expected+"\n"), expected)
        with self.assertRaises(ValueError):
            canonical_problem(instruction+"\n"+expected)

    def test_switch_runs_only_when_the_trunk_is_still_thinking(self):
        self.assertFalse(reaches_switch(result([1], 9), 2))
        self.assertFalse(reaches_switch(result([1, THINK_END, 2], 9), 2))
        self.assertTrue(reaches_switch(result([1, 2], 9), 2))

    def test_nested_cut_counts_the_tag_and_eos_separately(self):
        direct = result([THINK_END, 5, 6], 9)
        self.assertEqual(nested_restart(direct, 5, 2), {"tokens": [THINK_END, 5, 6], "eos_token": None})
        self.assertEqual(nested_restart(direct, 6, 2), {"tokens": [THINK_END, 5, 6], "eos_token": 9})

    def test_restart_keeps_spent_tokens_in_the_ledger(self):
        saved = output_row(result([THINK_END, 7], 9), "switch_restart", 5, prefix=[1, 2], switched=True)
        self.assertEqual(saved["tokens"], [1, 2, THINK_END, 7])
        self.assertEqual(saved["spent_before_switch"], 2)
        with self.assertRaises(ValueError):
            output_row(result([THINK_END, 7], 9), "switch_restart", 4, prefix=[1, 2], switched=True)

    def test_direct_is_scored_without_a_generated_reasoning_marker(self):
        class FakeTokenizer:
            def decode(self, ids, skip_special_tokens=False):
                return r"\boxed{42}"
        direct = output_row(result([7], 9), "no_think", 5)
        thinking = output_row(result([7], 9), "native", 5)
        self.assertTrue(score_policy({"gold": "42"}, direct, FakeTokenizer(), 99)["correct_eos"])
        self.assertFalse(score_policy({"gold": "42"}, thinking, FakeTokenizer(), 99)["correct_eos"])


if __name__ == "__main__":
    unittest.main()
