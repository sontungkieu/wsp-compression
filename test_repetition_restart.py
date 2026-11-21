"""Regression checks for charging the abandoned trunk and matching restart cuts."""
import unittest
from unittest.mock import patch

from repetition_baseline import astra_instruction
from repetition_cache_content_score import compare_values
from repetition_cache_retention import content_pair
from repetition_mediation import presence_scores
from repetition_phase_trace import measurements, repeated_ngram_fraction
from repetition_restart import nested_restart, output_row, reaches_switch, score_policy
from repetition_restart_cohort import canonical_problem
from repetition_restart_extension import CARD_SAMPLING
from repetition_restart_score import restart_decomposition
from repetition_restart_visible import score_visible
from repetition_state_data import THINK_END
from repetition_state_value import generated_span


def result(ids, eos=None):
    return {"id": "math500:example", "draw": 0, "tokens": ids, "eos_token": eos,
            "seed": 10, "post_seed": 20}


class TestRestartBudget(unittest.TestCase):
    def test_content_sensitivity_cancels_a_lucky_identical_guess(self):
        case = {"task":"math","variants":{"plus":{"row":{"gold":"38"}},"minus":{"row":{"gold":"48"}}}}
        first = {"value":"38.0","eos":True}
        second = {"value":"48","eos":True}
        lucky = compare_values(case,[first,first])
        self.assertEqual(lucky,{"own_correct_eos":1,"opposite_correct_eos":1,"sensitivity":0})
        self.assertEqual(compare_values(case,[first,second])["sensitivity"],1)

    def test_synthetic_pair_changes_only_the_scratchpad_value(self):
        class FakeTokenizer:
            def encode(self, text, add_special_tokens=False):
                return list(map(ord,text))
        with patch("repetition_cache_retention.render",return_value="<think>\n"):
            pair = content_pair(FakeTokenizer(),"math","synthetic:1",("12","17"))
        a,b = (pair["variants"][key] for key in ("plus","minus"))
        self.assertEqual(a["row"]["prompt"],b["row"]["prompt"])
        self.assertNotIn("12",a["row"]["prompt"])
        self.assertNotIn("17",b["row"]["prompt"])
        self.assertEqual((a["row"]["gold"],b["row"]["gold"]),("38","48"))
        self.assertEqual(len(a["completion_prefix"]),len(b["completion_prefix"]))
        self.assertEqual(len(pair["scaffold_changed_indices"]),1)

    def test_internal_cycle_is_distinguished_from_terminal_repetition(self):
        ids = list(range(7))*24+list(range(100,300))
        modes = measurements(ids)
        self.assertTrue(modes["exact_cycle_anywhere"])
        self.assertFalse(modes["exact_cycle_terminal"])
        self.assertGreater(modes["early_16"],modes["late_16"])
        self.assertEqual(repeated_ngram_fraction(list(range(128)),8),0)

    def test_other_tokenizer_uses_its_own_marker_at_switch_and_cap(self):
        marker = 151668
        native = result([4,marker,7], 151645)
        self.assertFalse(reaches_switch(native, 2, marker))
        self.assertTrue(reaches_switch(native, 2))
        self.assertEqual(generated_span([marker,7,8,151645], {151645}, 1, marker=marker),
                         ([marker,7], None))
        forced = output_row(result([marker,8], 151645), "question_e0", 3, marker=marker)
        self.assertEqual(forced["tokens"][0], marker)

    def test_model_card_sampler_settings_do_not_change_the_shared_sampler(self):
        class Logits:
            def float(self):
                return 7.0
        self.assertAlmostEqual(presence_scores(Logits(), True), (7.0-1.5)/.6)
        self.assertEqual(presence_scores(Logits(), True, CARD_SAMPLING), 5.0)

    def test_visible_stop_scores_only_before_the_turn_marker(self):
        class FakeTokenizer:
            def decode(self, ids, skip_special_tokens=False):
                return "".join({7:r"\boxed{42}",8:r"\boxed{10}"}[i] for i in ids)
        row = output_row(result([7,99,8], None), "no_think", 3)
        visible = score_visible({"gold":"42"}, row, FakeTokenizer(), 99)
        self.assertTrue(visible["correct_turn_without_eos"])

    def test_restart_effect_splits_into_early_gains_and_late_token_cost(self):
        rows = {}
        for prompt, values in {"early_good": (1,0,1), "early_bad": (0,1,0), "late": (0,1,0)}.items():
            for arm, result in zip(("native", "question_e0", "switch_restart"), values):
                rows[prompt,0,arm] = {"correct_eos": result}
        audit = restart_decomposition(rows, ("early_good", "early_bad", "late"), 1, {("late",0)})
        self.assertEqual(audit["early_native_gain_pairs"], 1)
        self.assertEqual(audit["early_native_loss_pairs"], 1)
        self.assertEqual(audit["switched_budget_loss_pairs"], 1)

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

    def test_numeric_score_accepts_a_worker_without_a_switch_annotation(self):
        class FakeTokenizer:
            def decode(self, ids, skip_special_tokens=False):
                return r"</think>\boxed{42}"
        record = result([THINK_END,7],9) | {"arm":"fixed_switch"}
        scored = score_policy({"gold":"42"},record,FakeTokenizer(),99)
        self.assertTrue(scored["correct_eos"])
        self.assertFalse(scored["switched"])


if __name__ == "__main__":
    unittest.main()
