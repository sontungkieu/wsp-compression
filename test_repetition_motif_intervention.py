import unittest
from types import SimpleNamespace
from unittest.mock import patch

from repetition_depth_aggregate import guard_summary
from repetition_motif_intervention import BREAK, CLOSE, EXTENDED_BASELINE_SHA256, RESTART, baseline_reader, outcome, source_cases


class ToyTokenizer:
    tokens = {100: "</think>", 9: r"\boxed{4}"}

    def decode(self, tokens, skip_special_tokens=False):
        return "".join(self.tokens.get(token, "x") for token in tokens)


class MotifInterventionTest(unittest.TestCase):
    def test_extended_reader_requires_the_explicit_pinned_worker(self):
        reader = object()
        module = SimpleNamespace(__file__="/fixture/extended.py", verified=reader)
        with patch.dict("sys.modules", {"repetition_followup_baseline_extended": module}):
            with patch("repetition_motif_intervention.sha", return_value="wrong"):
                with self.assertRaisesRegex(ValueError, "checksum differs"):
                    baseline_reader("baseline_extended")
            with patch("repetition_motif_intervention.sha", return_value=EXTENDED_BASELINE_SHA256):
                self.assertIs(baseline_reader("baseline_extended"), reader)
        with self.assertRaisesRegex(ValueError, "unregistered"):
            baseline_reader("auto")

    def test_frozen_prefix_uses_only_first_confirmation(self):
        header = {"scheme": [["native", 0, 16384]], "marker": 100}
        rows = {p: {"id": p, "gold": "4"} for p in ("repeat", "closed", "too_late")}
        raw = {
            ("repeat", "native", 0): {"tokens": [3, 11, 7] * 500, "render_sha256": "a"},
            ("closed", "native", 0): {"tokens": [3] * 40 + [100] + [5] * 200, "render_sha256": "b"},
            ("too_late", "native", 0): {"tokens": list(range(1000, 17000)) + [8] * 200, "render_sha256": "c"},
        }
        with patch("repetition_motif_intervention.verified", return_value=(header, rows, raw, {})):
            _, cases = source_cases(None, None, None, 2048, None)
        self.assertEqual([x["row"]["id"] for x in cases], ["repeat"])
        self.assertEqual(cases[0]["prefix_tokens"], 128)
        self.assertEqual(cases[0]["motif"][0], raw["repeat", "native", 0]["tokens"][128])

    def test_total_budget_charges_the_action_token_and_eos(self):
        case = {"row": {"gold": "4"}, "prefix_tokens": 128, "motif": [7]}
        header = {"marker": 100, "turn": 101}
        close = {"arm": CLOSE, "tokens": [100, 9], "eos_token": 101}
        cut = outcome(case, close, ToyTokenizer(), header, 2)
        self.assertFalse(cut["correct_eos"])
        self.assertTrue(cut["correct_without_eos"])
        self.assertEqual(cut["cost_from_start"], 130)
        self.assertTrue(outcome(case, close, ToyTokenizer(), header, 3)["correct_eos"])
        broken = {"arm": BREAK, "tokens": [8] + [7] * 128, "eos_token": None}
        self.assertTrue(outcome(case, broken, ToyTokenizer(), header, 129)["original_motif_tail"])
        restart = {"arm": RESTART, "tokens": [9], "eos_token": 101}
        self.assertTrue(outcome(case, restart, ToyTokenizer(), header, 2)["correct_eos"])

    def test_guard_aggregate_counts_natural_close_and_nonzero_actions(self):
        row = {"correct_eos": False, "wrong_eos": False, "cap": True,
               "correct_without_eos": False, "natural_close_token": None, "first_return_confirmed": 6}
        h = {"dataset": "math500", "split": "eval", "cap": 2048, "source_draw": 0,
             "draw_start": 60, "eligible": 1}
        cells = {BREAK: {"p": [row]}, "motif_guard": {"p": [
            row | {"correct_eos": True, "cap": False, "natural_close_token": 9, "first_return_confirmed": None}]},
                 CLOSE: {"p": [row | {"first_return_confirmed": None}]}}
        scores = [{"header": h, "windows": {"2048": {"cells": cells}},
                   "interventions": [{"steps": [{"opened_prefilter_support": True}]}]}]
        result = next(iter(guard_summary(scores).values()))
        self.assertEqual(result["nonzero_actions"], 1)
        self.assertEqual(result["prefilter_fallback"], 1)
        self.assertEqual(result["windows"]["2048"]["arms"]["motif_guard"]["natural_close"], 1)
        self.assertEqual(result["windows"]["2048"]["guard_minus_break"]["delta_pp"], 100)


if __name__ == "__main__":
    unittest.main()
