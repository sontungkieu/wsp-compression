import unittest
from unittest.mock import patch

from repetition_motif_intervention import BREAK, CLOSE, RESTART, outcome, source_cases


class ToyTokenizer:
    tokens = {100: "</think>", 9: r"\boxed{4}"}

    def decode(self, tokens, skip_special_tokens=False):
        return "".join(self.tokens.get(token, "x") for token in tokens)


class MotifInterventionTest(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
