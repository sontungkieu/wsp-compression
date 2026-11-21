"""Scorer and boundary regression checks without a model or GPU."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from repetition_state_data import THINK_END
from repetition_state_value import METRICS, generated_span, phase_input, summarize


class GeneratedSpanTest(unittest.TestCase):
    def test_padding_after_forced_cap_is_not_eos(self):
        self.assertEqual(generated_span([THINK_END, 1, 2, 9, 9], {9}, 2),
                         ([THINK_END, 1, 2], None))

    def test_eos_in_last_allowed_post_close_slot(self):
        self.assertEqual(generated_span([5, THINK_END, 1, 9, 9], {9}, 2),
                         ([5, THINK_END, 1], 9))

    def test_eos_before_reasoning_ends(self):
        self.assertEqual(generated_span([2, 9, 9], {9}, 2), ([2], 9))

    def test_copy_chat_terminator_is_outside_literal_answer(self):
        self.assertEqual(generated_span([1, 2, 8, 3, 9], {8, 9}), ([1, 2], 8))


class TimingCapTest(unittest.TestCase):
    def test_cap_is_read_from_the_frozen_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cohort = root / "cohort.json"
            cohort.write_text('{"eval": []}')
            self.assertEqual(phase_input(root, "labels-eval")[2], 512)
            cohort.write_text('{"eval": [], "labels_cap": 1536}')
            self.assertEqual(phase_input(root, "labels-eval")[2], 1536)
            cohort.write_text('{"eval": [], "labels_cap": 0}')
            with self.assertRaises(ValueError):
                phase_input(root, "labels-eval")

    def test_forced_close_arm_does_not_require_a_delayed_close_arm(self):
        fields = dict.fromkeys(METRICS, False) | {
            "think_step": None, "after_think_tokens": None, "forced_step": None,
        }
        matrix = ({"draws": 1, "arms": ["native", "e0"]},
                  {"p": {"stratum": "loop", "length": 800}},
                  {("p", 0, "native"): fields, ("p", 0, "e0"): fields | {"success": True}})
        with tempfile.TemporaryDirectory() as directory, patch("repetition_state_value.read_matrix", return_value=matrix):
            root = Path(directory)
            summarize(root, "replay", (("e0", "native"),))
            result = json.loads((root / "replay-summary.json").read_text())
            self.assertNotIn("labels", result)
            self.assertEqual(result["strata"]["all"]["paired"]["e0_minus_native"]["success"]["delta_pp"], 100.)


if __name__ == "__main__":
    unittest.main()
