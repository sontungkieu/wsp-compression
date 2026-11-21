"""Scorer and boundary regression checks without a model or GPU."""
from pathlib import Path
import tempfile
import unittest

from repetition_state_data import THINK_END
from repetition_state_value import generated_span, phase_input


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


if __name__ == "__main__":
    unittest.main()
