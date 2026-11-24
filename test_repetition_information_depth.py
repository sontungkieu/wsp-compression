import unittest

from repetition_information_depth import score_window, windows
from repetition_motif_path_audit import first_return


class FixtureTokenizer:
    def decode(self, token_ids, **kwargs):
        pieces = {7: "</think>", 8: r"\boxed{41}", 9: r"\boxed{42}"}
        return "".join(pieces[t] for t in token_ids)


class InformationDepthTest(unittest.TestCase):
    def test_nested_prefix_distinguishes_a_correct_answer_from_termination(self):
        trace = {"tokens": [7, 8], "eos_token": 1}
        row = {"gold": "41", "boxed_answer_required": True}
        before = score_window(row, trace, 2, FixtureTokenizer())
        stopped = score_window(row, trace, 3, FixtureTokenizer())
        self.assertTrue(before["correct_without_eos"])
        self.assertFalse(before["correct_eos"])
        self.assertTrue(stopped["correct_eos"])
        self.assertFalse(stopped["cap"])

    def test_registered_depths_are_nested_and_keep_the_largest_budget(self):
        self.assertEqual(windows(512), [128, 256, 512])
        self.assertEqual(windows(2048), [128, 256, 512, 1024, 2048])

    def test_first_return_requires_two_periods_and_allows_a_rotated_start(self):
        self.assertEqual(first_return([99, 2, 1, 2, 1], [1, 2]), (2, 5))
        self.assertEqual(first_return([99, 2, 1, 2], [1, 2]), (None, None))


if __name__ == "__main__":
    unittest.main()
