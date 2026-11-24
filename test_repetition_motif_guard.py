import importlib.util
import unittest

from repetition_motif_guard import blocked_next, completions, reallocate


class MotifGuardTest(unittest.TestCase):
    def test_blocks_all_rotations_within_window(self):
        moves = completions([7, 9, 7])
        self.assertEqual(blocked_next([9, 7], 3, moves, {99}), [7])
        self.assertEqual(blocked_next([7, 7], 3, moves, {99}), [9])
        self.assertEqual(blocked_next([7, 9], 3, moves, {99}), [7])
        self.assertFalse(blocked_next([9], 3, moves, {99}))
        self.assertFalse(blocked_next([3, 99, 9, 7], 3, moves, {99}))
        self.assertFalse(blocked_next([9, 7, 3, 3, 9, 7], 3, moves, {99}))

    def test_single_token_motif_guard_leaves_first_step_to_shared_intervention(self):
        moves = completions([7])
        self.assertFalse(blocked_next([], 1, moves, {99}))
        self.assertEqual(blocked_next([8], 1, moves, {99}), [7])
        self.assertFalse(blocked_next([8, 8], 1, moves, {99}))

    @unittest.skipUnless(importlib.util.find_spec("torch"), "torch runtime unavailable")
    def test_stop_mass_preserved_with_or_without_ordinary_filtered_support(self):
        import torch

        adjusted = torch.tensor([-4., 2., 0., 0.])
        for filtered, expect_fallback in (([.25, .5, .25, 0.], False), ([.25, .75, 0., 0.], True)):
            native = torch.tensor(filtered)
            edited, mass, fallback = reallocate(torch, native, adjusted, [1], [0])
            self.assertEqual(fallback, expect_fallback)
            self.assertEqual(mass, filtered[1])
            self.assertEqual(float(edited[0]), filtered[0])
            self.assertEqual(float(edited[1]), 0.)
            self.assertAlmostEqual(float(edited.sum()), 1.)


if __name__ == "__main__":
    unittest.main()
