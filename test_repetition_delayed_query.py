from collections import Counter
import json
from pathlib import Path
import tempfile
import unittest

from repetition_delayed_query import NONE, SIZES, THINK_END, assess, freeze, prefixes, tally


class CharacterTokenizer:
    all_special_ids = []

    def apply_chat_template(self, messages, **kwargs):
        return messages[0]["content"] + "\n<think>\n"

    def encode(self, text, **kwargs):
        return list(map(ord, text))

    def decode(self, ids, **kwargs):
        return "".join(map(chr, ids))


class DelayedQueryTests(unittest.TestCase):
    def test_frozen_pairs_change_one_record_without_revealing_query(self):
        tok = CharacterTokenizer()
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "panel.json"
            freeze(tok, path, "fixture")
            frozen = json.loads(path.read_text())
        self.assertEqual(frozen["tokenizer_sha256"], "fixture")
        for split, cases in frozen["panels"].items():
            self.assertEqual(len(cases), SIZES[split])
            if split != "canary":
                self.assertEqual(set(Counter((c["family"], c["slot"]) for c in cases).values()), {len(cases)//8})
            for case in cases:
                self.assertEqual(*prefixes(case, NONE))
                for first, second in zip(*(tok.decode(ids).splitlines() for ids in case["retained"]), strict=True):
                    if first != second:
                        self.assertIn("ABCD"[case["slot"]] + ":", first)
                self.assertTrue(all("Later query:" not in tok.decode(ids) for ids in case["retained"]))
                self.assertTrue(all(v[0] != v[1] for v in case["targets"].values()))

    def test_signed_outcome_penalizes_opposite_and_accounts_for_cut_eos(self):
        tok = CharacterTokenizer()
        case = {"query": {"apply": [11, 12]}, "targets": {"apply": [70, 72]}}
        row = {"mode": "apply", "variant": 0, "eos_token": 99}
        lead = [11, 12, THINK_END]
        correct = assess(tok, case, row | {"tokens": lead + tok.encode("70")}, 3)
        opposite = assess(tok, case, row | {"tokens": lead + tok.encode("72")}, 3)
        censored = assess(tok, case, row | {"tokens": lead + tok.encode("70")}, 2)
        self.assertTrue(correct["correct_eos"])
        self.assertTrue(opposite["opposite_eos"])
        self.assertTrue(censored["correct_without_eos"])
        self.assertEqual(tally([correct, opposite, censored])["delta"], 0)


if __name__ == "__main__":
    unittest.main()
