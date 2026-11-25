from collections import Counter
import json
from pathlib import Path
import tempfile
import unittest

from repetition_baseline import sha
from repetition_delayed_query import (
    KEEP, NONE, PAD_SPLIT, READ_ENDINGS, RESET, SIZES, THINK_END, assess, correct_eos_contrasts,
    freeze, freeze_after, freeze_match, prefixes, tally, whitespace_read,
)


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

    def test_bare_number_before_turn_is_visible_but_still_requires_configured_eos(self):
        tok = CharacterTokenizer()
        case = {"query": {"read": [11]}, "targets": {"read": [70, 72]}}
        turn = 999
        row = {"mode": "read", "variant": 0, "eos_token": 98,
               "tokens": [11, THINK_END, *tok.encode("70"), turn, *tok.encode("72")]}
        self.assertTrue(assess(tok, case, row, 6, turn)["correct_eos"])
        self.assertTrue(assess(tok, case, row, 5, turn)["correct_without_eos"])
        self.assertFalse(assess(tok, case, row | {"eos_token": None}, 6, turn)["correct_eos"])

    def test_matched_panel_changes_only_read_query_and_keeps_its_token_prefix(self):
        tok = CharacterTokenizer()
        with tempfile.TemporaryDirectory() as d:
            source, target = (Path(d) / name for name in ("source.json", "matched.json"))
            freeze(tok, source, "fixture")
            parent = json.loads(source.read_text())
            for cases in parent["panels"].values():
                for case in cases:
                    query = case["query"]
                    query["apply"] = [42] * (len(query["read"]) + len(READ_ENDINGS[0]) + 1)
            source.write_text(json.dumps(parent))
            freeze_match(tok, source, sha(source), target, "fixture")
            frozen = json.loads(target.read_text())
        for split, cases in frozen["panels"].items():
            for old, new in zip(parent["panels"][split], cases, strict=True):
                self.assertEqual({k: v for k, v in old.items() if k != "query"}, {
                    k: v for k, v in new.items() if k not in ("query", "original_read_sha256")})
                self.assertEqual(new["query"]["apply"], old["query"]["apply"])
                self.assertEqual(new["query"]["read"][:len(old["query"]["read"])], old["query"]["read"])
                self.assertEqual(len(new["query"]["read"]), len(new["query"]["apply"]))

    def test_matched_interaction_uses_correct_eos_without_rewarding_wrong_answers(self):
        rows = [{"id": "x", "draw": 0, "variant": 0, "mode": mode, "arm": arm,
                 "correct_eos": correct, "opposite_eos": not correct}
                for mode, arm, correct in (("read", KEEP, False), ("read", RESET, True),
                                           ("apply", KEEP, True), ("apply", RESET, False))]
        _, changes = correct_eos_contrasts(rows)
        self.assertEqual(changes["read"]["x",0,0], 1)
        self.assertEqual(changes["apply"]["x",0,0], -1)
        self.assertEqual(changes["read_minus_apply"]["x",0,0], 2)

    def test_whitespace_matching_preserves_the_first_token_and_the_entire_read_clause(self):
        tok = CharacterTokenizer()
        query = tok.encode("\nRead the saved register A.\n")
        padded = whitespace_read(tok, query, len(query) + 12)
        self.assertEqual(len(padded), len(query) + 12)
        self.assertEqual(padded[:1] + padded[13:], query)
        self.assertTrue(tok.decode(padded[1:13]).isspace())
        self.assertEqual(padded, whitespace_read(tok, query, len(query) + 12))

    def test_after_placement_moves_only_the_exact_pinned_whitespace_block(self):
        tok = CharacterTokenizer()
        with tempfile.TemporaryDirectory() as d:
            original, before, after = (Path(d) / name for name in ("original.json", "before.json", "after.json"))
            freeze(tok, original, "fixture")
            source = json.loads(original.read_text())
            for cases in source["panels"].values():
                for case in cases:
                    q = case["query"]
                    q["apply"] = [42] * (len(q["read"]) + 12)
            original.write_text(json.dumps(source))
            freeze_match(tok, original, sha(original), before, "fixture", padding="whitespace")
            freeze_after(tok, original, sha(original), before, sha(before), after, "fixture")
            front, back = (json.loads(x.read_text()) for x in (before, after))
            wrong = front["panels"]["canary"][0]["query"]["read"]
            wrong[2] = ord("X")
            before.write_text(json.dumps(front))
            with self.assertRaisesRegex(ValueError, "more than the whitespace placement"):
                freeze_after(tok, original, sha(original), before, sha(before), Path(d) / "bad.json", "fixture")
        for split, cases in back["panels"].items():
            for new, old in zip(cases, source["panels"][split], strict=True):
                expected = whitespace_read(tok, old["query"]["read"], len(old["query"]["apply"]))
                self.assertEqual(new["query"]["read"], old["query"]["read"] + expected[1:13])
                self.assertEqual(new["query"]["apply"], old["query"]["apply"])
                self.assertEqual(new["retained"], old["retained"])

    def test_split_placement_keeps_padding_order_and_both_queries_at_23_tokens(self):
        tok = CharacterTokenizer()
        with tempfile.TemporaryDirectory() as d:
            source, before, split = (Path(d) / name for name in ("source.json", "before.json", "split.json"))
            freeze(tok, source, "fixture")
            original = json.loads(source.read_text())
            for cases in original["panels"].values():
                for case in cases:
                    case["query"] = {"read": tok.encode("\nRead this\n"), "apply": [42] * 23}
            source.write_text(json.dumps(original))
            freeze_match(tok, source, sha(source), before, "fixture", padding="whitespace")
            freeze_after(tok, source, sha(source), before, sha(before), split, "fixture", 6)
            candidate = json.loads(split.read_text())
            padded = json.loads(before.read_text())
        self.assertEqual(candidate["protocol"], PAD_SPLIT)
        for name, cases in candidate["panels"].items():
            for old, front, moved in zip(original["panels"][name], padded["panels"][name], cases, strict=True):
                read = old["query"]["read"]
                pad = front["query"]["read"][1:13]
                self.assertEqual(moved, front | {"query": {"read": read[:1] + pad[:6] + read[1:] + pad[6:],
                                                          "apply": old["query"]["apply"]}})
                self.assertEqual(len(moved["query"]["read"]), 23)


if __name__ == "__main__":
    unittest.main()
