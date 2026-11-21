"""Regression fixtures for independent context/history and cap semantics."""
import sys
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

from repetition_baseline import panel
from repetition_budget_score import cut
from repetition_mediation import presence_ids
from repetition_scratchpad import ARMS, factor, verify_trace
from repetition_scratchpad_score import EFFECTS, paired_effect
from repetition_state_value import THINK_END, post_seed

KK, KR, DK, DR = ARMS


class ContextTest(unittest.TestCase):
    def test_context_and_presence_seed_are_independent(self):
        original = {"native_ids":[1,2,3,4,3],"completion_prefix":[3,4,3]}
        choices = factor(original)
        self.assertEqual(choices[KK]["context_ids"],choices[KR]["context_ids"])
        self.assertEqual(choices[DK]["context_ids"],choices[DR]["context_ids"])
        self.assertEqual(choices[DK]["context_ids"],[1,2])
        self.assertEqual(presence_ids(choices[KK]),[3,4])
        self.assertEqual(presence_ids(choices[DK]),[3,4])
        self.assertEqual(presence_ids(choices[KR]),[])
        self.assertEqual(presence_ids(choices[DR]),[])
        self.assertEqual(presence_ids(original),[3,4])

    def test_empty_scratchpad_has_identical_inputs_and_seed_sets(self):
        choices = factor({"native_ids":[1,2],"completion_prefix":[]})
        self.assertTrue(all(x["context_ids"]==[1,2] and presence_ids(x)==[] for x in choices.values()))


class TraceTest(unittest.TestCase):
    def test_nested_projection_never_uses_a_late_eos(self):
        raw = {"id":"p","draw":2,"tokens":[THINK_END]+[1]*256,"think_step":0,
               "eos_token":9,"eos":True,"forced_step":0,"post_seed":post_seed("p",2)}
        verify_trace(raw,512,{9})
        self.assertEqual(cut(raw,"post",256),(raw["tokens"],False))
        self.assertEqual(cut(raw,"post",257),(raw["tokens"],True))

    def test_eos_after_full_cap_is_rejected(self):
        raw = {"id":"p","draw":0,"tokens":[THINK_END]+[1]*512,"think_step":0,
               "eos_token":9,"eos":True,"forced_step":0,"post_seed":post_seed("p",0)}
        with self.assertRaises(ValueError):
            verify_trace(raw,512,{9})
        verify_trace(raw | {"eos_token":None,"eos":False},512,{9})


class FactorEffectTest(unittest.TestCase):
    def test_interaction_and_double_removal_signs(self):
        data = {("a",0,arm):{"success":arm==KK} for arm in ARMS}
        interaction = paired_effect(data,["a"],1,EFFECTS["context_history_interaction"],"success")
        removal = paired_effect(data,["a"],1,EFFECTS["drop_context_and_reset_history"],"success")
        self.assertEqual(interaction["delta_pp"],100.)
        self.assertEqual(removal["delta_pp"],-100.)


class DiscoveryPanelTest(unittest.TestCase):
    def test_explicit_panel_preserves_order_and_rejects_repeated_ids(self):
        arrow = ModuleType("pyarrow")
        parquet = ModuleType("pyarrow.parquet")
        arrow.parquet = parquet
        fake_rows = [{"question":f"question {i}","answer":f"#### {i}"} for i in range(40)]
        parquet.read_table = lambda _: SimpleNamespace(to_pylist=lambda: fake_rows)
        with patch.dict(sys.modules,{"pyarrow":arrow,"pyarrow.parquet":parquet}):
            ids = ["gsm8k:9","gsm8k:3"]
            rows = panel("unused",2,7,astra_math=True,explicit_ids=ids)
            self.assertEqual([r["id"] for r in rows],ids)
            self.assertEqual([r["gold"] for r in rows],["9","3"])
            with self.assertRaises(ValueError):
                panel("unused",2,7,astra_math=True,explicit_ids=ids[:1]*2)


if __name__=="__main__":
    unittest.main()
