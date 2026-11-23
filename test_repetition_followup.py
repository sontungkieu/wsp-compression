import unittest
from unittest.mock import patch

from repetition_followup_aggregate import reject_rate, risk_match
from repetition_followup_astra import context
from repetition_followup_baseline import consensus, scheme
from repetition_followup_evidence import evaluate_evidence, new_row
from repetition_followup_information import check_question_only
from repetition_followup_timing import decisions


class FollowupTest(unittest.TestCase):
    def test_draw_scheme_has_two_disjoint_draws_with_half_budget_for_agreement(self):
        self.assertEqual(scheme(8192,True,6),(
            ("native",6,8192),("no_think",6,8192),("no_think",7,4096)))
        case={"gold":"30"}
        with patch("repetition_followup_baseline.numeric_result",side_effect=(("30",21),("30.0",19))) as scorer:
            actual=consensus(case,{"draw":6},{"draw":7},120,None,1)
        self.assertEqual(actual,{"answered":True,"correct_eos":True,"wrong_eos":False,"cost":40})
        self.assertEqual([call.args[2] for call in scorer.call_args_list],[60,60])

    def test_question_only_parity_pairs_hidden_values_with_same_operation(self):
        traces=[{"tokens":[value],"eos_token":99} for value in (7,7,8,8)]
        check_question_only(traces)
        with self.assertRaises(ValueError):
            check_question_only([traces[0],traces[2],traces[1],traces[3]])

    def test_long_replay_ignores_triggers_that_exceed_total_budget(self):
        marker=99
        trace={"tokens":[5]*10000+[marker]+[2]*32}
        with patch("repetition_followup_timing.first_long_cycle",side_effect=len) as cycle:
            self.assertEqual(decisions(trace,marker,"exact",8192),8192)
        self.assertEqual(len(cycle.call_args.args[0]),8192)
        self.assertIsNone(decisions({"tokens":[0,marker]+[5]*512},marker,"512"))

    def test_published_edit_uses_full_native_panel_and_only_eligible_fixed_prefixes(self):
        marker=99
        models={"open":{"id":"open"},"closed":{"id":"closed"}}
        upstream={
            ("open","native",0):{"tokens":[5]*600,"render_sha256":"x"},
            ("closed","native",0):{"tokens":[marker]+[5]*600,"render_sha256":"y"},
        }
        all_rows,fixed=context(models,upstream,marker)
        self.assertEqual([x["row"]["id"] for x in all_rows],["open","closed"])
        self.assertEqual(len(fixed),1)
        self.assertEqual(fixed[0]["completion_prefix"],[5]*512)

    def test_record_instruction_keeps_the_operation_and_verbatim_evidence(self):
        old={"id":"a","gold":"41","prompt":"An integer counter snapshot was recorded in the scratchpad. "
             "Double the snapshot.\nVerified record: Counter snapshot is 20.5.\nAnswer in a box."}
        revised=new_row({"evidence_row":old})
        self.assertEqual(revised["row"]["prompt"],"Use the verified checkpoint record below. "
                         "Double the snapshot.\nVerified record: Counter snapshot is 20.5.\nAnswer in a box.")
        self.assertEqual(revised["row"]["gold"],old["gold"])
        self.assertEqual(revised["completion_prefix"],[])

    def test_evidence_scores_numeric_completion_after_forced_marker(self):
        row={"gold":"41","boxed_answer_required":True}
        self.assertTrue(evaluate_evidence(row,"</think>\\boxed{41}",True)["correct_eos"])
        self.assertTrue(evaluate_evidence(row,"</think>\\boxed{40}",True)["wrong_eos"])
        self.assertTrue(evaluate_evidence(row,"</think>UNKNOWN",True)["unknown"])
        self.assertFalse(evaluate_evidence(row,"\\boxed{41}",True)["correct_eos"])

    def test_thinning_keeps_cost_and_matches_dev_coverage_and_risk_when_feasible(self):
        correct={"answered":True,"correct_eos":True,"wrong_eos":False,"cost":37}
        wrong={"answered":True,"correct_eos":False,"wrong_eos":True,"cost":41}
        abstain={"answered":False,"correct_eos":False,"wrong_eos":False,"cost":99}
        native={"a":correct,"b":wrong}
        agreement={"a":correct,"b":abstain}
        selection=risk_match(agreement,{"direct":native,"agreement":agreement})
        self.assertEqual(selection["development_mismatch"],0)
        self.assertEqual(selection["development"]["answers"],1)
        self.assertEqual(reject_rate(native,0.,"direct")["b"]["cost"],41)
        self.assertFalse(reject_rate(native,0.,"direct")["b"]["wrong_eos"])


if __name__=="__main__":
    unittest.main()
