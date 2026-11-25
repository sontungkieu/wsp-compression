from pathlib import Path
import unittest

from repetition_delayed_campaign import compile_jobs
from repetition_delayed_length_campaign import DEADLINE, jobs_for
from repetition_followup_queue import Campaign


class DelayedCampaignTest(unittest.TestCase):
    def test_whitespace_holdout_waits_for_pilot_and_scores_with_corrected_archive(self):
        root = Path("/unused")
        old = (root / "old-panel.json", root / "old-worker.py",
               [(root / f"old-{index}.jsonl", "0" * 64) for index in range(8)])
        jobs = jobs_for(root, root / "worker.py", root / "panel.json", root / "model",
                        root / "data.parquet", archived=old, whitespace=True)
        by_id = {job["id"]: job for job in jobs}
        pilot = [job for job in jobs if job["kind"] == "gpu" and job["lane"] == "pilot"]
        self.assertEqual(len(pilot), 4)
        self.assertTrue(all(job["deps"] == ["DQL-canary-gate"] for job in pilot))
        heldout = [job for job in jobs if job["kind"] == "gpu" and job["lane"] == "eval"]
        self.assertTrue(all(job["deps"] == ["DQL-pilot-gate"] for job in heldout))
        reference = [job for job in jobs if job["lane"] == "archive"]
        self.assertEqual(len(reference), 8)
        self.assertTrue(all(job["kind"] == "cpu" and job["argv"][0] == "rescore" for job in reference))
        self.assertEqual(set(by_id["DQL-eval-summary"]["deps"]),
                         {job["id"] for job in reference} | {job["id"] + "-score" for job in heldout})

    def test_length_followup_uses_only_two_slots_and_waits_for_both_canaries(self):
        root = Path("/unused")
        jobs = jobs_for(root, root / "worker.py", root / "panel.json",
                        root / "model", root / "data.parquet")
        by_id = {job["id"]: job for job in jobs}
        self.assertEqual(by_id["DQL-canary-gate"]["deps"],
                         ["DQL-canary-000-score", "DQL-canary-001-score"])
        for job in jobs:
            self.assertIn(job["preferred"], (0, 1))
            if job["kind"] == "gpu":
                argv = job["argv"]
                self.assertEqual(argv[argv.index("--deadline") + 1], str(DEADLINE))
                if job["lane"] == "eval":
                    self.assertEqual(job["deps"], ["DQL-canary-gate"])
                    self.assertEqual(argv[argv.index("--draws") + 1], "2")

    def test_every_eval_shard_depends_on_the_complete_pilot_gate(self):
        root = Path("/unused")
        jobs = compile_jobs(root / "worker.py", root / "model", root / "panel.json",
                            root / "data.parquet", root, 100000)
        by_id = {job["id"]: job for job in jobs}
        pilot = [job for job in jobs if job["kind"] == "gpu" and job["lane"] == "pilot"]
        self.assertTrue(all(job["deps"] == ["DQ-canary-000-score", "DQ-canary-001-score"]
                            for job in pilot))
        self.assertEqual(set(by_id["DQ-pilot-summary"]["deps"]), {job["id"] + "-score" for job in pilot})
        holdout = [job for job in jobs if job["kind"] == "gpu" and job["lane"] == "eval"]
        ranges = []
        for job in holdout:
            self.assertEqual(job["deps"], ["DQ-pilot-summary"])
            argv = job["argv"]
            start = int(argv[argv.index("--start") + 1])
            count = int(argv[argv.index("--count") + 1])
            ranges.extend(range(start, start + count))
        self.assertEqual(ranges, list(range(64)))

        queue = Campaign(root, {"runtime": "", "deadline": 10**12, "devices": []})
        queue.jobs = by_id
        queue.mark = {name: {"stage": "queued"} for name in by_id}
        queue.mark["DQ-pilot-summary"]["stage"] = "failed"
        for job in jobs:
            if job["lane"] != "eval" and job["id"] != "DQ-pilot-summary":
                queue.mark[job["id"]]["stage"] = "completed"
        queue.schedule([], [])
        self.assertTrue(all(queue.mark[job["id"]]["stage"] == "blocked" for job in holdout))


if __name__ == "__main__":
    unittest.main()
