from pathlib import Path
import unittest

from repetition_delayed_campaign import compile_jobs
from repetition_followup_queue import Campaign


class DelayedCampaignTest(unittest.TestCase):
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
