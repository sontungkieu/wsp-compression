from pathlib import Path
import unittest
from unittest.mock import patch

from repetition_delayed_campaign import compile_jobs
from repetition_delayed_length_campaign import DEADLINE, jobs_for
from repetition_delayed_split_tail import tail_jobs
from repetition_followup_queue import Campaign


class DelayedCampaignTest(unittest.TestCase):
    def test_split_tail_only_replays_the_missing_eight_ids(self):
        root, source = Path("/tail"), Path("/workers")
        panel = Path("/archived/panel.json")
        templates = jobs_for(Path("/archived"), source / "repetition_delayed_query.py", panel,
                             Path("/model"), Path("/data"), technical_only=True, split_smoke=True)
        jobs = tail_jobs(root, source, {"jobs": templates}, 1234)
        self.assertEqual([job["id"] for job in jobs], [
            "DQL-eval-056", "DQL-eval-056-score", "DQL-eval-060", "DQL-eval-060-score"])
        for index, (gpu, score) in enumerate(zip(jobs[::2], jobs[1::2], strict=True)):
            args = gpu["argv"]
            self.assertEqual(
                {flag: args[args.index(flag) + 1] for flag in (
                    "--panel", "--start", "--count", "--draws", "--cap", "--deadline")},
                {"--panel": str(panel), "--start": str(56 + index * 4), "--count": "4",
                 "--draws": "2", "--cap": "512", "--deadline": "1234"})
            self.assertEqual(gpu["preferred"], index)
            self.assertEqual(score["deps"], [gpu["id"]])
            self.assertIn(f"{{sha:{gpu['id']}}}", score["argv"])

    def test_split_admission_respects_the_hard_cutoff_without_the_default_long_reserve(self):
        root = Path("/unused")
        config = {"runtime": "", "deadline": DEADLINE, "devices": ["gpu1", "gpu3"],
                  "admission_floor_seconds": 45, "admission_buffer_seconds": 30}
        queue = Campaign(root, config)
        jobs = jobs_for(root, root / "worker.py", root / "panel.json", root / "model",
                        root / "data.parquet", technical_only=True, split_smoke=True)
        holdout = [job for job in jobs if job["kind"] == "gpu"]
        self.assertEqual(len(holdout), 8)
        self.assertTrue(all(job["lane"] == "eval" and not job["deps"] for job in holdout))
        queue.jobs = {job["id"]: job for job in holdout}
        queue.mark = {job["id"]: {"stage": "queued"} for job in holdout}
        with patch("repetition_followup_queue.time.time", return_value=DEADLINE - 400):
            queue.schedule([], [])
        self.assertTrue(all(mark["stage"] == "queued" for mark in queue.mark.values()))
        with patch("repetition_followup_queue.time.time", return_value=DEADLINE - 350):
            queue.schedule([], [])
        self.assertTrue(all(mark["stage"] == "skipped_deadline" for mark in queue.mark.values()))

    def test_padding_position_holdout_uses_only_the_two_technical_canaries(self):
        root = Path("/unused")
        jobs = jobs_for(root, root / "worker.py", root / "panel.json", root / "model",
                        root / "data.parquet", technical_only=True)
        tests = {"DQL-canary-000-score", "DQL-canary-001-score"}
        self.assertFalse(any(job["lane"] == "pilot" or job["id"] == "DQL-canary-gate" for job in jobs))
        holdout = [job for job in jobs if job["kind"] == "gpu" and job["lane"] == "eval"]
        self.assertEqual(len(holdout), 8)
        self.assertTrue(all(set(job["deps"]) == tests for job in holdout))

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
