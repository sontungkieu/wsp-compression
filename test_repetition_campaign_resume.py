from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from repetition_campaign_resume import copy_pinned, save, sha, verify_completed
from repetition_followup_queue import Campaign


class ScopedInventoryTest(unittest.TestCase):
    def test_selected_inventory_excludes_other_device_processes(self):
        cards = [(str(index), f"gpu-{index}", "4", "0") for index in range(8)]
        devices = [card[1] for card in cards]
        apps = [("gpu-0", "100", "4096"), ("gpu-7", "200", "8192")]
        config = {"runtime": "", "host": "nemo", "devices": devices[2:], "host_devices": devices}
        queue = Campaign(Path("/unused"), config)
        with patch("repetition_followup_queue.socket.gethostname", return_value="nemo"), patch(
                "repetition_followup_queue.smi", side_effect=[cards, apps, cards[::-1], apps]):
            self.assertEqual(queue.inventory(), (cards[2:], apps[1:]))
            with self.assertRaisesRegex(ValueError, "identity changed"):
                queue.inventory()

    def test_legacy_mode_still_requires_full_roster(self):
        queue = Campaign(Path("/unused"), {"runtime": "", "host": "node", "devices": ["gpu-0"]})
        cards = [("0", "gpu-0", "4", "0"), ("1", "gpu-1", "4", "0")]
        with patch("repetition_followup_queue.socket.gethostname", return_value="node"), patch(
                "repetition_followup_queue.smi", side_effect=[cards, []]):
            with self.assertRaisesRegex(ValueError, "identity changed"):
                queue.inventory()


class VerifiedResumeTest(unittest.TestCase):
    def test_resume_reuses_only_matching_staged_files_and_receipts(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "jobs").mkdir()
            raw = root / "jobs" / "test.jsonl"
            raw.write_bytes(b"verified archived trajectory\n")
            fingerprint = sha(raw)
            receipt = {"output": str(raw), "sha256": fingerprint, "stage": "completed"}
            path = root / "jobs" / "test.receipt.json"
            save(path, receipt)
            save(path, receipt)
            copied = root / "staged.jsonl"
            copy_pinned(raw, copied, fingerprint)
            copy_pinned(raw, copied, fingerprint)
            self.assertEqual(verify_completed(root, "test", receipt), fingerprint)
            with self.assertRaises(FileExistsError):
                save(path, receipt | {"stage": "failed"})
            raw.write_bytes(b"incomplete data")
            with self.assertRaisesRegex(ValueError, "artifact or receipt differs"):
                verify_completed(root, "test", receipt)


if __name__ == "__main__":
    unittest.main()
