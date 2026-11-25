"""Admit the two optional crossed-value comparisons behind verified GPU release."""

import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

from repetition_delayed_campaign import DEVICES, HOST, load, preflight, save, sha
from repetition_delayed_timing_campaign import DEADLINE, SELECTED


WAVES = (
    ("2b", 1000, 670, "2b66b02a15e8b4e437c67c43cbbf21e9c7b44ce8914640a329d47c7bf01e5ec2",
     "8aed350297b386e0a3d8705a3f9b6afa5f280f6363b1609091cc54d27b33a029", ()),
    ("offset17", 780, 560, "b01c3f858a1c4eb20955ebe4d0e8cf60048121d5b408178f9b47542f58f5b889",
     "253bfba24d023dea78d76c81a97be50190b611b4ce5cb5007bc48fd6c12459a1", ("--offset", "17")),
)


def staged(base, suffix, launcher_sha, worker_sha):
    slug = f"repetition-delayed-query-value-cross-{suffix}-20261126-v1"
    root, source = base / "runs" / slug, base / "tmp" / slug
    launcher = source / "repetition_delayed_transfer_campaign.py"
    worker = source / "repetition_delayed_value_cross.py"
    if sha(launcher) != launcher_sha or sha(worker) != worker_sha:
        raise ValueError(f"Crossed-value source digest differs for {suffix}")
    if (root / "launch.json").exists():
        raise FileExistsError(f"Crossed-value wave already launched: {suffix}")
    return root, source, launcher


def await_release(root, minimum):
    release_path = root / "release.json"
    devices = [DEVICES[i] for i in SELECTED]
    while time.time() < DEADLINE - minimum:
        if release_path.is_file():
            release = load(release_path)
            if (release["host"] != HOST or release["status_sha256"] != sha(root / "status.json")
                    or release["campaign_compute"] or release["state"] != "completed"
                    or [row[1] for row in release["allocation"]] != devices):
                raise ValueError(f"Invalid predecessor completion receipt: {root.name}")
            if set(SELECTED) <= set(preflight()["idle"]):
                return sha(release_path)
        time.sleep(8)
    return None


def drive(base, restore):
    previous = restore
    for suffix, larger, smaller, launcher_sha, worker_sha, extra in WAVES:
        receipt = restore / f"final-chain-{suffix}.json"
        try:
            root, source, launcher = staged(base, suffix, launcher_sha, worker_sha)
            digest = await_release(previous, smaller + 20)
            remaining = DEADLINE - time.time()
            if digest is None or remaining < smaller + 20:
                save(receipt, {"state": "skipped_deadline", "predecessor": str(previous),
                               "remaining_seconds": int(remaining), "deadline": DEADLINE})
                continue
            count = 32 if remaining >= larger + 30 else 16
            subprocess.run([sys.executable, "-B", str(launcher), "--base", str(base), "--root", str(root),
                            "--source", str(source), "--count", str(count), *extra], check=True)
            save(receipt, {"state": "launched", "count": count, "predecessor_release_sha256": digest,
                           "launch": load(root / "launch.json")})
            previous = root
        except (ValueError, OSError, subprocess.CalledProcessError) as error:
            save(receipt, {"state": "stopped", "reason": str(error), "deadline": DEADLINE})
            raise


def submit(base, restore):
    preflight()
    expected = base / "runs" / "repetition-delayed-query-restore-holdout-20261126-v1"
    prior = load(expected / "config.json")
    devices = [DEVICES[i] for i in SELECTED]
    if (restore != expected or prior["deadline"] != DEADLINE or prior["host"] != HOST
            or prior["devices"] != devices or time.time() >= DEADLINE - 690):
        raise ValueError("Unexpected restoration allocation or insufficient remaining time")
    for suffix, _, _, launcher_sha, worker_sha, _ in WAVES:
        staged(base, suffix, launcher_sha, worker_sha)
    receipt = restore / "final-chain-launch.json"
    if receipt.exists():
        raise FileExistsError(receipt)
    with (restore / "final-chain.log").open("x") as log:
        proc = subprocess.Popen([sys.executable, "-B", str(Path(__file__).resolve()), "wait", "--base", str(base),
                                 "--restore-root", str(restore)], start_new_session=True, stdout=log,
                                stderr=subprocess.STDOUT)
    result = {"pid": proc.pid, "deadline": DEADLINE, "devices": devices,
              "order": [wave[0] for wave in WAVES], "requires_verified_predecessor_release": True}
    save(receipt, result)
    print(json.dumps(result), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=("wait", "submit"))
    p.add_argument("--base", type=Path, required=True)
    p.add_argument("--restore-root", type=Path, required=True)
    args = p.parse_args()
    action = submit if args.action == "submit" else drive
    action(args.base, args.restore_root)


if __name__ == "__main__":
    main()
