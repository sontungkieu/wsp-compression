"""Admit queued probes in order, after predecessor release and fresh GPU checks."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

from repetition_delayed_campaign import DEVICES, HOST, load, preflight, save, sha
from repetition_delayed_timing_campaign import DEADLINE, SELECTED


CAMPAIGNS = (
    ("transport", "repetition_delayed_transport", "9c41ea8bcdb102e33150b7812778dfc91e3165ed44c5244ac0009bfb6b3486f3",
     "ea70326d0c2247718b57792783d90e6c84be9370cda0159260254aa93463e51d"),
    ("numeric", "repetition_delayed_numeric", "97b75ca00d17b23cf1ca08887ee9ca38bd59b089b382f325d4a17ff48ee4a349",
     "7fd94b1f7fc717da92a3069dd6da3dafb083ddcace1baf02b2c155617fc9092e"),
    ("value-cross", "repetition_delayed_value_cross",
     "6d759c2e9497753b1b9564faaf17b44040ece45d67851e6b0e6be6a7b6910f2a",
     "402c54585979290867e500ebbe4f8d1e93cf4c40c5203ebe1ee8e9524407ec88"),
)


def wait_for_release(root):
    path = root / "release.json"
    while time.time() < DEADLINE - 700:
        if path.is_file():
            record = load(path)
            if (record["host"] != HOST or record["status_sha256"] != sha(root / "status.json")
                    or record["state"] != "completed" or record["campaign_compute"]
                    or [r[1] for r in record["allocation"]] != [DEVICES[i] for i in SELECTED]):
                raise ValueError(f"Predecessor did not complete with a verified release: {root.name}")
            if set(SELECTED) <= set(preflight()["idle"]):
                return sha(path)
        time.sleep(10)
    raise TimeoutError(f"Insufficient time after {root.name}")


def campaign_paths(base, stage, module, controller_sha, worker_sha):
    slug = f"repetition-delayed-query-{stage}-20261126-v1"
    root, source = base / "runs" / slug, base / "tmp" / slug
    controller, worker = source / f"{module}_campaign.py", source / f"{module}.py"
    if sha(controller) != controller_sha or sha(worker) != worker_sha:
        raise ValueError(f"Queued source digest differs for {stage}")
    return root, source, controller


def drive(base, timing_root):
    previous = timing_root
    for stage, module, launcher_pin, worker_pin in CAMPAIGNS:
        try:
            root, source, controller = campaign_paths(base, stage, module, launcher_pin, worker_pin)
            release_sha = wait_for_release(previous)
            if (root / "launch.json").exists():
                raise FileExistsError(f"{stage} was already launched")
            subprocess.run([sys.executable, "-B", str(controller), "--base", str(base), "--root", str(root),
                            "--source", str(source), "--worker-sha256", worker_pin], check=True)
            save(timing_root / f"probe-chain-{stage}.json",
                 {"state": "launched", "stage": stage, "predecessor": str(previous),
                  "predecessor_release_sha256": release_sha, "launch": load(root / "launch.json")})
        except (ValueError, OSError, subprocess.CalledProcessError) as error:
            save(timing_root / f"probe-chain-{stage}.json",
                 {"state": "stopped", "stage": stage, "reason": str(error), "deadline": DEADLINE})
            raise
        previous = root


def submit(base, timing_root):
    preflight()
    expected = base / "runs" / "repetition-delayed-query-reset-offset-20261126-v1"
    cfg = load(expected / "config.json")
    if (timing_root != expected or cfg["host"] != HOST or cfg["deadline"] != DEADLINE
            or cfg["devices"] != [DEVICES[i] for i in SELECTED] or time.time() >= DEADLINE - 700):
        raise ValueError("Unexpected timing allocation or no launch window")
    for args in CAMPAIGNS:
        root, _, _ = campaign_paths(base, *args)
        if (root / "launch.json").exists():
            raise FileExistsError(root / "launch.json")
    receipt = timing_root / "probe-chain-launch.json"
    if receipt.exists():
        raise FileExistsError(receipt)
    with (timing_root / "probe-chain.log").open("x") as log:
        proc = subprocess.Popen([sys.executable, "-B", str(Path(__file__).resolve()), "wait", "--base", str(base),
                                "--timing-root", str(timing_root)], start_new_session=True,
                                stdout=log, stderr=subprocess.STDOUT)
    result = {"pid": proc.pid, "deadline": DEADLINE, "devices": cfg["devices"],
              "order": [stage for stage, *_ in CAMPAIGNS],
              "requires_verified_predecessor_release": True}
    save(receipt, result)
    print(json.dumps(result))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=("submit", "wait"))
    p.add_argument("--base", type=Path, required=True)
    p.add_argument("--timing-root", type=Path, required=True)
    args = p.parse_args()
    if args.action == "submit":
        submit(args.base, args.timing_root)
    else:
        drive(args.base, args.timing_root)


if __name__ == "__main__":
    main()
