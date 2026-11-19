"""Run the two-node Qwen repetition campaign using shared, atomic receipts."""
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time

HOSTS = {"discovery": "rl-opd-cross-tok-test-8gpu-67670-siuu-0-0", "calibration": "anhhbt-eval-4-0-0"}
COUNTS = {"discovery": 8, "calibration": 4}
PHASES = ("main", "split", "long")
TARGET = 64
REVISION = "2fc06364715b967f1860aea9cf38778875588b17"
CONTROL_KEYS = ("http_proxy", "https_proxy", "all_proxy", "hf_token", "wandb_api_key")


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def load(path):
    return json.loads(path.read_text())


def save(path, value):
    temp = path.with_name(path.name + "." + str(os.getpid()) + ".tmp")
    with temp.open("x") as stream:
        json.dump(value, stream)
    temp.replace(path)


def smi(query):
    output = subprocess.check_output(["nvidia-smi", query, "--format=csv,noheader,nounits"], text=True, timeout=10)
    return [tuple(x.strip() for x in line.split(",")) for line in output.splitlines() if line.strip()]


def check_devices(devices):
    gpus = {uuid: (used, utilization) for _, uuid, used, utilization in
            smi("--query-gpu=index,uuid,memory.used,utilization.gpu")}
    busy = {uuid for uuid, _ in smi("--query-compute-apps=gpu_uuid,pid")}
    if any(gpus.get(uuid) != ("4", "0") or uuid in busy for uuid in devices):
        raise RuntimeError("assigned devices must be idle and match their UUID")


def read_shards(paths, deadline, discover_ids):
    from repetition_replay import terminal_cycle

    candidates = {}
    for path, fingerprint in paths:
        if digest(path) != fingerprint:
            raise ValueError(f"bad baseline digest for {path.name}")
        with path.open() as stream:
            manifest = json.loads(next(stream))
            rows = {row["id"]: row for row in manifest["panel"]}
            if (manifest["model_revision"] != REVISION or not manifest["astra_math_prompt"]
                    or manifest["decoder_settings"]["astra"]["presence_penalty"] != 1.5):
                raise ValueError("unmatched baseline")
            for line in stream:
                row = json.loads(line)
                if (row.get("record") != "rollout" or row["decoder"] != "astra"
                        or row["mode"] != "think" or row["id"] in discover_ids):
                    continue
                ids = row["token_ids"]
                loop = terminal_cycle(ids)
                if loop is None:
                    continue
                period, cutoff = loop
                candidates.setdefault(row["id"], {"row": rows[row["id"]], "completion_prefix": ids[:cutoff],
                                                   "period": period, "cycle": ids[cutoff - period:cutoff],
                                                   "origin": path.name})
        if time.time() >= deadline - 30:
            raise TimeoutError("deadline during cohort selection")
    return candidates


class Supervisor:
    def __init__(self, args):
        self.args = args
        self.src = Path(__file__).resolve().parent
        self.root = args.output
        self.lane = self.root / args.role
        self.devices = args.devices.split(",")
        self.children = {}
        self.state = {"host": socket.gethostname(), "stage": "admitted", "deadline": args.deadline, "batches": []}
        self.env = {key: value for key, value in os.environ.items() if key.lower() not in CONTROL_KEYS}
        self.env.update(PYTHONPATH=f"{args.stable}:{args.overlay}", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
                        TOKENIZERS_PARALLELISM="false", OMP_NUM_THREADS="2")

    def status(self, stage=None, **fields):
        if stage is not None:
            self.state["stage"] = stage
        self.state.update(fields, observed=time.time())
        save(self.lane / "status.json", self.state)

    def bounded(self, reserve=120):
        return time.time() < self.args.deadline - reserve

    def run_batch(self, label, plans, *, tolerate_failure=False):
        if not self.bounded():
            raise TimeoutError("no time left for new batch")
        devices = [self.devices[index] for index, _, _ in plans]
        check_devices(devices)
        receipt = {"name": label, "jobs": {}}
        self.state["batches"].append(receipt)
        outputs = []
        try:
            for index, output, argv in plans:
                device = self.devices[index]
                env = {**self.env, "CUDA_VISIBLE_DEVICES": device}
                with output.with_suffix(".log").open("x") as log:
                    proc = subprocess.Popen([sys.executable, "-B", *map(str, argv)], env=env, stdout=log,
                                            stderr=subprocess.STDOUT, start_new_session=True)
                self.children[index] = proc
                receipt["jobs"][index] = {"pid": proc.pid, "gpu": device, "output": str(output)}
            while self.bounded(20) and any(proc.poll() is None for proc in self.children.values()):
                foreign = {(u, int(p)) for u, p in smi("--query-compute-apps=gpu_uuid,pid")}
                for index, proc in self.children.items():
                    if any(u == self.devices[index] and p != proc.pid for u, p in foreign):
                        receipt["jobs"][index]["foreign_process"] = True
                        if proc.poll() is None:
                            proc.terminate()
                self.status(label)
                time.sleep(8)
        finally:
            for index, proc in self.children.items():
                if proc.poll() is None:
                    proc.terminate()
                try:
                    proc.wait(timeout=4)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait()
                row = receipt["jobs"][index]
                row["exit_code"] = proc.returncode
                output = Path(row["output"])
                if proc.returncode == 0 and output.is_file():
                    row["sha256"] = digest(output)
                    outputs.append((output, row["sha256"]))
            self.children.clear()
            self.status()
        if not tolerate_failure and len(outputs) != len(plans):
            raise RuntimeError(f"{label} failed: inspect worker receipts")
        return outputs

    def discover(self):
        initial = self.root / "holdout.json"
        status = load(self.args.baseline / "status.json")
        old = {job["name"]: job["sha256"] for job in status["jobs"] if job.get("status") == "completed"}
        excluded = set()
        for name in ("primary-a", "primary-b"):
            path = self.args.baseline / (name + ".jsonl")
            if digest(path) != old[name]:
                raise ValueError("discovery panel digest changed")
            with path.open() as stream:
                excluded.update(r["id"] for r in json.loads(next(stream))["panel"])
        long_shard = self.args.baseline / "long.jsonl"
        baseline_inputs = [(long_shard, old["long"])]
        for wave in range(5):
            if initial.is_file():
                break
            candidates = read_shards(baseline_inputs, self.args.deadline, excluded)
            self.status("assembling_holdout", heldout_loop_count=len(candidates))
            if len(candidates) >= TARGET:
                rows = sorted(candidates.values(), key=lambda item: hashlib.sha256(item["row"]["id"].encode()).digest())
                save(initial, {"cases": rows[:TARGET], "pool_size": len(rows), "target": TARGET,
                               "sources": [[str(path), sha] for path, sha in baseline_inputs]})
                break
            if wave == 4 or not self.bounded(420):
                raise RuntimeError(f"only {len(candidates)} independent loops, target is {TARGET}")
            plans = []
            for gpu in range(len(self.devices)):
                shard = 4 + wave * 8 + gpu
                file = self.lane / f"baseline-{shard:02d}.jsonl"
                command = [self.args.stable / "repetition_baseline.py", "--astra-math", "--mode", "think",
                           "--decoder", "astra", "--model", self.args.model, "--data", self.args.data,
                           "--output", file, "--deadline", self.args.deadline, "--shard", shard, "--count", 16,
                           "--batch", 8, "--cap", 8192]
                plans.append((gpu, file, command))
            baseline_inputs += self.run_batch(f"baseline_wave_{wave + 1}", plans, tolerate_failure=True)

    def wait_for(self, file, stage):
        while not file.is_file() and self.bounded(180):
            self.status(stage)
            time.sleep(8)
        if not file.is_file():
            raise TimeoutError(f"deadline waiting for {file.name}")
        return load(file)

    def replay(self, phase):
        script = self.src / "repetition_mediation.py"
        launcher = self.wait_for(self.src / "replay-ready.json", "waiting_for_replay_code")
        if digest(script) != launcher["sha256"]:
            raise ValueError("unexpected replay source hash")
        if phase == "dev":
            partitions, offset = 4, 0
        else:
            self.wait_for(self.root / "holdout.json", "waiting_for_holdout")
            self.wait_for(self.root / "dev-ready.json", "waiting_for_calibration")
            partitions, offset = 12, 0 if self.args.role == "discovery" else 8
        plans = []
        for gpu in range(len(self.devices)):
            file = self.lane / f"{phase}-{offset + gpu:02d}.jsonl"
            params = [script, "run", "--phase", phase, "--part", offset + gpu, "--partitions", partitions,
                      "--output", file, "--root", self.root, "--model", self.args.model, "--data", self.args.data,
                      "--baseline", self.args.baseline, "--parent", self.args.parent, "--deadline", self.args.deadline]
            plans.append((gpu, file, params))
        done = self.run_batch(phase, plans)
        save(self.lane / f"{phase}-done.json", {"outputs": [[str(path), sha] for path, sha in done]})
        if phase == "dev":
            self.summarize(phase)
            save(self.root / "dev-ready.json", {"summary_sha256": digest(self.root / "dev-summary.json")})
        elif self.args.role == "discovery":
            self.wait_for(self.root / "calibration" / f"{phase}-done.json", f"waiting_for_{phase}_peer")
            self.summarize(phase)

    def summarize(self, phase):
        env = {**self.env, "CUDA_VISIBLE_DEVICES": ""}
        command = [sys.executable, "-B", str(self.src / "repetition_mediation.py"), "summarize",
                   "--phase", phase, "--root", str(self.root)]
        subprocess.run(command, env=env, check=True, timeout=100)

    def run(self):
        try:
            if self.args.role == "discovery":
                self.discover()
            else:
                self.replay("dev")
            for phase in PHASES:
                if not self.bounded(600):
                    self.status("partial_deadline", remaining=[p for p in PHASES if not (self.lane / f"{p}-done.json").exists()])
                    return
                self.replay(phase)
            self.status("completed")
        except Exception as exc:
            self.status("failed", error=str(exc))
            raise


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=("launch", "supervise"))
    p.add_argument("--role", choices=HOSTS, required=True)
    for name in ("output", "model", "data", "stable", "overlay", "baseline", "parent"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--devices", required=True)
    p.add_argument("--deadline", type=int, required=True)
    args = p.parse_args()
    worker = Supervisor(args)
    if socket.gethostname() != HOSTS[args.role] or len(set(worker.devices)) != COUNTS[args.role] or not (
            120 < args.deadline - time.time() <= 18000):
        raise SystemExit("wrong host, device count, or hard deadline")
    if args.action == "launch":
        check_devices(worker.devices)
        worker.lane.mkdir()
        with (worker.lane / "supervisor.log").open("x") as log:
            proc = subprocess.Popen([sys.executable, "-B", __file__, "supervise", *sys.argv[2:]], env=worker.env,
                                    stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        print(json.dumps({"supervisor": proc.pid, "root": str(worker.root), "role": args.role,
                          "deadline": args.deadline, "devices": worker.devices}))
    else:
        worker.run()


if __name__ == "__main__":
    main()
