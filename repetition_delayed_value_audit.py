"""Check numerical readout parity and describe paired value flips by record group."""

import argparse
import hashlib
import json
from pathlib import Path
from statistics import mean


def load(path):
    return json.loads(path.read_text())


def sha(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def checked_rows(root, label, panel, field):
    job_root = root / "jobs"
    status_path = root / "status.json"
    release, status = load(root / "release.json"), load(status_path)
    if status["state"] != "completed" or release["status_sha256"] != sha(status_path):
        raise ValueError("Expected a completed, checksummed campaign")
    rows, fingerprints = {}, {}
    identity = None
    for start in range(0, len(panel), 8):
        name = f"{label}-eval-{start:03d}"
        job = status["jobs"][name]
        artifact = job_root / f"{name}.json"
        if (job["stage"] != "completed" or Path(job["output"]) != artifact
                or job["sha256"] != sha(artifact) or job != load(job_root / f"{name}.receipt.json")):
            raise ValueError(f"Unverified readout shard {name}")
        data = load(artifact)
        actual = [row["id"] for row in data["rows"]]
        expected = [row["id"] for row in panel[start:start + 8]]
        if actual != expected or data["start"] != start or data["count"] != len(actual) or rows.keys() & set(actual):
            raise ValueError(f"Mismatched cohort for {name}")
        header = (data["checkpoint"], data["tokenizer_sha256"], data["runtime"])
        if identity is not None and identity != header:
            raise ValueError(f"Changed checkpoint or runtime for {name}")
        identity = header
        rows.update((row["id"], row[field]) for row in data["rows"])
        fingerprints[name] = job["sha256"]
    return rows, fingerprints, identity


def audit(panel_path, numeric_root, cross_root):
    panel = load(panel_path)["panels"]["eval"]
    numeric, numeric_shas, identity = checked_rows(numeric_root, "DQN", panel, "placements")
    crossed, crossed_shas, cross_identity = checked_rows(cross_root, "DQX", panel, "arms")
    if identity != cross_identity:
        raise ValueError("Numerical readout and crossing require the same runtime and checkpoint")
    ids = [row["id"] for row in panel]
    field = "correct_minus_opposite_nats"
    placement = {pid: numeric[pid]["6+6"] for pid in ids}
    parity = [abs(x - y) for pid in ids for x, y in zip(placement[pid]["keep"][field],
                                                       crossed[pid]["keep"][field], strict=True)]
    groups = {"all": ids}
    for family in sorted({row["family"] for row in panel}):
        groups[family] = [row["id"] for row in panel if row["family"] == family]
    for slot in sorted({row["slot"] for row in panel}):
        groups[f"slot_{slot}"] = [row["id"] for row in panel if row["slot"] == slot]

    def measures(rows, ids, arm):
        paired = [(base, changed) for pid in ids for base, changed in
                  zip(rows[pid]["keep"][field], rows[pid][arm][field], strict=True)]
        return {"variants": len(paired), "keep_positive": sum(base > 0 for base, _ in paired),
                "arm_negative": sum(value < 0 for _, value in paired),
                "positive_to_negative": sum(base > 0 > value for base, value in paired),
                "negative_to_positive": sum(base < 0 < value for base, value in paired),
                "average_delta_nats": mean(value - base for base, value in paired)}

    return {"panel_sha256": sha(panel_path), "numeric_shards": numeric_shas, "crossed_shards": crossed_shas,
            "keep_parity": {"n": len(parity), "exact": sum(delta == 0 for delta in parity), "max_nats": max(parity)},
            "groups": {name: {"prompts": len(subset),
                              "crossed": {arm: measures(crossed, subset, arm) for arm in ("conv", "matrix", "kv")},
                              "reset": {arm: measures(placement, subset, arm) for arm in
                                        ("reset_0", "reset_7", "reset_17")}}
                       for name, subset in groups.items()}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for arg in ("panel", "numeric", "cross"):
        parser.add_argument("--" + arg, required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(audit(args.panel, args.numeric, args.cross)))


if __name__ == "__main__":
    main()
