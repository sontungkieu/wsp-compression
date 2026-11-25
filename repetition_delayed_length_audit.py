"""Check token-level parity of the unchanged APPLY control across delayed-query runs."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path


ARMS = ("keep", "reset_conv")
IDENTITY = ("tokenizer_sha256", "checkpoint", "runtime", "sampling")
UNCHANGED = ("seed", "post_seed", "prefix_sha256")


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def load(path):
    return json.loads(path.read_text())


def outcomes(root, state, name):
    mark = state[name]
    path = root / "jobs" / f"{name}.jsonl"
    if mark["stage"] != "completed" or Path(mark["output"]) != path or sha(path) != mark["sha256"]:
        raise ValueError(f"Invalid raw artifact receipt for {name}")
    with path.open() as stream:
        head, *entries, end = map(json.loads, stream)
    if head["split"] != "eval" or head["cap"] != 512 or head["draws"] != 2 or end["kind"] != "complete":
        raise ValueError(f"Wrong replay protocol in {name}")
    pairs = {}
    for row in entries:
        if row["kind"] != "rollout" or row["mode"] != "apply" or row["arm"] not in ARMS:
            continue
        key = tuple(row[k] for k in ("id", "draw", "variant", "arm"))
        if key in pairs:
            raise ValueError(f"Duplicate apply trace for {name}")
        pairs[key] = row
    if len(pairs) != head["count"] * head["draws"] * 2 * len(ARMS):
        raise ValueError(f"Incomplete apply traces for {name}")
    return head, pairs


def audit(root, archive):
    new = load(root / "status.json")["jobs"]
    old = load(archive / "status.json")["jobs"]
    by_arm = {arm: Counter() for arm in ARMS}
    compared = []
    for start in range(0, 64, 8):
        reference, candidate = f"DQ-eval-{start:03d}", f"DQL-eval-{start:03d}"
        if new[candidate]["stage"] != "completed":
            continue
        h_old, old_rows = outcomes(archive, old, reference)
        h_new, new_rows = outcomes(root, new, candidate)
        if old_rows.keys() != new_rows.keys() or any(h_old[k] != h_new[k] for k in IDENTITY):
            raise ValueError(f"Comparison used different cases or runtime in {candidate}")
        for key, r_old in old_rows.items():
            r_new = new_rows[key]
            if any(r_old[k] != r_new[k] for k in UNCHANGED):
                raise ValueError(f"Changed APPLY prefix or RNG seed in {candidate}")
            identical_tokens = r_old["tokens"] == r_new["tokens"]
            identical_eos = r_old["eos_token"] == r_new["eos_token"]
            by_arm[key[-1]].update(
                n=1, exact_continuation=int(identical_tokens and identical_eos),
                exact_token_ids=int(identical_tokens), exact_eos_token=int(identical_eos),
            )
        compared.append({"start": start, "original_raw_sha256": old[reference]["sha256"],
                         "matched_raw_sha256": new[candidate]["sha256"]})
    return {"complete": len(compared) == 8, "shards": compared, "apply": by_arm}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--archive", type=Path, required=True)
    args = p.parse_args()
    print(json.dumps(audit(args.root, args.archive)))


if __name__ == "__main__":
    main()
