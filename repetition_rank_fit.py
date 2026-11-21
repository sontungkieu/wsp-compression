"""Fit a prompt-weighted, no-intercept rank-one gate on frozen development data."""
import argparse
from collections import Counter
from pathlib import Path
import hashlib
import math

from repetition_mediation_queue import load, save
from repetition_rank_one import feature_rows

REGULARIZERS = (.01, .1, 1., 10.)
SUPPORT = 64
SPILL_WEIGHT = .5


def design(rows, labels, dose, np):
    weights = Counter(x["stratum"] for x in rows.values())
    vectors, target, weight, ids = [], [], [], []
    for p in sorted(rows):
        row = rows[p]
        group = row["stratum"]
        if group != "copy_zero":
            a = float(dose) * max(0., min(1., labels[p]["delta"]))
            vectors.append(row["x"])
            target.append(0. if group == "old_clean_eos" else a)
            weight.append(1. / weights[group])
            ids.append(p)
        for x in row["zeros"]:
            vectors.append(x)
            target.append(0.)
            weight.append(SPILL_WEIGHT / (len(row["zeros"]) * weights[group]))
            ids.append(p)
    return tuple(np.asarray(x) for x in (vectors, target, weight, ids))


def ridge(x, y, w, alpha, support, np):
    if x.ndim != 2 or not (len(x) and np.isfinite(x).all() and np.isfinite(y).all()) or not (w > 0).all():
        raise ValueError("invalid ridge fit input")
    score = np.abs(x.T @ (w * y))
    columns = np.lexsort((np.arange(x.shape[1]), -score))[:min(support, x.shape[1])]
    z = x[:, columns]
    half = np.sqrt(w)
    wz = z * half[:, None]
    gram = wz.T @ wz
    scale = max(float(np.trace(gram)) / len(columns), 1e-12)
    solved = np.linalg.solve(gram + alpha * scale * np.eye(len(columns)), wz.T @ (half * y))
    v = np.zeros(x.shape[1])
    v[columns] = solved
    return v, sorted(map(int, columns))


def split_id(p):
    return hashlib.sha256(("ridge-dev-fold:" + str(p)).encode()).digest()[0] % 3


def simple_features(item):
    p = max(1e-12, min(1. - 1e-12, item["p_first_native"]))
    return [math.log(p / (1. - p)), math.log1p(item["length"]), float(item["detector"])]


def simple_model(labels, np):
    keys = sorted(labels)
    x = np.asarray([simple_features(labels[k]) for k in keys])
    target = np.asarray([max(0., labels[k]["delta"]) if labels[k]["stratum"] != "old_clean_eos" else 0. for k in keys])
    mu = x.mean(0)
    scale = np.maximum(x.std(0), 1e-9)
    z = np.column_stack([np.ones(len(keys)), (x - mu) / scale])
    penalty = np.diag([0., 1., 1., 1.])
    b = np.linalg.solve(z.T @ z + penalty, z.T @ target)
    return {"mean": mu.tolist(), "std": scale.tolist(), "coef": b.tolist()}


def fit(root):
    import numpy as np

    data = feature_rows(root)
    labels = load(root / "labels-dev-summary.json")["labels"]
    dose = load(root / "direction-choice.json")["dose"]
    if set(labels) != {k for k, x in data.items() if x["stratum"] != "copy_zero"}:
        raise ValueError("ridge fit received a different development split")
    x, y, w, groups = design(data, labels, dose, np)
    fold = np.array([split_id(g) for g in groups])
    cv = {}
    for a in REGULARIZERS:
        held = []
        for k in sorted(set(fold)):
            keep = fold != k
            if not keep.any():
                raise ValueError("no independent fold left for ridge fit")
            v, _ = ridge(x[keep], y[keep], w[keep], a, SUPPORT, np)
            held += list(zip((x[~keep] @ v - y[~keep]) ** 2, w[~keep], strict=True))
        cv[a] = float(sum(err * wt for err, wt in held) / sum(wt for _, wt in held))
    alpha = min(REGULARIZERS, key=lambda a: (cv[a], -a))
    v, columns = ridge(x, y, w, alpha, SUPPORT, np)
    estimates = x @ v
    pos = y > 0
    positive = int(pos.sum())
    if not np.isfinite(estimates).all():
        raise ValueError("nonfinite gate")
    result = {"dose": dose, "alpha": alpha, "cv_weighted_mse": cv, "v": v.tolist(), "support": columns,
              "positive_rows": positive, "rows": len(y), "prompts": len(data), "intercept": 0.,
              "fit_mse": float(np.average((estimates - y) ** 2, weights=w)),
              "zero_target_rms": float(np.sqrt(np.mean(estimates[~pos] ** 2))),
              "simple_baseline": simple_model(labels, np)}
    save(root / "fit.json", result)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=("fit",))
    p.add_argument("--root", type=Path, required=True)
    args = p.parse_args()
    fit(args.root)


if __name__ == "__main__":
    main()
