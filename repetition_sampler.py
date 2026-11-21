"""Measure and intervene at the exact Astra first-token sampling boundary."""
from repetition_mediation import SAMPLING
from repetition_state_data import THINK_END

BOUND_TOL = .001


def temperature_scores(logits, items, torch):
    penalty = torch.zeros_like(logits, dtype=torch.bool)
    for i, item in enumerate(items):
        prior = sorted(set(item["completion_prefix"]))
        penalty[i, prior] = True
    return (logits.float() - SAMPLING["presence_penalty"] * penalty) / SAMPLING["temperature"]


class Boundary:
    def __init__(self, torch, tf, marker=THINK_END, k=SAMPLING["top_k"], p=SAMPLING["top_p"]):
        self.torch, self.marker, self.k = torch, marker, k
        self.top_k = tf.TopKLogitsWarper(top_k=k)
        self.top_p = tf.TopPLogitsWarper(top_p=p)

    def filters(self, scores):
        k = self.top_k(None, scores)
        return k, self.top_p(None, k)

    def q(self, scores, beta):
        altered = scores.clone()
        altered[:, self.marker] += beta
        _, p = self.filters(altered)
        return p.softmax(-1)[:, self.marker]

    def threshold(self, scores, target):
        torch = self.torch
        shape = scores.shape[:1]
        zero = torch.zeros(shape, device=scores.device)
        one = torch.ones_like(zero)
        good = (lambda q: q > 0) if not target else (lambda q: q >= target)
        q0 = self.q(scores, zero)
        done = good(q0)
        lo, hi = zero.clone(), torch.where(done, zero, one)
        for _ in range(10):
            more = ~good(self.q(scores, hi))
            if not bool(more.any()):
                break
            hi = torch.where(more, hi * 2, hi)
        if not bool(good(self.q(scores, hi)).all()):
            raise ValueError("could not bracket the exact sampler threshold")
        while float((hi - lo).max()) > BOUND_TOL:
            mid = (hi + lo) / 2
            reached = good(self.q(scores, mid))
            hi = torch.where(reached, mid, hi)
            lo = torch.where(reached, lo, mid)
        lower, upper = self.q(scores, lo), self.q(scores, hi)
        if bool((~done & good(lower)).any()) or not bool(good(upper).all()):
            raise ValueError("sampler threshold bracket failed verification")
        return torch.stack((lo, hi, lower, upper), -1).tolist()

    def measures(self, logits, scores):
        torch, token = self.torch, self.marker
        if not bool(torch.isfinite(scores).all()) or not bool(torch.isfinite(logits).all()):
            raise ValueError("nonfinite scores before sampler")
        k, p = self.filters(scores)
        adjusted = scores[:, token]
        others = scores.clone()
        others[:, token] = -torch.inf
        rival = others.topk(self.k).values[:, -1]
        measures = {
            "raw_logit": logits[:, token], "adjusted_logit": adjusted,
            "log_odds": adjusted - others.logsumexp(-1),
            "p_before": scores.softmax(-1)[:, token],
            "rank": 1 + (others > adjusted[:, None]).sum(-1),
            "ties": (others == adjusted[:, None]).sum(-1),
            "topk_gap": rival - adjusted,
            "in_k": torch.isfinite(k[:, token]), "in_p": torch.isfinite(p[:, token]),
            "p_after_k": k.softmax(-1)[:, token], "q": p.softmax(-1)[:, token],
        }
        thresholds = {"b_support": self.threshold(scores, 0), "b_half": self.threshold(scores, .5)}
        columns = {key: value.tolist() for key, value in measures.items()} | thresholds
        return [dict(zip(columns, values, strict=True)) for values in zip(*columns.values(), strict=True)]


def match_marker(probabilities, base, marker, requested, torch):
    """Keep the native conditionals of non-marker tokens after filtering."""
    q = torch.as_tensor(requested, dtype=probabilities.dtype, device=probabilities.device)
    if q.shape != probabilities.shape[:1] or not bool(((q >= 0) & (q <= 1)).all()):
        raise ValueError("invalid first-token marker mass")
    rest = probabilities.clone()
    rest[:, marker] = 0
    mass = rest.sum(-1, keepdim=True)
    fallback = (mass.squeeze(-1) == 0) & (q < 1)
    if bool(fallback.any()):
        other = base[fallback].clone()
        other[:, marker] = -torch.inf
        # When all filtered support is the marker, use the largest non-marker
        # logit. argmax fixes ties by taking the first vocabulary index.
        rest[fallback] = torch.zeros_like(other).scatter_(1, other.argmax(-1, keepdim=True), 1.)
        mass = rest.sum(-1, keepdim=True)
    rest *= ((1 - q[:, None]) / mass.clamp_min(torch.finfo(rest.dtype).tiny))
    rest[:, marker] = q
    return rest, fallback.tolist()
