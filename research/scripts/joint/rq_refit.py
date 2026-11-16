"""Realisability fixes tested offline on the logged RQeq / RQsafe / RQunif runs (hedge_bayes vs style bots).

Each decision record logs the style distributions q_s (4 dp) on the world-averaged matrix; the feedback (next record)
logs the observed column j. Logged q = (1-MIX) softmax(z/tau) + MIX/n, so inverting the mix and raising to a power
beta gives the style at temperature tau/beta exactly (for eq: a tempered LP strategy); a new mix m' is then added.
Scores (nats per scored turn, prequential): per-style log loss and the Bayes posterior's q-hat log loss for a grid of
(beta, mix, lambda), plus the fraction of turns the observed action lies outside the eq support (p_eq == MIX/n).
Usage: python rq_refit.py <results root with RQeq/ RQsafe/ RQunif/>
"""
import glob
import json
import os
import re
import sys

import numpy as np

STYLES = ("eq", "br_unif", "br_us", "safe", "uniform")
MIX = 0.05


def streams(root, tag, S=10):
    files = glob.glob(os.path.join(root, tag, "replays_headless", f"{tag}-s*-g*.A.decisions.jsonl"))
    seeds = sorted({int(re.search(r"-s(\d+)-g", f).group(1)) for f in files})
    for k in range(S):
        order = []
        for s in seeds[k::S]:
            for g in (1, 2):
                f = os.path.join(root, tag, "replays_headless", f"{tag}-s{s}-g{g}.A.decisions.jsonl")
                if os.path.exists(f):
                    order.append(f)
        yield k, order


def events(order):
    """(Q (5 x n) logged style dists, j) per scored feedback turn, in stream order."""
    for f in order:
        recs = [json.loads(l) for l in open(f)]
        d = None                                     # the pending decision = the latest earlier record with q
        for r in recs:
            fb = r.get("feedback")
            if fb and not fb.get("skipped") and "bayes" in fb and d is not None:
                Q = np.array([d["bayes"]["q"][s] for s in STYLES])
                assert Q.shape[1] == fb["bayes"]["n_cols"], (f, r["turn"])
                yield Q, fb["bayes"]["j"], fb["bayes"]["forced"]
            if "bayes" in r and "q" in r["bayes"]:
                d = r


def retemper(Q, beta, mix):
    n = Q.shape[1]
    if n == 1:
        return np.ones_like(Q)
    r = np.clip((Q - MIX / n) / (1 - MIX), 0, None)
    r = r ** beta
    r = r / np.maximum(r.sum(1, keepdims=True), 1e-300)
    r[r.sum(1) == 0] = 1 / n
    return (1 - mix) * r + mix / n


def run(root, tag):
    grid_beta = (0.25, 0.5, 1.0, 2.0)
    grid_mix = (0.05, 0.2, 0.5)
    grid_lam = (1.0, 0.98)
    ev = [(Q, j, forced) for _, order in streams(root, tag) for (Q, j, forced) in events(order)]
    # stream boundaries for the posterior: recompute per stream
    out = {}
    outside = np.mean([abs(Q[0, j] - MIX / Q.shape[1]) < 6e-5 for Q, j, _ in ev if Q.shape[1] > 1])
    print(f"{tag}: scored turns {len(ev)}; observed action outside eq support (p_eq == MIX/n): {outside:.3f}")
    for beta in grid_beta:
        for mix in grid_mix:
            L = np.zeros(len(STYLES)); T = 0
            for Q, j, _ in ev:
                Q2 = retemper(Q, beta, mix); L += -np.log(Q2[:, j]); T += 1
            for lam in grid_lam:
                tot = 0.0
                for _, order in streams(root, tag):
                    logp = np.full(len(STYLES), -np.log(len(STYLES)))
                    for Q, j, _ in events(order):
                        Q2 = retemper(Q, beta, mix)
                        P = np.exp(logp - logp.max()); P /= P.sum()
                        tot += -np.log(P @ Q2[:, j])
                        z = lam * logp + np.log(Q2[:, j]); logp = z - z.max()
                out[(beta, mix, lam)] = dict(qhat=tot / T, styles=dict(zip(STYLES, (L / T).round(4).tolist())))
    base = out[(1.0, 0.05, 1.0)]
    print(f"  baseline (beta 1, mix 0.05, lam 1): qhat {base['qhat']:.4f} styles {base['styles']}")
    best = min(out, key=lambda k: out[k]["qhat"])
    print(f"  best grid point {best}: qhat {out[best]['qhat']:.4f} styles {out[best]['styles']}")
    for k in sorted(out):
        print(f"    beta {k[0]:<5} mix {k[1]:<5} lam {k[2]:<5} qhat {out[k]['qhat']:.4f}  eq {out[k]['styles']['eq']:.3f} safe {out[k]['styles']['safe']:.3f} unif {out[k]['styles']['uniform']:.3f}")
    return {str(k): v for k, v in out.items()}


if __name__ == "__main__":
    root = sys.argv[1]
    res = {tag: run(root, tag) for tag in ("RQeq", "RQsafe", "RQunif")}
    json.dump(res, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "rq_refit.json"), "w"), indent=1)
