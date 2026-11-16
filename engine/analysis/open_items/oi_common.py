"""Shared helpers for the open-item analyses (read-only on engine/showdown/results)."""
import glob
import json
import math
import os
import re
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "synthesis"))
RES = os.path.abspath(os.path.join(HERE, "..", "..", "showdown", "results"))
OUT = HERE

# expected grid from engine/showdown/launch_open_items.sh: tag -> (A, B, first seed, last seed, streams)
GRID = {
    "LOG": ("hedge_bayes", "stock", 1, 300), "E3a": ("hedge", "switch:stock@8:exploit_top", 1, 160),
    "E3f": ("hedge_fs", "switch:stock@8:exploit_top", 1, 160),
    "XUmv": ("exploit_bandit_ucb1_mv", "bandit_ucb1_mv", 1, 300), "XU": ("exploit_bandit_ucb1", "bandit_ucb1", 1, 300),
    "XR": ("exploit_rmplus_avg", "rmplus_avg", 1, 300), "XEmv": ("exploit_bandit_exp3_mv", "bandit_exp3_mv", 1, 300),
    "E5t": ("top", "agentform", 1, 300), "E5s": ("agentform", "agentform", 1, 300),
    "SX1": ("bayes_safex", "stock", 1, 200), "SX3": ("bayes_safex", "stock", 1, 200),
    "SX10": ("bayes_safex", "stock", 1, 200), "SX30": ("bayes_safex", "stock", 1, 200),
    "RQeq": ("hedge_bayes", "style_eq", 1, 100), "RQsafe": ("hedge_bayes", "style_safe", 1, 100),
    "RQunif": ("hedge_bayes", "style_uniform", 1, 100),
}
SX_EPS = {"SX1": 0.001, "SX3": 0.003, "SX10": 0.01, "SX30": 0.03}
NSTREAMS = 10
FNRE = re.compile(r"^(?P<tag>.+)-s(?P<seed>\d+)-g(?P<game>[12])\.json$")


def rdir(tag):
    return os.path.join(RES, tag, "replays_headless")


def summary(tag):
    return json.load(open(os.path.join(RES, tag, f"{tag}_summary.json")))


def game_jsons(tag, with_decisions=False):
    """dict (seed, game) -> game record (json). Without with_decisions the big per-decision/input_log lists are dropped."""
    out = {}
    for p in sorted(glob.glob(os.path.join(rdir(tag), f"{tag}-s*-g*.json"))):
        m = FNRE.match(os.path.basename(p))
        if not m:
            continue
        d = json.load(open(p))
        if not with_decisions:
            for k in ("decisions", "input_log", "first_requests", "teams"):
                d.pop(k, None)
        out[(int(m["seed"]), int(m["game"]))] = d
    return out


def decisions(tag, seed, game, role="A"):
    p = os.path.join(rdir(tag), f"{tag}-s{seed}-g{game}.{role}.decisions.jsonl")
    if not os.path.exists(p):
        return None
    return [json.loads(l) for l in open(p) if l.strip()]


def stream_of(seed, first, n=NSTREAMS):
    """stream index under batch.split_streams: seeds[k::n] of the sorted list starting at `first`."""
    return (seed - first) % n


def score(rec, role="A"):
    w = rec.get("winner_role")
    return 1.0 if w == role else (0.0 if w in ("A", "B") else 0.5)


def mean_ci_normal(x):
    x = np.asarray(x, float)
    n = len(x)
    m = x.mean()
    se = x.std(ddof=1) / math.sqrt(n) if n > 1 else float("nan")
    return m, se, m - 1.96 * se, m + 1.96 * se, n


def cluster_ci(x, g):
    """Mean with cluster-robust (CR0 with G/(G-1) correction) SE over clusters g; t(G-1) interval."""
    from scipy import stats
    x = np.asarray(x, float)
    g = np.asarray(g)
    ug = np.unique(g)
    G = len(ug)
    m = x.mean()
    sums = np.array([(x[g == u] - m).sum() for u in ug])
    se = math.sqrt(G / (G - 1) * (sums ** 2).sum()) / len(x)
    h = stats.t.ppf(0.975, G - 1) * se
    return m, se, m - h, m + h, G


def phi_logit(v):
    return 2.0 / (1.0 + np.exp(-np.asarray(v, float) / 98.0)) - 1.0


def phi_tanh(v):
    return np.tanh(np.asarray(v, float) / 400.0)


def game_value_lp(A):
    """Value of the zero-sum matrix game A (row player maximises), via scipy linprog."""
    from scipy.optimize import linprog
    A = np.asarray(A, float)
    n, m = A.shape
    shift = A.min()
    B = A - shift + 1.0
    # min sum(u) s.t. B^T u >= 1, u>=0 ; value = 1/sum(u)
    r = linprog(np.ones(n), A_ub=-B.T, b_ub=-np.ones(m), bounds=(0, None), method="highs")
    return 1.0 / r.fun + shift - 1.0


def dump(name, obj):
    with open(os.path.join(OUT, name), "w") as f:
        json.dump(obj, f, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
