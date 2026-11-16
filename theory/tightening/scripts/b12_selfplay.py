"""B12: posterior learners playing each other on a fixed stage matrix (M0), 57 real stage matrices.

Pairings (row player x column player), T rounds, realised (sampled) actions, posteriors updated on the realised
opponent action exactly as behavior.Posterior (lam = 1):
  SS : bayes_safex vs bayes_safex (both eps = 0.003, 5 styles each, p_ref = one-hot safex as in run_ourbot)
  SSm: same, column player sees A + noise (sigma_m * ptp(A)) -- world-sampling mismatch proxy
  HS : Hedge over pure rows (anytime eta_t = sqrt(8 ln m / t), rewards (A[:,b]-min)/ptp) vs bayes_safex column
  SH : bayes_safex row vs Hedge over pure columns
Recorded per matrix: per-round payoff band (x_t^T A y_t - v)/ptp, last-iterate and time-average exploitability of each
side (in ptp units), last and time-averaged posteriors, MAP switches, and for HS/SH the Hedge side's regret.
"""
import json
import os
import sys

import numpy as np
from scipy.optimize import linprog

HERE = os.path.dirname(os.path.abspath(__file__))
MATS = "/private/tmp/claude-501/-Users-biofanat-Documents-classes-f26-cs535/2bbe670b-2f97-4d9a-bfad-e1139e0352c6/scratchpad/experts_ext/real_matrices.npz"
STYLES = ("eq", "br_unif", "br_us", "safe", "uniform")
TAU, MIX, EPS = 0.1, 0.05, 0.003
T = int(sys.argv[1]) if len(sys.argv) > 1 else 400
rng = np.random.default_rng(1)


def lp(A):
    m, n = A.shape
    # max v s.t. x^T A >= v, sum x = 1
    c = np.zeros(m + 1); c[-1] = -1
    A_ub = np.hstack([-A.T, np.ones((n, 1))]); b_ub = np.zeros(n)
    A_eq = np.hstack([np.ones((1, m)), np.zeros((1, 1))])
    r = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=[1], bounds=[(0, None)] * m + [(None, None)], method="highs")
    x = np.clip(r.x[:m], 0, None); x /= x.sum()
    v = r.x[-1]
    # column equilibrium via the dual
    c2 = np.zeros(n + 1); c2[-1] = 1
    r2 = linprog(c2, A_ub=np.hstack([A, -np.ones((m, 1))]), b_ub=np.zeros(m), A_eq=np.hstack([np.ones((1, n)), np.zeros((1, 1))]),
                 b_eq=[1], bounds=[(0, None)] * n + [(None, None)], method="highs")
    y = np.clip(r2.x[:n], 0, None); y /= y.sum()
    return x, y, v


def safex(A, v, q, eps=EPS):
    m, n = A.shape
    r = float(np.ptp(A)) or 1.0
    res = linprog(-(A @ q), A_ub=-A.T, b_ub=-(v - eps * r) * np.ones(n), A_eq=np.ones((1, m)), b_eq=[1],
                  bounds=[(0, None)] * m, method="highs")
    if res.status != 0:
        return None
    x = np.clip(res.x, 0, None); return x / x.sum()


def softmax(z):
    e = np.exp(z - z.max()); return e / e.sum()


def style_dists(A, p, y_eq):
    """Opponent-column styles for the row player holding A (own perspective, maximiser). Same formula as behavior."""
    n = A.shape[1]; u = np.full(n, 1 / n)
    if n == 1:
        return np.ones((5, 1))
    tau = TAU * float(np.ptp(A)) or 1.0
    raw = np.stack([y_eq, softmax(-A.mean(0) / tau), softmax(-(p @ A) / tau), softmax(-A.max(0) / tau), u])
    return (1 - MIX) * raw + MIX * u


class Safex:
    """bayes_safex on own matrix A (rows = own actions, maximiser)."""

    def __init__(self, A):
        self.A = A; self.x_eq, self.y_eq, self.v = lp(A)
        self.logp = np.full(5, -np.log(5)); self.hist = []
        q0 = np.zeros(A.shape[1]); q0[int(A.mean(0).argmin())] = 1
        self.p_ref = safex(A, self.v, q0)
        if self.p_ref is None:
            self.p_ref = self.x_eq
        self.qs = style_dists(A, self.p_ref, self.y_eq)        # fixed matrix: styles fixed (p_ref fixed)

    def probs(self):
        w = np.exp(self.logp - self.logp.max()); return w / w.sum()

    def act(self):
        P = self.probs(); qhat = P @ self.qs
        x = safex(self.A, self.v, qhat)
        return self.x_eq if x is None else x

    def update(self, j):
        z = self.logp + np.log(self.qs[:, j]); self.logp = z - (z.max() + np.log(np.exp(z - z.max()).sum()))


class Hedge:
    def __init__(self, A):
        self.A = A; self.m = A.shape[0]; self.cum = np.zeros(self.m); self.t = 0; self.r = float(np.ptp(A)) or 1.0
        self.regret = 0.0

    def act(self):
        eta = np.sqrt(8 * np.log(self.m) / max(self.t, 1))
        return softmax(eta * self.cum)

    def update(self, j, x):
        rew = (self.A[:, j] - self.A.min()) / self.r
        self.cum += rew; self.t += 1; self.regret += rew.max() - 0  # placeholder, regret computed outside


def expl_row(A, x, v):        # exploitability of a row mixture in ptp units
    return (v - (x @ A).min()) / (float(np.ptp(A)) or 1.0)


def expl_col(A, y, v):
    return ((A @ y).max() - v) / (float(np.ptp(A)) or 1.0)


def run(A, pairing, sigma_m=0.0):
    m, n = A.shape
    r = float(np.ptp(A)) or 1.0
    x_eq, y_eq, v = lp(A)
    Acol = -(A + sigma_m * r * rng.normal(size=A.shape)).T if sigma_m else -A.T    # column player's own view
    row = Safex(A) if pairing in ("SS", "SSm", "SH") else Hedge(A)
    col = Safex(Acol) if pairing in ("SS", "SSm", "HS") else Hedge(-A.T)
    xs, ys, pay = [], [], []
    maps_r, maps_c = [], []
    post_r, post_c = [], []
    cum_row_rew = np.zeros(m); cum_col_rew = np.zeros(n); hedge_rew = 0.0
    for t in range(T):
        x = row.act(); y = col.act()
        xs.append(x); ys.append(y); pay.append((x @ A @ y - v) / r)
        i = rng.choice(m, p=x / x.sum()); j = rng.choice(n, p=y / y.sum())
        if isinstance(row, Safex):
            row.update(j); post_r.append(row.probs()); maps_r.append(int(row.probs().argmax()))
        else:
            rew = (A[:, j] - A.min()) / r; row.cum += rew; row.t += 1; cum_row_rew += rew; hedge_rew += x @ rew
        if isinstance(col, Safex):
            col.update(i); post_c.append(col.probs()); maps_c.append(int(col.probs().argmax()))
        else:
            rew = (-A[i, :] + A.max()) / r; col.cum += rew; col.t += 1; cum_col_rew += rew; hedge_rew += y @ rew
    xs = np.array(xs); ys = np.array(ys); pay = np.array(pay)
    out = dict(shape=[m, n], v=v, pay_mean=float(pay.mean()), pay_min=float(pay.min()), pay_max=float(pay.max()),
               last_expl_row=expl_row(A, xs[-1], v), last_expl_col=expl_col(A, ys[-1], v),
               avg_expl_row=expl_row(A, xs.mean(0), v), avg_expl_col=expl_col(A, ys.mean(0), v),
               max_last_expl_row=float(max(expl_row(A, x, v) for x in xs)),
               max_last_expl_col=float(max(expl_col(A, y, v) for y in ys)),
               pure_saddle=bool(abs(A.min(1).max() - A.max(0).min()) < 1e-9))
    if post_r:
        P = np.array(post_r)
        out.update(row_last_post=P[-1].round(4).tolist(), row_avg_post=P.mean(0).round(4).tolist(),
                   row_map_switches=int((np.diff(maps_r) != 0).sum()), row_map_last=STYLES[maps_r[-1]])
    if post_c:
        P = np.array(post_c)
        out.update(col_last_post=P[-1].round(4).tolist(), col_avg_post=P.mean(0).round(4).tolist(),
                   col_map_switches=int((np.diff(maps_c) != 0).sum()), col_map_last=STYLES[maps_c[-1]])
    if isinstance(row, Hedge):
        out["hedge_regret_per_t"] = float((cum_row_rew.max() - hedge_rew) / T)
    if isinstance(col, Hedge):
        out["hedge_regret_per_t"] = float((cum_col_rew.max() - hedge_rew) / T)
    return out


d = np.load(MATS, allow_pickle=True)
mats = [np.asarray(d[f], float) for f in d.files]
mats = [A for A in mats if A.shape[0] >= 2 and A.shape[1] >= 2]
print(f"{len(mats)} real matrices, T = {T}")
results = {}
for pairing, sm in (("SS", 0.0), ("SSm", 0.1), ("SSm", 0.3), ("HS", 0.0), ("SH", 0.0)):
    key = pairing if not sm else f"{pairing}{sm}"
    res = [run(A, pairing, sm) for A in mats]
    results[key] = res
    f = lambda k: np.array([r[k] for r in res])
    print(f"\n--- {key} ---  eps = {EPS} (so the safex floor is {EPS} ptp)")
    print(f"payoff band (x^T A y - v)/ptp: mean {f('pay_mean').mean():+.4f}, min over all rounds {f('pay_min').min():+.4f}, max {f('pay_max').max():+.4f}")
    print(f"row: last-iterate expl max over t {f('max_last_expl_row').max():.4f} (median of per-matrix max {np.median(f('max_last_expl_row')):.4f}); time-avg expl mean {f('avg_expl_row').mean():.4f} max {f('avg_expl_row').max():.4f}")
    print(f"col: last-iterate expl max over t {f('max_last_expl_col').max():.4f} (median {np.median(f('max_last_expl_col')):.4f}); time-avg expl mean {f('avg_expl_col').mean():.4f} max {f('avg_expl_col').max():.4f}")
    for side in ("row", "col"):
        if f"{side}_last_post" in res[0]:
            last = np.array([r[f"{side}_last_post"] for r in res]); avg = np.array([r[f"{side}_avg_post"] for r in res])
            maps = [r[f"{side}_map_last"] for r in res]
            print(f"{side} posterior: mean last {dict(zip(STYLES, last.mean(0).round(3)))}; mean time-avg {dict(zip(STYLES, avg.mean(0).round(3)))}")
            print(f"   final MAP counts {dict((s, maps.count(s)) for s in STYLES)}; MAP switches median {np.median([r[f'{side}_map_switches'] for r in res]):.0f} max {max(r[f'{side}_map_switches'] for r in res)}; P_T(MAP) >= 0.99 on {int((last.max(1) >= 0.99).sum())}/{len(res)}")
    if "hedge_regret_per_t" in res[0]:
        print(f"Hedge side per-round regret (ptp units): mean {f('hedge_regret_per_t').mean():.4f}, bound 1.5*sqrt(ln k/(2T)) ~ {1.5*np.sqrt(np.log(7)/(2*T)):.4f} (k=7)")
json.dump(results, open(os.path.join(HERE, "out_b12_selfplay.json"), "w"))
