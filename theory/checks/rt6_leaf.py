"""Section R6 checks: (a) affine invariance of nash.rm_plus and eps_s scaling; (b) the inversion lower bound
on calibrated eps_v and the a = 0 criterion; one mis-scored terminal cell; exact terminals;
(c) a_zero_states fractions in results/endgame/eps.jsonl; (d) the loop example (Prop R6c).
Run: engine/.venv/bin/python theory/checks/rt6_leaf.py"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import linprog

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "engine"))


def rm_plus(M, iters=1000):           # verbatim copy of engine/nash.py:47 (avoids importing poke_engine)
    m, n = M.shape
    Rx, Ry = np.zeros(m), np.zeros(n); sx, sy = np.zeros(m), np.zeros(n)
    x, y = np.full(m, 1 / m), np.full(n, 1 / n)
    for t in range(1, iters + 1):
        u = M @ y
        Rx = np.maximum(Rx + u - x @ u, 0)
        x = Rx / Rx.sum() if Rx.sum() > 0 else np.full(m, 1 / m)
        w = -(x @ M)
        Ry = np.maximum(Ry + w - y @ w, 0)
        y = Ry / Ry.sum() if Ry.sum() > 0 else np.full(n, 1 / n)
        sx += t * x; sy += t * y
    return sx / sx.sum(), sy / sy.sum()


def val(M):
    """Row maximin by LP."""
    n, m = M.shape
    c = np.r_[np.zeros(n), -1]
    A = np.c_[-M.T, np.ones(m)]
    r = linprog(c, A_ub=A, b_ub=np.zeros(m), A_eq=np.r_[np.ones(n), 0][None], b_eq=[1],
                bounds=[(0, None)] * n + [(None, None)], method="highs")
    return -r.fun


def calibrate(M, Q):                   # same LP as engine/endgame_eps.py:131
    m, q = M.ravel(), Q.ravel(); one = np.ones_like(m)
    A = np.r_[np.c_[m, one, -one], np.c_[-m, -one, -one]]
    r = linprog([0, 0, 1], A_ub=A, b_ub=np.r_[q, -q], bounds=[(0, None), (None, None), (0, None)], method="highs")
    return r.x[0], r.x[1], r.x[2]


def inversion_lb(M, Q):
    m, q = M.ravel(), Q.ravel()
    dq = q[:, None] - q[None, :]; inv = m[:, None] <= m[None, :]
    return max(0.0, (dq * inv).max() / 2)


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    print("(a) affine invariance of nash.rm_plus (copy) and eps_s scaling")
    worst, ratio = 0.0, []
    for _ in range(200):
        n, m = rng.integers(2, 8, 2); M = rng.normal(size=(n, m)) * rng.uniform(1, 200)
        a, c = rng.uniform(1e-3, 1e3), rng.normal() * 500
        x1, _ = rm_plus(M, 300); x2, _ = rm_plus(a * M + c, 300)
        worst = max(worst, np.abs(x1 - x2).max())
        es1 = val(M) - (x1 @ M).min(); es2 = val(a * M + c) - (x2 @ (a * M + c)).min()
        if es1 > 1e-9:
            ratio.append(es2 / (a * es1))
    print(f"   max |x(M) - x(aM+c)| over 200 random (M,a,c): {worst:.2e}; eps_s(aM+c)/(a eps_s(M)) in "
          f"[{min(ratio):.6f}, {max(ratio):.6f}]")
    print("(b) calibrated eps_v vs the inversion lower bound and the a = 0 criterion")
    viol, crit_ok, crit_n = 0, 0, 0
    for _ in range(500):
        n, m = rng.integers(2, 6, 2); Q = np.tanh(rng.normal(size=(n, m)))
        M = 200 * np.arctanh(np.clip(Q, -.99, .99)) + rng.normal(size=(n, m)) * rng.uniform(0, 300)
        a, c, e = calibrate(M, Q)
        viol += e < inversion_lb(M, Q) - 1e-9
        i, j = Q.argmax(), Q.argmin()
        if M.ravel()[i] <= M.ravel()[j]:
            crit_n += 1; crit_ok += abs(e - np.ptp(Q) / 2) < 1e-7
    print(f"   eps_cal < inversion bound: {viol}/500; argmax-Q cell scored <= argmin-Q cell: {crit_n} cases, "
          f"eps_cal = range(Q*)/2 in {crit_ok}/{crit_n}")
    # one mis-scored terminal: live cells Q* = tanh(M/215), plus a won terminal (Q*=+1) scored by material (-50)
    Ml = np.linspace(-215, 215, 11); Ql = np.tanh(Ml / 215 * 1.5)
    for mt in (185, 0, -50):
        M = np.r_[Ml, mt][None]; Q = np.r_[Ql, 1.0][None]
        a, c, e = calibrate(M, Q)
        print(f"   won terminal scored {mt:4d}: calibrated a={a:.5f}, eps_cal={e:.3f}, inversion LB={inversion_lb(M, Q):.3f}")
    M = np.r_[Ml, 0][None]; Q = np.r_[Ql, 1.0][None]; a, c, e = calibrate(M[:, :-1], Q[:, :-1])
    print(f"   same live cells with the terminal set exact (excluded from calibration): a={a:.5f}, eps_cal={e:.3f}")
    # criterion case: a won terminal (Q*=+1, material score +20) below the worst live cell (Q*=-0.2, score +60)
    M = np.array([[20., 60., 40.], [-10., 0., 30.]]); Q = np.array([[1.0, -0.2, 0.3], [0.1, 0.2, 0.6]])
    a, c, e = calibrate(M, Q)
    print(f"   won terminal (+1) scored 20 < worst cell (Q*=-0.2) scored 60: a={a:.5f}, eps_cal={e:.3f}, range(Q*)/2={np.ptp(Q) / 2:.3f}")
    Mx = M.copy(); Mx[0, 0] = 400.0; a, c, e = calibrate(Mx, Q)
    print(f"   same matrix, terminal scored exactly at the top (400): a={a:.5f}, eps_cal={e:.3f}")
    # exact terminals: cell error <= P(live child) * live leaf error
    bad = 0
    for _ in range(500):
        k = 6; P = rng.dirichlet(np.ones(k)); term = rng.random(k) < .5
        Vs = rng.uniform(-1, 1, k); Vs[term] = rng.choice([-1., 1.], term.sum())
        Vh = Vs + (~term) * rng.uniform(-.5, .5, k)
        err = abs(P @ Vh - P @ Vs); bound = P[~term].sum() * np.abs(Vh - Vs)[~term].max(initial=0)
        bad += err > bound + 1e-12
    print(f"   exact terminals: |Qhat - Q*| > P(live) * max live leaf error in {bad}/500 random cells")
    print("(c) a = 0 states in results/endgame/eps.jsonl (per-state calibration slope exactly 0)")
    seen = set()
    for l in (ROOT / "engine/results/endgame/eps.jsonl").read_text().splitlines():
        r = json.loads(l)
        if "a_zero_states" in r and r["case"] not in seen:
            seen.add(r["case"])
            print(f"   {r['case'][:48]:48s} a=0 in {r['a_zero_states']:5d}/{r['n_live']:5d} live states "
                  f"({r['a_zero_states'] / r['n_live']:.0%}); max/mean eps_v,cal {r['eps_v_cal'][0]:.3f}/{r['eps_v_cal'][1]:.3f}")
    print("(d) loop example: agent {E,C}, opponent {E,C}; (E,.) -> 0.3, (C,E) -> 0.8, (C,C) -> s")
    Qof = lambda V: np.array([[0.3, 0.3], [0.8, V]])
    fps = [V for V in np.linspace(-1, 1, 2001) if abs(val(Qof(V)) - V) < 1e-9]
    print(f"   Shapley fixed points on a grid: [{min(fps):.3f}, {max(fps):.3f}] ({len(fps)} grid points)")
    V = 0.0
    for _ in range(50):
        V = val(Qof(V))
    print(f"   value iteration from 0 (= finite-horizon values with 0 at the cap): {V:.3f}")
    x = np.array([0., 1.])     # stage maximin of Q(0.3): row C (weakly dominant), min_b = 0.3
    print(f"   stage-wise agent x=C at V*=0.3: min_b x.Q(0.3)[:,b] = {(x @ Qof(0.3)).min():.3f} = val {val(Qof(0.3)):.3f}")
    W = 0.0                     # BR evaluation with endless play = 0: W = min_b x.[0.3, 0.3 ; 0.8, W]
    for _ in range(200):
        W = min(x @ Qof(W)[:, 0], x @ Qof(W)[:, 1])
    print(f"   best response vs stationary C, endless = 0 (iteration from 0): W = {W:.3f}; exploitability {0.3 - W:.3f}")
    Wv = 0.3
    for _ in range(200):
        Wv = min(x @ Qof(Wv)[:, 0], x @ Qof(Wv)[:, 1])
    print(f"   the same evaluation started from V* (endless valued at V*): W = {Wv:.3f}  <- the misleading fixed point")
    for H in (1, 2, 5, 300):
        Vt, strat = 0.0, []
        for t in reversed(range(H)):
            Q = Qof(Vt); v = val(Q); strat.append("E" if (Q[1] - v).min() < -1e-12 else "C or E"); Vt = v
        print(f"   cap H={H:3d}: backward-induction value {Vt:.3f}; last-stage agent must play {strat[0]}")
