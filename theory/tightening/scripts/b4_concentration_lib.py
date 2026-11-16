"""Shared pieces of b4_concentration.py (lp, safex, style_dists, the 57 matrices and their style matrices Q)."""
"""B4 quantitative: how fast the style posterior concentrates when the opponent plays a style (realisable), with a
world-sampling mismatch proxy, on the 57 real stage matrices. Also the Bhattacharyya bound and the switch (tracking)
experiment for lam = 1 vs fixed share vs discount-then-predict.

Model: turns are i.i.d. draws of a real stage matrix A (ours). The opponent plays style s* computed on its own matrix
A' = A + sigma_m * ptp(A) * N(0,1) (fresh noise each turn; sigma_m = 0 is exact realisability); b_t ~ q_{s*}(A').
Our posterior uses q_s(A) with p_ref = the one-hot safex strategy (bayes_safex's p_ref). Prior uniform, lam = 1.
  1. pairwise separation on the 57 matrices: Bhattacharyya distance -ln BC(q_s*, q_s) and KL(q_s* || q_s)
  2. concentration: P_T(s*) and MAP accuracy at T in {25, 50, 100, 250, 600} for s* in {eq, safe, uniform} x sigma_m
  3. KL-closest style under mismatch (which style the posterior should converge to; Berk)
  4. bound check at sigma_m = 0: sum_t (1 - P_{t-1}(s*)) vs sum_{s != s*} sum_t prod_{u<t} BC_u(s*, s)  and vs
     sum_s 1/(1 - rho_s) with rho_s = max over matrices of BC
  5. switch: truth s_a for t0 turns then s_b; flip time of the MAP for lam = 1 (prediction t0 * mu_a / mu_b by N1),
     fixed share alpha = 0.01 (B2) and discount-then-predict lam = 0.98 (B3(b)).
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
rng = np.random.default_rng(7)


def lp(A):
    m, n = A.shape
    c = np.zeros(m + 1); c[-1] = -1
    r = linprog(c, A_ub=np.hstack([-A.T, np.ones((n, 1))]), b_ub=np.zeros(n), A_eq=np.hstack([np.ones((1, m)), np.zeros((1, 1))]),
                b_eq=[1], bounds=[(0, None)] * m + [(None, None)], method="highs")
    x = np.clip(r.x[:m], 0, None); x /= x.sum(); v = r.x[-1]
    c2 = np.zeros(n + 1); c2[-1] = 1
    r2 = linprog(c2, A_ub=np.hstack([A, -np.ones((m, 1))]), b_ub=np.zeros(m), A_eq=np.hstack([np.ones((1, n)), np.zeros((1, 1))]),
                 b_eq=[1], bounds=[(0, None)] * n + [(None, None)], method="highs")
    y = np.clip(r2.x[:n], 0, None); y /= y.sum()
    return x, y, v


def safex(A, v, q, eps=EPS):
    m, n = A.shape; r = float(np.ptp(A)) or 1.0
    res = linprog(-(A @ q), A_ub=-A.T, b_ub=-(v - eps * r) * np.ones(n), A_eq=np.ones((1, m)), b_eq=[1], bounds=[(0, None)] * m, method="highs")
    if res.status != 0:
        return None
    x = np.clip(res.x, 0, None); return x / x.sum()


def softmax(z):
    e = np.exp(z - z.max()); return e / e.sum()


def style_dists(A, p=None):
    n = A.shape[1]; u = np.full(n, 1 / n)
    if n == 1:
        return np.ones((5, 1))
    x_eq, y_eq, v = lp(A)
    if p is None:
        q0 = np.zeros(n); q0[int(A.mean(0).argmin())] = 1
        p = safex(A, v, q0)
        if p is None:
            p = x_eq
    tau = TAU * float(np.ptp(A)) or 1.0
    raw = np.stack([y_eq, softmax(-A.mean(0) / tau), softmax(-(p @ A) / tau), softmax(-A.max(0) / tau), u])
    return (1 - MIX) * raw + MIX * u


d = np.load(MATS, allow_pickle=True)
mats = [np.asarray(d[f], float) for f in d.files if np.asarray(d[f]).shape[1] >= 2 and np.asarray(d[f]).shape[0] >= 2]
Q = [style_dists(A) for A in mats]

