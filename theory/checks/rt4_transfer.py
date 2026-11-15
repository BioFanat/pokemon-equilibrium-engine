"""Cor R4 (surrogate-to-true transfer along a trajectory): random stage matrices Q*, calibrated surrogates
A = Q* + E with |E| <= eps_t, random x_t, q_t, columns b_t; checks (a), (b), (c) of Cor R4.
Run: engine/.venv/bin/python theory/checks/rt4_transfer.py"""
import numpy as np
from scipy.optimize import linprog


def val(M):
    n, m = M.shape
    r = linprog(np.r_[np.zeros(n), -1], A_ub=np.c_[-M.T, np.ones(m)], b_ub=np.zeros(m),
                A_eq=np.r_[np.ones(n), 0][None], b_eq=[1], bounds=[(0, None)] * n + [(None, None)], method="highs")
    return -r.fun


rng = np.random.default_rng(0)
worst = np.full(3, np.inf)
for trial in range(200):
    T = rng.integers(1, 30); lhs = np.zeros(3); rhs = np.zeros(3)
    for t in range(T):
        n, m = rng.integers(2, 5, 2); eps = rng.uniform(0, .5)
        Q = rng.uniform(-1, 1, (n, m)); A = Q + rng.uniform(-eps, eps, (n, m))
        e = np.abs(A - Q).max()
        x, q = rng.dirichlet(np.ones(n)), rng.dirichlet(np.ones(n)); b = rng.integers(m)
        lhs += [x @ Q[:, b], (x - q) @ Q[:, b], x @ Q[:, b] - val(Q)]
        rhs += [x @ A[:, b] - e, (x - q) @ A[:, b] - 2 * e, x @ A[:, b] - val(A) - 2 * e]
    worst = np.minimum(worst, lhs - rhs)
print("min over 200 random trajectories of (lhs - rhs) for (a), (b), (c):", np.round(worst, 6), "(all must be >= 0)")
