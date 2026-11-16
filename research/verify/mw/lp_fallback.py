"""Brute-force: can ha.safex_strategy_worlds fall back (res.status != 0)? Also checks K=1 equivalence with
nash.safex_strategy and that the safety constraint holds. Random + degenerate matrices shaped like logged decisions."""
import os, sys
import numpy as np
from scipy.optimize import linprog
ENG = "/Users/biofanat/Documents/classes/f26/cs535/final-integration/engine"
sys.path.insert(0, ENG); sys.path.insert(0, ENG + "/showdown")
import hidden_agents as ha
import nash

rng = np.random.default_rng(0)
fails, viol, k1_diff, n = 0, 0, [], 0
orig = ha.linprog
status = []


def spy(*a, **k):
    r = orig(*a, **k)
    status.append(r.status)
    return r


ha.linprog = spy
for trial in range(3000):
    m = int(rng.integers(2, 14)); K = 4
    kind = trial % 5
    Ms = []
    for k in range(K):
        nk = int(rng.integers(1, 14))
        if kind == 0:
            M = rng.normal(0, 100, (m, nk))
        elif kind == 1:
            M = np.round(rng.normal(0, 100, (m, nk)) / 50) * 50          # many ties
        elif kind == 2:
            M = np.zeros((m, nk))                                         # ptp = 0
        elif kind == 3:
            M = rng.normal(0, 1e4, (m, nk)); M[:, 0] = M[:, 0] * 1e-6     # bad scaling
        else:
            M = np.tile(rng.normal(0, 100, (1, nk)), (m, 1))              # identical rows
        Ms.append(M)
    w = rng.dirichlet(np.ones(K))
    qs = [rng.dirichlet(np.ones(M.shape[1])) for M in Ms]
    status.clear()
    x = ha.safex_strategy_worlds(Ms, w, qs, 0.003)
    n += 1
    if any(s != 0 for s in status):
        fails += 1
    x = np.clip(x, 0, None); x /= x.sum()
    xaf, vaf = ha.agent_form_lp(Ms, w)
    r = float(np.ptp(np.concatenate([M.ravel() for M in Ms]))) or 1.0
    if ha.f_b(Ms, w, x) < vaf - 0.003 * r - 1e-6 * max(1, abs(vaf)):
        viol += 1
    if kind == 0 and trial < 1000:   # K=1 equivalence
        status.clear()
        x1 = ha.safex_strategy_worlds([Ms[0]], [1.0], [qs[0]], 0.003)
        x2 = nash.safex_strategy(Ms[0], 0.003, qs[0])
        v1 = float(x1 @ Ms[0] @ qs[0]); v2 = float(np.clip(x2, 0, None) / np.clip(x2, 0, None).sum() @ Ms[0] @ qs[0])
        k1_diff.append(abs(v1 - v2) / (1 + abs(v2)))
print(f"trials={n} LP non-zero status (fallback) in {fails}; safety-constraint violations {viol}; "
      f"K=1 vs nash.safex_strategy objective rel diff max={max(k1_diff):.2e} over {len(k1_diff)}")
