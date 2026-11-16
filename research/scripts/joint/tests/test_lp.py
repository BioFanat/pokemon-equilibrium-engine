"""safex_strategy_worlds on random multiworld instances: the agent-form guarantee Σ_k w_k min_j (M_kᵀ x)_j >= v_AF -
eps * ptp holds within 1e-7 (absolute; payoffs ~ N(0, 100)); with eps = 0 the guarantee equals v_AF (x is agent-form optimal) and the
objective is >= that of agent_form_lp's own x (which is feasible); the objective never decreases in eps.
  ~/sandbox/bot-venv/bin/python research/scripts/joint/tests/test_lp.py"""
import _path  # noqa: F401
import numpy as np

import hidden_agents as ha

rng = np.random.default_rng(2)
worst = worst0 = worst_abs = 0.0
n = 0
for trial in range(300):
    K, m = int(rng.integers(1, 6)), int(rng.integers(2, 10))
    Ms = [np.round(rng.normal(0, 100, (m, int(rng.integers(1, 10)))), 2) for _ in range(K)]
    w = rng.dirichlet(np.ones(K))
    qs = [rng.dirichlet(np.ones(M.shape[1])) for M in Ms]
    x_af, v_af = ha.agent_form_lp(Ms, w)
    r = float(np.ptp(np.concatenate([M.ravel() for M in Ms]))) or 1.0
    obj = lambda x: float(sum(wk * (x @ M @ q) for wk, M, q in zip(w, Ms, qs)))
    prev = -np.inf
    for eps in (0.0, 0.003, 0.03, 0.3):
        x = np.asarray(ha.safex_strategy_worlds(Ms, w, qs, eps))
        assert x.min() >= -1e-9 and abs(x.sum() - 1) < 1e-9
        g = ha.f_b(Ms, w, x)
        slack = (v_af - eps * r) - g                    # > 0 means violated
        worst, worst_abs = max(worst, slack / r), max(worst_abs, slack)
        assert slack <= 1e-7, (trial, eps, slack)
        if eps == 0:
            worst0 = max(worst0, abs(g - v_af) / r)
            assert abs(g - v_af) <= 1e-7, (trial, g, v_af)
            assert obj(x) >= obj(x_af) - 1e-7 * r
        assert obj(x) >= prev - 1e-7 * r
        prev = obj(x)
        n += 1
print(f"LP: {n} solves; max guarantee violation {worst_abs:.2e} abs, {worst:.2e} / ptp; eps=0 max |guarantee - v_AF| / ptp {worst0:.2e}")
print("PASS")
