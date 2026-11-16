"""K = 1 reduction: the membership-aware prediction / update equal style_dists / decide / Posterior.update on that one
matrix (bitwise), with and without DTP, and safex_strategy_worlds equals nash.safex_strategy on it (objective value,
guarantee; x up to LP ties).
  ~/sandbox/bot-venv/bin/python research/scripts/joint/tests/test_one_world.py"""
import _path  # noqa: F401
import numpy as np

import behavior
import hidden_agents as ha
import nash

rng = np.random.default_rng(0)
n_pred = n_lp = 0
x_same = 0
for trial in range(200):
    m, n = rng.integers(2, 9), rng.integers(2, 10)
    M = np.round(rng.normal(0, 100, (m, n)), 2)
    cols = [[f"c{j}" for j in range(n)]]
    labels = list(cols[0])                            # avg_matrix with K = 1: Abar = M, labels = cols[0]
    p = rng.dirichlet(np.ones(m))
    for dtp in (False, True):
        post_old = behavior.Posterior(lam=0.98, dtp=dtp)
        post_new = behavior.WorldPosterior(lam=0.98, dtp=dtp)
        for step in range(5):
            qs, qhat, rec = behavior.decide(M, p, post_old, labels)
            qsk, q, qk, mwrec = behavior.decide_worlds([M], cols, np.ones(1), labels, p, post_new)
            assert np.array_equal(qsk[0], qs), "style_dists_worlds != style_dists"
            assert np.array_equal(q, qhat) and np.array_equal(qk[0], qhat), "prediction != q-hat"
            j = int(rng.integers(n))
            lik_o = post_old.update(qs, j)
            lik_n = post_new.update_worlds(qsk, {0: j}, np.ones(1))
            assert np.array_equal(lik_o, lik_n) and np.array_equal(post_old.logp, post_new.logp), "update differs"
            n_pred += 1
    q = rng.dirichlet(np.ones(n))
    for eps in (0.0, 0.003, 0.05):
        xs = np.asarray(nash.safex_strategy(M, eps, q))
        xw = np.asarray(ha.safex_strategy_worlds([M], [1.0], [q], eps))
        v = nash.lp(M)[2]
        r = float(np.ptp(M)) or 1.0
        assert abs(xs @ M @ q - xw @ M @ q) < 1e-7 * max(1.0, abs(xs @ M @ q)), (trial, eps, xs @ M @ q, xw @ M @ q)
        assert (xw @ M).min() >= v - eps * r - 1e-7 * r, "guarantee violated"
        x_same += int(np.abs(xs - xw).max() < 1e-7)
        n_lp += 1
print(f"one-world prediction/update: {n_pred} checks bitwise equal (DTP off and on)")
print(f"one-world safex LP: {n_lp} instances, objective equal (rel 1e-7) and guarantee met; "
      f"x equal within 1e-7 on {x_same}/{n_lp} (rest: alternative optima)")
print("PASS")
