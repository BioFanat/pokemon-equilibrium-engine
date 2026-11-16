"""Membership: a label absent from world k gets no mass from world k, in the prediction and in the likelihood; the
prediction sums to 1 over the union; MIX keeps every likelihood > 0 when some world has the label.
  ~/sandbox/bot-venv/bin/python research/scripts/joint/tests/test_membership.py"""
import _path  # noqa: F401
import numpy as np

import behavior

rng = np.random.default_rng(1)
pool = [f"a{i}" for i in range(12)]
checks = 0
for trial in range(300):
    K, m = int(rng.integers(2, 6)), int(rng.integers(2, 7))
    cols = [list(rng.choice(pool, int(rng.integers(1, 8)), replace=False)) for _ in range(K)]
    labels = list(dict.fromkeys(c for cs in cols for c in cs))
    Ms = [rng.normal(0, 50, (m, len(c))) for c in cols]
    w = rng.dirichlet(np.ones(K))
    post = behavior.WorldPosterior()
    post.logp = np.log(rng.dirichlet(np.ones(len(behavior.STYLES))))
    qsk, q, qk, _ = behavior.decide_worlds(Ms, cols, w, labels, rng.dirichlet(np.ones(m)), post)
    assert abs(q.sum() - 1) < 1e-12
    for i, lab in enumerate(labels):
        expect = sum(w[k] * qk[k][cols[k].index(lab)] for k in range(K) if lab in cols[k])
        assert abs(q[i] - expect) < 1e-12, "world without the label contributed"
        # dropping every world that lacks the label from the mixture changes nothing
        jk = {k: cols[k].index(lab) for k in range(K) if lab in cols[k]}
        lik = behavior.lik_worlds(qsk, jk, w)
        ref = sum(w[k] * qsk[k][:, j] for k, j in jk.items())
        assert np.array_equal(lik, ref) and (lik > 0).all()
        assert abs(post.probs() @ lik - q[i]) < 1e-12, "likelihood and prediction disagree"
        checks += 1
# hand case: label 'b' only in world 1 -> q(b) = w_1 q̂_1(b), world 0 gives 0
M0, M1 = np.array([[1.0, 2.0], [3.0, 0.0]]), np.array([[1.0, 5.0], [3.0, -1.0]])
post = behavior.WorldPosterior()
qsk, q, qk, _ = behavior.decide_worlds([M0, M1], [["a", "c"], ["a", "b"]], np.array([0.5, 0.5]), ["a", "c", "b"],
                                       np.array([0.5, 0.5]), post)
assert q[2] == 0.5 * qk[1][1] and q[1] == 0.5 * qk[0][1] and abs(q[0] - 0.5 * (qk[0][0] + qk[1][0])) < 1e-15
print(f"membership: {checks} (label, instance) checks + hand case; absent worlds contribute exactly 0")
print("PASS")
