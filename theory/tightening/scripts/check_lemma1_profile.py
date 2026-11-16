"""Numeric check of the profile-restricted one-step lemma (Lemma 1').
For Q (true), B (estimate), x eps_s-optimal in B, x* maximin in Q, yhat minimax in B:
   loss(x) := val Q - min_y x^T Q y  <=  eps_s + x*^T (Q - B) yhat + max_y x^T (B - Q) y  <=  eps_s + 2 ||Q - B||_inf.
Also checks the exact three-term split for an arbitrary column sigma:
   V - x^T Q sigma = (val Q - val B) + (val B - x^T B sigma) + x^T (B - Q) sigma.
cwd engine; engine/.venv/bin/python check_lemma1_profile.py
"""
import sys, numpy as np
sys.path.insert(0, "/Users/biofanat/Documents/classes/f26/cs535/final/engine")
import nash
rng = np.random.default_rng(0)
viol = tight = 0; ratios = []; worst = 0.0
for trial in range(2000):
    n, m = rng.integers(2, 7, size=2)
    Q = rng.uniform(-1, 1, (n, m))
    kind = trial % 4
    if kind == 0:   E = rng.uniform(-0.3, 0.3, (n, m))            # dense noise
    elif kind == 1: E = rng.uniform(-0.3, 0.3, (n, 1)) * np.ones((1, m))   # row-only error
    elif kind == 2: E = np.ones((n, 1)) * rng.uniform(-0.3, 0.3, (1, m))   # column-only error
    else:           E = np.zeros((n, m)); E[rng.integers(n), rng.integers(m)] = rng.uniform(-1, 1)  # one cell
    B = Q + E
    xB, yB, vB = nash.lp(B); xQ, yQ, vQ = nash.lp(Q)
    xB = np.clip(xB, 0, None); xB /= xB.sum(); yB = np.clip(yB, 0, None); yB /= yB.sum()
    xQ = np.clip(xQ, 0, None); xQ /= xQ.sum()
    eps_s = max(0.0, vB - (xB @ B).min())
    loss = vQ - (xB @ Q).min()
    bud3 = eps_s + xQ @ (Q - B) @ yB + (xB @ (B - Q)).max()
    bud2 = eps_s + 2 * np.abs(E).max()
    if loss > bud3 + 1e-7: viol += 1; worst = max(worst, loss - bud3)
    if bud3 > bud2 + 1e-7: viol += 1
    ratios.append((bud3 + 1e-12) / (bud2 + 1e-12))
    sigma = rng.dirichlet(np.ones(m))
    split = (vQ - vB) + (vB - xB @ B @ sigma) + xB @ (B - Q) @ sigma
    assert abs(split - (vQ - xB @ Q @ sigma)) < 1e-9
r = np.array(ratios)
print(f"violations of loss <= bud3 <= bud2: {viol}/2000 (worst excess {worst:.2e}); split identity exact")
print(f"bud3/bud2 ratio: median {np.median(r):.3f}, 10% {np.percentile(r,10):.3f}, 90% {np.percentile(r,90):.3f}, max {r.max():.3f}")
