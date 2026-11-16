"""Independent numeric checks for bayes_dynamics.md T3 (claim 12) and T4(a) band (claim 11).
Uses the coded behavior.style_dists / Posterior and nash.safex_strategy / nash.lp on the 57 real matrices.
python v_t3_b12.py
"""
import sys
import numpy as np
sys.path.insert(0, "/Users/biofanat/Documents/classes/f26/cs535/final/engine/showdown")
sys.path.insert(0, "/Users/biofanat/Documents/classes/f26/cs535/final/engine")
import nash
from behavior import style_dists, Posterior, STYLES

MATS = "/private/tmp/claude-501/-Users-biofanat-Documents-classes-f26-cs535/2bbe670b-2f97-4d9a-bfad-e1139e0352c6/scratchpad/experts_ext/real_matrices.npz"
d = np.load(MATS, allow_pickle=True)
mats = [np.asarray(d[f], float) for f in d.files if np.asarray(d[f]).shape[0] >= 2 and np.asarray(d[f]).shape[1] >= 2]
print(f"{len(mats)} matrices; cols median {np.median([A.shape[1] for A in mats])}")
rng = np.random.default_rng(3)
EPS = 0.003

# ---- T4(a) band: two safex players on a shared matrix, arbitrary predictions q
worst_band = 0; worst_expl = 0
for A in mats:
    x_eq, y_eq, v = nash.lp(A); r = float(np.ptp(A)) or 1.0
    for _ in range(50):
        qc = rng.dirichlet(np.ones(A.shape[1]) * 0.3); qr = rng.dirichlet(np.ones(A.shape[0]) * 0.3)
        x = np.clip(nash.safex_strategy(A, EPS, qc), 0, None); x /= x.sum()
        y = np.clip(nash.safex_strategy(-A.T, EPS, qr), 0, None); y /= y.sum()
        band = (x @ A @ y - v) / r
        worst_band = max(worst_band, abs(band))
        nash_gap = ((A @ y).max() - x @ A @ y + x @ A @ y - (x @ A).min()) / r
        worst_expl = max(worst_expl, nash_gap)
print(f"[T4a] max |x'Ay - v|/range = {worst_band:.5f} (claim <= {EPS}); max NE gap/range = {worst_expl:.5f} (claim <= {2*EPS})")

# ---- T3: mean-BC bound vs realised sum_t (1 - P_{t-1}(s*)), i.i.d. matrices, exact realisability
Qs = []
for A in mats:
    p = np.clip(nash.safex_strategy(A, EPS), 0, None); p /= p.sum()
    Qs.append(style_dists(A, p))
BC = np.array([[[np.sqrt(q[a] * q[b]).sum() for b in range(5)] for a in range(5)] for q in Qs])
T, NP = 600, 40
for s in [0, 3, 4, 1]:
    others = [b for b in range(5) if b != s]
    rbar = [BC[:, s, b].mean() for b in others]
    rsup = [BC[:, s, b].max() for b in others]
    bound = sum(1 / (1 - r) for r in rbar)
    lhs = []
    for path in range(NP):
        post = Posterior(lam=1.0); tot = 0.0
        for t in range(T):
            q = Qs[rng.integers(len(Qs))]
            tot += 1 - post.probs()[s]
            j = rng.choice(q.shape[1], p=q[s] / q[s].sum())
            post.update(q, j)
        lhs.append(tot)
    lhs = np.array(lhs)
    print(f"[T3] truth {STYLES[s]:8s}: realised mean {lhs.mean():.2f} (SE {lhs.std(ddof=1)/np.sqrt(NP):.2f}, max {lhs.max():.2f}) "
          f"| bound(v) sum 1/(1-mean BC) = {bound:.1f} | sup rho_s = {np.round(rsup,3)}")
