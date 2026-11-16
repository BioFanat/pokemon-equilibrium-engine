"""B3(c): the CODED forgetting (predict with P_{t-1} ∝ exp(-D_{t-1}), then D_t = lam*D_{t-1} + l_t) vs the discounted
regret bound sum_t lam^{T-t} (lhat_t - l_{s,t}) <= ln k.

1. Exact 2-step counterexample (k styles; comparator = style 0):
     t=1: style 1 pays 0, every other style pays B0;   t=2: style 0 pays 0, every other style pays B.
   Closed form of the discounted regret vs style 0, optimised over B0 in [0, B]; threshold lam*(B) above which
   the 2-step example no longer exceeds ln k when losses are capped at B = ln(n/MIX).
2. The same paths pushed through behavior.Posterior(lam) (the shipped class) to confirm the closed form.
3. Log-space hill-climbing search over loss matrices in [0, B]^{T x k} for T in {2,3,4,6} (no overflow; the old
   b3c_search.py 'inf' was exp underflow with losses of 500+ nats).
4. Data-dependent bound  DiscReg_T(s) <= ln k + (1-lam) sum_t lam^{T-t} range_s D_{t-1}(s)  checked on every path.
5. Realisable 2-turn example on the 57 real stage matrices (style matrices from behavior.style_dists).
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path[:0] = [os.path.join(ROOT, "engine"), os.path.join(ROOT, "engine", "showdown")]
import behavior  # noqa: E402

MATS = "/private/tmp/claude-501/-Users-biofanat-Documents-classes-f26-cs535/2bbe670b-2f97-4d9a-bfad-e1139e0352c6/scratchpad/experts_ext/real_matrices.npz"
MIX = 0.05
rng = np.random.default_rng(0)


def logsumexp(z):
    m = z.max()
    return m + np.log(np.exp(z - m).sum())


def disc_regret(L, lam, coded=True, return_bound=False):
    """Discounted regret vs every style (vector) for the coded variant (or discount-then-predict); log-space."""
    T, k = L.shape
    D = np.zeros(k)
    lh = np.zeros(T)
    slack = 0.0
    for t in range(T):
        z = -(D if coded else lam * D)
        logP = z - logsumexp(z)
        lh[t] = -logsumexp(logP - L[t])              # mix loss -ln sum_s P(s) e^{-l_s}
        slack = lam * slack + (1 - lam) * (D.max() - D.min())
        D = lam * D + L[t]
    w = lam ** (T - 1 - np.arange(T))
    reg = w @ lh - w @ L
    return (reg, slack) if return_bound else reg


def two_step_closed(k, lam, B0, B):
    """Exact discounted regret vs style 0 of the 2-step example (coded variant)."""
    a = np.log(1 + (k - 1) * np.exp(-B0))
    lh1 = np.log(k) - a
    lh2 = a - np.log(np.exp(-B0) + (1 + (k - 2) * np.exp(-B0)) * np.exp(-B))
    return lam * (lh1 - B0) + lh2


def two_step_L(k, B0, B):
    L = np.full((2, k), B, dtype=float)
    L[0] = B0; L[0, 1] = 0.0
    L[1, 0] = 0.0
    return L


def posterior_path(L, lam):
    """Run the shipped class behavior.Posterior on a loss matrix (style likelihoods e^{-l}); returns lhat per turn."""
    post = behavior.Posterior(lam=lam)
    out = []
    for t in range(L.shape[0]):
        qs = np.exp(-L[t])[:, None]                  # one column: likelihood of the observed column per style
        out.append(-np.log(float(post.predict(qs)[0])))
        post.update(qs, 0)
    return np.array(out)


def H(lam):
    return -lam * np.log(lam) - (1 - lam) * np.log(1 - lam)


print("=== 1. exact 2-step example, k = 5, cap B = ln(n/MIX) ===")
k = 5
for n in (13, 18, 27, 44):
    B = np.log(n / MIX)
    line = []
    for lam in (0.3, 0.5, 0.7, 0.9, 0.95, 0.98, 0.99):
        grid = np.linspace(0, B, 20001)
        vals = two_step_closed(k, lam, grid, B)
        i = vals.argmax()
        line.append(f"lam {lam}: R*={vals[i]:.3f} (B0={grid[i]:.2f}){' VIOL' if vals[i] > np.log(k) + 1e-9 else '     '}")
    # threshold lam*: largest lam whose optimised 2-step regret exceeds ln k
    lams = np.linspace(0.5, 0.9999, 5000)
    ok = [two_step_closed(k, l, np.linspace(0, B, 4001), B).max() > np.log(k) for l in lams]
    lam_star = lams[np.where(ok)[0].max()] if any(ok) else None
    approx = None
    # predicted threshold from  B = ln k + H(lam)/(1-lam)
    f = lambda l: np.log(k) + H(l) / (1 - l) - B
    lo, hi = 0.5, 0.99999
    for _ in range(100):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if f(mid) < 0 else (lo, mid)
    approx = lo
    print(f"n={n:2d} B={B:.3f}  lam*(exact grid)={lam_star:.4f}  lam*(B = ln k + H/(1-lam))={approx:.4f}")
    for s in line:
        print("   ", s)
print("closed-form vs asymptotic (B -> inf, B0 = B): lam ln k + (1-lam) B;  capped optimum ~ lam ln k + (1-lam) B - H(lam)")
for lam in (0.5, 0.7, 0.9):
    B = np.log(44 / MIX)
    grid = np.linspace(0, B, 20001); vals = two_step_closed(k, lam, grid, B)
    print(f"  lam {lam}: optimum {vals.max():.4f}, B0*={grid[vals.argmax()]:.3f} vs B - ln(lam/(1-lam)) = {B - np.log(lam / (1 - lam)):.3f};"
          f" approx {lam * np.log(k) + (1 - lam) * B - H(lam):.4f}")

print("\n=== 2. the shipped class behavior.Posterior(lam) on the optimised 2-step paths ===")
for lam in (0.3, 0.5, 0.7, 0.9, 0.95):
    for n in (13, 44):
        B = np.log(n / MIX)
        B0 = min(B, B - np.log(lam / (1 - lam))) if lam > 0.5 else B
        L = two_step_L(k, B0, B)
        lh = posterior_path(L, lam)
        w = lam ** np.array([1, 0])
        R = w @ lh - w @ L[:, 0]
        R2 = disc_regret(L, lam)[0]
        print(f"lam {lam} n {n}: shipped-class regret vs style 0 = {R:.4f}  (log-space fn {R2:.4f}; closed form "
              f"{two_step_closed(k, lam, B0, B):.4f}; ln5 = {np.log(5):.4f}){'  VIOLATION' if R > np.log(5) else ''}")
    # the (0,C,C,C,C) variant with uncapped C (free losses): regret -> lam ln k + (1-lam) B0 for every lam < 1
L = two_step_L(k, 6.0, 60.0)
for lam in (0.9, 0.99, 0.999):
    print(f"free losses (C=60), lam {lam}: {disc_regret(L, lam)[0]:.4f} vs ln5 {np.log(5):.4f}  (formula {lam*np.log(5)+(1-lam)*6.0:.4f})")

print("\n=== 3. log-space search, losses in [0, B], B = ln(44/MIX) = %.3f ===" % np.log(44 / MIX))
B = np.log(44 / MIX)


def search(T, k, lam, iters=4000):
    best, bestL = -np.inf, None
    for restart in range(6):
        z = rng.normal(0, 2, (T, k))
        L = B / (1 + np.exp(-z)); cur = disc_regret(L, lam).max()
        step = 1.5
        for it in range(iters):
            z2 = z + rng.normal(0, step, (T, k)) * (rng.random((T, k)) < 0.3)
            L2 = B / (1 + np.exp(-z2)); v = disc_regret(L2, lam).max()
            if v > cur:
                z, cur, L = z2, v, L2
            if it % 1000 == 999:
                step *= 0.6
        if cur > best:
            best, bestL = cur, L
    return best, bestL


for lam in (0.5, 0.7, 0.9, 0.95):
    for T, kk in ((2, 2), (2, 5), (3, 5), (4, 5), (6, 5)):
        best, L = search(T, kk, lam)
        two = two_step_closed(kk, lam, np.linspace(0, B, 4001), B).max()
        print(f"lam {lam} T {T} k {kk}: max disc. regret {best:.3f} (ln k {np.log(kk):.3f}; optimised 2-step {two:.3f})")
        if T <= 3 and kk == 5 and lam == 0.9:
            print("    losses\n", L.round(2))

print("\n=== 4. data-dependent bound ln k + (1-lam) sum_t lam^{T-t} range(D_{t-1}) on random + adversarial paths ===")
worst = -np.inf
for lam in (0.5, 0.7, 0.9, 0.99):
    for trial in range(300):
        T = rng.integers(2, 40); kk = rng.choice([2, 3, 5])
        L = rng.uniform(0, B, (T, kk)) if trial % 2 else B * (rng.random((T, kk)) < 0.5)
        reg, slack = disc_regret(L, lam, return_bound=True)
        worst = max(worst, reg.max() - (np.log(kk) + slack))
    L = two_step_L(5, B - np.log(lam / (1 - lam)) if lam > 0.5 else B, B)
    reg, slack = disc_regret(L, lam, return_bound=True)
    print(f"lam {lam}: 2-step example regret {reg[0]:.3f}  bound {np.log(5) + slack:.3f}  (tightness {reg[0]/(np.log(5)+slack):.2f})")
print(f"max (regret - bound) over 1200 random paths: {worst:.2e}  (<= 0 means the bound held)")

print("\n=== 5. realisable 2-turn example on the 57 real stage matrices (style matrices from behavior.style_dists) ===")
d = np.load(MATS, allow_pickle=True)
mats = [d[f] for f in d.files]
Qs = []
for A in mats:
    A = np.asarray(A, float)
    if A.shape[1] < 2:
        continue
    p = np.full(A.shape[0], 1.0 / A.shape[0])
    Qs.append(np.log(behavior.style_dists(A, p)))      # (5 x n) log-likelihoods
# losses of each style for each (matrix, column): l[s] = -log q_s(col)
cols = [(-Q[:, j]) for Q in Qs for j in range(Q.shape[1])]
cols = np.array(cols)                                   # (N, 5)
print(f"{len(cols)} (matrix, column) loss vectors; max loss {cols.max():.3f}; min loss {cols.min():.3f}")
for lam in (0.5, 0.7, 0.9, 0.95):
    best = (-np.inf, None)
    # try all ordered pairs of loss vectors (N^2 ~ 1e5): regret vs each style
    for i in range(len(cols)):
        L1 = cols[i]
        # vectorised over the second turn
        D1 = L1.copy()
        logP0 = np.full(5, -np.log(5))
        lh1 = -logsumexp(logP0 - L1)
        z = -D1; logP1 = z - logsumexp(z)
        lh2 = -np.array([logsumexp(logP1 - L2) for L2 in cols])
        reg = lam * (lh1 - L1)[None, :] + (lh2[:, None] - cols)   # (N, 5)
        j, s = np.unravel_index(reg.argmax(), reg.shape)
        if reg[j, s] > best[0]:
            best = (reg[j, s], (i, j, s))
    (i, j, s) = best[1]
    print(f"lam {lam}: max real 2-turn discounted regret {best[0]:.3f} vs ln5 {np.log(5):.3f} "
          f"{'VIOLATION' if best[0] > np.log(5) else 'no violation'}; comparator {behavior.STYLES[s]}; "
          f"turn-1 losses {cols[i].round(2)}, turn-2 losses {cols[j].round(2)}")
