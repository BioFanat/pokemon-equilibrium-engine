"""Theorem B / B' / Remark 1 (full_game.md §5): brute-force checks on random small instances.
S = sup_x |f_bhat(x) - f_K(x)| is computed EXACTLY by LP enumeration over pure column profiles
(f is concave piecewise-linear; -f = max over profiles of linear), cross-checked by dense sampling."""
import itertools, numpy as np
from scipy.optimize import linprog

def g(M, x): return (M.T @ x).min()
def f(Ms, w, x): return sum(wi * g(M, x) for wi, M in zip(w, Ms) if wi > 0)

def agent_form(Ms, w):
    n, K = Ms[0].shape[0], len(Ms); A = []
    for k, M in enumerate(Ms):
        blk = np.zeros((M.shape[1], n + K)); blk[:, :n] = -M.T; blk[:, n + k] = 1; A.append(blk)
    r = linprog(np.r_[np.zeros(n), -np.asarray(w)], A_ub=np.vstack(A), b_ub=np.zeros(sum(M.shape[1] for M in Ms)),
                A_eq=np.r_[np.ones(n), np.zeros(K)][None], b_eq=[1], bounds=[(0, None)] * n + [(None, None)] * K, method="highs")
    x = np.clip(r.x[:n], 0, None); return x / x.sum(), -r.fun

def sup_diff(Ms, w1, w2):
    """max_x f_{w1}(x) - f_{w2}(x): enumerate pure profiles for the w2 part (its negation is a max of linear fns)."""
    idx = [i for i in range(len(Ms)) if w2[i] > 0]
    best = -np.inf
    for prof in itertools.product(*[range(Ms[i].shape[1]) for i in idx]):
        lin = sum(w2[i] * Ms[i][:, j] for i, j in zip(idx, prof))   # f_{w2}(x) <= lin.x ; f_{w1}-lin.x is concave
        # max_x f_{w1}(x) - lin.x  : LP with per-type epigraph
        n, T = Ms[0].shape[0], len(Ms); A, rhs = [], []
        for k, M in enumerate(Ms):
            if w1[k] == 0: continue
            blk = np.zeros((M.shape[1], n + T)); blk[:, :n] = -M.T; blk[:, n + k] = 1; A.append(blk)
        c = np.r_[lin, -np.asarray(w1)]
        r = linprog(c, A_ub=np.vstack(A), b_ub=np.zeros(sum(b.shape[0] for b in A)), A_eq=np.r_[np.ones(n), np.zeros(T)][None],
                    b_eq=[1], bounds=[(0, None)] * n + [(None if w1[k] > 0 else 0, None if w1[k] > 0 else 0) for k in range(T)], method="highs")
        best = max(best, -r.fun)
    return best

def S_exact(Ms, w1, w2): return max(sup_diff(Ms, w1, w2), sup_diff(Ms, w2, w1))

rng = np.random.default_rng(0)
def instance(n, T, mcols, Delta):
    lo = rng.uniform(-1, 1); Ms = [lo + Delta * rng.uniform(0, 1, (n, rng.integers(1, mcols + 1))) for _ in range(T)]
    return Ms

# ---------- Theorem B
worst = 0; viol = 0; N = 1500; worst_tv = 0
for it in range(N):
    n, T, K, Delta = rng.integers(2, 4), rng.integers(2, 4), rng.integers(1, 8), 2.0
    Ms = instance(n, T, 3, Delta)
    b = rng.dirichlet(np.ones(T)); bh = rng.dirichlet(np.ones(T)) if rng.random() < 0.7 else b.copy()
    ks = rng.choice(T, size=K, p=bh); w = np.bincount(ks, minlength=T) / K
    ev = rng.uniform(0, 0.2) * rng.random()
    Mh = [M + rng.uniform(-ev, ev, M.shape) for M in Ms]
    xh, vh = agent_form(Mh, w)
    if rng.random() < 0.5:                       # deliberately suboptimal output -> eps_s > 0
        xh = 0.7 * xh + 0.3 * rng.dirichlet(np.ones(n))
    eps_s = vh - f(Mh, w, xh); eps_v = max(np.abs(a - c).max() for a, c in zip(Mh, Ms))
    tv = 0.5 * np.abs(b - bh).sum(); S = S_exact(Ms, bh, w)
    _, vb = agent_form(Ms, b)
    lhs = vb - f(Ms, b, xh); rhs = eps_s + 2 * Delta * tv + 2 * eps_v + 2 * S
    Dact = max(M.max() for M in Ms) - min(M.min() for M in Ms)
    rhs_tight = eps_s + 2 * Dact * tv + 2 * eps_v + 2 * S
    if lhs > rhs_tight + 1e-7: viol += 1
    worst = max(worst, lhs / rhs_tight if rhs_tight > 1e-12 else 0)
    # targets' constant with R = half-range: 2R TV = Delta TV
    if tv > 0: worst_tv = max(worst_tv, (lhs - eps_s - 2 * eps_v - 2 * S) / (Dact * tv))
print(f"Theorem B: {N} instances, violations={viol}, max lhs/rhs={worst:.3f} (rhs uses actual range Delta)")
print(f"  max (lhs - eps_s - 2eps_v - 2S)/(Delta*TV) = {worst_tv:.3f}  (>1 would show 'Delta*TV' = targets' 2R*TV with R=half-range fails)")

# ---------- Remark 1 tightness example
for t in (0.1, 0.3, 0.5):
    D = 2.0
    Ms = [np.array([[D / 2], [-D / 2]]), np.array([[-D / 2], [D / 2]]), t * D / (2 * (1 - t)) * np.array([[1.0], [-1.0]])]
    b = np.array([t, 0, 1 - t]); bh = np.array([0, t, 1 - t])
    xs = np.linspace(0, 1, 101); fbh = [f(Ms, bh, np.array([s, 1 - s])) for s in xs]
    _, vb = agent_form(Ms, b)
    print(f"Remark 1 t={t}: max|f_bhat| on simplex={np.max(np.abs(fbh)):.2e}, v(b)={vb:.4f} (tD={t*D}), f_b(e2)={f(Ms,b,np.array([0.,1])):.4f}, loss/(2D TV)={(vb-f(Ms,b,np.array([0.,1])))/(2*D*t):.4f}")

# ---------- Theorem B' : S vs bound, and dense-sampling cross-check of S_exact
print("\nTheorem B': S (exact LP) vs bounds; Delta = entry range")
for (n, T, K) in [(2, 3, 9), (3, 4, 9), (3, 4, 25), (4, 5, 16), (2, 6, 100)]:
    Ss, mism = [], 0
    for rep in range(300):
        Delta = 2.0; Ms = instance(n, T, 3, Delta); bh = rng.dirichlet(np.ones(T))
        ks = rng.choice(T, size=K, p=bh); w = np.bincount(ks, minlength=T) / K
        S = S_exact(Ms, bh, w)
        if rep < 20:
            X = rng.dirichlet(np.ones(n) * 0.3, size=4000); X = np.vstack([X, np.eye(n)])
            Sd = max(abs(f(Ms, bh, x) - f(Ms, w, x)) for x in X)
            if Sd > S + 1e-7: mism += 1
        Dact = max(M.max() for M in Ms) - min(M.min() for M in Ms)
        Ss.append(S / Dact)
    Ss = np.array(Ss); delta = 0.1
    bnd_i = np.sqrt(2 * (n * np.log(K) + np.log(2 / delta)) / K); bnd_ii = np.sqrt(2 * (n * np.log(K) + np.log(2)) / K)
    print(f"  n={n} T={T} K={K}: E[S]/Delta={Ss.mean():.3f} (bound ii {bnd_ii:.3f});  q90 S/Delta={np.quantile(Ss,0.9):.3f} (bound i, delta=.1: {bnd_i:.3f});"
          f" max={Ss.max():.3f}; dense-sampling > exact: {mism}/20")
