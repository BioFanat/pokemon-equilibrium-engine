"""B3(c), realisable version: sequences of T real (matrix, column) loss vectors (style matrices from behavior.style_dists
on the 57 real stage matrices, p_ref uniform) that make the CODED forgetting exceed the discounted bound ln 5.
Hill-climbing over index sequences (each turn = one of the 569 real loss vectors)."""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path[:0] = [os.path.join(ROOT, "engine"), os.path.join(ROOT, "engine", "showdown")]
import behavior  # noqa: E402

MATS = "/private/tmp/claude-501/-Users-biofanat-Documents-classes-f26-cs535/2bbe670b-2f97-4d9a-bfad-e1139e0352c6/scratchpad/experts_ext/real_matrices.npz"
rng = np.random.default_rng(3)
d = np.load(MATS, allow_pickle=True)
cols = []
for f in d.files:
    A = np.asarray(d[f], float)
    if A.shape[1] < 2:
        continue
    Qlog = np.log(behavior.style_dists(A, np.full(A.shape[0], 1.0 / A.shape[0])))
    cols += [-Qlog[:, j] for j in range(A.shape[1])]
cols = np.array(cols)
N = len(cols)


def lse(z):
    m = z.max(); return m + np.log(np.exp(z - m).sum())


def disc_regret(L, lam):
    T, k = L.shape; D = np.zeros(k); lh = np.zeros(T)
    for t in range(T):
        logP = -D - lse(-D); lh[t] = -lse(logP - L[t]); D = lam * D + L[t]
    w = lam ** (T - 1 - np.arange(T)); return (w @ lh - w @ L).max()


def shipped(L, lam):
    post = behavior.Posterior(lam=lam); lh = []
    for t in range(L.shape[0]):
        qs = np.exp(-L[t])[:, None]; lh.append(-np.log(float(post.predict(qs)[0]))); post.update(qs, 0)
    w = lam ** (L.shape[0] - 1 - np.arange(L.shape[0])); return (w @ np.array(lh) - w @ L).max()


for lam in (0.5, 0.7, 0.9, 0.95):
    for T in (2, 3, 4, 6, 8):
        best, bidx = -np.inf, None
        for restart in range(8):
            idx = rng.integers(N, size=T); cur = disc_regret(cols[idx], lam)
            for it in range(1500):
                idx2 = idx.copy(); pos = rng.integers(T); idx2[pos] = rng.integers(N)
                v = disc_regret(cols[idx2], lam)
                if v >= cur:
                    idx, cur = idx2, v
            if cur > best:
                best, bidx = cur, idx.copy()
        print(f"lam {lam} T {T}: best realisable discounted regret {best:.3f} (shipped class {shipped(cols[bidx], lam):.3f}) vs ln5 1.609 {'VIOLATION' if best > np.log(5) else ''}")
        if best > np.log(5) and T <= 4:
            print("   loss vectors (eq, br_unif, br_us, safe, uniform):\n", cols[bidx].round(2))
