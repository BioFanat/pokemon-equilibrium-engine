"""Prop R1: incremental anytime MW (logw += eta_t r_t, eta_t = c/sqrt(t)) has linear regret on the
two-phase sequence; regret/T -> kappa(alpha) = min(2 alpha, 1 - 2 alpha)^+ ; FTRL form stays O(sqrt T).
Also runs engine/exp3_live.Learner itself in its incremental branch (eta=None forced to incremental)
for the alpha = 0.3 headline. numpy only.  Run: engine/.venv/bin/python theory/checks/rt1_incremental_mw.py"""
import numpy as np


def kappa(a):
    return max(0.0, min(2 * a, 1 - 2 * a))


def run(T, alpha, k=3, form="incremental"):
    """Rewards: expert 0 gets 1 on t <= floor(alpha T), expert 1 gets 1 afterwards, others 0.
    Weights at round t use rewards of rounds < t. Returns (regret, learner reward)."""
    c = np.sqrt(8 * np.log(k))
    t = np.arange(1, T + 1)
    T1 = int(np.floor(alpha * T))
    R = np.zeros((T, k)); R[:T1, 0] = 1; R[T1:, 1] = 1
    if form == "incremental":            # logw_t = sum_{s<t} eta_s r_s, eta_s = c/sqrt(s)
        S = np.vstack([np.zeros(k), np.cumsum(R * (c / np.sqrt(t))[:, None], 0)[:-1]])
    else:                                # code's FTRL indexing: round t uses eta_{t-1} * L_{t-1}
        L = np.vstack([np.zeros(k), np.cumsum(R, 0)[:-1]])
        eta = np.r_[c, c / np.sqrt(t[:-1])]
        S = L * eta[:, None]
    W = np.exp(S - S.max(1, keepdims=True)); W /= W.sum(1, keepdims=True)
    got = float((W * R).sum())
    return R.sum(0).max() - got, got


if __name__ == "__main__":
    print(f"{'alpha':>5s} {'kappa':>6s} " + " ".join(f"{'T=' + str(T):>10s}" for T in (10**4, 10**5, 10**6))
          + "   FTRL regret/sqrt(T ln k) at T=1e6")
    for a in (0.05, 0.1, 0.2, 0.25, 0.3, 0.4, 0.45, 0.5, 0.6):
        rs = [run(T, a)[0] / T for T in (10**4, 10**5, 10**6)]
        f = run(10**6, a, form="ftrl")[0] / np.sqrt(10**6 * np.log(3))
        print(f"{a:5.2f} {kappa(a):6.3f} " + " ".join(f"{r:10.4f}" for r in rs) + f"   {f:.4f}")
    for k in (2, 3, 10):
        print(f"k={k:2d} alpha=0.3 T=1e5: incremental regret/T = {run(10**5, .3, k)[0] / 1e5:.4f}")
    # the shipped Learner's incremental branch, for the headline 0.40 T (exp3_live with logw += eta_t r)
    import sys
    sys.path.insert(0, __file__.rsplit("/theory/", 1)[0] + "/engine")
    try:
        import exp3_live
        L = exp3_live.Learner(alpha=1e-300)   # alpha>0 routes to the incremental branch; share ~ 0
        T, got, cum = 10**4, 0.0, np.zeros(3)
        A = np.array([[400.], [0.]])
        for t in range(1, T + 1):
            r = np.array([1., 0, 0]) if t <= 0.3 * T else np.array([0, 1., 0])
            got += L.weights() @ r; cum += r
            L.update(A, np.stack([[x, 1 - x] for x in r]), 0)
        print(f"exp3_live.Learner incremental branch, alpha=0.3, T=1e4: regret/T = {(cum.max() - got) / T:.4f}")
    except Exception as e:  # poke_engine missing etc.
        print("exp3_live import skipped:", e)
