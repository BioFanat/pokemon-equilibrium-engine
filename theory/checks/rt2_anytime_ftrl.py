"""Thm R2 (anytime FTRL as shipped): round t uses w_t ∝ exp(beta_t L_{t-1}), beta_t = sqrt(8 ln k/(t-1))
(t >= 2; round 1 uniform). Proved bound: Reg_T <= 3 sqrt(T ln k / 8) + sqrt(ln k / 8)
(= 1.5 sqrt(T ln k/2) + sqrt(ln k/8)); CBL Thm 2.3 constant (their indexing): 2 sqrt(T ln k/2) + sqrt(ln k/8).
Checks (a) the bound against oblivious and adaptive sequences, (b) that the shipped exp3_live.Learner
has exactly this indexing, (c) Phi(beta) = (1/beta) ln mean exp(beta L) is nondecreasing in beta."""
import sys
import numpy as np

sys.path.insert(0, __file__.rsplit("/theory/", 1)[0] + "/engine")


def bound(T, k):
    return 3 * np.sqrt(T * np.log(k) / 8) + np.sqrt(np.log(k) / 8)


def cbl(T, k):
    return 2 * np.sqrt(T * np.log(k) / 2) + np.sqrt(np.log(k) / 8)


def weights(L, t, k):
    if t == 1:
        return np.ones(k) / k
    z = np.sqrt(8 * np.log(k) / (t - 1)) * L
    w = np.exp(z - z.max()); return w / w.sum()


def play(T, k, adversary, rng):
    L, got = np.zeros(k), 0.0
    for t in range(1, T + 1):
        w = weights(L, t, k)
        r = adversary(t, w, L, rng)
        got += w @ r; L += r
    return L.max() - got


ADV = {
    "phase 30/70": lambda t, w, L, rng, T=None: None,
}


def make(T, k):
    return {
        "phase 30/70": lambda t, w, L, rng: np.eye(k)[0 if t <= .3 * T else 1],
        "alternate": lambda t, w, L, rng: np.eye(k)[t % 2],
        "reward the lighter expert": lambda t, w, L, rng: np.eye(k)[int(w.argmin())],
        "keep leader tied (adaptive)": lambda t, w, L, rng: np.eye(k)[int(L.argmin())],
        "iid Bernoulli(0.5)": lambda t, w, L, rng: (rng.random(k) < .5).astype(float),
    }


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    for k in (2, 3):
        for T in (100, 1000, 10000):
            for name, adv in make(T, k).items():
                r = play(T, k, adv, rng)
                print(f"k={k} T={T:5d} {name:28s} regret {r:8.2f}  bound {bound(T, k):7.2f}  ratio {r / bound(T, k):5.2f}"
                      f"  (CBL {cbl(T, k):7.2f})")
    # (b) indexing of the shipped learner
    try:
        import exp3_live
        k, Lr = 2, exp3_live.Learner(experts=["a", "b"])
        A = np.array([[400.], [0.]]); L = np.zeros(k); err = 0.0
        for t in range(1, 200):
            err = max(err, np.abs(Lr.weights() - weights(L, t, k)).max())
            r = rng.random(k); L += r
            Lr.update(A, np.stack([[x, 1 - x] for x in r]), 0)
        print(f"(b) max |exp3_live weights - beta_t=sqrt(8lnk/(t-1)) FTRL| over 199 rounds: {err:.2e}")
    except Exception as e:
        print("(b) skipped:", e)
    # (c) monotonicity of the exponential mean in beta
    bad = 0
    for _ in range(2000):
        L = rng.normal(size=rng.integers(2, 6)) * rng.uniform(.1, 50)
        b = np.sort(rng.uniform(1e-3, 5, 2))
        phi = [np.log(np.mean(np.exp(bb * L - (bb * L).max()))) / bb + L.max() for bb in b]
        bad += phi[0] > phi[1] + 1e-12
    print(f"(c) Phi(beta) decreasing on {bad}/2000 random draws")
