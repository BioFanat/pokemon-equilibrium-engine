"""Independent check of bayes_dynamics.md T1 (claims 9-10) against the coded engine/showdown/behavior.Posterior.
python v_b3c.py
"""
import sys, math
import numpy as np
from scipy.optimize import brentq, minimize_scalar
sys.path.insert(0, "/Users/biofanat/Documents/classes/f26/cs535/final/engine/showdown")
sys.path.insert(0, "/Users/biofanat/Documents/classes/f26/cs535/final/engine")
from behavior import Posterior, MIX

K = 5
rng = np.random.default_rng(7)


def coded_regret(L, lam, s, post_cls=Posterior, fix=False):
    """anytime-final discounted regret of style s for loss matrix L (T x K), using the coded class.
    qs = exp(-L_t) as a K x 1 'style matrix', observed column 0."""
    p = post_cls(lam=lam)
    T = len(L); R = 0.0
    for t in range(T):
        q = np.exp(-L[t])[:, None]
        lhat = -math.log(float(p.predict(q)[0]))
        R = lam * R + (lhat - L[t, s])
        if fix:
            z = p.lam * (p.logp + np.log(q[:, 0]))
            p.logp = z - (z.max() + np.log(np.exp(z - z.max()).sum()))
        else:
            p.update(q, 0)
    return R


def own_regret(L, lam, s, mode):
    """own implementation: mode 'coded' = predict with exp(-D_{t-1}); 'dtp' = exp(-lam D_{t-1})."""
    D = np.zeros(L.shape[1]); R = 0.0; rngD = []
    for t in range(len(L)):
        w = np.exp(-(D if mode == "coded" else lam * D)); w /= w.sum()
        lhat = -math.log(float(w @ np.exp(-L[t])))
        rngD.append(D.max() - D.min())
        R = lam * R + lhat - L[t, s]
        D = lam * D + L[t]
    return R, np.array(rngD)


def closed_form(lam, B0, B, k=K):
    a = math.log(1 + (k - 1) * math.exp(-B0))
    return lam * (math.log(k) - a - B0) + a - math.log(math.exp(-B0) + (1 + (k - 2) * math.exp(-B0)) * math.exp(-B))


def two_turn(B0, B, k=K):
    L = np.full((2, k), B0); L[0, 1] = 0.0
    L[1] = B; L[1, 0] = 0.0
    return L


# 1. closed form vs coded class vs own implementation
err = 0
for _ in range(200):
    lam = rng.uniform(0.05, 0.999); B = rng.uniform(1, 8); B0 = rng.uniform(0, B)
    L = two_turn(B0, B)
    c1 = closed_form(lam, B0, B); c2 = coded_regret(L, lam, 0); c3 = own_regret(L, lam, 0, "coded")[0]
    err = max(err, abs(c1 - c2), abs(c1 - c3))
print(f"[1] 2-turn closed form vs behavior.Posterior vs own: max abs diff {err:.2e}")


def best2(lam, B):
    r = minimize_scalar(lambda b0: -closed_form(lam, b0, B), bounds=(0, B), method="bounded", options={"xatol": 1e-10})
    g = max(np.linspace(0, B, 2001), key=lambda b0: closed_form(lam, b0, B))
    return max(-r.fun, closed_form(lam, g, B))


H = lambda l: -l * math.log(l) - (1 - l) * math.log(1 - l)
print("[2] thresholds lam*_2(n) (k=5, MIX=%.2f):" % MIX)
for n in [13, 18, 27, 44]:
    B = math.log(n / MIX)
    Brng = math.log(n * (1 - MIX) / MIX + 1)  # attainable per-turn loss range: max -ln(MIX/n) minus min -ln(1-MIX+MIX/n)
    lstar = brentq(lambda l: best2(l, B) - math.log(K), 0.5, 0.99999)
    lstar_rng = brentq(lambda l: best2(l, Brng) - math.log(K), 0.5, 0.99999)
    lform = brentq(lambda l: math.log(K) + H(l) / (1 - l) - B, 0.5, 0.99999)
    print(f"   n={n}: B={B:.3f} lam*_exact={lstar:.4f} formula={lform:.4f} R*(0.9)={best2(0.9,B):.3f} | "
          f"attainable range {Brng:.3f}: lam*={lstar_rng:.4f} R*(0.9)={best2(0.9,Brng):.3f}")

# 3. hill-climb T-step sequences with the coded class at B=6.78 (losses in [0,B]); lam=0.95 and 0.9
B = math.log(44 / MIX)


def climb(lam, T, iters=4000, restarts=6):
    best = -1e9; bestL = None
    for r in range(restarts):
        L = rng.choice([0.0, B], size=(T, K)); L[:, 0] = rng.choice([0.0, B], size=T)
        cur = max(own_regret(L, lam, s, "coded")[0] for s in range(K))
        for it in range(iters):
            L2 = L.copy(); i, j = rng.integers(T), rng.integers(K)
            L2[i, j] = np.clip(L2[i, j] + rng.normal() * (B / 3 if it < iters / 2 else B / 20), 0, B)
            v = max(own_regret(L2, lam, s, "coded")[0] for s in range(K))
            if v >= cur:
                L, cur = L2, v
        if cur > best:
            best, bestL = cur, L
    s = int(np.argmax([own_regret(bestL, lam, s, "coded")[0] for s in range(K)]))
    return best, coded_regret(bestL, lam, s), bestL


for lam in [0.95, 0.9]:
    for T in [2, 3, 6]:
        v, vc, L = climb(lam, T)
        print(f"[3] lam={lam} T={T}: hill-climb max discounted regret {v:.3f} (coded class re-eval {vc:.3f}); ln5={math.log(5):.3f}")

# 4. DTP fix (z = lam*(logp + log lik)): discounted regret <= ln k on random + adversarial sequences
worst = -1e9
for trial in range(300):
    lam = rng.uniform(0.3, 0.99); T = rng.integers(2, 30)
    L = rng.choice([0.0, B], size=(T, K)) if trial % 2 else rng.uniform(0, B, size=(T, K))
    for s in range(K):
        worst = max(worst, coded_regret(L, lam, s, fix=True))
        assert abs(coded_regret(L, lam, s, fix=True) - own_regret(L, lam, s, "dtp")[0]) < 1e-9
# adversarial: climb on the fixed variant
for lam in [0.5, 0.9, 0.95]:
    for T in [2, 4]:
        L = rng.uniform(0, B, size=(T, K)); cur = max(coded_regret(L, lam, s, fix=True) for s in range(K))
        for it in range(1500):
            L2 = L.copy(); L2[rng.integers(T), rng.integers(K)] = rng.uniform(0, B)
            v = max(coded_regret(L2, lam, s, fix=True) for s in range(K))
            if v >= cur: L, cur = L2, v
        worst = max(worst, cur)
print(f"[4] fixed update (z=lam*(logp+loglik)) == own DTP; max discounted regret {worst:.4f} vs ln5 {math.log(5):.4f}")

# 5. data-dependent bound R <= ln k + (1-lam) sum lam^{T-t} range D_{t-1}
mx = -1e9; ratios = []
for trial in range(2000):
    lam = rng.uniform(0.05, 0.99); T = rng.integers(1, 25)
    L = rng.choice([0.0, B], size=(T, K)) if trial % 3 == 0 else rng.uniform(0, B, size=(T, K)) * rng.uniform(0, 3)
    for s in range(K):
        R, rD = own_regret(L, lam, s, "coded")
        bound = math.log(K) + (1 - lam) * sum(lam ** (T - 1 - t) * rD[t] for t in range(T))
        mx = max(mx, R - bound)
print(f"[5] data-dependent bound: max(R - bound) over random sequences {mx:.2e} (<=0 required)")
for lam in [0.5, 0.7, 0.9, 0.95]:
    for Bc in [math.log(13 / MIX), B]:
        r = minimize_scalar(lambda b0: -closed_form(lam, b0, Bc), bounds=(0, Bc), method="bounded")
        L = two_turn(r.x, Bc); R, rD = own_regret(L, lam, 0, "coded")
        bound = math.log(K) + (1 - lam) * sum(lam ** (1 - t) * rD[t] for t in range(2))
        ratios.append(R / bound)
print(f"    tightness R/bound on the optimal 2-turn examples: {min(ratios):.2f}-{max(ratios):.2f}")
