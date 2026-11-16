"""B4 quantitative: how fast the style posterior concentrates when the opponent plays a style (realisable), with a
world-sampling mismatch proxy, on the 57 real stage matrices. Also the Bhattacharyya bound and the switch (tracking)
experiment for lam = 1 vs fixed share vs discount-then-predict.

Model: turns are i.i.d. draws of a real stage matrix A (ours). The opponent plays style s* computed on its own matrix
A' = A + sigma_m * ptp(A) * N(0,1) (fresh noise each turn; sigma_m = 0 is exact realisability); b_t ~ q_{s*}(A').
Our posterior uses q_s(A) with p_ref = the one-hot safex strategy (bayes_safex's p_ref). Prior uniform, lam = 1.
  1. pairwise separation on the 57 matrices: Bhattacharyya distance -ln BC(q_s*, q_s) and KL(q_s* || q_s)
  2. concentration: P_T(s*) and MAP accuracy at T in {25, 50, 100, 250, 600} for s* in {eq, safe, uniform} x sigma_m
  3. KL-closest style under mismatch (which style the posterior should converge to; Berk)
  4. bound check at sigma_m = 0: sum_t (1 - P_{t-1}(s*)) vs sum_{s != s*} sum_t prod_{u<t} BC_u(s*, s)  and vs
     sum_s 1/(1 - rho_s) with rho_s = max over matrices of BC
  5. switch: truth s_a for t0 turns then s_b; flip time of the MAP for lam = 1 (prediction t0 * mu_a / mu_b by N1),
     fixed share alpha = 0.01 (B2) and discount-then-predict lam = 0.98 (B3(b)).
"""
import json
import os
import sys

import numpy as np
from scipy.optimize import linprog

HERE = os.path.dirname(os.path.abspath(__file__))
MATS = "/private/tmp/claude-501/-Users-biofanat-Documents-classes-f26-cs535/2bbe670b-2f97-4d9a-bfad-e1139e0352c6/scratchpad/experts_ext/real_matrices.npz"
STYLES = ("eq", "br_unif", "br_us", "safe", "uniform")
TAU, MIX, EPS = 0.1, 0.05, 0.003
NPATH = int(sys.argv[1]) if len(sys.argv) > 1 else 40
rng = np.random.default_rng(7)


def lp(A):
    m, n = A.shape
    c = np.zeros(m + 1); c[-1] = -1
    r = linprog(c, A_ub=np.hstack([-A.T, np.ones((n, 1))]), b_ub=np.zeros(n), A_eq=np.hstack([np.ones((1, m)), np.zeros((1, 1))]),
                b_eq=[1], bounds=[(0, None)] * m + [(None, None)], method="highs")
    x = np.clip(r.x[:m], 0, None); x /= x.sum(); v = r.x[-1]
    c2 = np.zeros(n + 1); c2[-1] = 1
    r2 = linprog(c2, A_ub=np.hstack([A, -np.ones((m, 1))]), b_ub=np.zeros(m), A_eq=np.hstack([np.ones((1, n)), np.zeros((1, 1))]),
                 b_eq=[1], bounds=[(0, None)] * n + [(None, None)], method="highs")
    y = np.clip(r2.x[:n], 0, None); y /= y.sum()
    return x, y, v


def safex(A, v, q, eps=EPS):
    m, n = A.shape; r = float(np.ptp(A)) or 1.0
    res = linprog(-(A @ q), A_ub=-A.T, b_ub=-(v - eps * r) * np.ones(n), A_eq=np.ones((1, m)), b_eq=[1], bounds=[(0, None)] * m, method="highs")
    if res.status != 0:
        return None
    x = np.clip(res.x, 0, None); return x / x.sum()


def softmax(z):
    e = np.exp(z - z.max()); return e / e.sum()


def style_dists(A, p=None):
    n = A.shape[1]; u = np.full(n, 1 / n)
    if n == 1:
        return np.ones((5, 1))
    x_eq, y_eq, v = lp(A)
    if p is None:
        q0 = np.zeros(n); q0[int(A.mean(0).argmin())] = 1
        p = safex(A, v, q0)
        if p is None:
            p = x_eq
    tau = TAU * float(np.ptp(A)) or 1.0
    raw = np.stack([y_eq, softmax(-A.mean(0) / tau), softmax(-(p @ A) / tau), softmax(-A.max(0) / tau), u])
    return (1 - MIX) * raw + MIX * u


d = np.load(MATS, allow_pickle=True)
mats = [np.asarray(d[f], float) for f in d.files if np.asarray(d[f]).shape[1] >= 2 and np.asarray(d[f]).shape[0] >= 2]
Q = [style_dists(A) for A in mats]
print(f"{len(mats)} real matrices; columns median {np.median([A.shape[1] for A in mats]):.0f}")

print("\n=== 1. pairwise separation on the 57 matrices (per-turn, nats) ===")
BC = np.zeros((len(mats), 5, 5)); KL = np.zeros((len(mats), 5, 5))
for i, q in enumerate(Q):
    for a in range(5):
        for b in range(5):
            BC[i, a, b] = np.sqrt(q[a] * q[b]).sum()
            KL[i, a, b] = (q[a] * np.log(q[a] / q[b])).sum()
DB = -np.log(np.clip(BC, 1e-12, 1))
print("Bhattacharyya distance -ln BC: mean over matrices (rows = true s*, cols = s)")
print("        " + " ".join(f"{s:>8}" for s in STYLES))
for a in range(5):
    print(f"{STYLES[a]:>8}" + " ".join(f"{DB[:, a, b].mean():8.3f}" for b in range(5)))
print("KL(q_s* || q_s): mean over matrices")
for a in range(5):
    print(f"{STYLES[a]:>8}" + " ".join(f"{KL[:, a, b].mean():8.3f}" for b in range(5)))
print("rho_s = max over matrices of BC(s*, s) (uniform separation needed for the geometric bound):")
for a in range(5):
    print(f"{STYLES[a]:>8}" + " ".join(f"{BC[:, a, b].max():8.3f}" for b in range(5)) +
          f"   | fraction of matrices with BC > 0.99 to some other style: {(BC[:, a, [b for b in range(5) if b != a]].max(1) > 0.99).mean():.2f}")
print("br_unif vs br_us: identical when p_ref is uniform; here p_ref = one-hot safex, DB mean %.3f" % DB[:, 1, 2].mean())


def simulate(truth, sigma_m, T, lam=1.0, alpha=0.0, dtp=False, switch_to=None, t0=None, seed=0):
    """One path; returns the pre-update posterior P_{t-1}(.) for all t and the realised BC products per other style."""
    r = np.random.default_rng(seed)
    logp = np.full(5, -np.log(5)); P = np.zeros((T, 5)); bcprod = np.ones((T, 5))
    cur = np.ones(5)
    for t in range(T):
        i = r.integers(len(mats)); A = mats[i]; q = Q[i]
        s = truth if (switch_to is None or t < t0) else switch_to
        if sigma_m > 0:
            A2 = A + sigma_m * float(np.ptp(A)) * r.normal(size=A.shape)
            q_true = style_dists(A2)[s]
        else:
            q_true = q[s]
        b = r.choice(A.shape[1], p=q_true / q_true.sum())
        w = np.exp(logp - logp.max()); w /= w.sum(); P[t] = w
        bcprod[t] = cur
        cur = cur * np.sqrt(q * q[s]).sum(1) / 1.0   # realised BC_t(s*, .) per other style on this turn's matrix
        base = (lam * logp) if dtp else logp
        if dtp:
            z = lam * (logp + np.log(q[:, b]))
        else:
            z = lam * logp + np.log(q[:, b])
        z = z - (z.max() + np.log(np.exp(z - z.max()).sum()))
        if alpha > 0:
            pz = np.exp(z); pz = (1 - alpha) * pz + alpha / 5; z = np.log(pz)
        logp = z
    return P, bcprod


print("\n=== 2. concentration under realisability with mismatch proxy (NPATH = %d paths; T = 600) ===" % NPATH)
T = 600; checks = (25, 50, 100, 250, 600)
summary = {}
for sname in ("eq", "safe", "uniform", "br_unif"):
    s = STYLES.index(sname)
    for sigma_m in (0.0, 0.05, 0.1, 0.2, 0.4):
        if sname == "br_unif" and sigma_m not in (0.0, 0.1):
            continue
        Ps = np.array([simulate(s, sigma_m, T, seed=100 * k + s)[0] for k in range(NPATH)])   # (NPATH, T, 5)
        line = []
        for c in checks:
            Pt = Ps[:, c - 1, :]
            line.append(f"T={c}: P(s*)={Pt[:, s].mean():.3f} MAP ok {(Pt.argmax(1) == s).mean():.2f}")
        final_map = np.bincount(Ps[:, -1, :].argmax(1), minlength=5)
        avg_wrong = (1 - Ps[:, :, s]).sum(1).mean()
        summary[f"{sname}_{sigma_m}"] = dict(P=Ps[:, [c - 1 for c in checks], s].mean(0).tolist(), map_ok=[(Ps[:, c - 1, :].argmax(1) == s).mean() for c in checks],
                                             final_map=final_map.tolist(), sum_1mP=avg_wrong)
        print(f"truth {sname:>7} sigma_m {sigma_m:.2f}: " + "; ".join(line) + f" | final MAP counts {dict(zip(STYLES, final_map.tolist()))} | mean sum_t(1-P_{{t-1}}(s*)) = {avg_wrong:.2f}")

print("\n=== 3. KL-closest style under mismatch: E[KL(q_s*(A') || q_s(A))] averaged over 57 matrices x 20 noise draws ===")
for sname in ("eq", "safe", "uniform"):
    s = STYLES.index(sname)
    for sigma_m in (0.05, 0.1, 0.2, 0.4):
        kl = np.zeros(5); cnt = 0
        for i, A in enumerate(mats):
            for rep in range(20):
                A2 = A + sigma_m * float(np.ptp(A)) * rng.normal(size=A.shape)
                qt = style_dists(A2)[s]
                kl += np.array([(qt * np.log(qt / Q[i][b])).sum() for b in range(5)]); cnt += 1
        kl /= cnt
        print(f"truth {sname:>7} sigma_m {sigma_m:.2f}: " + " ".join(f"{STYLES[b]}={kl[b]:.3f}" for b in range(5)) + f"  -> closest {STYLES[kl.argmin()]}; margin to runner-up {np.sort(kl)[1]-kl.min():.3f} nats/turn")

print("\n=== 4. Bhattacharyya bound at sigma_m = 0:  sum_t (1 - P_{t-1}(s*))  <=  sum_{s != s*} sum_t prod_{u<t} BC_u(s*, s) ===")
for sname in ("eq", "safe", "uniform", "br_unif"):
    s = STYLES.index(sname)
    lhs, rhs = [], []
    for k in range(NPATH):
        P, bcp = simulate(s, 0.0, T, seed=100 * k + s)
        lhs.append((1 - P[:, s]).sum()); rhs.append(sum(bcp[:, b].sum() for b in range(5) if b != s))
    rho = [BC[:, s, b].max() for b in range(5) if b != s]
    geo = sum(1 / (1 - r_) if r_ < 1 else np.inf for r_ in rho)
    mean_bc = [BC[:, s, b].mean() for b in range(5) if b != s]
    geo_mean = sum(1 / (1 - r_) for r_ in mean_bc)
    print(f"truth {sname:>7}: realised sum_t(1-P) mean {np.mean(lhs):.2f} (max {np.max(lhs):.2f}); realised-BC-product bound mean {np.mean(rhs):.2f} (max {np.max(rhs):.2f}); "
          f"worst-case rho bound sum 1/(1-rho_s) = {geo:.1f}; with mean BC instead: {geo_mean:.1f}; Pinsker-route B4(ii) sqrt(2 T ln5)/2 = {np.sqrt(2*T*np.log(5))/2:.1f}")

print("\n=== 5. switch experiment: truth s_a for t0 = 300 turns, then s_b until T = 1200 (MAP flip time after t0) ===")
for sa, sb in (("safe", "br_unif"), ("eq", "safe"), ("uniform", "eq"), ("br_unif", "safe")):
    a, b = STYLES.index(sa), STYLES.index(sb)
    mu_a = KL[:, a, b].mean(); mu_b = KL[:, b, a].mean()       # per-turn drift of the a-vs-b log-lik gap before/after
    res = {}
    for name, kw in (("lam=1", dict()), ("fixed share a=0.01", dict(alpha=0.01)), ("DTP lam=0.98", dict(lam=0.98, dtp=True))):
        flips = []
        for k in range(NPATH):
            P, _ = simulate(a, 0.0, 1200, switch_to=b, t0=300, seed=500 + k, **kw)
            after = P[300:, :].argmax(1)
            ok = np.where(after == b)[0]
            # first time the MAP is b and stays b for 20 turns
            stay = [i for i in ok if (after[i:i + 20] == b).all()]
            flips.append(stay[0] if stay else 900)
        res[name] = (np.mean(flips), np.median(flips), np.max(flips))
    print(f"{sa} -> {sb}: KL drift before {mu_a:.3f}/turn, after {mu_b:.3f}/turn; N1 prediction for lam=1: t0*mu_a/mu_b = {300*mu_a/mu_b:.0f} turns; "
          + "; ".join(f"{n}: mean {m:.0f} median {md:.0f} max {mx:.0f}" for n, (m, md, mx) in res.items()))
json.dump(summary, open(os.path.join(HERE, "out_b4_concentration.json"), "w"), indent=1)
