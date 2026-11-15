"""Numeric checks for theory/hedge_experts.md (Hedge + fixed-share experts learner, exp3_live.py).

numpy only, one process. Stage matrices are 2 x m so val and an exact maximin row strategy are
computed in closed form (the row payoff min_j (x a_1j + (1-x) a_2j) is concave piecewise linear in x;
its max is at x in {0, 1} or at a pairwise crossing).

Experts, as in exp3_live.Learner.plan: [top move of the equilibrium, the equilibrium, safest pure row].
Here "the equilibrium" is RM+ (alternating, linear averaging, as nash.rm_plus) with T iterations,
so eps_s > 0 and is measured exactly as val(A) - min_b q*.A[:, b].

Tests
  A  as coded (per-turn normalisation by range(A_t), eta=0.2, alpha=0.002), adaptive best-responding
     opponent b_t = argmin_b p_t.A_t[:, b], ranges log-uniform over 3 decades:
     the 1/Delta_t-weighted Claim 1 holds with Reg = ln k/eta + S(eta/8 + ln(1/(1-alpha))/eta).
  B  fixed normaliser Dbar = 2 (entries in [-1, 1]), alpha = 0, tuned eta = sqrt(8 ln k / S):
     raw Claim 1 holds with Dbar*sqrt(S ln k / 2); sweep S to see the sqrt(S) order.
  C  counterexample to raw Claim 1 under per-turn normalisation (code's own experts): an adaptive
     opponent punishes the non-equilibrium experts on high-range turns and repays them on low-range
     turns; raw deficit grows linearly. Same opponent vs fixed normaliser: sublinear.
  D  composition (Claim 2) on a random layered stochastic game with Ahat = Q* + noise (|.| <= eps_v):
     telescoping identity and avg-per-game deficit <= mean_len*(eps_s+2eps_v) + Dbar*Reg/N.
Run: engine/.venv/bin/python theory/checks/hedge_check.py
"""
import numpy as np

K = 3  # experts


def solve_2xm(A):
    """Exact val and maximin row strategy of a 2 x m matrix (row player maximises)."""
    a, b = A[0], A[1]
    xs = [0.0, 1.0]
    m = A.shape[1]
    for i in range(m):
        for j in range(i + 1, m):
            den = (a[i] - b[i]) - (a[j] - b[j])
            if abs(den) > 1e-15:
                x = (b[j] - b[i]) / den
                if 0.0 <= x <= 1.0:
                    xs.append(x)
    xs = np.array(xs)
    vals = (np.outer(xs, a) + np.outer(1 - xs, b)).min(1)
    i = int(vals.argmax())
    return float(vals[i]), np.array([xs[i], 1 - xs[i]])


def rm_plus(A, T):
    """Alternating RM+ with linear averaging (same scheme as engine/nash.py:rm_plus)."""
    n, m = A.shape
    qx, qy = np.zeros(n), np.zeros(m)
    x, y = np.ones(n) / n, np.ones(m) / m
    sx, sy = np.zeros(n), np.zeros(m)
    for t in range(1, T + 1):
        u = A @ y
        qx = np.maximum(qx + u - x @ u, 0)
        x = qx / qx.sum() if qx.sum() > 0 else np.ones(n) / n
        v = -(x @ A)
        qy = np.maximum(qy + v - y @ v, 0)
        y = qy / qy.sum() if qy.sum() > 0 else np.ones(m) / m
        sx += t * x
        sy += t * y
    return sx / sx.sum()


def experts(A, T=60):
    x = rm_plus(A, T)
    eye = np.eye(A.shape[0])
    return np.stack([eye[int(x.argmax())], x, eye[int(A.min(1).argmax())]])


class Learner:
    """exp3_live.Learner with a switch for the normaliser: per-turn range (as coded) or fixed Dbar."""

    def __init__(self, eta, alpha, dbar=None, lo=None):
        self.logw, self.eta, self.alpha, self.dbar, self.lo = np.zeros(K), eta, alpha, dbar, lo

    def weights(self):
        w = np.exp(self.logw - self.logw.max())
        return w / w.sum()

    def rewards(self, A, Q, b):
        if self.dbar is None:
            return (Q @ A[:, b] - A.min()) / (np.ptp(A) or 1.0)
        return (Q @ A[:, b] - self.lo) / self.dbar

    def update(self, A, Q, b):
        self.logw += self.eta * self.rewards(A, Q, b)
        w = (1 - self.alpha) * self.weights() + self.alpha / K
        self.logw = np.log(w)


def reg_bound(S, eta, alpha):
    """Fixed-comparator regret bound of Hedge + fixed share (derived in hedge_experts.md, Lemma H)."""
    return np.log(K) / eta + S * eta / 8 + S * np.log(1 / (1 - alpha)) / eta


def run_stream(rng, S, learner, scale_fn, m_range=(2, 5)):
    """S turns of random 2 x m stage matrices; adaptive opponent best-responds to p in A.
    Returns dict of accumulated quantities."""
    acc = dict(w_def=0.0, raw_def=0.0, eps_s_max=0.0, raw_def_val=0.0, dmax=0.0,
               ereg=np.zeros(K))
    for _ in range(S):
        m = int(rng.integers(*m_range, endpoint=True))
        A = scale_fn(rng) * rng.uniform(-1, 1, size=(2, m))
        Q = experts(A)
        val, _ = solve_2xm(A)
        p = learner.weights() @ Q
        b = int((p @ A).argmin())          # adaptive: sees p
        g = float((Q[1] @ A).min())        # guarantee of the equilibrium expert = val - eps_s
        d = np.ptp(A)
        acc["eps_s_max"] = max(acc["eps_s_max"], (val - g) / d)
        acc["dmax"] = max(acc["dmax"], d)
        pay = float(p @ A[:, b])
        acc["w_def"] += (g - pay) / d
        acc["raw_def"] += g - pay
        r = learner.rewards(A, Q, b)
        acc["ereg"] += r - learner.weights() @ r   # normalised regret vs each expert
        learner.update(A, Q, b)
    return acc


def test_A():
    print("== A: as coded (per-turn normalisation, eta=0.2, alpha=0.002), adaptive BR opponent")
    S, eta, alpha = 2000, 0.2, 0.002
    worst = -np.inf
    for seed in range(5):
        rng = np.random.default_rng(seed)
        L = Learner(eta, alpha)
        acc = run_stream(rng, S, L, lambda r: 10 ** r.uniform(0, 3))
        B = reg_bound(S, eta, alpha)
        ok = acc["w_def"] <= B and acc["ereg"].max() <= B
        worst = max(worst, acc["w_def"] / B)
        print(f"  seed {seed}: weighted deficit {acc['w_def']:8.2f}  max expert regret "
              f"{acc['ereg'].max():8.2f}  bound {B:7.2f}  max eps_s/range {acc['eps_s_max']:.3g}  "
              f"{'OK' if ok else 'VIOLATED'}")
    print(f"  max weighted deficit / bound = {worst:.3f}")


def test_B():
    print("== B: fixed normaliser Dbar=2, alpha=0, tuned eta; raw Claim 1 and sqrt(S) order")
    for S in [100, 400, 1600, 6400]:
        eta = np.sqrt(8 * np.log(K) / S)
        ratios, viol = [], 0
        for seed in range(8):
            rng = np.random.default_rng(1000 + seed)
            L = Learner(eta, 0.0, dbar=2.0, lo=-1.0)
            acc = run_stream(rng, S, L, lambda r: 10 ** r.uniform(-3, 0))  # entries in [-1, 1]
            R = np.sqrt(S * np.log(K) / 2)
            viol += acc["raw_def"] > 2.0 * R or acc["ereg"].max() > R
            ratios.append(acc["ereg"].max() / np.sqrt(S * np.log(K)))
        print(f"  S={S:5d}: max regret/sqrt(S ln k) = {max(ratios):.3f} (bound 0.707); "
              f"violations {viol}/8")


def test_C():
    print("== C: counterexample to raw Claim 1 under per-turn normalisation")
    eps = 0.01
    H = np.array([[1.0, -1.0], [-1.0, 1.0]])   # matching pennies, val 0
    for norm in ["per-turn", "fixed Dbar=2"]:
        out = []
        for S in [400, 1600, 6400]:
            eta = np.sqrt(8 * np.log(K) / S)
            L = Learner(eta, 0.0) if norm == "per-turn" else Learner(eta, 0.0, dbar=2.0, lo=-1.0)
            raw = 0.0
            for t in range(S):
                A = H if t % 2 == 0 else eps * H
                Q = experts(A)
                p = L.weights() @ Q
                top = int(Q[0].argmax())
                # high stakes: punish the top/safest row; low stakes: reward it (bait)
                b = (1 - top) if t % 2 == 0 else top
                raw += float((Q[1] @ A).min()) - float(p @ A[:, b])
                L.update(A, Q, b)
            out.append((S, raw, 2.0 * np.sqrt(S * np.log(K) / 2)))
        print(f"  {norm:13s}: " + "; ".join(f"S={S} raw deficit {r:7.1f} vs Dmax*Reg {B:5.1f}"
                                            for S, r, B in out))


def test_D():
    print("== D: composition on a random layered stochastic game (2 x 3 stages)")
    rng = np.random.default_rng(7)
    Hh, nS, m, eps_v, p_end = 5, 4, 3, 0.05, 0.3
    # transitions: (t, s, a, b) -> distribution over next live states, plus termination prob
    P = rng.dirichlet(np.ones(nS), size=(Hh, nS, 2, m))
    Uend = rng.uniform(-1, 1, size=(Hh, nS, 2, m))     # payoff if the game ends after this turn
    UH = rng.uniform(-1, 1, size=nS)                   # payoff at the horizon
    V = np.zeros((Hh + 1, nS)); V[Hh] = UH
    Qs = np.zeros((Hh, nS, 2, m))
    for t in reversed(range(Hh)):
        for s in range(nS):
            Qs[t, s] = p_end * Uend[t, s] + (1 - p_end) * P[t, s] @ V[t + 1]
            V[t, s] = solve_2xm(Qs[t, s])[0]
    Ahat = Qs + rng.uniform(-eps_v, eps_v, size=Qs.shape)
    EX = np.array([[experts(Ahat[t, s], T=300) for s in range(nS)] for t in range(Hh)])
    EPS = np.array([[solve_2xm(Ahat[t, s])[0] - (EX[t, s, 1] @ Ahat[t, s]).min()
                     for s in range(nS)] for t in range(Hh)])
    N = 4000
    S_tot = 0
    for kind in ["Hedge alpha=0 (fixed Dbar=2)", "as coded (per-turn, alpha=.002)"]:
        S_exp = N * 3.0  # rough planning length for tuning eta
        L = (Learner(np.sqrt(8 * np.log(K) / S_exp), 0.0, dbar=2.0, lo=-1.0 - eps_v)
             if kind.startswith("Hedge") else Learner(0.2, 0.002))
        if kind.startswith("Hedge"):
            L.dbar = 2.0 + 2 * eps_v
        gap, delta, budget, S_tot, surr = [], 0.0, 0.0, 0, 0.0
        for g in range(N):
            s, tot_delta = int(rng.integers(nS)), 0.0
            v0 = V[0, s]
            for t in range(Hh):
                A, Q = Ahat[t, s], EX[t, s]
                p = L.weights() @ Q
                b = int((p @ Qs[t, s]).argmin())   # adaptive myopic exploiter in the TRUE stage game
                tot_delta += V[t, s] - float(p @ Qs[t, s][:, b])
                budget += EPS[t, s] + 2 * eps_v
                surr += float((Q[1] @ A).min()) - float(p @ A[:, b])
                L.update(A, Q, b)
                S_tot += 1
                a = int(rng.random() < p[1])
                if rng.random() < p_end:
                    u = Uend[t, s, a, b]; break
                s = int(rng.choice(nS, p=P[t, s, a, b]))
            else:
                u = UH[s]
            gap.append(v0 - u)
            delta += tot_delta
        gap = np.array(gap)
        se = gap.std() / np.sqrt(N)
        if kind.startswith("Hedge"):
            R = L.dbar * (np.log(K) / L.eta + L.eta * S_tot / 8)
            print(f"  {kind}: mean len {S_tot / N:.2f}; avg(V*-u) = {gap.mean():.4f} +- {2 * se:.4f}; "
                  f"avg sum delta_t = {delta / N:.4f} (identity: equal in expectation); "
                  f"bound len*(eps_s+2eps_v) + Dbar*Reg/N = {budget / N:.4f} + {R / N:.4f} = "
                  f"{(budget + R) / N:.4f}; surrogate deficit {surr:.1f} <= Dbar*Reg {R:.1f}")
        else:
            print(f"  {kind}: mean len {S_tot / N:.2f}; avg(V*-u) = {gap.mean():.4f} +- {2 * se:.4f}; "
                  f"avg sum delta_t = {delta / N:.4f}; len*(eps_s+2eps_v) = {budget / N:.4f}")


if __name__ == "__main__":
    test_A()
    test_B()
    test_C()
    test_D()
