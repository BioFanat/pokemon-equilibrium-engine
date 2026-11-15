"""Adversarial check of Theorem L3, Corollaries L3a/L3b and Proposition L3c (search_rl_links.md).

L3: random finite-horizon zero-sum stochastic games (lk_common.Game), a fixed random Markov agent pi,
the opponent's MDP M_pi built exactly. The exploiter runs per-state Hedge on losses
sigma_{k+1}(b|t,s) ∝ exp(-eta sum_{j<=k} Qhat^{sigma_j}_t(s,b)), sigma_1 uniform, with exact Q^{sigma_k}
(backward induction) plus critic noise (random or adversarial), Qhat clipped to [-1,1].
Checked for every K <= Kmax: the PDL identity of step 1, Theorem L3 with the stated regret bound
ln|B(s)|/eta + eta K/2 and with realised per-state regret, the H-bar first-term bound, L3a, L3b.
L3c: the combination-lock chain; exact V*_0 and expl; simulated MC-critic per-state Hedge exploiter.
Run: engine/.venv/bin/python theory/checks/lk3_rl_hedge.py
"""
from __future__ import annotations

import numpy as np

from lk_common import Game, maximin


def mdp(game, x):
    """M_pi: per (t,s): term-payoff vector c[b], continuation matrix N[b, s'] (sub-stochastic)."""
    M = {}
    for (t, s), cell in game.cells.items():
        xa = x[t, s]
        c = xa @ (cell["pt"] * cell["u"])                                   # [nB]
        N = np.einsum("a,ab,abk->bk", xa, 1 - cell["pt"], cell["P"])        # [nB, nxt]
        if t + 1 >= game.H:
            N = np.zeros_like(N)                                            # cap pays 0
        M[t, s] = (c, N)
    return M


def evaluate(game, M, sig):
    """Q^sigma_t(s, .) and V^sigma_t(s) by backward induction."""
    Q, V = {}, {}
    for t in reversed(range(game.H)):
        for s in range(game.nst[t]):
            c, N = M[t, s]
            Vn = np.array([V[t + 1, k] for k in range(N.shape[1])]) if t + 1 < game.H else np.zeros(N.shape[1])
            Q[t, s] = c + N @ Vn
            V[t, s] = float(sig[t, s] @ Q[t, s])
    return Q, V


def best_response(game, M):
    sig, V = {}, {}
    for t in reversed(range(game.H)):
        for s in range(game.nst[t]):
            c, N = M[t, s]
            Vn = np.array([V[t + 1, k] for k in range(N.shape[1])]) if t + 1 < game.H else np.zeros(N.shape[1])
            q = c + N @ Vn
            sig[t, s] = np.eye(len(q))[int(q.argmin())]
            V[t, s] = float(q.min())
    return sig, V


def occupancy(game, M, sig):
    d = {(0, 0): 1.0}
    for t in range(game.H - 1):
        for k in range(game.nst[t + 1]):
            d[t + 1, k] = 0.0
        for s in range(game.nst[t]):
            c, N = M[t, s]
            nxt = d[t, s] * (sig[t, s] @ N)
            for k in range(len(nxt)):
                d[t + 1, k] += nxt[k]
    return d


def run_l3(game, x, Kmax, eps, noise, rng, comparators):
    M = mdp(game, x)
    keys = list(game.cells)
    nB = {key: game.cells[key]["nB"] for key in keys}
    m = max(nB.values())
    eta = np.sqrt(2 * np.log(m) / Kmax)
    brs, Vbr = best_response(game, M)
    cum = {key: np.zeros(nB[key]) for key in keys}
    stats = {name: dict(reg=dict((key, 0.0) for key in keys), crit=0.0, sumV=0.0, worst_ratio=-np.inf,
                        min_slack=np.inf, min_slack_real=np.inf, pdl=0.0, hbar_ok=None, sum_exact=0.0)
             for name in comparators}
    comp = {name: (sig, occupancy(game, M, sig), evaluate(game, M, sig)[1][0, 0]) for name, sig in comparators.items()}
    sumV = 0.0
    eps_crit, Cmax, crit_br = 0.0, 0.0, 0.0
    for k in range(1, Kmax + 1):
        sig = {key: np.exp(-eta * (cum[key] - cum[key].min())) for key in keys}
        sig = {key: v / v.sum() for key, v in sig.items()}
        Q, V = evaluate(game, M, sig)
        dk = occupancy(game, M, sig)
        Qhat, epsk = {}, {}
        for key in keys:
            if noise == "random":
                z = rng.uniform(-eps, eps, size=nB[key])
            else:                                   # adversarial: hide the BR action
                z = eps * np.ones(nB[key]); z[brs[key].argmax()] = -eps
                z = -z
            Qhat[key] = np.clip(Q[key] + z, -1, 1)
            epsk[key] = float(np.abs(Qhat[key] - Q[key]).max())
            cum[key] += Qhat[key]
        sumV += V[0, 0]
        for t in range(game.H):
            eps_crit = max(eps_crit, sum(dk[t, s] * epsk[t, s] for s in range(game.nst[t])))
        dbr = comp["BR"][1]
        Cmax = max(Cmax, max(dbr[key] / dk[key] for key in keys if dbr[key] > 0))
        crit_br += sum(dbr[key] * epsk[key] for key in keys)
        for name, (sc, dc, Vc) in comp.items():
            st = stats[name]
            pdl = sum(dc[key] * float((sig[key] - sc[key]) @ Q[key]) for key in keys)
            st["pdl"] = max(st["pdl"], abs((V[0, 0] - Vc) - pdl))
            for key in keys:
                st["reg"][key] += float((sig[key] - sc[key]) @ Qhat[key])
            st["crit"] += sum(dc[key] * epsk[key] for key in keys)
            lhs = sumV / k - Vc
            reg_bound = {key: np.log(nB[key]) / eta + eta * k / 2 for key in keys}
            rhs = sum(dc[key] * reg_bound[key] for key in keys) / k + 2 * st["crit"] / k
            rhs_real = sum(dc[key] * st["reg"][key] for key in keys) / k + 2 * st["crit"] / k
            st["min_slack"] = min(st["min_slack"], rhs - lhs)
            st["min_slack_real"] = min(st["min_slack_real"], rhs_real - lhs)
            if rhs > 0:
                st["worst_ratio"] = max(st["worst_ratio"], lhs / rhs)
            if k == Kmax:
                Hbar = sum(dc.values())
                first = sum(dc[key] * reg_bound[key] for key in keys) / k
                st["hbar_ok"] = (first, Hbar * np.sqrt(2 * np.log(m) / Kmax))
                st["lhs_final"], st["rhs_final"] = lhs, rhs
    # L3a with sigma_bar (uniform iterate) and comparator BR
    V0 = game.V[0, 0]
    expl = V0 - Vbr[0, 0]
    gap = V0 - sumV / Kmax
    slack = stats["BR"]["rhs_final"]
    l3a = (expl - slack <= gap + 1e-12, gap <= expl + 1e-12, expl, gap, slack)
    l3b = (2 * crit_br / Kmax, 2 * game.H * Cmax * eps_crit)
    return stats, l3a, l3b


def chain(D, nB):
    """Combination lock as an explicit game: (V*_0, expl(pi)) by backward induction / LP."""
    # s_D: matching pennies, agent rows (heads, tails), pi = heads.
    mp = np.array([[1.0, -1.0], [-1.0, 1.0]])
    VD = maximin(mp)[1]
    Vstar, Vpi = VD, float(mp[0].min())           # BR to pure heads gets -1
    for j in reversed(range(D)):
        q_star = np.array([Vstar] + [0.0] * (nB - 1))      # b*_j continues, others end with 0
        q_pi = np.array([Vpi] + [0.0] * (nB - 1))
        Vstar = maximin(q_star[None, :])[1]               # forced agent: min over b
        Vpi = float(q_pi.min())
    return Vstar, Vstar - Vpi


def sim_l3c(D, nB, N, trials, rng, eta=1.0):
    """MC-critic per-state Hedge exploiter (no bonus). Returns fraction of trials with no reach of s_D
    in the first N games, and whether the policy stayed exactly uniform until the first reach."""
    misses, stayed = 0, True
    for _ in range(trials):
        sums = np.zeros((D + 1, nB)); cnt = np.zeros((D + 1, nB))
        reached = False
        for g in range(N):
            qhat = np.divide(sums, np.maximum(cnt, 1))
            pol = np.exp(-eta * (g * qhat - (g * qhat).min(1, keepdims=True)))
            pol /= pol.sum(1, keepdims=True)
            if not reached and np.abs(pol - 1 / nB).max() > 1e-15:
                stayed = False
            path, ret = [], 0.0
            for j in range(D):
                b = int(rng.choice(nB, p=pol[j]))
                path.append((j, b))
                if b != 0:
                    break
            else:
                b = int(rng.choice(2, p=pol[D, :2] / pol[D, :2].sum()))
                ret = 1.0 if b == 0 else -1.0             # pi plays heads
                path.append((D, b))
                reached = True
            for (j, b) in path:
                sums[j, b] += ret; cnt[j, b] += 1
            if reached:
                break
        misses += not reached
    return misses / trials, stayed


def l1_sandwich(game, x):
    """L1(b): E_{sigma_g}[sum e_t] <= expl(pi) <= E_{sigma_BR}[sum e_t], Markov pi, exact occupancies."""
    M = mdp(game, x)
    e = {key: game.V[key] - float((x[key] @ game.Q[key]).min()) for key in game.cells}
    sg = {key: np.eye(game.cells[key]["nB"])[int((x[key] @ game.Q[key]).argmin())] for key in game.cells}
    sbr, Vbr = best_response(game, M)
    lo = sum(occupancy(game, M, sg)[k] * e[k] for k in e)
    hi = sum(occupancy(game, M, sbr)[k] * e[k] for k in e)
    return lo, game.V[0, 0] - Vbr[0, 0], hi


def example_l1b(delta, modified):
    """Example L1b: s0 forced agent, b1 ends (payoff -delta, or 1-delta in the modified version),
    b2 -> s1; s1 matching pennies (pi = heads) or rows (1,1),(1,-1) with pi = row 2 (modified)."""
    S1 = np.array([[1.0, 1.0], [1.0, -1.0]]) if modified else np.array([[1.0, -1.0], [-1.0, 1.0]])
    xs1 = np.array([0.0, 1.0]) if modified else np.array([1.0, 0.0])
    V1 = maximin(S1)[1]
    v1 = (1 - delta) if modified else -delta
    Q0 = np.array([v1, V1]); V0 = Q0.min()
    e0 = V0 - Q0.min(); e1 = V1 - (xs1 @ S1).min()
    lower = e0 if Q0.argmin() == 0 else e0 + e1           # sigma_g ends the game iff b1 is its argmin
    expl = V0 - min(v1, (xs1 @ S1).min())
    upper = e0 + e1
    return lower, expl, upper


def main():
    print("[L3] per-state Hedge exploiter vs fixed random Markov agent; exact Q^sigma; checked at every K<=Kmax")
    worst = dict(ratio=(-np.inf, None), slack=(np.inf, None), slack_real=(np.inf, None), pdl=0.0, hbar=0.0)
    l3a_ok, l3b_worst = True, 0.0
    for gi in range(6):
        H = [3, 4, 5, 6, 4, 5][gi]
        game = Game(np.random.default_rng(500 + gi), H=H, S=3, pforced=0.1)
        rg = np.random.default_rng(600 + gi)
        x = {key: rg.dirichlet(np.ones(c["nA"])) for key, c in game.cells.items()}
        M = mdp(game, x)
        comparators = {"BR": best_response(game, M)[0],
                       "rand": {key: rg.dirichlet(np.ones(c["nB"])) for key, c in game.cells.items()}}
        for Kmax in (10, 50, 200):
            for eps, noise in ((0.0, "random"), (0.05, "random"), (0.2, "random"), (0.2, "adv")):
                stats, l3a, l3b = run_l3(game, x, Kmax, eps, noise, np.random.default_rng([gi, Kmax, int(eps * 100)]), comparators)
                tag = (gi, H, Kmax, eps, noise)
                for name, st in stats.items():
                    if st["worst_ratio"] > worst["ratio"][0]: worst["ratio"] = (st["worst_ratio"], tag + (name,))
                    if st["min_slack"] < worst["slack"][0]: worst["slack"] = (st["min_slack"], tag + (name,))
                    if st["min_slack_real"] < worst["slack_real"][0]: worst["slack_real"] = (st["min_slack_real"], tag + (name,))
                    worst["pdl"] = max(worst["pdl"], st["pdl"])
                    f, b = st["hbar_ok"]
                    worst["hbar"] = max(worst["hbar"], f - b)
                l3a_ok &= l3a[0] and l3a[1]
                l3b_worst = max(l3b_worst, l3b[0] - l3b[1])
                if Kmax == 200 and noise == "random" and eps == 0.05 and gi < 3:
                    st = stats["BR"]
                    print(f"  game {gi} H={H}: expl={l3a[2]:.4f}  V*_0-V(pi,sigma_bar)={l3a[3]:.4f}  L3 slack(RHS)={l3a[4]:.4f}"
                          f"  LHS={st['lhs_final']:.4f}  crit term={l3b[0]:.4f} <= 2HC eps_crit={l3b[1]:.4f}")
    print(f"  PDL identity (step 1) max |err| over all k, comparators: {worst['pdl']:.2e}")
    print(f"  Theorem L3 (stated bound) max LHS/RHS: {worst['ratio'][0]:.4f} at {worst['ratio'][1]}")
    print(f"  Theorem L3 (stated bound) min RHS-LHS: {worst['slack'][0]:.4f} at {worst['slack'][1]}")
    print(f"  Theorem L3 (realised per-state regret) min RHS-LHS: {worst['slack_real'][0]:.2e} at {worst['slack_real'][1]}")
    print(f"  first term <= Hbar sqrt(2 ln m/K) at K=Kmax: max excess {worst['hbar']:.2e} (<=0 means holds)")
    print(f"  L3a sandwich holds in all cells: {l3a_ok}")
    print(f"  L3b critic term - 2HC eps_crit, max: {l3b_worst:.3e} (<=0 means holds)")

    print("\n[L1] sandwich E_g[sum e] <= expl <= E_BR[sum e] on the 6 L3 games (fixed random Markov pi)")
    for gi in range(6):
        H = [3, 4, 5, 6, 4, 5][gi]
        game = Game(np.random.default_rng(500 + gi), H=H, S=3, pforced=0.1)
        rg = np.random.default_rng(600 + gi)
        x = {key: rg.dirichlet(np.ones(c["nA"])) for key, c in game.cells.items()}
        lo, ex, hi = l1_sandwich(game, x)
        print(f"  game {gi}: lower={lo:.4f}  expl={ex:.4f}  upper={hi:.4f}  holds={lo <= ex + 1e-12 <= hi + 2e-12}")
    for mod in (False, True):
        lo, ex, hi = example_l1b(0.1, mod)
        print(f"  Example L1b{' (modified)' if mod else ''}, delta=0.1: lower={lo:.3f} expl={ex:.3f} upper={hi:.3f}"
              f"  upper-lower={hi - lo:.3f}  expl-lower={ex - lo:.3f}")

    print("\n[L3c] combination lock")
    for D, nB in ((3, 2), (4, 3), (6, 2)):
        V0, expl = chain(D, nB)
        print(f"  D={D} |B|={nB}: V*_0={V0:.3g}  expl(pi)={expl:.3g}  per-game hit prob |B|^-D={nB ** -D:.4f}")
    rng = np.random.default_rng(77)
    for D, nB, N, trials in ((3, 2, 5, 20000), (4, 3, 30, 6000), (6, 2, 20, 6000)):
        frac, stayed = sim_l3c(D, nB, N, trials, rng)
        se = np.sqrt(frac * (1 - frac) / trials)
        print(f"  D={D} |B|={nB} N={N}: P(no reach in N games) = {frac:.4f} +- {1.96 * se:.4f}"
              f"  exact (1-|B|^-D)^N = {(1 - nB ** -D) ** N:.4f}  claimed lower bound 1-N|B|^-D = {1 - N * nB ** -D:.4f}"
              f"  policy uniform until first reach: {stayed}")


if __name__ == "__main__":
    main()
