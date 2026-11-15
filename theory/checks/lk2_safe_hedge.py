"""Adversarial check of Theorem L2 (search_rl_links.md): Hedge over experts in the full game.

Ground truth: random finite-horizon zero-sum stochastic games (lk_common.Game), exact Q*, V* by LP.
Leaf: A_hat = clip(Q* + noise, -1, 1), noise fixed per (t,s), |noise| <= eps_v (clipping keeps the bound).
Experts: equilibrium of A_hat (LP), its argmax ("top move"), maximin-pure ("safest"). k = 3.
Learner: exp3_live.Learner FTRL indexing (after update t: logw = sqrt(8 ln k / t) * cum; uniform at turn 1).
  variant 'norm': reward (q.A_hat e_b + 1)/2 in [0,1]   (code with Delta-bar = payoff range 2)
  variant 'raw' : reward q.A_hat e_b in [-1,1]           (same eta on unnormalised payoffs)
Opponents (all see p_i before moving): exploiter in A_hat, myopic Q*-exploiter, fixed random Markov,
minimax-optimal in Q*, greedy regret-maximiser; plus 'cheat' (sees the realised action a_i: outside L2).
Checks per run (pathwise, every prefix): (H) regret <= R(T'); (a)/(c) pathwise forms; forced-turn sign;
exact telescoping identity sum_g (V*_0 - u_g) = sum_i l_i(x_i,b_i) - sum_i D_i; MC: mean sum D ~ 0.
Run: engine/.venv/bin/python theory/checks/lk2_safe_hedge.py
"""
from __future__ import annotations

import numpy as np

from lk_common import Game, maximin

K_EXP = 3
LNK = np.log(K_EXP)


def R_bound(T):
    """Claimed R(T) = 2 (1.5 sqrt(T ln k / 2) + sqrt(ln k / 8)) on the [-1,1] scale."""
    return 2 * (1.5 * np.sqrt(T * LNK / 2) + np.sqrt(LNK / 8))


def R_raw_provable(T):
    """R2a's general bound for step beta_t applied to range-2 rewards: ln k/beta_T + (1/2) sum beta_t."""
    T = np.asarray(T, float)
    out = np.zeros_like(T)
    for i, TT in enumerate(np.atleast_1d(T)):
        n = int(TT)
        if n == 0:
            continue
        b = np.sqrt(8 * LNK / np.maximum(np.arange(n), 1))     # beta_1 := beta_2
        out.flat[i] = LNK / b[-1] + 0.5 * b.sum()
    return out


class Learner:
    def __init__(self, variant):
        self.variant, self.t, self.cum, self.logw = variant, 0, np.zeros(K_EXP), np.zeros(K_EXP)

    def weights(self):
        w = np.exp(self.logw - self.logw.max())
        return w / w.sum()

    def update(self, rewards_payoff):
        self.t += 1
        r = (rewards_payoff + 1) / 2 if self.variant == "norm" else rewards_payoff
        self.cum += r
        self.logw = np.sqrt(8 * LNK / self.t) * self.cum


def build_leaf(game, eps_v, rng):
    Ah, experts, eps_s = {}, {}, 0.0
    for key, Q in game.Q.items():
        A = np.clip(Q + rng.uniform(-eps_v, eps_v, size=Q.shape), -1, 1)
        Ah[key] = A
        x, v = maximin(A)
        eye = np.eye(A.shape[0])
        experts[key] = np.stack([x, eye[int(x.argmax())], eye[int(A.min(1).argmax())]])
        eps_s = max(eps_s, v - float((x @ A).min()))
        assert np.abs(A - Q).max() <= eps_v + 1e-12
    return Ah, experts, eps_s


def opponent_move(kind, key, p, A, Qs, qs, rand_pol, a_real, rng):
    """Returns (b, y) with y the opponent's mixed strategy given the history (before a is drawn)."""
    m = A.shape[1]
    eye = np.eye(m)
    if kind == "expl_hat":
        b = int((p @ A).argmin())
    elif kind == "expl_q":
        b = int((p @ Qs).argmin())
    elif kind == "regret":
        b = int(((qs - p) @ A).max(0).argmax())
    elif kind == "cheat":                      # sees the realised action: violates a_i independent of b_i
        b = int(Qs[a_real].argmin())
        return b, None
    else:
        y = rand_pol[key] if kind == "random" else MINIMAX[id(Qs)]
        b = int(rng.choice(m, p=y))
        return b, y
    return b, eye[b]


MINIMAX = {}


def run(game, Ah, experts, eps_v, kind, variant, persistent, G, rng, rand_pol):
    L = Learner(variant)
    ups, regs, rhs_a, rhs_c, lhs_path, forced_l = [], [], [], [], [], 0.0
    cum_q = np.zeros(K_EXP)                 # sum_i (q_e - p_i)^T A_hat e_b (payoff units)
    cum_lhat = np.zeros(K_EXP)              # sum_i lhat(q_e, b_i)
    cum_l = 0.0                             # sum over learner turns of l_i(p_i, b_i) in Q*
    T = 0
    sum_loss, sum_ell, sum_D = 0.0, 0.0, 0.0
    min_slack_H, min_slack_a, min_slack_c = np.inf, np.inf, np.inf
    max_ratio_H = 0.0
    game_regs, game_T = [], []
    for g in range(G):
        if not persistent:
            L = Learner(variant)
            gq, gT = np.zeros(K_EXP), 0
        t, s = 0, 0
        V0 = game.V[0, 0]
        while True:
            key = (t, s)
            Qs, A, qs = game.Q[key], Ah[key], experts[key]
            nA = Qs.shape[0]
            forced = nA == 1
            p = np.ones(1) if forced else L.weights() @ qs
            a = int(rng.choice(nA, p=p / p.sum()))
            b, _ = opponent_move(kind, key, p, A, Qs, qs, rand_pol, a, rng)
            done, pay, s2 = game.step(t, s, a, b, rng)
            vnext = game.next_value(t, s, a, b, done, pay, s2)
            ell = game.V[key] - float(p @ Qs[:, b])
            sum_ell += ell
            sum_D += vnext - float(p @ Qs[:, b])
            if forced:
                forced_l += ell
            else:
                T += 1
                rew = qs @ A[:, b]
                cum_q += rew - float(p @ A[:, b])
                val_hat = maximin(A)[1] if key not in VALS else VALS[key]
                cum_lhat += val_hat - rew
                cum_l += ell
                if not persistent:
                    gq += rew - float(p @ A[:, b]); gT += 1
                L.update(rew)
                R = R_bound(T)
                regT = cum_q.max()
                min_slack_H = min(min_slack_H, R - regT)
                max_ratio_H = max(max_ratio_H, regT / R)
                if persistent:
                    min_slack_a = min(min_slack_a, cum_lhat.min() + R + 2 * eps_v * T - cum_l)
                    min_slack_c = min(min_slack_c, -cum_l - (-cum_lhat).max() + R + 2 * eps_v * T)
            if done:
                sum_loss += V0 - pay
                break
            t, s = t + 1, s2
        if not persistent:
            game_regs.append(gq.max()); game_T.append(gT)
    ident_err = abs(sum_loss - (sum_ell - sum_D))
    out = dict(T=T, loss=sum_loss, ell=sum_ell, D=sum_D, ident_err=ident_err, forced_l=forced_l,
               lhat_min=cum_lhat.min(), lhat_negmax=(-cum_lhat).max(), cum_l=cum_l, reg=cum_q.max(),
               min_slack_H=min_slack_H, max_ratio_H=max_ratio_H, min_slack_a=min_slack_a,
               min_slack_c=min_slack_c)
    if not persistent:
        out["reset_reg"] = float(np.sum(game_regs))
        out["reset_bound"] = float(sum(R_bound(x) for x in game_T if x > 0))
        out["reset_game_ratio"] = float(max(r / R_bound(x) for x, r in zip(game_T, game_regs) if x > 0)) if any(game_T) else 0.0
        out["reset_game_slack"] = float(min(R_bound(x) - r for x, r in zip(game_T, game_regs) if x > 0)) if any(game_T) else np.inf
    return out


VALS = {}


def abstract_alternation():
    """Oblivious alternating rewards in [-1,1]: expert 1 gets +1 on odd turns, expert 2 on even turns,
    the rest -1. Pure (H) test for both variants at k = 3."""
    print("\n[H, abstract] alternating rewards in [-1,1], k=3: regret / R(T)")
    for T in (100, 1000, 10000, 100000):
        for variant in ("norm", "raw"):
            L = Learner(variant)
            cum_e, learner = np.zeros(3), 0.0
            for i in range(1, T + 1):
                r = -np.ones(3); r[0 if i % 2 else 1] = 1.0
                learner += float(L.weights() @ r)
                cum_e += r
                L.update(r)
            reg = cum_e.max() - learner
            print(f"  T={T:6d} {variant:4s}: regret={reg:9.3f}  R(T)={R_bound(T):8.3f}  ratio={reg / R_bound(T):.3f}"
                  f"  (raw provable bound {float(R_raw_provable(T)):8.3f})")


def main():
    master = np.random.default_rng(535)
    G, RUNS = 60, 16
    kinds = ["expl_hat", "expl_q", "random", "minimax", "regret", "cheat"]
    agg = {}
    worst = dict(H=(np.inf, None), a=(np.inf, None), c=(np.inf, None), ratio=(0, None),
                 ident=(0, None), rratio=(0, None), forced=(-np.inf, None), reset=(np.inf, None))
    for gi in range(4):
        game = Game(np.random.default_rng(1000 + gi), H=[4, 5, 6, 5][gi], S=3)
        for eps_v in (0.0, 0.15, 0.4):
            Ah, experts, eps_s = build_leaf(game, eps_v, np.random.default_rng(2000 + gi))
            VALS.clear(); VALS.update({k: maximin(A)[1] for k, A in Ah.items()})
            for k, Qs in game.Q.items():
                MINIMAX[id(Qs)] = maximin(-Qs.T)[0]
            rand_pol = {k: np.random.default_rng(3000 + gi).dirichlet(np.ones(Q.shape[1])) for k, Q in game.Q.items()}
            for kind in kinds:
                for variant in ("norm", "raw"):
                    for persistent in (True, False):
                        rows = [run(game, Ah, experts, eps_v, kind, variant, persistent, G,
                                    np.random.default_rng([gi, int(eps_v * 100), kinds.index(kind), int(variant == 'raw'), int(persistent), r]), rand_pol)
                                for r in range(RUNS)]
                        tag = (gi, eps_v, kind, variant, persistent)
                        agg[tag] = (rows, eps_s)
                        legit = kind != "cheat"
                        for r in rows:
                            if legit and variant == "norm":
                                if persistent and r["min_slack_H"] < worst["H"][0]: worst["H"] = (r["min_slack_H"], tag)
                                if persistent and r["max_ratio_H"] > worst["ratio"][0]: worst["ratio"] = (r["max_ratio_H"], tag)
                                if not persistent and r["reset_game_ratio"] > worst["rratio"][0]: worst["rratio"] = (r["reset_game_ratio"], tag)
                                if persistent and r["min_slack_a"] < worst["a"][0]: worst["a"] = (r["min_slack_a"], tag)
                                if persistent and r["min_slack_c"] < worst["c"][0]: worst["c"] = (r["min_slack_c"], tag)
                                if not persistent and r["reset_game_slack"] < worst["reset"][0]: worst["reset"] = (r["reset_game_slack"], tag)
                            if r["ident_err"] > worst["ident"][0]: worst["ident"] = (r["ident_err"], tag)
                            if r["forced_l"] > worst["forced"][0]: worst["forced"] = (r["forced_l"], tag)
    print(f"games: 4 random (H=4..6, 3 states/stage, 1-4 agent / 2-4 opp actions); eps_v in 0, .15, .4; "
          f"G={G} games per run, {RUNS} runs per cell")
    print("\n[pathwise, variant norm, legit opponents]")
    print(f"  (d) reset: max over games of Reg_g/R(tau_g): {worst['rratio'][0]:.4f}  at {worst['rratio'][1]}")
    print(f"  (H) persistent runs: min slack R(T')-Reg(T') over all prefixes/runs: {worst['H'][0]:.4f}  at {worst['H'][1]}")
    print(f"  (H) persistent runs: max Reg(T')/R(T') ratio over all prefixes: {worst['ratio'][0]:.4f}  at {worst['ratio'][1]}")
    print(f"  (a) pathwise min slack: {worst['a'][0]:.4f}  at {worst['a'][1]}")
    print(f"  (c) pathwise min slack: {worst['c'][0]:.4f}  at {worst['c'][1]}")
    print(f"  (d) reset: min over games of R(tau_g)-Reg_g: {worst['reset'][0]:.4f}  at {worst['reset'][1]}")
    print(f"  telescoping identity max |err| (all runs): {worst['ident'][0]:.2e}")
    print(f"  forced turns: max sum of l_i over forced turns (must be <= 0): {worst['forced'][0]:.3e}")

    # raw variant: does the claimed R(T) still hold?
    rr = max((r["max_ratio_H"], tag) for tag, (rows, _) in agg.items() for r in rows
             if tag[3] == "raw" and tag[2] != "cheat" and tag[4])
    print(f"\n[variant raw, persistent] max Reg/R(T) ratio over all prefixes/runs: {rr[0]:.4f} at {rr[1]}")

    # expectation-level (a), (b), (c) and the martingale term, per cell (MC with 95% CI)
    print("\n[expectation level, persistent, variant norm]  cells with worst (a) MC slack per opponent")
    print("  opp       eps_v  E[sum(V*-u)] +-ci   E[RHS_a]   E[RHS_b]   E[sum D] +-ci   E[lhs_c] E[RHS_c]")
    for kind in kinds:
        best = None
        for tag, (rows, eps_s) in agg.items():
            if tag[2] != kind or tag[3] != "norm" or not tag[4]:
                continue
            loss = np.array([r["loss"] for r in rows]); D = np.array([r["D"] for r in rows])
            T = np.array([r["T"] for r in rows], float)
            rhs_a = np.array([r["lhat_min"] for r in rows]) + R_bound(T) + 2 * tag[1] * T
            rhs_b = (eps_s + 2 * tag[1]) * T + R_bound(T)
            rhs_c = np.array([r["lhat_negmax"] for r in rows]) - R_bound(T) - 2 * tag[1] * T
            ci = 1.96 * loss.std(ddof=1) / np.sqrt(len(loss))
            slack = rhs_a.mean() - (loss.mean() + ci)
            row = (slack, tag, loss.mean(), ci, rhs_a.mean(), rhs_b.mean(), D.mean(),
                   1.96 * D.std(ddof=1) / np.sqrt(len(D)), -loss.mean(), rhs_c.mean())
            if best is None or slack < best[0]:
                best = row
        sl, tag, lm, ci, ra, rb, dm, dci, lc, rc = best
        print(f"  {kind:9s} {tag[1]:.2f}  {lm:8.3f} +-{ci:6.3f}  {ra:9.3f}  {rb:9.3f}  {dm:8.3f} +-{dci:6.3f}  {lc:8.3f} {rc:8.3f}")

    zs = []
    for tag, (rows, _) in agg.items():
        D = np.array([r["D"] for r in rows])
        zs.append((abs(D.mean()) / (D.std(ddof=1) / np.sqrt(len(D)) + 1e-300), tag))
    leg = sorted(z for z, tag in zs if tag[2] != "cheat"); ch = sorted(z for z, tag in zs if tag[2] == "cheat")
    print(f"\n[martingale term] |mean sum D|/se over {len(leg)} legit cells: max {leg[-1]:.2f}, "
          f"frac>1.96 = {np.mean(np.array(leg) > 1.96):.3f}; cheat cells: min {ch[0]:.2f}, median {np.median(ch):.2f}")
    print("\n[(d) persistent vs reset, variant norm, eps_v=0.15, game 0]: mean realised regret, bound, sum(V*-u)")
    for kind in kinds[:-1]:
        P, Rr = agg[(0, 0.15, kind, "norm", True)][0], agg[(0, 0.15, kind, "norm", False)][0]
        print(f"  {kind:9s} persistent reg={np.mean([r['reg'] for r in P]):7.2f} bound={np.mean([R_bound(r['T']) for r in P]):7.2f}"
              f" loss={np.mean([r['loss'] for r in P]):7.2f} | reset reg={np.mean([r['reset_reg'] for r in Rr]):7.2f}"
              f" bound={np.mean([r['reset_bound'] for r in Rr]):7.2f} loss={np.mean([r['loss'] for r in Rr]):7.2f}")
    abstract_alternation()
    alternation_unclipped()
    cheat_long()


def alternation_unclipped():
    """Leaf outside [-1,1] (A_hat = Q* + noise unclipped): rewards +-(1+eps_v), normalised as (r+1)/2."""
    print("\n[H, abstract] alternation with rewards +-(1+eps_v) (unclipped leaf), variant norm, T=1e5")
    for e in (0.15, 0.4):
        L = Learner("norm"); cum_e, learner, T = np.zeros(3), 0.0, 100000
        for i in range(1, T + 1):
            r = -(1 + e) * np.ones(3); r[0 if i % 2 else 1] = 1 + e
            learner += float(L.weights() @ r); cum_e += r; L.update(r)
        reg = cum_e.max() - learner
        print(f"  eps_v={e}: regret={reg:8.2f}  R(T)={R_bound(T):8.2f}  ratio={reg / R_bound(T):.3f}"
              f"  (R(T)*(1+eps_v) = {R_bound(T) * (1 + e):8.2f})")


def cheat_long():
    """Opponent that sees the realised action (outside L2's setting): (b) fails for long matches."""
    game = Game(np.random.default_rng(1000), H=4, S=3)
    Ah, experts, eps_s = build_leaf(game, 0.0, np.random.default_rng(2000))
    VALS.clear(); VALS.update({k: maximin(A)[1] for k, A in Ah.items()})
    print("\n[scope] 'cheat' opponent (sees realised a_i), game 0, eps_v=0, variant norm, persistent")
    for G in (200, 1000, 4000):
        rows = [run(game, Ah, experts, 0.0, "cheat", "norm", True, G, np.random.default_rng([7, G, r]), {})
                for r in range(4)]
        loss = np.mean([r["loss"] for r in rows]); T = np.mean([r["T"] for r in rows])
        rhs_b = np.mean([(eps_s) * r["T"] + R_bound(r["T"]) for r in rows])
        print(f"  G={G:5d}: E[sum(V*-u)]={loss:8.2f}  RHS_b={rhs_b:8.2f}  E[sum D]={np.mean([r['D'] for r in rows]):8.2f}  T={T:.0f}")


if __name__ == "__main__":
    main()
