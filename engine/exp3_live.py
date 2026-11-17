"""Live experts learner with per-turn counterfactual feedback (full information).

Experts = Equilibrium top move, Equilibrium sampled, Safest move. All three come from one full matrix
per turn, so the learner costs the same as one matrix bot. Each turn it plays the mixture
p = Σ_e w_e q_e of the experts' strategies q_e, then, after seeing the opponent's actual move b,
gives every expert the reward r_e = (q_e · A[:, b] − min A) / Δ̄, clipped to [0, 1] (A = own-perspective
matrix, Δ̄ = 400 fixed). Weights: Hedge, log w_e += η_t r_e with η_t = sqrt(8 ln k / t); optional fixed share
(mix α back to uniform) to track an opponent that changes. X6 used range(A) per turn, η = 0.2, α = 0.002. Compare: bandit EXP3 on win/loss only gets one bit per battle.

Opponents, in sequence: MCTS 2,000 iterations for N1 battles, then an exploiter for N2 battles that
knows the learner's current mixture p and plays the best reply argmin_b p · A[:, b] in the same matrix.
Run: .venv/bin/python exp3_live.py --seed S [--n1 40 --n2 40]   (X6 reproduction: --eta 0.2 --alpha 0.002 --scale 0) -> results/exp3/live_seed<S>.jsonl
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import poke_engine as pe

import nash
import pkbattle as pb

EXPERTS = ["Equilibrium, top move", "Equilibrium, sampled", "Safest move"]
OUT = Path(__file__).resolve().parent / "results" / "exp3"


def expert_rewards(A, Q, b, scale=400.0):
    """Existing bounded matrix-surrogate reward, shared by single/world-averaged feedback."""
    norm = scale if scale else (np.ptp(A) or 1.0)
    return np.clip((Q @ A[:, b] - A.min()) / norm, 0.0, 1.0)


class Learner:
    """Hedge over the experts. Defaults follow theory/hedge_experts.md (Prop. E1(b), Thm E2):
    fixed normaliser scale = Δ̄ (rewards (q_e·A[:,b] − min A)/Δ̄, clipped to [0,1]; subtracting the per-turn
    min A is common to all experts before clipping; saturation can remove reward differences),
    anytime step η_t = sqrt(8 ln k / t), no fixed share.
    Δ̄ = 400 covers ~99% of full_matrix ranges (median 166 on the 12-species pool, 203 on randbats; p99 327/405).
    The X6 setting (per-turn normaliser, η = 0.2, α = 0.002) is Learner(eta=0.2, alpha=0.002, scale=None):
    the per-turn normaliser has a linear-regret counterexample, and constant η = 0.2 gives linear regret."""

    def __init__(self, eta=None, alpha=0.0, scale=400.0, experts=EXPERTS):
        """`experts`: names of the experts (only their number matters to update(); plan() is the
        3-expert pool above, other pools build their own Q and call update directly)."""
        self.k = len(experts)
        self.logw, self.eta, self.alpha, self.scale, self.t = np.zeros(self.k), eta, alpha, scale, 0
        self.cum = np.zeros(self.k)                # cumulative normalised reward per expert

    def weights(self):
        w = np.exp(self.logw - self.logw.max())
        return w / w.sum()

    def plan(self, state, side):
        """Returns (own options, A, expert strategies Q [3 x n], mixture p)."""
        M, s1, s2 = nash.full_matrix(state)
        A, own = (M, s1) if side == 1 else (-M.T, s2)
        x, _ = nash.rm_plus(A, 1000)
        x = np.clip(x, 0, None); x /= x.sum()
        eye = np.eye(len(own))
        Q = np.stack([eye[int(x.argmax())], x, eye[int(A.min(1).argmax())]])
        return own, A, Q, self.weights() @ Q

    def update(self, A, Q, b):
        self.update_rewards(expert_rewards(A, Q, b, self.scale))

    def update_rewards(self, r):
        """Apply one full-information update from a bounded reward vector (one observation)."""
        self.t += 1
        self.cum += r
        if self.eta is None and not self.alpha:
            # anytime Hedge in FTRL form: w_t ∝ exp(η_t · cumulative reward), η_t = sqrt(8 ln k / t)
            # (Cesa-Bianchi & Lugosi 2006, Thm 2.3). The incremental form logw += η_t r_t is NOT
            # equivalent and has linear regret (early leads are never overturned).
            self.logw = np.sqrt(8 * np.log(self.k) / self.t) * self.cum
            return
        eta = self.eta if self.eta is not None else np.sqrt(8 * np.log(self.k) / self.t)
        self.logw += eta * r                       # constant η (incremental = FTRL), or fixed share
        if self.alpha:
            w = (1 - self.alpha) * self.weights() + self.alpha / self.k
            self.logw = np.log(w)


def exploiter_reply(state, lside, opp_opts, p=None, A=None):
    """Best reply (one turn deep) to the learner's mixture p over its options; a forced learner
    (one option) is the pure strategy [1]. Minimizes the learner's payoff in A (own perspective)."""
    if len(opp_opts) == 1:
        return opp_opts[0]
    if A is None:
        M, _, _ = nash.full_matrix(state)
        A = M if lside == 1 else -M.T
    if p is None:
        p = np.ones(A.shape[0]) / A.shape[0]
        assert A.shape[0] == 1, "p is required when the learner has several options"
    return opp_opts[int((p @ A).argmin())]


def battle(learner, opp_kind, lside, rng, max_turns=300):
    state = pb.random_state(rng)
    mcts = pb.mcts_policy(2000)
    for turn in range(1, max_turns + 1):
        o1, o2 = pb.options(state)
        own_opts, opp_opts = (o1, o2) if lside == 1 else (o2, o1)
        if len(own_opts) > 1:
            own, A, Q, p = learner.plan(state, lside)
            a = rng.choices(own, weights=p)[0]
            ob = (exploiter_reply(state, lside, opp_opts, p, A) if opp_kind == "exploiter"
                  else mcts(state, opp_opts, 3 - lside, rng))
            learner.update(A, Q, opp_opts.index(ob))
        else:
            a = own_opts[0]
            ob = (exploiter_reply(state, lside, opp_opts) if opp_kind == "exploiter"
                  else mcts(state, opp_opts, 3 - lside, rng))
        m1, m2 = (a, ob) if lside == 1 else (ob, a)
        state = pb.step(state, m1, m2, rng)
        a1, a2 = pb.alive(state.side_one), pb.alive(state.side_two)
        if a1 == 0 or a2 == 0:
            w = 0 if a1 == a2 else (1 if a1 else 2)
            return (0.5 if w == 0 else float(w == lside)), turn
    return 0.5, max_turns


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n1", type=int, default=40)
    ap.add_argument("--n2", type=int, default=40)
    ap.add_argument("--eta", type=float, default=None, help="constant step (X6 used 0.2); default anytime sqrt(8 ln k/t)")
    ap.add_argument("--alpha", type=float, default=0.0, help="fixed share (X6 used 0.002)")
    ap.add_argument("--scale", type=float, default=400.0, help="fixed normaliser; 0 = per-turn range (X6)")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    rng, L = random.Random(10_000 + a.seed), Learner(eta=a.eta, alpha=a.alpha, scale=a.scale or None)
    with open(OUT / f"live_seed{a.seed}.jsonl", "w") as f:
        for i in range(a.n1 + a.n2):
            kind = "mcts" if i < a.n1 else "exploiter"
            score, turns = battle(L, kind, 1 + i % 2, rng)
            f.write(json.dumps(dict(seed=a.seed, battle=i, opponent=kind, score=score, turns=turns, eta=a.eta,
                                    weights=dict(zip(EXPERTS, L.weights().round(4).tolist())))) + "\n")
            f.flush()


if __name__ == "__main__":
    main()
