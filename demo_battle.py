"""Demo: one battle of the equilibrium agent (side 1) vs MCTS 2,000 iterations (side 2), printing the per-turn solve.

Each turn: build the full payoff matrix, solve it with RM+, print the matrix size, the mixed strategy and the moves.
  .venv/bin/python demo_battle.py [--seed 3] [--out demo.txt]
"""
from __future__ import annotations

import argparse
import json
import random

import numpy as np

import nash
import pkbattle as pb


def mon(side) -> str:
    p = side.pokemon[int(str(side.active_index)[-1])]
    return f"{str(p.id).lower():<12} {100 * p.hp / p.maxhp:5.0f}% HP"


def snap(side) -> dict:
    """Active species, HP fraction, and the team's remaining count."""
    p = side.pokemon[int(str(side.active_index)[-1])]
    return dict(species=str(p.id).lower(), hp=round(p.hp / p.maxhp, 4) if p.maxhp else 0.0,
                alive=pb.alive(side), team=[str(q.id).lower() for q in side.pokemon],
                hps=[round(q.hp / q.maxhp, 3) if q.maxhp else 0.0 for q in side.pokemon])


def label(side, move: str) -> str:
    """'switch → x' for switches, the move name otherwise."""
    names = [str(p.id).lower() for p in side.pokemon]
    return f"switch → {move.lower()}" if move.lower() in names else move


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--out", default=None)
    ap.add_argument("--json", default=None, help="per-turn records (matrix, strategies, HP) for the visual demo")
    a = ap.parse_args()
    rng, mcts, lines, recs = random.Random(a.seed), pb.mcts_policy(2000), [], []
    say = lambda s="": (print(s), lines.append(s))
    state = pb.random_state(rng)
    say("Equilibrium agent (us)  vs  MCTS, 2,000 iterations")
    for turn in range(1, 301):
        o1, o2 = pb.options(state)
        say(f"── Turn {turn} " + "─" * 46)
        say(f"  us:  {mon(state.side_one)}      them: {mon(state.side_two)}")
        rec = dict(turn=turn, us=snap(state.side_one), them=snap(state.side_two))
        if len(o1) > 1:
            M, s1, s2 = nash.full_matrix(state)
            x, y = nash.rm_plus(M, 1000)
            rec.update(M=M.round(2).tolist(), s1=[label(state.side_one, o) for o in s1],
                       s2=[label(state.side_two, o) for o in s2], x=x.round(4).tolist(), y=y.round(4).tolist())
            x = np.clip(x, 0, None); x /= x.sum()
            say(f"  payoff matrix {M.shape[0]}×{M.shape[1]}  →  RM+, 1,000 iterations")
            top = np.argsort(-x)[:3]
            say("  mixed strategy: " + "   ".join(f"{label(state.side_one, s1[i])} {x[i]:.2f}" for i in top if x[i] >= 0.01))
            m1 = s1[int(rng.choices(range(len(s1)), weights=x)[0])]
        else:
            m1 = o1[0]
            say("  forced move")
        m2 = mcts(state, o2, 2, rng)
        rec.update(m1=label(state.side_one, m1), m2=label(state.side_two, m2))
        recs.append(rec)
        say(f"  we play: {label(state.side_one, m1):<22} they play: {label(state.side_two, m2)}")
        state = pb.step(state, m1, m2, rng)
        n1, n2 = pb.alive(state.side_one), pb.alive(state.side_two)
        if n1 == 0 or n2 == 0:
            say("─" * 56)
            say("Result: " + ("draw" if n1 == n2 else ("we win" if n1 else "MCTS wins")) + f" in {turn} turns")
            recs.append(dict(turn=turn + 1, us=snap(state.side_one), them=snap(state.side_two),
                             result="draw" if n1 == n2 else ("win" if n1 else "loss")))
            break
    if a.json:
        json.dump(recs, open(a.json, "w"))
    if a.out:
        open(a.out, "w").write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
