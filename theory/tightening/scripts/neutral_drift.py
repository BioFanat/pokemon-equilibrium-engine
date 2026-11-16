"""Neutral-path control for the B12 drift channel.

Plays the target 'Equilibrium, sampled' (rm_plus(full_matrix, 1000) average, sampled; the W1 target
`nash:rm+:0:1000:sample`) on a random seat against a NON-adversarial opponent, and logs per turn exactly
what analyze_b12.py needs for the telescoping identity (target perspective):
  v = LP value of the target's depth-2 matrix A_t, xa_j = (x A)_{j}, a_ij = A[i, j], g = v - min_j (xA)_j, r = ptp(A)
plus the outcome z in {-1, 0, +1}. Opponents: 'mcts2000' (pb.mcts_policy(2000)) or 'self' (same sampled agent).
Usage (cwd engine, engine/.venv/bin/python):
  OMP_NUM_THREADS=1 python neutral_drift.py N_GAMES WORKERS SEED0 {mcts2000|self} out.jsonl
"""
from __future__ import annotations

import json
import os
import random
import sys
import time
from multiprocessing import get_context

ENGINE = "/Users/biofanat/Documents/classes/f26/cs535/final/engine"
sys.path.insert(0, ENGINE)
os.chdir(ENGINE)

OPP = "mcts2000"


def sampled_strategy(M):
    import numpy as np, nash
    x, _ = nash.rm_plus(M, 1000)
    x = np.clip(x, 0, None)
    return x / x.sum()


def game(args):
    seed, opp_name = args
    import numpy as np, nash, pkbattle as pb
    rng = random.Random(seed)
    st = pb.random_state(rng)
    tside = 1 + (seed % 2)                  # target seat alternates with the seed
    opp = pb.mcts_policy(2000) if opp_name == "mcts2000" else None
    per_turn, winner, t0 = [], None, time.perf_counter()
    for turn in range(1, 301):
        M, s1, s2 = nash.full_matrix(st)      # single option listing: pe.id's order is not stable across calls
        o1, o2 = s1, s2
        moves = {}
        for side, opts in ((1, o1), (2, o2)):
            G = M if side == 1 else -M.T
            is_target = side == tside
            if len(opts) == 1:
                moves[side] = opts[0]
                if is_target:
                    per_turn.append(dict(turn=turn, trivial=True))
                continue
            if is_target or opp is None:
                x = sampled_strategy(G)
                k = rng.choices(range(len(opts)), weights=x)[0]
                moves[side] = opts[k]
                if is_target:
                    moves["_x"], moves["_k"], moves["_G"] = x, k, G
            else:
                moves[side] = opp(st, opts, side, rng)
        if "_x" in moves:
            x, k, G = moves.pop("_x"), moves.pop("_k"), moves.pop("_G")
            oside = 3 - tside
            oopts = o1 if oside == 1 else o2
            j = oopts.index(moves[oside])
            xa = x @ G
            v = float(nash.lp(G)[2])
            per_turn.append(dict(turn=turn, v=v, xa_j=float(xa[j]), a_ij=float(G[k, j]), g=float(v - xa.min()),
                                 r=float(np.ptp(G)), mixed=bool((x > 1e-6).sum() > 1)))
        st = pb.step(st, moves[1], moves[2], rng)
        a1, a2 = pb.alive(st.side_one), pb.alive(st.side_two)
        if a1 == 0 or a2 == 0:
            winner = 0 if a1 == a2 else (1 if a1 else 2)
            break
    capped = winner is None
    z = 0.0 if (capped or winner == 0) else (1.0 if winner == tside else -1.0)
    return dict(seed=seed, opp=opp_name, tside=tside, winner=0 if capped else winner, capped=capped, z=z,
                turns=turn, wall_s=time.perf_counter() - t0, per_turn=per_turn)


def main():
    n, workers, seed0, opp, out = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), sys.argv[4], sys.argv[5]
    t0 = time.perf_counter()
    with get_context("spawn").Pool(workers) as pool, open(out, "w") as f:
        for g in pool.imap_unordered(game, [(seed0 + i, opp) for i in range(n)]):
            f.write(json.dumps(g) + "\n"); f.flush()
            print(f"seed {g['seed']} seat {g['tside']} z {g['z']:+.0f} turns {g['turns']} {g['wall_s']:.1f}s", flush=True)
    print(f"done {n} games in {time.perf_counter()-t0:.0f}s")


if __name__ == "__main__":
    main()
