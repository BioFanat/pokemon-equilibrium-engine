"""Self-play positions labelled with the game outcome, for the value net.

Both sides play poke-engine MCTS (1,000 iterations, most-visited move; one search per turn serves
both sides), each side independently replaced by a uniformly random legal option with prob. eps.
Every turn records encode(state) before the moves and the engine heuristic h = nash._leaf(state,
'mid') (the baseline value_net.py compares against). z = +1 / 0 (draw or 300-turn cap) / -1
from side one's perspective.

  nice -n 10 .venv/bin/python value_data.py --games 50 --workers 6 --seed 0
Output: results/value/data.npz with X (float32), z, h, game_id.
"""
from __future__ import annotations

import argparse
import random
import time
from multiprocessing import get_context
from pathlib import Path

import numpy as np

OUT = Path(__file__).parent / "results" / "value"
ITERS, EPS, MAX_TURNS = 1000, 0.1, 300


def play_game(task):
    """task = (game_id, seed) -> (X, h, z, game_id) for one self-play game."""
    import poke_engine as pe
    import features
    import nash
    import pkbattle as pb
    g, seed = task
    rng = random.Random(seed)
    state, X, H, z = pb.random_state(rng), [], [], 0
    for _ in range(MAX_TURNS):
        o1, o2 = pb.options(state)
        X.append(features.encode(state))
        H.append(nash._leaf(state, "mid"))
        r = pe.mcts(state, 0, ITERS, 1) if max(len(o1), len(o2)) > 1 else None
        moves = []
        for opts, res in ((o1, r and r.s1), (o2, r and r.s2)):
            if len(opts) == 1:
                moves.append(opts[0])
            elif rng.random() < EPS:
                moves.append(rng.choice(opts))
            else:
                moves.append(pb._canon(max(res, key=lambda x: x.visits).move_choice))
        state = pb.step(state, moves[0], moves[1], rng)
        a1, a2 = pb.alive(state.side_one), pb.alive(state.side_two)
        if a1 == 0 or a2 == 0:
            z = 0 if a1 == a2 else (1 if a1 else -1)
            break
    return np.stack(X), np.array(H, np.float32), z, g


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=50)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=str(OUT / "data.npz"))
    args = ap.parse_args()
    tasks = [(g, 1_000_003 * (args.seed + 1) + g) for g in range(args.games)]
    Xs, Hs, Zs, Gs = [], [], [], []
    t0 = time.perf_counter()
    with get_context("spawn").Pool(min(args.workers, 6)) as pool:
        for j, (X, h, z, g) in enumerate(pool.imap_unordered(play_game, tasks, chunksize=4), 1):
            Xs.append(X); Hs.append(h); Zs.append(np.full(len(X), z, np.int8)); Gs.append(np.full(len(X), g, np.int32))
            if j % 200 == 0 or j == len(tasks):
                n = sum(len(x) for x in Xs)
                print(f"  {j}/{len(tasks)} games, {n} positions, {time.perf_counter() - t0:.0f}s", flush=True)
    dt = time.perf_counter() - t0
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, X=np.concatenate(Xs), h=np.concatenate(Hs), z=np.concatenate(Zs),
                        game_id=np.concatenate(Gs))
    n, zg = sum(len(x) for x in Xs), np.array([z[0] for z in Zs])
    print(f"{args.games} games, {n} positions in {dt:.1f}s ({n / dt:.0f} pos/s); "
          f"z: +1 {int((zg == 1).sum())}, 0 {int((zg == 0).sum())}, -1 {int((zg == -1).sum())} -> {args.out}")


if __name__ == "__main__":
    main()
