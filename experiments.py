"""Head-to-head matches between engine agents, in parallel, on Gen 9 Random Battle teams.

Agent specs (strings, so they can be shipped to worker processes):
  random
  nash:<solver>:<ms>[:<iters>][:argmax][:legacy]
                                 per-turn equilibrium of nash.engine_matrix (unpruned depth-2; <ms> only
                                 used with :legacy), solver rm+ | omwu | lp; sampled, or its argmax
  safest:<ms>[:legacy]           pure maximin on the same matrix (deterministic)
  :legacy = the old pruned pe.id matrix with NaN->0 (buggy; how E1-E3 of 2026-11-13 were run --
            their records carry the spec WITHOUT the suffix, see results/experiments/NOTE.md)
  mcts_it:<N>                    poke-engine MCTS, N iterations, plays the most-visited move
  lbr:<argmax|sample|safest>     local best response to that full-matrix agent (knows its algorithm)
  wb2:<argmax|sample>[:K:supp]   white-box depth-2 exploiter of that equilibrium agent (nash.wb2_policy;
                                 top-K own actions, default 5, target support to mass supp, default 0.95)
  mcts_ms:<ms>                   poke-engine MCTS, time budget instead of iterations
  vnet[:argmax|:sample]          RM+ equilibrium of nash.vnet_matrix (depth 1, value-net leaves from
                                 results/value/vnet.npz); its argmax (default) or the sampled mix
  foulplay_it:<N> / foulplay_ms:<N>  foul-play's selection rule on one MCTS run (nash.foulplay_strategy):
                                 sample among options with >= 75% of the top visit fraction
  lbr:foulplay_it:<N> | lbr:foulplay_ms:<N> | lbr:mcts   local best response to those MCTS agents
                                 (lbr:mcts = to mcts_it:2000's most-visited move); q re-estimated by its own run
  hedge2                         Hedge over {foulplay_it:2000, equilibrium sampled}, weights reset per game
  xhedge2                        white-box exploiter of hedge2 (paired with it in play_pair, see XHedge2)

Design: games come in pairs. Both games of a pair use the same seed, so the same two teams and
the same first random draws, with A on side one in the first game and on side two in the second.
The unit of analysis is the pair (score in [0, 1] per pair, a draw counts as half), so the CI
accounts for team luck being shared within a pair.

  python experiments.py E1 [--pairs N] [--workers W]     # see MATRIX below
  python experiments.py summary                          # table over results/experiments/*.jsonl
"""
from __future__ import annotations

import argparse
import json
import math
import random
import time
from multiprocessing import get_context
from pathlib import Path

OUT = Path(__file__).parent / "results" / "experiments"

# experiment -> list of (A, B, pairs). 200 pairs = 400 games.
MATRIX = {
    "E0": [("nash:rm+:50:1000", "random", 10)],                     # smoke test
    "E1fix": [("safest:0", "mcts_it:2000", 200),                    # rerun of E1 on the fixed matrix
              ("nash:rm+:0:1000:argmax", "mcts_it:2000", 200),
              ("nash:rm+:0:1000", "mcts_it:2000", 200),
              ("nash:rm+:0:1000", "safest:0", 200)],
    "X1": [("nash:rm+:0:1000:argmax", "mcts_ms:75", 100),          # time-matched MCTS (~72 ms/move)
           ("safest:0", "mcts_ms:75", 100)],
    "V1": [("vnet", "mcts_ms:35", 100),                            # value-net leaves (~30 ms/move median)
           ("vnet", "mcts_it:2000", 100),
           ("vnet", "nash:rm+:0:1000:argmax", 100)],
    "W1": [("wb2:argmax:4:0.8", "nash:rm+:0:1000:argmax", 100),    # confirmatory: white-box depth-2
           ("wb2:sample:4:0.8", "nash:rm+:0:1000", 100)],          # exploiter, fresh seeds (SEED_BASE);
                                                                   # K=4/80%: K=5/95% sampled was 2.4x slower
    "E6q": [("nash:rm+:0:1000:argmax", "lbr:argmax", 30),        # quick exploitability: local best response
            ("nash:rm+:0:1000", "lbr:sample", 30),
            ("safest:0", "lbr:safest", 30)],
    "E7": [("foulplay_it:2000", "nash:rm+:0:1000", 50),            # foul-play-style MCTS head-to-head
           ("foulplay_it:2000", "nash:rm+:0:1000:argmax", 50),
           ("foulplay_it:2000", "mcts_it:2000", 50),               # sanity: ~0.5
           ("foulplay_ms:75", "nash:rm+:0:1000:argmax", 50)],      # time-matched to the matrix bot
    "E6f": [("foulplay_it:2000", "lbr:foulplay_it:2000", 30),      # exploitability of the MCTS agents
            ("mcts_it:2000", "lbr:mcts", 30)],
    "E7t": [("foulplay_ms:75", "nash:rm+:0:1000", 50)],               # time-matched vs equilibrium sampled
    "E6ft": [("foulplay_ms:75", "lbr:foulplay_ms:75", 30)],           # exploitability of time-matched foul-play
    "H1": [("hedge2", "mcts_it:2000", 50),                         # Hedge over {foul-play, equilibrium}
           ("hedge2", "xhedge2", 30)],
    "E1": [("nash:rm+:50:1000", "random", 50),
           ("nash:rm+:50:1000", "safest:50", 200),
           ("nash:rm+:50:1000", "mcts_it:2000", 200),
           ("safest:50", "mcts_it:2000", 200)],
    "E2": [(f"nash:rm+:{ms}:1000", f"mcts_ms:{ms}", 200) for ms in (10, 50, 200)],
    "E3": [("nash:rm+:50:100", "nash:lp:50", 200),
           ("nash:rm+:50:1000", "nash:lp:50", 200)],
}

# experiment -> offset added to its seeds (default 0), so its pairs use teams no other run has seen.
# Without it seeds depend only on (matchup index, pair index), shared by every experiment.
SEED_BASE = {"W1": 50_000_000}


def make_agent(spec: str):
    import numpy as np
    import poke_engine as pe
    import nash
    import pkbattle as pb

    kind, *a = spec.split(":")
    legacy, argmax = "legacy" in a, "argmax" in a
    a = [x for x in a if x not in ("legacy", "argmax")]
    if kind == "random":
        return pb.random_policy
    if kind == "safest":
        return nash.safest_policy(int(a[0]), legacy)
    if kind == "nash":
        solver = {"rm+": "rm_plus"}.get(a[0], a[0])
        return nash.nash_policy(solver, int(a[1]), int(a[2]) if len(a) > 2 else 1000,
                                "argmax" if argmax else "sample", legacy)
    if kind == "vnet":
        return nash.vnet_policy(1000, "sample" if "sample" in a else "argmax")
    if kind == "lbr":                                  # lbr:argmax | lbr:sample | lbr:safest | lbr:mcts | lbr:foulplay_*:N
        return nash.lbr_policy(spec.split(":", 1)[1])
    if kind == "wb2":                                  # wb2:argmax | wb2:sample [:K:supp_mass]
        n = [x for x in a if x != "sample"]
        return nash.wb2_policy("argmax" if argmax else "sample", *([int(n[0]), 0.95, float(n[1])] if n else []))
    if kind in ("foulplay_it", "foulplay_ms"):
        n = int(a[0])
        ms, it = (0, n) if kind == "foulplay_it" else (n, 0)

        def pol(state, opts, side, rng):
            if len(opts) == 1:
                return opts[0]
            own, q = nash.foulplay_strategy(state, side, ms, it)
            return rng.choices(own, weights=q)[0]
        return pol
    if kind == "hedge2":
        return Hedge2()
    if kind == "xhedge2":
        return XHedge2()
    if kind in ("mcts_it", "mcts_ms"):
        n = int(a[0])
        ms, it = (0, n) if kind == "mcts_it" else (n, 0)

        def pol(state, opts, side, rng):
            if len(opts) == 1:
                return opts[0]
            r = pe.mcts(state, ms, it, 1)
            res = r.s1 if side == 1 else r.s2
            return pb._canon(max(res, key=lambda x: x.visits).move_choice)
        return pol
    raise ValueError(spec)


class Hedge2:
    """Hedge over two experts, from one full matrix A (own perspective) and one MCTS run per turn:
      q_F = foul-play's distribution (nash.foulplay_strategy, 2000 iterations),
      q_N = RM+ equilibrium of A (1000 iterations), clipped and normalised.
    Plays p = w . Q. After each non-forced turn, from the opponent's realised move b, every expert gets
    exp3_live's reward r_e = clip((q_e . A[:, b] - min A) / 400, 0, 1), anytime eta = sqrt(8 ln 2 / t),
    no fixed share: the theory/hedge_experts.md Prop. E1(b) setting (fixed normaliser, an equilibrium
    expert in the pool). One object per game (make_agent is called per game), so weights reset per game.
    Caveat: the reward judges the MCTS expert by the heuristic matrix A. E1's safety therefore holds
    relative to val(A), and the learner favours MCTS only when MCTS's moves look good under A, not when
    they win games the matrix misjudges.
    The turn's plan is cached on the state object so a paired XHedge2 sees the same p (and A)."""
    EXPERTS = ["Foul-play MCTS", "Equilibrium, sampled"]

    def __init__(self):
        import exp3_live
        self.learner = exp3_live.Learner(experts=self.EXPERTS)
        self.cache = None                          # (state, side, own, opp, A, Q, p) of the current turn
        self.pending = None                        # (A, Q, opp options) awaiting observe()

    def weights(self):
        return self.learner.weights()

    def plan(self, state, side):
        """(own options, opponent options, A, Q, p) for this turn; Q is None on a forced turn."""
        import numpy as np
        import nash
        if self.cache is not None and self.cache[0] is state and self.cache[1] == side:
            return self.cache[2:]
        M, s1, s2 = nash.full_matrix(state)
        (A, own, opp) = (M, s1, s2) if side == 1 else (-M.T, s2, s1)
        if len(own) == 1:
            Q, p = None, np.ones(1)
        else:
            x, _ = nash.rm_plus(A, 1000)
            x = np.clip(x, 0, None); x /= x.sum()
            fo, qf = nash.foulplay_strategy(state, side, 0, 2000)
            assert fo == own, (fo, own)
            Q = np.stack([qf, x])
            p = self.weights() @ Q
        self.cache = (state, side, own, opp, A, Q, p)  # holding `state` keeps its id from being reused
        return self.cache[2:]

    def __call__(self, state, opts, side, rng):
        if len(opts) == 1:
            self.pending = None
            return opts[0]
        own, opp, A, Q, p = self.plan(state, side)
        self.pending = (A, Q, opp)
        return rng.choices(own, weights=p)[0]

    def observe(self, own_move, opp_move):
        if self.pending is not None:
            A, Q, opp = self.pending
            self.learner.update(A, Q, opp.index(opp_move))
            self.pending = None


class XHedge2:
    """White-box exploiter of Hedge2: play_pair hands it the learner object (pair_with), it reads the
    learner's current mixture p (the learner's own cached plan for this turn, so the same MCTS draw) and
    plays argmin_b p . A[:, b] (exp3_live.exploiter_reply). In games where it is side one it moves first
    and triggers the learner's plan, so the plan's cost lands in ITS ms/move, not the learner's."""

    def __init__(self):
        self.learner = None

    def pair_with(self, other):
        if isinstance(other, Hedge2):
            self.learner = other

    def __call__(self, state, opts, side, rng):
        import exp3_live
        import pkbattle as pb
        assert self.learner is not None, "xhedge2 must be paired with hedge2"
        if len(opts) == 1:
            return opts[0]
        lside = 3 - side
        if len(pb.options(state)[lside - 1]) == 1:  # forced learner: pure strategy [1]
            return exp3_live.exploiter_reply(state, lside, opts)
        _, _, A, _, p = self.learner.plan(state, lside)
        return exp3_live.exploiter_reply(state, lside, opts, p, A)


def timed(pol, clock):
    def f(state, opts, side, rng):
        t = time.perf_counter()
        m = pol(state, opts, side, rng)
        clock[0] += time.perf_counter() - t
        clock[1] += 1
        return m
    if hasattr(pol, "observe"):
        f.observe = pol.observe
    return f


def play_pair(task):
    """task = (exp, A, B, pair_index, seed) -> one record per game."""
    import pkbattle as pb
    exp, A, B, k, seed = task
    recs = []
    for a_side in (1, 2):
        ca, cb = [0.0, 0], [0.0, 0]
        ra, rb = make_agent(A), make_agent(B)
        for x, y in ((ra, rb), (rb, ra)):          # pairing hook (xhedge2 reads hedge2's state)
            if hasattr(x, "pair_with"):
                x.pair_with(y)
        pa, pb_ = timed(ra, ca), timed(rb, cb)
        p1, p2 = (pa, pb_) if a_side == 1 else (pb_, pa)

        def log(turn, m1, m2):                     # agents with observe() see both realised moves
            if hasattr(p1, "observe"):
                p1.observe(m1, m2)
            if hasattr(p2, "observe"):
                p2.observe(m2, m1)
        rng = random.Random(seed)
        t = time.perf_counter()
        try:
            w, turns = pb.play(p1, p2, rng, log=log)
            err = None
        except Exception as e:                     # record, don't kill the run
            w, turns, err = -1, 0, repr(e)[:300]
        score = None if w < 0 else (0.5 if w == 0 else float(w == a_side))
        recs.append(dict(exp=exp, A=A, B=B, pair=k, seed=seed, a_side=a_side, winner=w,
                         score_A=score, turns=turns, error=err, wall_s=time.perf_counter() - t,
                         ms_per_move_A=1e3 * ca[0] / max(1, ca[1]),
                         ms_per_move_B=1e3 * cb[0] / max(1, cb[1])))
        for key, ag in (("weights_A", ra), ("weights_B", rb)):
            if hasattr(ag, "weights"):              # learners: final expert weights of this game
                recs[-1][key] = dict(zip(ag.EXPERTS, ag.weights().round(4).tolist()))
    return recs


def run(exp: str, workers: int, pairs: int | None):
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"{exp}.jsonl"
    done = set()
    if path.exists():                              # resume: skip pairs already recorded
        for line in path.read_text().splitlines():
            r = json.loads(line)
            done.add((r["A"], r["B"], r["pair"]))
    tasks = []
    for i, (A, B, n) in enumerate(MATRIX[exp]):
        for k in range(pairs or n):
            if (A, B, k) not in done:
                tasks.append((exp, A, B, k, SEED_BASE.get(exp, 0) + 1_000_003 * (i + 1) + k))
    random.Random(0).shuffle(tasks)                # interleave matchups so partial runs are balanced
    print(f"{exp}: {len(tasks)} pairs to play ({len(done)} already done), {workers} workers", flush=True)
    t0 = time.perf_counter()
    with get_context("spawn").Pool(workers) as pool, path.open("a") as f:
        for j, recs in enumerate(pool.imap_unordered(play_pair, tasks), 1):
            for r in recs:
                f.write(json.dumps(r) + "\n")
            f.flush()
            if j % 20 == 0 or j == len(tasks):
                print(f"  {j}/{len(tasks)} pairs, {time.perf_counter() - t0:.0f}s", flush=True)


def summary():
    rows = {}
    for p in sorted(OUT.glob("*.jsonl")):
        for line in p.read_text().splitlines():
            r = json.loads(line)
            rows.setdefault((r["exp"], r["A"], r["B"]), []).append(r)
    print(f"{'exp':4s} {'A':20s} {'B':16s} {'games':>5s} {'err':>3s} {'A score':>8s} {'95% CI':>15s} "
          f"{'W-L-D':>11s} {'turns':>5s} {'ms/mv A':>7s} {'ms/mv B':>7s}")
    for (exp, A, B), rs in sorted(rows.items()):
        ok = [r for r in rs if r["score_A"] is not None]
        pairs = {}
        for r in ok:
            pairs.setdefault(r["pair"], []).append(r["score_A"])
        ps = [sum(v) / len(v) for v in pairs.values()]
        n = len(ps)
        m = sum(ps) / n if n else float("nan")
        sd = math.sqrt(sum((x - m) ** 2 for x in ps) / (n - 1)) if n > 1 else float("nan")
        h = 1.96 * sd / math.sqrt(n) if n > 1 else float("nan")
        w = sum(r["score_A"] == 1 for r in ok); l = sum(r["score_A"] == 0 for r in ok); d = len(ok) - w - l
        mean = lambda k: sum(r[k] for r in ok) / max(1, len(ok))
        print(f"{exp:4s} {A:20s} {B:16s} {len(ok):5d} {len(rs) - len(ok):3d} {m:8.3f} "
              f"[{m - h:.3f}, {m + h:.3f}] {f'{w}-{l}-{d}':>11s} {mean('turns'):5.0f} "
              f"{mean('ms_per_move_A'):7.1f} {mean('ms_per_move_B'):7.1f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("exp", choices=[*MATRIX, "summary"])
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--pairs", type=int, default=None, help="override pairs per matchup")
    a = ap.parse_args()
    summary() if a.exp == "summary" else run(a.exp, a.workers, a.pairs)
