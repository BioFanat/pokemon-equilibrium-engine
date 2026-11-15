"""Hedge2 with weights carried across games (the theory/hedge_experts.md Thm E2 setting), vs the
per-game reset used in H1.

One Hedge2 learner plays a stream of games: N1 games vs mcts_it:2000, then N2 games vs XHedge2
paired with that same learner (the opponent switches mid-stream). Seats alternate (learner on side
1 + i % 2), and game i of seed S uses the deterministic game seed 1_000_003 * (100 + S) + i, so the
persistent and --reset streams of a seed play the same teams. --reset recreates the learner every
game (control: the H1 setting). The anytime step eta_t = sqrt(8 ln 2 / t) keeps counting turns
across games in the persistent stream.

  .venv/bin/python hedge_stream.py --seeds 0 1 2 3 [--n1 30 --n2 30] [--reset] [--workers 4]
  .venv/bin/python hedge_stream.py summary          # table over results/experiments/hedge_stream_*.jsonl
-> results/experiments/hedge_stream_seed<S>[_reset].jsonl, one row per game.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import time
from multiprocessing import get_context

from experiments import OUT, Hedge2, XHedge2, make_agent


class LoggedHedge2(Hedge2):
    """Hedge2 that records the foul-play weight w_F used on each of its non-forced turns."""

    def __init__(self):
        super().__init__()
        self.pf = []

    def __call__(self, state, opts, side, rng):
        if len(opts) > 1:
            self.pf.append(float(self.weights()[0]))
        return super().__call__(state, opts, side, rng)


def play_game(learner, opp, lside, seed):
    import pkbattle as pb
    p1, p2 = (learner, opp) if lside == 1 else (opp, learner)

    def log(turn, m1, m2):
        if hasattr(p1, "observe"):
            p1.observe(m1, m2)
        if hasattr(p2, "observe"):
            p2.observe(m2, m1)
    w, turns = pb.play(p1, p2, random.Random(seed), log=log)
    return (0.5 if w == 0 else float(w == lside)), turns


def stream(task):
    """task = (seed, n1, n2, reset) -> writes the stream's JSONL, returns its path."""
    S, n1, n2, reset = task
    path = OUT / f"hedge_stream_seed{S}{'_reset' if reset else ''}.jsonl"
    learner = None
    with path.open("w") as f:
        for i in range(n1 + n2):
            if learner is None or reset:
                learner = LoggedHedge2()
            learner.pf = []
            kind = "mcts" if i < n1 else "exploiter"
            if kind == "mcts":
                opp = make_agent("mcts_it:2000")
            else:
                opp = XHedge2()
                opp.pair_with(learner)
            lside, gseed = 1 + i % 2, 1_000_003 * (100 + S) + i
            t = time.perf_counter()
            score, turns = play_game(learner, opp, lside, gseed)
            f.write(json.dumps(dict(seed=S, game=i, opponent=kind, reset=reset, lside=lside, game_seed=gseed,
                                    score=score, turns=turns, t_learner=learner.learner.t,
                                    weights=dict(zip(Hedge2.EXPERTS, learner.weights().round(4).tolist())),
                                    mean_pF=(sum(learner.pf) / len(learner.pf)) if learner.pf else None,
                                    wall_s=time.perf_counter() - t)) + "\n")
            f.flush()
    return str(path)


T975 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365}


def ci(xs, t=False):
    """Mean and 95% half-width; t=True uses Student t (for few units, e.g. seeds), else 1.96."""
    n = len(xs)
    m = sum(xs) / n
    sd = math.sqrt(sum((x - m) ** 2 for x in xs) / (n - 1)) if n > 1 else float("nan")
    return m, (T975.get(n - 1, 1.96) if t else 1.96) * sd / math.sqrt(n)


def summary():
    rows = [json.loads(l) for p in sorted(OUT.glob("hedge_stream_seed*.jsonl")) for l in p.read_text().splitlines()]
    F = Hedge2.EXPERTS[0]
    print(f"{'mode':10s} {'phase':9s} {'games':>5s} {'score':>6s} {'95% CI (games)':>16s} "
          f"{'95% CI (seeds)':>16s} {'end w_F':>7s} {'mean p_F':>8s}")
    for reset in (False, True):
        for kind in ("mcts", "exploiter"):
            rs = [r for r in rows if r["reset"] == reset and r["opponent"] == kind]
            if not rs:
                continue
            m, h = ci([r["score"] for r in rs])
            seeds = sorted({r["seed"] for r in rs})
            sm, sh = ci([sum(r["score"] for r in rs if r["seed"] == s) / sum(r["seed"] == s for r in rs)
                         for s in seeds], t=True)
            wf = sum(r["weights"][F] for r in rs) / len(rs)
            pf = [r["mean_pF"] for r in rs if r["mean_pF"] is not None]
            print(f"{'reset' if reset else 'persistent':10s} {kind:9s} {len(rs):5d} {m:6.3f} "
                  f"[{m - h:.3f}, {m + h:.3f}] [{sm - sh:.3f}, {sm + sh:.3f}] {wf:7.3f} {sum(pf) / len(pf):8.3f}")
    print("\npersistent - reset, paired on (seed, game) = same teams and first draws:")
    key = {(r["reset"], r["seed"], r["game"]): r for r in rows}
    for kind in ("mcts", "exploiter"):
        d = [r["score"] - key[(True, r["seed"], r["game"])]["score"] for r in rows
             if not r["reset"] and r["opponent"] == kind and (True, r["seed"], r["game"]) in key]
        if d:
            m, h = ci(d)
            print(f"  {kind:9s} {len(d):3d} pairs  diff {m:+.3f} [{m - h:+.3f}, {m + h:+.3f}]")
    print("\nend-of-game w_F, mean over seeds, by game index:")
    for reset in (False, True):
        by = {}
        for r in rows:
            if r["reset"] == reset:
                by.setdefault(r["game"], []).append(r["weights"][F])
        print(f"  {'reset' if reset else 'persistent':10s} " +
              " ".join(f"{g}:{sum(v) / len(v):.2f}" for g, v in sorted(by.items())))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", nargs="?", default="run", choices=["run", "summary"])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--n1", type=int, default=30)
    ap.add_argument("--n2", type=int, default=30)
    ap.add_argument("--reset", action="store_true", help="recreate the learner every game (control)")
    ap.add_argument("--workers", type=int, default=4, help="processes, one stream (seed) each")
    a = ap.parse_args()
    if a.cmd == "summary":
        return summary()
    OUT.mkdir(parents=True, exist_ok=True)
    tasks = [(s, a.n1, a.n2, a.reset) for s in a.seeds]
    t0 = time.perf_counter()
    with get_context("spawn").Pool(min(a.workers, len(tasks))) as pool:
        for p in pool.imap_unordered(stream, tasks):
            print(f"  {p} done, {time.perf_counter() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
