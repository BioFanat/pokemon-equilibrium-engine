"""Paired-seed hidden-information matches (each side sees only its own View/Belief).

Same design as experiments.py: pairs share a seed (same teams), A plays side one then side two;
the pair is the unit of analysis. Output: results/hidden/<exp>.jsonl; with --log-decisions (or
exp E5) per-decision blend records go to results/hidden/E5_decisions.jsonl.

  python hidden_experiments.py H0 --workers 2
  python hidden_experiments.py E4 --workers 10 [--log-decisions]
  python hidden_experiments.py summary
"""
from __future__ import annotations

import argparse
import json
import math
import random
import time
from multiprocessing import get_context
from pathlib import Path

OUT = Path(__file__).parent / "results" / "hidden"
DEC = OUT / "E5_decisions.jsonl"

MATRIX = {
    "H0": [("agentform:4:50", "random", 10)],
    "E4": [("agentform:4:200", "pimc:4:200", 200),
           ("agentform:4:200", "mostlikely:200", 200),
           ("pimc:4:200", "mostlikely:200", 200),
           ("oracle:200", "agentform:4:200", 200)],
    "E4am": [("agentform:4:200:argmax", "pimc:4:200:argmax", 50),     # E4 with argmax play
             ("agentform:4:200:argmax", "mostlikely:200:argmax", 50),
             ("pimc:4:200:argmax", "mostlikely:200:argmax", 50),
             ("oracle:200:argmax", "agentform:4:200:argmax", 50)],
    "X2": [("agentform:8:200:argmax", "agentform:4:200:argmax", 50)],   # do more worlds help?
    "X5": [("agentform:4:200:argmax", "xlbr:agentform:4:200:argmax", 20),   # hidden-info exploiter
           ("agentform:4:200", "xlbr:agentform:4:200", 20)],
    "E5": [("agentform:4:200", "pimc:4:200", 20)],          # small run with logging forced on
}


def play_hidden(agent1, agent2, rng, max_turns=300, clocks=None):
    """Returns (winner 1/2/0, turns, decision records)."""
    import hidden as hd
    import pkbattle as pb
    state = pb.random_state(rng)
    tr = {1: hd.Tracker(1, state), 2: hd.Tracker(2, state)}
    agents = {1: agent1, 2: agent2}
    decs = []
    for turn in range(1, max_turns + 1):
        opts = dict(zip((1, 2), pb.options(state)))
        ch = {}
        for s in (1, 2):
            view = hd.make_view(state, tr[s])
            belief = hd.Belief(view)
            t = time.perf_counter()
            if getattr(agents[s], "kind", "") == "xlbr":
                ov = hd.make_view(state, tr[3 - s])
                agents[s].opp = (ov, hd.Belief(ov))
            n0, l0 = len(agents[s].records), agents[s].log_s
            ch[s] = agents[s].act(state, view, belief, opts[s], s, rng)
            if clocks is not None and len(opts[s]) > 1:
                clocks[s][0] += time.perf_counter() - t - (agents[s].log_s - l0)
                clocks[s][1] += 1
            for r in agents[s].records[n0:]:
                decs.append({**r, "turn": turn})
        branches = __import__("poke_engine").generate_instructions(state, ch[1], ch[2])
        pick = rng.choices(branches, weights=[b.percentage for b in branches])[0]
        new = state.apply_instructions(pick)
        for s in (1, 2):
            tr[s].update(state, ch[1], ch[2], pick, new)
        state = new
        a1, a2 = pb.alive(state.side_one), pb.alive(state.side_two)
        if a1 == 0 or a2 == 0:
            return (0 if a1 == a2 else (1 if a1 else 2)), turn, decs
    return 0, max_turns, decs


def play_pair(task):
    import hidden_agents as ha
    exp, A, B, k, seed, log = task
    recs, decs = [], []
    for a_side in (1, 2):
        ga, gb = ha.Agent(A, log), ha.Agent(B, log)
        rng = random.Random(seed)
        clocks = {1: [0.0, 0], 2: [0.0, 0]}
        t = time.perf_counter()
        try:
            w, turns, d = play_hidden(ga, gb, rng, clocks=clocks) if a_side == 1 else play_hidden(gb, ga, rng, clocks=clocks)
            err = None
        except Exception as e:
            import traceback
            w, turns, d, err = -1, 0, [], traceback.format_exc()[-600:]
        ca, cb = (clocks[1], clocks[2]) if a_side == 1 else (clocks[2], clocks[1])
        score = None if w < 0 else (0.5 if w == 0 else float(w == a_side))
        recs.append(dict(exp=exp, A=A, B=B, pair=k, seed=seed, a_side=a_side, winner=w, score_A=score,
                         turns=turns, error=err, wall_s=time.perf_counter() - t,
                         ms_per_move_A=1e3 * ca[0] / max(1, ca[1]), ms_per_move_B=1e3 * cb[0] / max(1, cb[1])))
        decs += [dict(exp=exp, A=A, B=B, pair=k, a_side=a_side, **r) for r in d]
    return recs, decs


def run(exp, workers, pairs, log, only=None):
    OUT.mkdir(parents=True, exist_ok=True)
    log = log or exp == "E5"
    path = OUT / f"{exp}.jsonl"
    done = set()
    if path.exists():
        for line in path.read_text().splitlines():
            r = json.loads(line)
            done.add((r["A"], r["B"], r["pair"]))
    tasks = [(exp, A, B, k, 1_000_003 * (i + 1) + k, log)
             for i, (A, B, n) in enumerate(MATRIX[exp]) if only is None or i in only
             for k in range(pairs or n) if (A, B, k) not in done]
    random.Random(0).shuffle(tasks)
    print(f"{exp}: {len(tasks)} pairs to play ({len(done)} done), {workers} workers, log={log}", flush=True)
    t0 = time.perf_counter()
    with get_context("spawn").Pool(workers) as pool, path.open("a") as f, DEC.open("a") as fd:
        for j, (recs, decs) in enumerate(pool.imap_unordered(play_pair, tasks), 1):
            for r in recs:
                f.write(json.dumps(r) + "\n")
            for r in decs:
                fd.write(json.dumps(r) + "\n")
            f.flush(); fd.flush()
            if j % 5 == 0 or j == len(tasks):
                print(f"  {j}/{len(tasks)} pairs, {time.perf_counter() - t0:.0f}s", flush=True)


def summary():
    rows = {}
    for p in sorted(OUT.glob("*.jsonl")):
        if p == DEC:
            continue
        for line in p.read_text().splitlines():
            r = json.loads(line)
            rows.setdefault((r["exp"], r["A"], r["B"]), []).append(r)
    print(f"{'exp':4s} {'A':18s} {'B':16s} {'games':>5s} {'err':>3s} {'A score':>8s} {'95% CI':>15s} "
          f"{'W-L-D':>9s} {'turns':>5s} {'ms/mv A':>7s} {'ms/mv B':>7s} {'s/game':>6s}")
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
        w = sum(r["score_A"] == 1 for r in ok); l = sum(r["score_A"] == 0 for r in ok)
        mean = lambda k: sum(r[k] for r in ok) / max(1, len(ok))
        print(f"{exp:4s} {A:18s} {B:16s} {len(ok):5d} {len(rs) - len(ok):3d} {m:8.3f} [{m - h:.3f}, {m + h:.3f}] "
              f"{f'{w}-{l}-{len(ok) - w - l}':>9s} {mean('turns'):5.0f} {mean('ms_per_move_A'):7.1f} "
              f"{mean('ms_per_move_B'):7.1f} {mean('wall_s'):6.1f}")
    if DEC.exists() and DEC.stat().st_size:
        ds = [json.loads(l) for l in DEC.read_text().splitlines()]
        loss = [(d["v_AF"] - d["v_blend"]) / max(d["payoff_range"], 1e-9) for d in ds]
        ratio = [(d["v_AF"] - d["v_blend"]) / (4 * d["delta_H"]) for d in ds if d["delta_H"] > 1e-9]
        loss.sort()
        print(f"E5 decisions: {len(ds)}, frac loss>1e-6 {sum(x > 1e-6 for x in loss) / len(loss):.2f}, "
              f"median normalised loss {loss[len(loss) // 2]:.4f}, max {loss[-1]:.4f}, "
              f"max loss/(4 delta_H) {max(ratio, default=0):.3f}, "
              f"argmax differs {sum(not d['argmax_same'] for d in ds) / len(ds):.2f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("exp", choices=[*MATRIX, "summary"])
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--pairs", type=int, default=None)
    ap.add_argument("--log-decisions", action="store_true")
    ap.add_argument("--only", type=str, default=None, help="comma-separated matchup indices (seeds unchanged)")
    a = ap.parse_args()
    summary() if a.exp == "summary" else run(a.exp, a.workers, a.pairs, a.log_decisions,
                                                    {int(i) for i in a.only.split(",")} if a.only else None)
