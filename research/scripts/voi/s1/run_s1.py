"""E1 / E2 / E4 evaluation of S1 (research/info_gathering_concealment.md §4), per decision, engine simulation.

Games: hidden_experiments.play_pair (paired seeds, both sides swapped), K = 4, OURBOT_DET_LEAF=1. Both seats play
today's agent A0 ('agentform:4', sampled); the A-seat is wrapped (S1LogAgent, installed at runtime in the worker by
replacing hidden_agents.Agent with a factory — no project file is edited) so that at every decision with > 1 legal
option it ALSO runs info_planner.plan on the SAME K worlds A0 used, and logs A0, A1, A2 (cell formulas 'rep' =
report §3.2, primary, and 'pw' = per-world) and A0L (clairvoyant children, LP leaf). A0's action is played, so the
rng stream and the game are exactly A0 self-play; in a pair the a_side=2 game replays the a_side=1 game (identical
agents and seed), so a pair yields the logged decisions of both sides of one game. Analysis time is excluded from
the play clocks (log_s).

Usage: OURBOT_DET_LEAF=1 python run_s1.py --pairs 30 --workers 2 --minutes 85 [--out out/s1_eval.jsonl]
After the deadline, running games finish with A0 only (no further analysis) and queued pairs are skipped.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ENGINE = os.path.abspath(os.path.join(HERE, "..", "..", "..", "..", "engine"))
sys.path.insert(0, ENGINE)
os.environ.setdefault("OURBOT_DET_LEAF", "1")

SPEC_A = "s1log:agentform:4:200"
SPEC_B = "agentform:4:200"
SEED0 = 5_000_011


def _install(deadline):
    import numpy as np
    import hidden as hd
    import hidden_agents as ha
    import info_planner as ip
    import nash

    Orig = ha.Agent

    class S1LogAgent(Orig):
        def __init__(self, spec, log=False):
            super().__init__(spec.split(":", 1)[1], False)
            self.spec = spec

        def strategy(self, true_state, view, belief, legal, side, rng):
            if len(legal) == 1:
                return np.full(1, 1.0)
            worlds = [hd.determinize(true_state, view, belief, rng) for _ in range(self.K)]
            c0, t0 = time.process_time(), time.perf_counter()
            Ms = [ha.align(*ha.own_matrix(*nash.engine_matrix(s, max(1, self.ms // self.K)), side)[:2], legal)
                  for s in worlds]
            w = np.full(self.K, 1 / self.K)
            x0, v0 = ha.agent_form_lp(Ms, w)
            a0_cpu, a0_wall = time.process_time() - c0, time.perf_counter() - t0
            if time.time() > deadline:
                return x0
            c1, t1 = time.process_time(), time.perf_counter()
            leaf0 = dict(nash.LEAF_STATS)
            try:
                res = ip.plan(worlds, side, legal, set(view.reveals), w=w, cache={})
                err = None
            except Exception:
                import traceback
                res, err = None, traceback.format_exc()[-800:]
            s1_cpu, s1_wall = time.process_time() - c1, time.perf_counter() - t1
            self.log_s += s1_wall
            allv = np.concatenate([M.ravel() for M in Ms])
            rec = dict(side=side, legal=list(legal), K=self.K, payoff_range=float(allv.max() - allv.min()),
                       A0=dict(x=x0.tolist(), v=float(v0), V=(sum(wk * M.min(1) for wk, M in zip(w, Ms))).tolist(),
                               cpu=a0_cpu, wall=a0_wall),
                       s1_cpu=s1_cpu, s1_wall=s1_wall, error=err,
                       leaf=dict(calls=nash.LEAF_STATS["calls"] - leaf0["calls"],
                                 non_d1=nash.LEAF_STATS["non_d1"] - leaf0["non_d1"],
                                 fallback=nash.LEAF_STATS["fallback"] - leaf0["fallback"]),
                       own_tera_available=not any(p.terastallized for p in hd._sides(true_state, side)[0].pokemon))
            if res is not None:
                for k in ("A1_rep", "A2_rep", "A1_pw", "A2_pw", "A0L"):
                    r = res[k]
                    rec[k] = dict(x=r["x"].tolist(), v=r["v"], V=r["V"].tolist(),
                                  range=float(np.ptp(np.concatenate([R.ravel() for R in r["R"]]))))
                g = res["groups"]
                multi = [x for x in g if len(x["worlds"]) > 1]
                rec.update(expand_cpu=res["ex"].secs_cpu, d1_calls=res["ex"].stats["d1_calls"],
                           lp_secs=res["lp_secs"], root_secs=res["root_secs"],
                           keys=ip.key_stats(res["ex"].nodes, self.K), keys_report=ip.key_stats(res["ex"].nodes, self.K, "report"),
                           n_groups_multi=len(multi), groups_dV_pos=sum(x["V1"] - x["V2"] > 1e-9 * max(1.0, abs(x["V1"])) for x in multi),
                           groups_aligned=sum(x["aligned"] > 0 for x in g))
                for k in ("keys", "keys_report"):           # keep distributions compact
                    for f in ("keys_per_cell_switch", "keys_per_cell_move"):
                        v = rec[k][f]
                        rec[k][f] = dict(n=len(v), hist={str(h): v.count(h) for h in sorted(set(v))})
            self.records.append(rec)
            return x0

    def factory(spec, log=False):
        return S1LogAgent(spec, log) if spec.startswith("s1log:") else Orig(spec, log)

    ha.Agent = factory


DEADLINE = [float("inf")]


def _init(deadline):
    DEADLINE[0] = deadline
    _install(deadline)


def _task(task):
    import hidden_experiments as he
    if time.time() > DEADLINE[0]:
        return dict(pair=task[3], seed=task[4], skipped=True, wall_s=0.0, games=[], decisions=[])
    t = time.perf_counter()
    recs, decs = he.play_pair(task)
    return dict(pair=task[3], seed=task[4], wall_s=time.perf_counter() - t, games=recs, decisions=decs)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", type=int, default=30)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--minutes", type=float, default=85)
    ap.add_argument("--first", type=int, default=0)
    ap.add_argument("--out", default=os.path.join(HERE, "out", "s1_eval.jsonl"))
    a = ap.parse_args()
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    t_end = time.time() + 60 * a.minutes
    from multiprocessing import get_context
    tasks = [("S1", SPEC_A, SPEC_B, k, SEED0 + 7919 * k, False) for k in range(a.first, a.first + a.pairs)]
    print(f"{len(tasks)} pairs, {a.workers} workers, deadline in {a.minutes} min -> {a.out}", flush=True)
    t0 = time.perf_counter()
    n_dec = 0
    with get_context("spawn").Pool(a.workers, initializer=_init, initargs=(t_end,)) as pool, open(a.out, "a") as f:
        pending = list(tasks)
        it = pool.imap_unordered(_task, pending)
        for g in it:
            if g.get("skipped"):
                continue
            f.write(json.dumps(g) + "\n"); f.flush()
            d = [x for x in g["decisions"] if "A1_rep" in x]
            n_dec += len(d)
            print(f"pair {g['pair']} seed {g['seed']} decisions {len(d)} (total {n_dec}) | {g['wall_s']:.0f}s pair, "
                  f"{time.perf_counter() - t0:.0f}s elapsed", flush=True)
