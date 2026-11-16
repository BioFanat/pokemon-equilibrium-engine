"""Per-decision eta-hat in the full engine (6v6 Random Battles, hidden information), no new leaf.

At every decision of an `agentform:K` agent the K per-world depth-2 matrices M_k are already built.
  eta_hat_set  = mean_k val(M_k) - v_AF(M_1..K, uniform)        (strategy-fusion value over full-set worlds)
  eta_hat_tera = mean_th val(M_th) - v_AF(M_th, uniform)        (same, worlds = MAP sets with only the active
                                                                 opponent's tera type varied over its role's
                                                                 teraTypes; skipped if <2 types or already tera'd)
Both are >= 0 by construction (Jensen) and lower-bound eta(t-1, s_t, b_t) of Theorem C with K sampled worlds.
Normalised by that decision's payoff range. Also writes sum_t eta_hat per game.

Usage: eta_hat_engine.py --games 12 --workers 4 --out eta_hat.jsonl
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from dataclasses import replace

import numpy as np

ENGINE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "engine")
sys.path.insert(0, os.path.abspath(ENGINE))

import hidden as hd  # noqa: E402
import hidden_agents as ha  # noqa: E402
import nash  # noqa: E402
import randbats as rb  # noqa: E402


def stable_engine_matrix(st):
    """engine_matrix twice until two consecutive evaluations agree (pe.id(child,0) is load-sensitive)."""
    prev = nash.engine_matrix(st, 0)
    for _ in range(6):
        cur = nash.engine_matrix(st, 0)
        if cur[1:] == prev[1:] and np.array_equal(cur[0], prev[0]):
            return cur
        STABLE_RETRIES[0] += 1
        prev = cur
    raise RuntimeError("matrix did not stabilise")


STABLE_RETRIES = [0]


class EtaAgent(ha.Agent):
    def __init__(self, spec):
        super().__init__(spec)
        self.records, self.log_s = [], 0.0

    def _matrices(self, true_state, view, belief, legal, side, rng):
        states = [hd.determinize(true_state, view, belief, rng) for _ in range(self.K)]
        return [ha.align(*ha.own_matrix(*stable_engine_matrix(s), side)[:2], legal) for s in states]

    def strategy(self, true_state, view, belief, legal, side, rng):
        if self.kind == "random" or len(legal) == 1:
            return np.full(len(legal), 1 / len(legal))
        t0 = time.perf_counter()
        Ms = self._matrices(true_state, view, belief, legal, side, rng)
        b = np.full(len(Ms), 1 / len(Ms))
        x_af, v_af = ha.agent_form_lp(Ms, b)
        t1 = time.perf_counter()
        vals = [ha.lp_x(M)[1] for M in Ms]
        allv = np.concatenate([M.ravel() for M in Ms])
        rec = dict(side=side, K=len(Ms), v_AF=v_af, retries=STABLE_RETRIES[0], v_worlds=vals, eta_hat_set=float(np.mean(vals) - v_af),
                   payoff_range=float(allv.max() - allv.min()), n_legal=len(legal))
        # tera-only worlds at the same position
        opp_act = int(view.opp_fields["active_index"])
        pub = view.opp_public[opp_act]
        sr = view.reveals[opp_act]
        if not pub["terastallized"]:
            s = belief.map_set(opp_act)
            teras = rb.sets()[s.species]["roles"][s.role]["teraTypes"]
            if len(teras) >= 2:
                fixed = {i: belief.map_set(i) for i in view.reveals}
                fixed.update(belief.sample_unrevealed(random.Random(0)))
                Mt = []
                for th in teras:
                    fb = hd.FixedBelief(view, {**fixed, opp_act: replace(s, tera=th)})
                    st = hd.determinize(true_state, view, fb, rng)
                    Mt.append(ha.align(*ha.own_matrix(*stable_engine_matrix(st), side)[:2], legal))
                vt = [ha.lp_x(M)[1] for M in Mt]
                _, v_af_t = ha.agent_form_lp(Mt, np.full(len(Mt), 1 / len(Mt)))
                allt = np.concatenate([M.ravel() for M in Mt])
                rec.update(k_tera=len(teras), teras=list(teras), v_AF_tera=v_af_t, v_worlds_tera=vt,
                           eta_hat_tera=float(np.mean(vt) - v_af_t), payoff_range_tera=float(allt.max() - allt.min()),
                           opp_species=s.species)
        self.records.append(rec)
        self.log_s += time.perf_counter() - t1
        return x_af


def one_game(args):
    seed, spec = args
    from hidden_experiments import play_hidden
    rng = random.Random(seed)
    a, b = EtaAgent(spec), EtaAgent(spec)
    t = time.perf_counter()
    w, turns, _ = play_hidden(a, b, rng)
    recs = [dict(seed=seed, agent=1, **r) for r in a.records] + [dict(seed=seed, agent=2, **r) for r in b.records]
    return dict(seed=seed, winner=w, turns=turns, wall_s=time.perf_counter() - t, decisions=recs)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=8)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--spec", default="agentform:4:200")
    ap.add_argument("--out", default="eta_hat.jsonl")
    a = ap.parse_args()
    from multiprocessing import get_context
    with get_context("spawn").Pool(a.workers) as pool, open(a.out, "a") as f:
        for g in pool.imap_unordered(one_game, [(10_007 * (i + 1), a.spec) for i in range(a.games)]):
            f.write(json.dumps(g) + "\n"); f.flush()
            d = g["decisions"]
            es = [r["eta_hat_set"] / r["payoff_range"] for r in d if r["payoff_range"] > 0]
            et = [r["eta_hat_tera"] / r["payoff_range_tera"] for r in d if "eta_hat_tera" in r and r["payoff_range_tera"] > 0]
            print(f"seed {g['seed']} turns {g['turns']} dec {len(d)} eta_set/range mean {np.mean(es):.4f} max {np.max(es):.4f} | "
                  f"tera n={len(et)} mean {np.mean(et) if et else float('nan'):.4f} max {np.max(et) if et else float('nan'):.4f} | {g['wall_s']:.0f}s", flush=True)
