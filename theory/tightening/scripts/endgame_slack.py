"""Where Theorem A's slack lives on the exact (abstracted) endgames.

Reuses engine/endgame_eps.py (enumeration, Shapley, per-state matrices, best response, chain).
For the agent "equilibrium sampled" (and "equilibrium top move"), seats 1 and 2, along the exact
best-response chain (longest tie-break = endgame_eps's primary fields), it computes chain-weighted sums
E[sum_t r(s_t)] of several per-state quantities on the [-1, 1] payoff scale:

  ell   = V*(s) - x^T Q* e_{j_BR}            true per-stage loss (L1(a) identity: sums to expl exactly)
  e     = val Q* - min_j (x^T Q*)_j          stage exploitability in Q* (L1(b) upper sandwich, >= ell)
  bud2  = eps_s + 2 eps_v^cal(s)             Theorem A per-state budget (= endgame_eps's refined_bound)
  bud3  = eps_s + x*^T (Q* - B) yhat + max_j (x^T (B - Q*))_j     profile-restricted Lemma 1
          (x* maximin in Q*, yhat minimax in B = aM + c, the per-state calibrated matrix)
  T1 = V* - val B,  T2 = val B - x^T B e_j,  T3 = x^T (B - Q*) e_j     exact split of ell (T1+T2+T3 = ell)
  evcal = eps_v^cal(s)                       for the visited-state mean vs max

Usage (cwd = engine, engine/.venv/bin/python):  python endgame_slack.py <case indices, e.g. 2,4,0> <out.json>
"""
from __future__ import annotations

import json
import os
import sys
import time
from multiprocessing import Pool

ENGINE = "/Users/biofanat/Documents/classes/f26/cs535/final/engine"
sys.path.insert(0, ENGINE)
os.chdir(ENGINE)

import numpy as np
import poke_engine as pe

import scipy.sparse as sp
import scipy.sparse.linalg as spla

import endgame_eps as ee
import solve_endgame as se


def signed_sum(B, r, s0, reach):
    """E[sum_t r(s_t)] along the absorbing chain B for a signed r (endgame_eps.expected_sum rejects
    negative totals); inf if the reachable chain never exits."""
    Br = B.tocsr()[reach][:, reach].tocsc()
    rr, n = np.asarray(r, dtype=float)[reach], len(reach)
    i0 = int(np.searchsorted(reach, s0))
    try:
        x = np.atleast_1d(spla.spsolve(sp.identity(n, format="csc") - Br, rr))
    except Exception:
        return float("inf")
    if not np.all(np.isfinite(x)):
        return float("inf")
    res = np.abs((sp.identity(n) - Br) @ x - rr).max()
    return float(x[i0]) if res < 1e-6 * max(1.0, np.abs(x).max()) else float("inf")

AGENTS = ["equilibrium sampled", "equilibrium top move"]


def per_state_safe(args):
    """endgame_eps.per_state, but tolerant of pe.id's (rare) unstable option order: retry until two
    consecutive option listings agree and permute M to that order by label. Flags the state."""
    import pkbattle as pb
    import nash
    s, s_str, Qs = args
    for attempt in range(6):
        try:
            return ee.per_state(args)
        except AssertionError:
            pass
    st = pe.State.from_string(s_str)
    M, s1, s2 = nash.full_matrix(st)
    o1, o2 = pb.options(st)
    if sorted(s1) == sorted(o1) and sorted(s2) == sorted(o2):
        M = M[[s1.index(a) for a in o1]][:, [s2.index(b) for b in o2]]
    out = _per_state_from_matrix(s, M, Qs)
    out["unstable"] = True
    return out


def _per_state_from_matrix(s, M, Qs):
    import numpy as np
    assert M.shape == Qs.shape, (M.shape, Qs.shape)
    a, c, ev_cal = ee.calibrate(M, Qs)
    Mc = a * M + c
    out = {"s": s, "M": M, "a": a, "c": c, "ev_raw": float(np.abs(M - Qs).max()), "ev_cal": ev_cal,
           "strat": {}, "eps_s": {}}
    for seat, G, Gc, Gq in [(1, M, Mc, Qs), (2, -M.T, -Mc.T, -Qs.T)]:
        strat = ee.agent_strategies(G)
        for name, x in strat.items():
            out["eps_s"][(seat, name)] = ee.stage_gap(Gc, x)
        xe = ee.eq_strategy(Gq)
        strat["exact equilibrium"] = xe
        out["eps_s"][(seat, "exact equilibrium")] = ee.stage_gap(Gq, xe)
        out["strat"][seat] = strat
    return out


def case_name(n1, n2, hpf, pp, hb):
    return f"{'+'.join(n1)} vs {'+'.join(n2)} @ {hpf:.0%}" + (" [PP*]" if pp else "") + (f" [HP/{hb}]" if hb else "")


def run(case, cap, pool):
    n1, n2, hpf, pp, hb = case
    name = case_name(*case)
    se.ABSTRACT_PP, se.HP_BUCKETS = pp, hb
    t0 = time.perf_counter()
    start = se.endgame(n1, n2, hpf)
    graph, rep, complete = ee.enumerate_with_states(start, cap)
    assert complete, name
    g = ee.Game(graph)
    s0 = g.idx[se.state_key(start)]
    V, sweeps, sdelta = ee.shapley(g)
    allQ = g.P @ V
    Qs = [allQ[g.off[s]:g.off[s + 1]].reshape(g.shape[s]) for s in range(g.nl)]
    res = [None] * g.nl
    for r in pool.imap_unordered(per_state_safe, [(s, rep[k], Qs[s]) for s, k in enumerate(g.live)], chunksize=32):
        res[r["s"]] = r
    ev_cal = np.array([r["ev_cal"] for r in res])
    n_unstable = sum(1 for r in res if r.get("unstable"))
    print(f"unstable option order at {n_unstable} states (M permuted by label)", flush=True)
    print(f"{name}: {len(graph):,} states ({g.nl:,} live), V*(s0)={V[s0]:+.4f}, shapley {sweeps} sweeps "
          f"delta {sdelta:.1e}, prep {time.perf_counter()-t0:.0f}s", flush=True)
    rows = []
    for seat in (1, 2):
        tval = g.tval if seat == 1 else -g.tval
        vstar = V[:g.nl] if seat == 1 else -V[:g.nl]
        for agent in AGENTS:
            strat = [r["strat"][seat][agent] for r in res]
            eps_s = np.array([r["eps_s"][(seat, agent)] for r in res])
            A, roff = ee.agent_matrix(g, seat, strat)
            U, it, br_delta = ee.best_response(g, A, roff, tval)
            expl = float(vstar[s0] - U[s0])
            pol = ee.horizon_policy(g, A, roff, U, True)
            B = ee.chain(g, A, pol)
            reach = ee.reachable(B, s0)
            H = ee.expected_sum(B, np.ones(g.nl), s0, reach)
            q = {k: np.zeros(g.nl) for k in ["ell", "e", "bud2", "bud3", "T1", "T2", "T3", "evcal", "pos_ell", "neg_ell"]}
            for s in reach:
                r = res[s]
                M, a, c = r["M"], r["a"], r["c"]
                if seat == 1:
                    Gq, Gc = Qs[s], a * M + c
                else:
                    Gq, Gc = -Qs[s].T, -(a * M + c).T
                x = strat[s]
                j = int(pol[s] - roff[s])
                xq, xc = x @ Gq, x @ Gc
                vq, vc = ee.game_value(Gq), ee.game_value(Gc)
                xstar = ee.eq_strategy(Gq)
                yhat = ee.eq_strategy(-Gc.T)        # minimax column strategy of Gc
                q["ell"][s] = vstar[s] - xq[j]
                q["e"][s] = vq - xq.min()
                q["bud2"][s] = eps_s[s] + 2 * ev_cal[s]
                q["bud3"][s] = eps_s[s] + float(xstar @ (Gq - Gc) @ yhat) + float((xc - xq).max())
                q["T1"][s] = vstar[s] - vc
                q["T2"][s] = vc - xc[j]
                q["T3"][s] = xc[j] - xq[j]
                q["evcal"][s] = ev_cal[s]
                q["pos_ell"][s] = max(q["ell"][s], 0.0)
                q["neg_ell"][s] = min(q["ell"][s], 0.0)
            sums = {k: signed_sum(B, v, s0, reach) for k, v in q.items()}
            row = {"case": name, "agent": agent, "seat": seat, "n_live": g.nl, "n_reach": int(len(reach)),
                   "V*": float(vstar[s0]), "expl": expl, "H": H, "br_delta": br_delta,
                   "sum": sums,
                   "evcal_reach_max": float(ev_cal[reach].max()), "evcal_reach_mean": float(ev_cal[reach].mean()),
                   "evcal_all_max": float(ev_cal.max()),
                   "eps_s_reach_max": float(eps_s[reach].max())}
            rows.append(row)
            print(f"  seat {seat} {agent:22s} expl {expl:+.4f} H {H:6.2f} | sum ell {sums['ell']:+.4f} "
                  f"(pos {sums['pos_ell']:.3f} neg {sums['neg_ell']:+.3f}) e {sums['e']:.4f} | bud2 {sums['bud2']:.3f} "
                  f"bud3 {sums['bud3']:.3f} | T1 {sums['T1']:+.3f} T2 {sums['T2']:+.3f} T3 {sums['T3']:+.3f} "
                  f"| evcal reach max {ev_cal[reach].max():.3f} mean {ev_cal[reach].mean():.3f}", flush=True)
    return rows


def main():
    idx = [int(i) for i in sys.argv[1].split(",")] if len(sys.argv) > 1 else [2, 4]
    out = sys.argv[2] if len(sys.argv) > 2 else "/dev/stdout"
    rows = []
    with Pool(ee.WORKERS) as pool:
        for i in idx:
            rows += run(ee.CASES[i], 20000, pool)
    with open(out, "w") as f:
        json.dump(rows, f, indent=1)


if __name__ == "__main__":
    main()
