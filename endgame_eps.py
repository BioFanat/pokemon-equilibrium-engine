"""Leaf error, exact exploitability and the theory bound on small exactly-solvable endgames.

For each endgame (solve_endgame.py's exact graph + Shapley value V*), with all values on the
[-1, 1] scale (side-one perspective, Q*_s[i,j] = sum p (2V*(s') - 1)):

  * eps_v(s)  leaf error of the agent's stage matrix M_s = nash.full_matrix(s) vs Q*_s:
              raw max|M - Q*|; per-state calibrated min_{a>=0,c} max|aM + c - Q*| (LP);
              one global (a, c) per endgame (LP over all cells).
  * expl      exact exploitability of a per-turn agent x_s (built from M_s) as side one / two:
              opponent best response W(s) = min_j sum_i x_i sum_b p W(s') by value iteration,
              expl = V*(s0) - W(s0) (seat two by symmetry).
  * H         expected turns to a terminal under (agent, best response), ties broken toward
              longer games; eps_s(s) = val(M~_s) - min_j x M~_s[:, j] on the calibrated matrix
              M~ = aM + c; bound = H (max eps_s + 2 max eps_v) and the refined bound
              E[sum_t eps_s(s_t) + 2 eps_v(s_t)] along the (agent, best response) chain.

Run: .venv/bin/python endgame_eps.py [cap [case indices to append]]  ->  results/endgame/eps.jsonl
"""
from __future__ import annotations

import json
import os
import sys
import time
import warnings
from multiprocessing import Pool

import numpy as np
import poke_engine as pe
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.optimize import linprog

import nash
import pkbattle as pb
import solve_endgame as se

AGENTS = ["equilibrium sampled", "equilibrium top move", "safest", "exact equilibrium"]
TIE = 1e-6          # best-response tie tolerance on the [-1, 1] scale
WORKERS = 6


# ---------------------------------------------------------------- graph

def enumerate_with_states(start: pe.State, max_states: int):
    """solve_endgame.enumerate_states plus rep[key] = string of the first state seen with that key
    (needed to rebuild states when PP/HP abstraction merges positions)."""
    key0 = se.state_key(start)
    graph, rep, frontier, objs = {}, {}, [key0], {key0: start}
    while frontier:
        key = frontier.pop()
        if key in graph:
            continue
        st = objs.pop(key)
        tv = se.terminal(st)
        if tv is not None:
            graph[key] = tv
            continue
        o1, o2 = pb.options(st)
        base = st.to_string()
        rep[key] = base
        trans = {}
        for i, m1 in enumerate(o1):
            for j, m2 in enumerate(o2):
                outs = []
                for br in pe.generate_instructions(st, m1, m2):
                    nxt = pe.State.from_string(base).apply_instructions(br)
                    k = se.state_key(nxt)
                    outs.append((br.percentage / 100.0, k))
                    if k not in graph and k not in objs:
                        objs[k] = nxt
                        frontier.append(k)
                trans[(i, j)] = outs
        graph[key] = (len(o1), len(o2), trans)
        if len(graph) > max_states:
            return graph, rep, False
    return graph, rep, True


class Game:
    """Indexed graph: states 0..N-1 (live first), cell rows (s, i, j) in a sparse transition matrix."""

    def __init__(self, graph):
        self.live = [k for k, v in graph.items() if isinstance(v, tuple)]
        self.term = [k for k, v in graph.items() if not isinstance(v, tuple)]
        self.keys = self.live + self.term
        self.idx = {k: n for n, k in enumerate(self.keys)}
        self.nl = len(self.live)
        self.tval = np.array([2 * graph[k] - 1 for k in self.term], dtype=float)
        self.shape = [graph[k][:2] for k in self.live]
        self.off = np.r_[0, np.cumsum([a * b for a, b in self.shape])].astype(int)
        r, c, d = [], [], []
        for s, k in enumerate(self.live):
            n1, n2, trans = graph[k]
            for (i, j), outs in trans.items():
                for p, k2 in outs:
                    r.append(self.off[s] + i * n2 + j); c.append(self.idx[k2]); d.append(p)
        self.P = sp.csr_matrix((d, (r, c)), shape=(self.off[-1], len(self.keys)))

    def full(self, v_live):
        return np.r_[v_live, self.tval]


def game_value(Q: np.ndarray) -> float:
    lo, hi = Q.min(1).max(), Q.max(0).min()
    return float(lo) if hi - lo < 1e-12 else se.matrix_game_value(Q)


def shapley(g: Game, tol=1e-9, max_iter=5000):
    """Jacobi Shapley iteration on the [-1, 1] scale, init 0 (= 0.5 in solve_endgame's units).
    Pure-saddle stage games are valued in closed form; mixed ones by LP, skipped if Q is unchanged."""
    V = g.full(np.zeros(g.nl))
    lastQ, lastv = [None] * g.nl, np.zeros(g.nl)
    for it in range(max_iter):
        allQ = g.P @ V
        new = np.empty(g.nl)
        for s in range(g.nl):
            Q = allQ[g.off[s]:g.off[s + 1]].reshape(g.shape[s])
            if lastQ[s] is not None and np.abs(Q - lastQ[s]).max() < 1e-13:
                new[s] = lastv[s]; continue
            new[s] = game_value(Q); lastQ[s] = Q; lastv[s] = new[s]
        delta = float(np.abs(new - V[:g.nl]).max())
        V[:g.nl] = new
        if delta < tol:
            return V, it + 1, delta
    return V, max_iter, delta


# ---------------------------------------------------------------- per-state work (pool)

def calibrate(M, Q):
    """min_{a>=0, c, t} t  s.t. |a M + c - Q| <= t  (vars a, c, t)."""
    m, q = M.ravel(), Q.ravel()
    one = np.ones_like(m)
    A = np.r_[np.c_[m, one, -one], np.c_[-m, -one, -one]]
    r = linprog([0, 0, 1], A_ub=A, b_ub=np.r_[q, -q], bounds=[(0, None), (None, None), (0, None)],
                method="highs")
    a, c, t = r.x
    return a, c, float(np.abs(a * M + c - Q).max())


def eq_strategy(G):
    """Exact maximin strategy of G (row player): pure if G has a saddle, else LP."""
    lo, hi = G.min(1).max(), G.max(0).min()
    if hi - lo < 1e-12:
        x = np.zeros(G.shape[0]); x[int(G.min(1).argmax())] = 1
        return x
    x = np.clip(nash.lp(G)[0], 0, None)
    return x / x.sum()


def agent_strategies(G):
    """Row-player strategies of the three matrix agents on their own (estimated) matrix G."""
    x = np.clip(nash.rm_plus(G, 1000)[0], 0, None); x /= x.sum()
    top = np.zeros_like(x); top[int(x.argmax())] = 1
    safe = np.zeros_like(x); safe[int(G.min(1).argmax())] = 1
    return {"equilibrium sampled": x, "equilibrium top move": top, "safest": safe}


def stage_gap(G, x):
    """One-sided suboptimality of row strategy x in G: val(G) - min_j (x G)_j."""
    return max(0.0, game_value(G) - float((x @ G).min()))


def per_state(args):
    s, s_str, Qs = args
    st = pe.State.from_string(s_str)
    M, s1, s2 = nash.full_matrix(st)
    o1, o2 = pb.options(st)
    assert s1 == o1 and s2 == o2, "full_matrix option order != pb.options order"
    assert M.shape == Qs.shape, (M.shape, Qs.shape)
    a, c, ev_cal = calibrate(M, Qs)
    Mc = a * M + c
    out = {"s": s, "M": M, "a": a, "c": c, "ev_raw": float(np.abs(M - Qs).max()), "ev_cal": ev_cal,
           "strat": {}, "eps_s": {}}
    for seat, G, Gc, Gq in [(1, M, Mc, Qs), (2, -M.T, -Mc.T, -Qs.T)]:
        strat = agent_strategies(G)
        for name, x in strat.items():
            out["eps_s"][(seat, name)] = stage_gap(Gc, x)
        xe = eq_strategy(Gq)
        strat["exact equilibrium"] = xe
        out["eps_s"][(seat, "exact equilibrium")] = stage_gap(Gq, xe)
        out["strat"][seat] = strat
    return out


# ---------------------------------------------------------------- best response / horizon

def agent_matrix(g: Game, seat: int, strat):
    """Sparse A with rows = (state, opponent action) and A @ U = expected next value given the agent's
    mix; plus row offsets per state. Seat 2 rows are side one's actions."""
    blocks, roff = [], [0]
    for s in range(g.nl):
        n1, n2 = g.shape[s]
        Ps = g.P[g.off[s]:g.off[s + 1]]
        x = strat[s]
        if seat == 1:   # cell (i, j) row i*n2+j; opponent picks j: weight x_i
            W = sp.kron(x[None, :], sp.identity(n2), format="csr")
        else:           # opponent picks i: weight y_j
            W = sp.kron(sp.identity(n1), x[None, :], format="csr")
        blocks.append(W @ Ps)
        roff.append(roff[-1] + W.shape[0])
    return sp.vstack(blocks, format="csr"), np.array(roff)


def best_response(g: Game, A, roff, tval, tol=1e-8, max_iter=20000):
    """Value iteration from 0 (infinite play = draw); returns (U, sweeps, final max change)."""
    U = np.r_[np.zeros(g.nl), tval]
    for it in range(max_iter):
        new = np.minimum.reduceat(A @ U, roff[:-1])
        d = np.abs(new - U[:g.nl]).max()
        U[:g.nl] = new
        if d < tol:
            break
    return U, it + 1, float(d)


def horizon_policy(g: Game, A, roff, U, longest: bool, max_iter=3000):
    """Best-response tie-break: among rows within TIE of the min, pick the one with the longest
    (longest=True) or shortest expected game, by value iteration on T = 1 + ext_tied(A T) from 0.
    The returned policy is then evaluated exactly (expected_sum), which decides finiteness."""
    E = A @ U
    mins = np.minimum.reduceat(E, roff[:-1])
    tied = E <= np.repeat(mins, np.diff(roff)) + TIE
    T = np.zeros(A.shape[1])
    pad = -np.inf if longest else np.inf
    red = np.maximum.reduceat if longest else np.minimum.reduceat
    for it in range(max_iter):
        new = 1 + red(np.where(tied, A @ T, pad), roff[:-1])
        d = np.abs(new - T[:g.nl]).max()
        T[:g.nl] = new
        if d < 1e-9:
            break
    AT = np.where(tied, A @ T, pad)
    pick = np.argmax if longest else np.argmin
    return np.array([roff[s] + int(pick(AT[roff[s]:roff[s + 1]])) for s in range(g.nl)])


def times(H, e):
    """H * e with an infinite horizon giving an infinite (vacuous) bound, even when e = 0."""
    return H * e if np.isfinite(H) else float("inf")


def chain(g: Game, A, pol):
    """Absorbing chain of (agent, chosen best response) on live states: B = A[pol, :nl]."""
    return A[pol][:, :g.nl].tocsc()


def expected_sum(B, r, s0, reach):
    """E[sum_t r(s_t)] from s0, solving (I - B) x = r on the states reachable from s0 (unreachable
    closed loops would make the full system singular); inf if a reachable loop never exits."""
    Br = B.tocsr()[reach][:, reach].tocsc()
    rr, n = np.asarray(r, dtype=float)[reach], len(reach)
    i0 = int(np.searchsorted(reach, s0))
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", spla.MatrixRankWarning)
            x = np.atleast_1d(spla.spsolve(sp.identity(n, format="csc") - Br, rr))
    except (RuntimeError, np.linalg.LinAlgError):
        return float("inf")
    if not np.all(np.isfinite(x)) or x[i0] < 0 or x[i0] > 1e8:
        return float("inf")
    res = np.abs((sp.identity(n) - Br) @ x - rr).max()
    return float(x[i0]) if res < 1e-6 * max(1.0, np.abs(x).max()) else float("inf")


def reachable(B, s0):
    seen, stack, Bc = {s0}, [s0], B.tocsr()
    while stack:
        s = stack.pop()
        row = Bc[s]
        for t in row.indices[row.data > 0]:
            if t not in seen:
                seen.add(t); stack.append(t)
    return np.array(sorted(seen))


# ---------------------------------------------------------------- driver

# (side one, side two, hp fraction, abstract PP, HP buckets). No case fits 20k states exactly:
# without abstraction even garchomp/rotomwash at 8% HP and kingambit/clefable at 10% exceed it (PP
# combinatorics), so every case runs with PP abstracted plus HP buckets (finest that fit).
CASES = [
    (["garchomp"], ["rotomwash"], 1.0, True, 20),                  # 10,878 states
    (["garchomp"], ["rotomwash"], 0.4, True, 20),                  # 7,268
    (["greattusk"], ["gholdengo"], 1.0, True, 10),                 # 1,836
    (["greattusk"], ["gholdengo"], 0.5, True, 10),
    (["gholdengo"], ["heatran"], 1.0, True, 10),                   # 4,555
    (["greattusk"], ["rotomwash"], 1.0, True, 10),                 # 19,592
]
# Known to exceed the cap (probed, not re-enumerated in the run to save CPU).
SKIPPED = [
    ("kingambit vs clefable @ 100%", "exceeds 20k exact, with PP abstraction, and with PP + HP/10"),
    ("garchomp+toxapex vs rotomwash+kingambit @ 40%", "exceeds 20k exact, with PP, and with PP + HP/10"),
    ("garchomp+toxapex vs rotomwash+kingambit @ 100%", "not attempted: the 40% variant already exceeds with PP + HP/10"),
]


def stats(v, idx):
    v = np.asarray(v)[idx]
    return float(v.max()), float(v.mean())


def run_case(n1, n2, hpf, pp, hb, cap, pool, out):
    name = f"{'+'.join(n1)} vs {'+'.join(n2)} @ {hpf:.0%}" + (" [PP*]" if pp else "") + (f" [HP/{hb}]" if hb else "")
    se.ABSTRACT_PP, se.HP_BUCKETS = pp, hb
    t0 = time.perf_counter()
    start = se.endgame(n1, n2, hpf)
    graph, rep, complete = enumerate_with_states(start, cap)
    te = time.perf_counter() - t0
    if not complete:
        print(f"{name}: exceeded cap {cap:,} ({te:.0f}s) -> skipped", flush=True)
        out.write(json.dumps({"case": name, "skipped": f"exceeded cap {cap}"}) + "\n"); out.flush()
        return
    g = Game(graph)
    quirk = 0          # rebuild check: the stored string maps back to its key. poke_engine's
    for k in g.live:   # from_string adds a spurious NONE volatile next to PARTIALLYTRAPPED: tolerated, counted
        k2 = se.state_key(pe.State.from_string(rep[k]))
        if k2 != k:
            assert k2.replace("=NONE:", "=") == k, k
            quirk += 1
        elif not (pp or hb):
            assert pe.State.from_string(k).to_string() == k
    s0 = g.idx[se.state_key(start)]
    t1 = time.perf_counter()
    V, sweeps, sdelta = shapley(g)
    allQ = g.P @ V
    Qs = [allQ[g.off[s]:g.off[s + 1]].reshape(g.shape[s]) for s in range(g.nl)]
    ts = time.perf_counter() - t1
    t2 = time.perf_counter()
    res = [None] * g.nl
    for r in pool.imap_unordered(per_state, [(s, rep[k], Qs[s]) for s, k in enumerate(g.live)], chunksize=32):
        res[r["s"]] = r
    tp = time.perf_counter() - t2
    # global calibration
    Mall = np.concatenate([r["M"].ravel() for r in res]); Qall = np.concatenate([q.ravel() for q in Qs])
    ag, cg, _ = calibrate(Mall, Qall)
    ev_glob = np.array([np.abs(ag * r["M"] + cg - Qs[s]).max() for s, r in enumerate(res)])
    ev_raw = np.array([r["ev_raw"] for r in res]); ev_cal = np.array([r["ev_cal"] for r in res])
    a0 = int(sum(r["a"] < 1e-9 for r in res))
    print(f"{name}: {len(graph):,} states ({g.nl:,} live), V*(s0)={V[s0]:+.4f} (P1 win {(V[s0]+1)/2:.4f}), "
          f"enum {te:.0f}s, shapley {sweeps} sweeps {ts:.0f}s, matrices {tp:.0f}s, a=0 at {a0} states", flush=True)
    for seat in (1, 2):
        tval = g.tval if seat == 1 else -g.tval
        vstar = V[s0] if seat == 1 else -V[s0]
        for agent in AGENTS:
            strat = [r["strat"][seat][agent] for r in res]
            eps_s = np.array([r["eps_s"][(seat, agent)] for r in res])
            evc = np.zeros(g.nl) if agent == "exact equilibrium" else ev_cal
            A, roff = agent_matrix(g, seat, strat)
            U, it, br_delta = best_response(g, A, roff, tval)
            expl = float(vstar - U[s0])
            loss = eps_s + 2 * evc
            allidx = np.arange(g.nl)
            row = {"case": name, "agent": agent, "seat": seat, "n_states": len(graph), "n_live": g.nl,
                   "abstract_pp": pp, "hp_buckets": hb, "roundtrip_quirk_states": quirk,
                   "V*": float(vstar), "W": float(U[s0]), "expl": expl, "br_sweeps": it, "br_delta": br_delta,
                   "shapley_sweeps": sweeps, "shapley_delta": sdelta,
                   "eps_v_raw": stats(ev_raw, allidx), "eps_v_cal": stats(evc, allidx),
                   "eps_v_glob": stats(ev_glob, allidx), "global_a_c": [float(ag), float(cg)],
                   "a_zero_states": a0, "eps_s_all": stats(eps_s, allidx),
                   "bound_allmax": None}
            # tie-break toward longer games = primary fields (spec); shorter games = *_short fields
            for longest, sfx in [(True, ""), (False, "_short")]:
                B = chain(g, A, horizon_policy(g, A, roff, U, longest))
                reach = reachable(B, s0)
                H = expected_sum(B, np.ones(g.nl), s0, reach)
                es_r = stats(eps_s, reach)
                row.update({f"H{sfx}": H, f"h_finite{sfx}": bool(np.isfinite(H)), f"n_reach{sfx}": int(len(reach)),
                            f"eps_v_raw_reach{sfx}": stats(ev_raw, reach), f"eps_v_cal_reach{sfx}": stats(evc, reach),
                            f"eps_v_glob_reach{sfx}": stats(ev_glob, reach), f"eps_s_reach{sfx}": es_r,
                            f"bound{sfx}": times(H, es_r[0] + 2 * float(evc[reach].max())),
                            f"refined_bound{sfx}": expected_sum(B, loss, s0, reach) if np.isfinite(H) else float("inf")})
                if longest:
                    row["bound_allmax"] = times(H, row["eps_s_all"][0] + 2 * float(evc.max()))
            row["ok"] = bool(all(row[k] >= expl - 1e-6 for k in
                                 ["bound", "refined_bound", "bound_short", "refined_bound_short"])
                             and row["bound"] >= row["eps_s_reach"][0] - 1e-9 and br_delta < 1e-8)
            if agent == "exact equilibrium":
                row["ok"] = row["ok"] and abs(expl) < 1e-4
            out.write(json.dumps(row) + "\n"); out.flush()
            r_ = lambda k: row[k] / expl if expl > 1e-9 else float("inf")
            print(f"  seat{seat} {agent:21s} expl {expl:7.4f} | long: H {row['H']:7.2f} bound {row['bound']:7.3f} "
                  f"refined {row['refined_bound']:7.3f} | short: H {row['H_short']:7.2f} "
                  f"eps_v cal max/mean {row['eps_v_cal_reach_short'][0]:.3f}/{row['eps_v_cal_reach_short'][1]:.3f} "
                  f"eps_s max {row['eps_s_reach_short'][0]:.3f} bound {row['bound_short']:7.3f} "
                  f"refined {row['refined_bound_short']:7.3f} ratio {r_('refined_bound_short'):7.1f}"
                  + ("" if row["ok"] else "   <-- CHECK FAILED"), flush=True)


def main():
    cap = int(sys.argv[1]) if len(sys.argv) > 1 else 20000
    only = [int(i) for i in sys.argv[2].split(",")] if len(sys.argv) > 2 else None   # append these CASES
    os.makedirs("results/endgame", exist_ok=True)
    t0 = time.perf_counter()
    with Pool(WORKERS) as pool, open("results/endgame/eps.jsonl", "a" if only else "w") as out:
        for name, why in ([] if only else SKIPPED):
            print(f"{name}: skipped ({why})", flush=True)
            out.write(json.dumps({"case": name, "skipped": why}) + "\n")
        for case in (CASES if only is None else [CASES[i] for i in only]):
            run_case(*case, cap, pool, out)
    print(f"total wall time {time.perf_counter() - t0:.0f}s")


if __name__ == "__main__":
    main()
