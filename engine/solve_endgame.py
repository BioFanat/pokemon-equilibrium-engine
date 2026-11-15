"""Exact solve of small perfect-information endgames on poke-engine.

Enumerates every position reachable from a start state (all joint actions x all chance
outcomes), then runs Shapley value iteration: V(s) = value of the matrix game
Q_s[a, b] = sum_outcomes p * V(s'), with V = 1 (side one wins) / 0 (loses) at terminals.
Each matrix game is solved exactly by LP. This is the exact game value (win probability
for side one under optimal play by both sides), restricted to positions within `max_states`.
"""
from __future__ import annotations

import re
import sys
import time

import numpy as np
import poke_engine as pe
from scipy.optimize import linprog

import pkbattle as pb


def terminal(state: pe.State):
    a1, a2 = pb.alive(state.side_one), pb.alive(state.side_two)
    if a1 and a2:
        return None
    return 0.5 if a1 == a2 else float(a1 > 0)


ABSTRACT_PP = False
HP_BUCKETS = 0   # 0 = exact HP; k > 0 = merge positions whose HP falls in the same 1/k-of-max bucket


def state_key(st: pe.State) -> str:
    s = st.to_string()
    if ABSTRACT_PP:
        s = re.sub(r"(;(?:true|false);)\d+", r"\1*", s)
    if HP_BUCKETS:
        for side in (st.side_one, st.side_two):
            for p in side.pokemon:
                if p.maxhp > 0 and p.hp > 0:
                    b = min(HP_BUCKETS - 1, p.hp * HP_BUCKETS // p.maxhp)
                    s = s.replace(f",{p.hp},{p.maxhp},", f",B{b},{p.maxhp},", 1)
    return s


def enumerate_states(start: pe.State, max_states: int):
    """BFS over the reachable graph. Returns dict key -> (opts1, opts2, {(i,j): [(p, key')]}) or terminal value."""
    key0 = state_key(start)
    graph, frontier, objs = {}, [key0], {key0: start}
    while frontier:
        key = frontier.pop()
        if key in graph:
            continue
        st = objs.pop(key)
        tv = terminal(st)
        if tv is not None:
            graph[key] = tv
            continue
        o1, o2 = pb.options(st)
        base = st.to_string()
        trans = {}
        for i, m1 in enumerate(o1):
            for j, m2 in enumerate(o2):
                outs = []
                for br in pe.generate_instructions(st, m1, m2):
                    nxt = pe.State.from_string(base).apply_instructions(br)
                    k = state_key(nxt)
                    outs.append((br.percentage / 100.0, k))
                    if k not in graph and k not in objs:
                        objs[k] = nxt
                        frontier.append(k)
                trans[(i, j)] = outs
        graph[key] = (len(o1), len(o2), trans)
        if len(graph) > max_states:
            return graph, False
    return graph, True


def matrix_game_value(Q: np.ndarray) -> float:
    """max_x min_j (Q^T x)_j via LP."""
    m, n = Q.shape
    c = np.zeros(m + 1); c[-1] = -1
    A_ub = np.hstack([-Q.T, np.ones((n, 1))])
    A_eq = np.hstack([np.ones((1, m)), np.zeros((1, 1))])
    r = linprog(c, A_ub=A_ub, b_ub=np.zeros(n), A_eq=A_eq, b_eq=[1],
                bounds=[(0, None)] * m + [(None, None)], method="highs")
    return -r.fun


def shapley_value_iteration(graph, tol=1e-6, max_iter=500):
    V = {k: (v if not isinstance(v, tuple) else 0.5) for k, v in graph.items()}
    live = [k for k, v in graph.items() if isinstance(v, tuple)]
    for it in range(max_iter):
        delta = 0.0
        for k in live:
            n1, n2, trans = graph[k]
            Q = np.empty((n1, n2))
            for (i, j), outs in trans.items():
                Q[i, j] = sum(p * V[k2] for p, k2 in outs)
            v = matrix_game_value(Q)
            delta = max(delta, abs(v - V[k])); V[k] = v
        if delta < tol:
            return V, it + 1
    return V, max_iter


def endgame(names1, names2, hp_frac=1.0):
    def side(ns):
        mons = [pb.make_pokemon(n, hp_frac=hp_frac) for n in ns]
        return pe.Side(pokemon=mons + [pe.Pokemon.create_fainted() for _ in range(6 - len(mons))])
    return pe.State(side_one=side(names1), side_two=side(names2))


if __name__ == "__main__":
    cap = int(sys.argv[1]) if len(sys.argv) > 1 else 20000
    ABSTRACT_PP = len(sys.argv) > 2 and "pp" in sys.argv[2]
    HP_BUCKETS = int(sys.argv[3]) if len(sys.argv) > 3 else 0
    print("PP abstracted:", ABSTRACT_PP, "| HP buckets:", HP_BUCKETS or "exact")
    cases = [
        (["garchomp"], ["rotomwash"], 1.0),
        (["kingambit"], ["clefable"], 1.0),
        (["garchomp"], ["rotomwash"], 0.4),
        (["garchomp", "toxapex"], ["rotomwash", "kingambit"], 0.4),
        (["garchomp", "toxapex"], ["rotomwash", "kingambit"], 1.0),
    ]
    for n1, n2, hpf in cases:
        t = time.perf_counter()
        graph, complete = enumerate_states(endgame(n1, n2, hpf), cap)
        te = time.perf_counter() - t
        live = sum(isinstance(v, tuple) for v in graph.values())
        line = f"{'+'.join(n1)} vs {'+'.join(n2)} @ {hpf:.0%} HP: {len(graph):,} states ({live:,} non-terminal), enum {te:.1f}s"
        if not complete:
            print(line + f"  -> exceeded cap {cap:,}, not solved")
            continue
        t = time.perf_counter()
        V, iters = shapley_value_iteration(graph)
        start = state_key(endgame(n1, n2, hpf))
        print(line + f"  -> exact value (P[side one wins]) = {V[start]:.4f}, {iters} sweeps, {time.perf_counter()-t:.1f}s")
