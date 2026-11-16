"""Gate (a): S1 reduces to the depth-2 agent-form value when no information set merges anything.

Reference (independent loop): for each world, M_k[i, j] = sum_b p_b val(full_d1(child_b)) over the TRUE legal rows
(hidden_agents.align / own_matrix conventions), then hidden_agents.agent_form_lp(M_k, w). That is today's depth-2
agent-form computation with the LP leaf in place of nash._leaf's midpoint leaf (S1 needs the game value of the child
matrix; with the midpoint leaf the two agree only at pure saddles — see test_full_d1's leaf-change line).

 (a1) every node made distinguishable (unique keys), K in {1, 2, 4}: A1 == A2 == A0L == reference for the root value,
      the per-root-action values V(a), both cell formulas (rep, pw).
 (a2) K = 1 with the real public keys: A1 == A2 (one world: nothing to update); A1 == reference when the decision has
      no within-world merges, A1 <= reference always (merging can only lower the decider's value).
Run: OURBOT_DET_LEAF=1 python test_a_reduction.py
"""
from dataclasses import replace

from _common import check, decisions, finish, np, pe, ha, pb
import info_planner as ip

TOL = 1e-6


def reference(worlds, side, legal, w):
    Ms = []
    for st in worlds:
        s1, s2 = pb.options(st)
        M = np.empty((len(s1), len(s2)))
        for i, a in enumerate(s1):
            for j, b in enumerate(s2):
                M[i, j] = sum(br.percentage / 100.0 * ip.value_of(ip.full_d1(st.apply_instructions(br))[0])[1]
                              for br in pe.generate_instructions(st, a, b))
        Mo, rows, _ = ha.own_matrix(M, s1, s2, side)
        Ms.append(ha.align(Mo, rows, legal))
    x, v = ha.agent_form_lp(Ms, w)
    return v, sum(wk * M.min(1) for wk, M in zip(w, Ms))


def run_arms(nodes, ex, legal, w):
    sg = ip.solve_groups(nodes, w)
    out = {}
    for arm in ip.ARMS:
        for form in ("rep", "pw"):
            x, v, Va = ip.root_solve(ip.root_matrices(nodes, sg["arms"][arm][form], ex.cols, ex.own, legal, len(w)), w)
            out[f"{arm}_{form}"] = (v, Va)
    x, v, Va = ip.root_solve(ip.root_matrices(nodes, sg["A0L"], ex.cols, ex.own, legal, len(w)), w)
    out["A0L"] = (v, Va)
    return out


pos = decisions(201, 3, every=5, K=4) + decisions(202, 3, every=5, K=4)
print(f"{len(pos)} decisions")
worst = 0.0
for K in (1, 2, 4):
    for state, side, view, legal, worlds in pos:
        ws = worlds[:K]
        w = np.full(K, 1 / K)
        ex = ip.expand(ws, side, legal, set(view.reveals), key_report=False)
        uniq = [replace(nd, key=("uniq", i)) for i, nd in enumerate(ex.nodes)]
        vref, Vref = reference(ws, side, legal, w)
        scale = max(1.0, float(np.abs(Vref).max()))
        for name, (v, Va) in run_arms(uniq, ex, legal, w).items():
            d = max(abs(v - vref), float(np.abs(Va - Vref).max())) / scale
            worst = max(worst, d)
            if d > TOL:
                check(f"(a1) K={K} {name}", False, f"rel diff {d:.2e}")
check("(a1) all-distinguishable S1 == depth-2 AF reference (all arms, K=1,2,4)", worst <= TOL, f"(max rel diff {worst:.1e})")

within_any, a2_eq, a2_le, a2_n, eq_free = 0, True, True, 0, True
for state, side, view, legal, worlds in pos:
    ws = worlds[:1]
    w = np.ones(1)
    ex = ip.expand(ws, side, legal, set(view.reveals), key_report=False)
    st = ip.key_stats(ex.nodes, 1)
    r = run_arms(ex.nodes, ex, legal, w)
    vref, Vref = reference(ws, side, legal, w)
    scale = max(1.0, float(np.abs(Vref).max()))
    a2_n += 1
    a2_eq &= all(abs(r[f"A1_{f}"][0] - r[f"A2_{f}"][0]) <= TOL * scale and
                 np.allclose(r[f"A1_{f}"][1], r[f"A2_{f}"][1], atol=TOL * scale) for f in ("rep", "pw"))
    a2_le &= bool(np.all(r["A1_rep"][1] <= Vref + TOL * scale))
    if st["within_world_frac"] > 0:
        within_any += 1
    else:
        eq_free &= abs(r["A1_rep"][0] - vref) <= TOL * scale and np.allclose(r["A1_rep"][1], Vref, atol=TOL * scale)
check("(a2) K=1 public keys: A1 == A2", a2_eq, f"({a2_n} decisions)")
check("(a2) K=1: A1 <= reference for every root action", a2_le)
check("(a2) K=1 without within-world merges: A1 == reference", eq_free,
      f"({a2_n - within_any} such decisions; {within_any} had within-world merges)")
finish()
