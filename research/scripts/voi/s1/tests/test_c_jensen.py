"""Gate (c): E1's Jensen property V_A1(a) >= V_A2(a) on random small instances (and on a few real decisions).

Random instances: K in {2,3,4} worlds (random prior), 2-4 root rows, per-world opponent columns drawn from a shared
label pool (so some labels are missing in some worlds), 1-4 chance branches per cell with keys from a small alphabet
(so groups merge across and within worlds), random continuation matrices whose row labels are drawn from a shared pool
(so the colmin alignment fires), random immediate payoffs.
Each cell formula satisfies exactly one of the two natural bounds, and each is asserted where it is a theorem:
  rep (report §3.2, R_k = sum_o P(o|k) V(o)): Jensen. Per group V1 >= V2 and V_A1(a) >= V_A2(a) for every root action
      (also at a* = argmax x_A1). The clairvoyance bound V_A0L(a) >= V_A1(a) is NOT a theorem here (world k's cell
      takes the cross-world group average, which can exceed its own clairvoyant value): violation rate reported.
  pw (per-world, R_k = world k's own payoff under x_o): clairvoyance bound V_A0L(a) >= V_A1(a) >= ... asserted;
      Jensen is not a theorem: violation rate reported.
Run: OURBOT_DET_LEAF=1 python test_c_jensen.py
"""
import random

from _common import check, decisions, finish, np
import info_planner as ip

TOL = 1e-7


def instance(rng):
    K = rng.randint(2, 4)
    w = np.array([rng.random() + 0.2 for _ in range(K)]); w /= w.sum()
    legal = [f"a{i}" for i in range(rng.randint(2, 4))]
    pool = [f"b{j}" for j in range(5)]
    rpool = ["x", "y", "z", "u"]
    cols = [sorted(rng.sample(pool, rng.randint(1, 4))) for _ in range(K)]
    own = [list(legal) if rng.random() < 0.85 else legal[:-1] for _ in range(K)]
    nodes = []
    for k in range(K):
        for a in own[k]:
            for b in cols[k]:
                nb = rng.randint(1, 4)
                p = np.array([rng.random() + 0.05 for _ in range(nb)]); p /= p.sum()
                for pi in p:
                    rows = sorted(rng.sample(rpool, rng.randint(2, 4)))
                    nc = rng.randint(1, 3)
                    M = np.array([[rng.uniform(-1, 1) for _ in range(nc)] for _ in rows])
                    nodes.append(ip.Node(k, a, b, float(pi), (a, b, rng.randint(0, 2)), M, rows, r=rng.uniform(-.2, .2)))
    return nodes, w, legal, cols, own


rng = random.Random(0)
n_inst, bad_group, bad_rep, bad_a0l, pw_viol, pw_tot, a_star_bad, rep_a0l = 2000, 0, 0, 0, 0, 0, 0, 0
for _ in range(n_inst):
    nodes, w, legal, cols, own = instance(rng)
    sg = ip.solve_groups(nodes, w)
    bad_group += sum(g["V1"] < g["V2"] - TOL for g in sg["groups"])
    Va = {}
    for arm in ip.ARMS:
        for form in ("rep", "pw"):
            x, v, V = ip.root_solve(ip.root_matrices(nodes, sg["arms"][arm][form], cols, own, legal, len(w)), w)
            Va[f"{arm}_{form}"] = (x, V)
    _, _, V0 = ip.root_solve(ip.root_matrices(nodes, sg["A0L"], cols, own, legal, len(w)), w)
    bad_rep += int(np.any(Va["A1_rep"][1] < Va["A2_rep"][1] - TOL))
    bad_a0l += int(np.any(V0 < Va["A1_pw"][1] - TOL))
    rep_a0l += int(np.any(V0 < Va["A1_rep"][1] - TOL))
    s = int(np.argmax(Va["A1_rep"][0]))
    a_star_bad += Va["A1_rep"][1][s] < Va["A2_rep"][1][s] - TOL
    pw_tot += 1
    pw_viol += int(np.any(Va["A1_pw"][1] < Va["A2_pw"][1] - TOL))
check("(c) per-group V_A1(o) >= V_A2(o) (A2 evaluated under the live posterior)", bad_group == 0, f"({bad_group} violations)")
check("(c) rep: V_A1(a) >= V_A2(a) for every root action", bad_rep == 0, f"({n_inst} instances, {bad_rep} with a violation)")
check("(c) rep: V_A1(a*) >= V_A2(a*) at a* = argmax x_A1", a_star_bad == 0)
check("(c) pw: V_A0L(a) >= V_A1(a) (clairvoyant children bound)", bad_a0l == 0, f"({bad_a0l} violations)")
print(f"    pw (Jensen not a theorem): instances with some V_A1(a) < V_A2(a): {pw_viol}/{pw_tot}")
print(f"    rep (clairvoyance bound not a theorem): instances with some V_A0L(a) < V_A1(a): {rep_a0l}/{n_inst}")

real_bad = 0
pos = decisions(301, 3, every=6, K=4)
for state, side, view, legal, worlds in pos:
    res = ip.plan(worlds, side, legal, set(view.reveals))
    sc = max(1.0, float(np.abs(res["A1_rep"]["V"]).max()))
    real_bad += int(np.any(res["A1_rep"]["V"] < res["A2_rep"]["V"] - 1e-6 * sc))
check("(c) rep: V_A1(a) >= V_A2(a) on real decisions", real_bad == 0, f"({len(pos)} decisions)")
finish()
