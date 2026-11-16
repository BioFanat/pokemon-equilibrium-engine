"""Gate (b): S1's solver on the exact scouting toy (research/scripts/voi/toy_scout.py).

Each toy root action becomes a root row, the opponent is passive at the root (one column 'pass' per world), and every
(world theta, observation o) branch becomes a Node with key (action, o), continuation matrix M_theta and immediate
payoff r. Expected (toy_scout's hand-checked values): A1 recovers v_isc — wait 1/2, probe 1-c (probe worth 1/2 > c iff
chosen), coin 1/2-c (zero VOI), noisy 0.8-c, hitprobe - hit = 1/2 — and A0L recovers the fused (clairvoyant) values.
Control A2 (frozen posterior on the group's support): coin == A1 (the §3.4 regression: zero-VOI reveal == frozen value),
noisy 1/2-c (the soft update is worth 0.3), probe == A1 (a perfect reveal eliminates worlds; hard eliminations are
credited to both arms by construction). Both cell formulas (rep, pw) agree here (passive opponent).
Run: python test_b_toy.py
"""
from _common import check, finish, np
import info_planner as ip
import toy_scout as ts

C, D = (0.0, 0.3, 0.49, 0.5, 0.51, 0.9), (-0.2, 0.0, 0.3, 1.5)
actions = [("wait", 0)] + [(k, c) for k in ("probe", "noisy", "coin") for c in C] + [(k, d) for k in ("hit", "hitprobe") for d in D]
label = lambda a: f"{a[0]}:{a[1]}"
legal = [label(a) for a in actions]
nodes = []
for k, th in enumerate(ts.TYPES):
    for a in actions:
        for o, p, r in ts.branches(a, th):
            nodes.append(ip.Node(k, label(a), "pass", p, (label(a), o), ts.M[th], ["cA", "cB"], r=r))
w = np.array([ts.B0[t] for t in ts.TYPES])
sg = ip.solve_groups(nodes, w)
V = {}
for arm in ip.ARMS:
    for form in ("rep", "pw"):
        Rs = ip.root_matrices(nodes, sg["arms"][arm][form], [["pass"]] * 2, [legal] * 2, legal, 2)
        V[f"{arm}_{form}"] = dict(zip(legal, ip.root_solve(Rs, w)[2]))
Rs = ip.root_matrices(nodes, sg["A0L"], [["pass"]] * 2, [legal] * 2, legal, 2)
V["A0L"] = dict(zip(legal, ip.root_solve(Rs, w)[2]))

def eq(name, got, want):
    check(name, abs(got - want) < 1e-9, f"got {got:+.6f} want {want:+.6f}")

for form in ("rep", "pw"):
    A1, A2 = V[f"A1_{form}"], V[f"A2_{form}"]
    eq(f"[{form}] A1 wait", A1["wait:0"], 0.5)
    eq(f"[{form}] A1 == toy v_isc(wait)", A1["wait:0"], ts.v_isc(("wait", 0)))
    for c in C:
        eq(f"[{form}] A1 probe c={c}", A1[f"probe:{c}"], 1 - c)
        eq(f"[{form}] A1 probe value over wait c={c} (= 1/2 - c)", A1[f"probe:{c}"] - A1["wait:0"], 0.5 - c)
        check(f"[{form}] A1 prefers probe iff c < 1/2 (c={c})", (A1[f"probe:{c}"] > A1["wait:0"] + 1e-12) == (c < 0.5))
        eq(f"[{form}] A1 coin (zero VOI) c={c}", A1[f"coin:{c}"], 0.5 - c)
        eq(f"[{form}] A1 noisy c={c}", A1[f"noisy:{c}"], 0.8 - c)
        eq(f"[{form}] A2 coin == A1 coin (frozen == live at zero VOI) c={c}", A2[f"coin:{c}"], A1[f"coin:{c}"])
        eq(f"[{form}] A2 noisy (soft update removed) c={c}", A2[f"noisy:{c}"], 0.5 - c)
        eq(f"[{form}] A2 probe == A1 probe (hard elimination credited to both) c={c}", A2[f"probe:{c}"], 1 - c)
        eq(f"A0L probe == toy fused c={c}", V["A0L"][f"probe:{c}"], ts.v_fused(("probe", c)))
    for d in D:
        eq(f"[{form}] A1 info value hitprobe - hit d={d}", A1[f"hitprobe:{d}"] - A1[f"hit:{d}"], 0.5)
        eq(f"A0L (fused) info value hitprobe - hit d={d}", V["A0L"][f"hitprobe:{d}"] - V["A0L"][f"hit:{d}"], 0.0)
eq("A0L wait == toy fused (clairvoyant 1)", V["A0L"]["wait:0"], ts.v_fused(("wait", 0)))
finish()
