"""Exact toy for scouting option (a): one-step lookahead over PUBLIC observation branches with posterior re-solve,
one continuation strategy per information set (no strategy fusion), plus the two controls of workstream G.

Game.  Opponent type theta in {A, B}, prior b0 = (1/2, 1/2); the opponent knows theta, we do not.
  Stage 1 (after the root): we pick c in {cA, cB}, the opponent picks beta in {1, 2};  u = 1[c matches theta]
      (matrices M_A = [[1,1],[0,0]], M_B = [[0,0],[1,1]]).  Known type: value 1.  Unknown (b0): AF value 1/2.
      => perfect reveal is worth exactly 1/2 at b0 (Theorem C's eta(b0) = sum b0 V(delta) - V(b0) = 1 - 1/2).
  Root actions (opponent passive at the root, so the toy is exact and hand-checkable):
      wait            no observation, no cost
      probe(c)        observation o = theta (perfect), cost c
      noisy(c)        o = theta w.p. 0.8, flipped w.p. 0.2, cost c         -> posterior (0.8, 0.2): continuation 0.8
      coin(c)         o = independent fair coin (zero-VOI reveal), cost c   -> must be worth exactly -c + 1/2
      hit(d)          physical effect +d, no observation                   (matched no-information control)
      hitprobe(d)     physical effect +d AND o = theta                      -> info value = V(hitprobe) - V(hit) = 1/2

Planners compared at the root:
  isc   information-set-correct one-step lookahead: group successors by public o, posterior b'(theta) ∝ b0 P(o|a,theta),
        continuation = AF-LP over the types consistent with o at b', root value = sum_o P(o|a) [cost + V(b'_o)].
  fused per-world (strategy-fusion) lookahead: each world theta is solved with theta known -> every root action is
        worth cost + 1, so the planner never probes (and values `wait` at 1 instead of 1/2).

Checks (asserted): V_isc(probe c) = 1 - c; V_isc(wait) = 1/2; probe chosen iff c < 1/2; V_isc(coin c) = 1/2 - c
(zero VOI); V_isc(noisy c) = 0.8 - c; V_isc(hitprobe d) - V_isc(hit d) = 1/2 for every d; fused values as stated.
Run: python toy_scout.py
"""
from __future__ import annotations

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "..", "..", "engine")))
import hidden_agents as ha  # noqa: E402  (agent_form_lp is the project's own AF solver)

TYPES = ["A", "B"]
M = {"A": np.array([[1.0, 1.0], [0.0, 0.0]]), "B": np.array([[0.0, 0.0], [1.0, 1.0]])}
B0 = {"A": 0.5, "B": 0.5}


def branches(action, theta):
    """Public observation branches (o, prob, immediate payoff) of a root action in world theta."""
    kind, par = action
    other = "B" if theta == "A" else "A"
    if kind == "wait":
        return [("none", 1.0, 0.0)]
    if kind == "probe":
        return [(theta, 1.0, -par)]
    if kind == "noisy":
        return [(theta, 0.8, -par), (other, 0.2, -par)]
    if kind == "coin":
        return [("H", 0.5, -par), ("T", 0.5, -par)]
    if kind == "hit":
        return [("none", 1.0, par)]
    if kind == "hitprobe":
        return [(theta, 1.0, par)]
    raise ValueError(kind)


def value_at(b):
    """AF-LP continuation value at belief b (dict type -> prob), one strategy for the information set."""
    ts = [t for t in TYPES if b[t] > 0]
    _, v = ha.agent_form_lp([M[t] for t in ts], [b[t] for t in ts])
    return float(v)


def v_isc(action):
    """Information-set-correct one-step lookahead: group by public observation, posterior re-solve."""
    joint = {}                                         # o -> {theta: P(theta) P(o | a, theta)}, immediate payoff
    for t in TYPES:
        for o, p, r in branches(action, t):
            joint.setdefault(o, {})[t] = joint.get(o, {}).get(t, 0.0) + B0[t] * p
            joint[o]["_r"] = r
    total = 0.0
    for o, d in joint.items():
        r = d.pop("_r")
        po = sum(d.values())
        post = {t: d.get(t, 0.0) / po for t in TYPES}
        total += po * (r + value_at(post))
    return total


def v_fused(action):
    """Per-world (strategy-fusion) lookahead: each world solved with its type known."""
    return sum(B0[t] * sum(p * (r + float(ha.lp_x(M[t])[1])) for _, p, r in branches(action, t)) for t in TYPES)


if __name__ == "__main__":
    ok = True

    def check(name, got, want):
        global ok
        good = abs(got - want) < 1e-9
        ok &= good
        print(f"{'ok ' if good else 'BAD'} {name:34s} got {got:+.6f} want {want:+.6f}")

    check("eta(b0) = sum b V(delta) - V(b0)", sum(B0[t] * float(ha.lp_x(M[t])[1]) for t in TYPES) - value_at(B0), 0.5)
    check("isc wait", v_isc(("wait", 0)), 0.5)
    check("fused wait (clairvoyant)", v_fused(("wait", 0)), 1.0)
    for c in (0.0, 0.3, 0.49, 0.5, 0.51, 0.9):
        check(f"isc probe c={c}", v_isc(("probe", c)), 1 - c)
        check(f"fused probe c={c}", v_fused(("probe", c)), 1 - c)
        chosen = v_isc(("probe", c)) > v_isc(("wait", 0)) + 1e-12
        check(f"isc chooses probe iff c<1/2 (c={c})", float(chosen), float(c < 0.5))
        check(f"fused never probes (c={c})", float(v_fused(("probe", c)) > v_fused(("wait", 0))), 0.0)
        check(f"isc coin (zero VOI) c={c}", v_isc(("coin", c)), 0.5 - c)
        check(f"isc noisy c={c}", v_isc(("noisy", c)), 0.8 - c)
    for d in (-0.2, 0.0, 0.3, 1.5):
        check(f"isc info value hitprobe-hit d={d}", v_isc(("hitprobe", d)) - v_isc(("hit", d)), 0.5)
        check(f"fused info value hitprobe-hit d={d}", v_fused(("hitprobe", d)) - v_fused(("hit", d)), 0.0)
    print("ALL OK" if ok else "FAILURES")
    sys.exit(0 if ok else 1)
