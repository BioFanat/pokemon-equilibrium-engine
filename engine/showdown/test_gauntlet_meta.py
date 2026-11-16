"""Checks gauntlet.meta_nash on small games. Run: ~/sandbox/bot-venv/bin/python engine/showdown/test_gauntlet_meta.py"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np                                                                       # noqa: E402

from gauntlet import meta_bots, meta_nash                                                # noqa: E402


def game(scores):
    """Upper-triangle scores {(i, j): score of i vs j} -> skew-symmetric P = score - 0.5."""
    n = 1 + max(max(k) for k in scores)
    P = np.zeros((n, n))
    for (i, j), s in scores.items():
        P[i, j], P[j, i] = s - 0.5, 0.5 - s
    return P


def check(name, P, want):
    r = meta_nash(P)
    assert np.allclose(r["sigma"], want, atol=1e-6), (name, r["sigma"])
    assert abs(r["value"]) <= 1e-7, (name, r["value"])
    assert r["check"], (name, r["exploitability"])
    assert np.all(r["payoff"] <= 1e-7), (name, r["payoff"])
    print(f"ok  {name}: sigma={np.round(r['sigma'], 4)} value={r['value']:.1e} "
          f"payoff={np.round(r['payoff'], 4)} note={r['note']}")
    return r


def main():
    check("rock-paper-scissors", game({(0, 1): 1, (1, 2): 1, (0, 2): 0}), [1 / 3] * 3)
    check("one dominant bot", game({(0, 1): 0.8, (0, 2): 0.9, (0, 3): 0.7, (1, 2): 0.6, (1, 3): 0.3, (2, 3): 0.5}),
          [1, 0, 0, 0])
    # RPS among 0-2, bot 3 loses to all of them: sigma_3 = 0, its Nash-averaged payoff < 0
    r = check("4 bots, one dominated", game({(0, 1): 0.9, (1, 2): 0.9, (0, 2): 0.1,
                                             (0, 3): 0.6, (1, 3): 0.7, (2, 3): 0.8}), [1 / 3, 1 / 3, 1 / 3, 0])
    assert r["payoff"][3] < -1e-3 and r["support"] == [0, 1, 2]
    # two identical bots: every split between them is an equilibrium, which the dual reports
    r = meta_nash(game({(0, 1): 0.5, (0, 2): 0.7, (1, 2): 0.7}))
    assert r["check"] and r["sigma"][2] <= 1e-6, r
    print(f"ok  two identical bots: sigma={np.round(r['sigma'], 4)} note={r['note']}")
    assert r["note"], r
    d = {("a", "b"): dict(seeds={1: [1, 1]}), ("a", "c"): dict(seeds={1: [0, 1]}), ("b", "c"): dict(seeds={})}
    assert meta_bots(["a", "b", "c"], d) == (["a", "b"], ["c"])
    print("ok  meta_bots: drops the later bot of a seedless pairing")
    print("all passed")


if __name__ == "__main__":
    main()
