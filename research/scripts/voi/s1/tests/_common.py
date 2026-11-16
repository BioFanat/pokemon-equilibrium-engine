"""Shared setup for the S1 gate tests: engine path, det leaf, real decision positions from A0 self-play."""
from __future__ import annotations

import os
import random
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", "..", "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "engine"))
sys.path.insert(0, os.path.join(ROOT, "research", "scripts", "voi"))
os.environ.setdefault("OURBOT_DET_LEAF", "1")

import numpy as np  # noqa: E402
import poke_engine as pe  # noqa: E402
import hidden as hd  # noqa: E402
import hidden_agents as ha  # noqa: E402
import nash  # noqa: E402
import pkbattle as pb  # noqa: E402

FAILS: list[str] = []


def check(name, ok, detail=""):
    print(f"{'ok ' if ok else 'BAD'} {name} {detail}", flush=True)
    if not ok:
        FAILS.append(name)


def decisions(seed, n, every=3, K=4, max_turns=80):
    """Up to n decision positions (state, side, view, legal, worlds) from uniform-random self-play with Trackers."""
    rng = random.Random(seed)
    state = pb.random_state(rng)
    tr = {1: hd.Tracker(1, state), 2: hd.Tracker(2, state)}
    out, c = [], 0
    for _ in range(max_turns):
        opts = dict(zip((1, 2), pb.options(state)))
        for s in (1, 2):
            if len(opts[s]) > 1:
                c += 1
                if c % every == 0 and len(out) < n:
                    view = hd.make_view(state, tr[s])
                    belief = hd.Belief(view)
                    worlds = [hd.determinize(state, view, belief, rng) for _ in range(K)]
                    out.append((state, s, view, opts[s], worlds))
        ch = {s: rng.choice(opts[s]) for s in (1, 2)}
        br = pe.generate_instructions(state, ch[1], ch[2])
        pick = rng.choices(br, weights=[b.percentage for b in br])[0]
        new = state.apply_instructions(pick)
        for s in (1, 2):
            tr[s].update(state, ch[1], ch[2], pick, new)
        state = new
        if not pb.alive(state.side_one) or not pb.alive(state.side_two) or len(out) >= n:
            break
    return out


def finish():
    print("ALL OK" if not FAILS else f"FAILURES: {FAILS}")
    sys.exit(0 if not FAILS else 1)
