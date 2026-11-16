"""Descriptive tera-reveal statistics from Showdown headless replays (no equilibrium reference).

Reads <results>/<RUN>/replays_headless/*.json: `input_log` lines like `>p1 move X terastallize`, `sides`
(role -> p1/p2), `methods` (role -> method), `turns`. Reports per method: fraction of games with a tera,
turn of first tera (absolute and as a fraction of game length), and the number of own mons still alive at
the tera turn (from the `|faint|` count in the output log when present, else omitted).

Usage: tera_logs.py --runs AF AF16 E2 --max 150
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import random
import re

import numpy as np

RES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "engine", "showdown", "results")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", default=["AF", "AF16", "E2"])
    ap.add_argument("--max", type=int, default=150)
    a = ap.parse_args()
    stats = collections.defaultdict(lambda: dict(games=0, tera=0, turn=[], frac=[]))
    for run in a.runs:
        d = os.path.join(RES, run, "replays_headless")
        if not os.path.isdir(d):
            continue
        files = sorted(f for f in os.listdir(d) if f.endswith(".json"))
        random.Random(0).shuffle(files)
        for f in files[: a.max]:
            g = json.load(open(os.path.join(d, f)))
            if g.get("error") or not g.get("turns"):
                continue
            roles = {v: k for k, v in g["sides"].items() if k in ("A", "B")}       # p1 -> A
            method = {p: f"{run}:{g['methods'][r]}" for p, r in roles.items()}
            turn, first = 0, {}
            for line in g["input_log"]:
                m = re.match(r">(p[12]) (move|switch) ", line)
                if not m:
                    continue
                if m.group(1) == "p1":
                    turn += 1          # one p1 command per turn (approximate; forced switches add commands)
                if "terastallize" in line and m.group(1) not in first:
                    first[m.group(1)] = turn
            for p, meth in method.items():
                s = stats[meth]
                s["games"] += 1
                if p in first:
                    s["tera"] += 1
                    s["turn"].append(first[p])
                    s["frac"].append(first[p] / max(1, turn))
    print(f"{'method':>22} {'games':>5} {'P(tera)':>8} {'first tera turn med':>20} {'as frac of game med':>20}")
    for meth, s in sorted(stats.items()):
        print(f"{meth:>22} {s['games']:>5} {s['tera'] / max(1, s['games']):8.2f} {np.median(s['turn']) if s['turn'] else float('nan'):20.1f} "
              f"{np.median(s['frac']) if s['frac'] else float('nan'):20.2f}")
