"""Load curve with repeats: fraction of pe.id(c, 0) calls that return depth_searched != 1, per kind, per load level.
Also re-checks that every non-terminal 6v6 deeper result is bit-identical to an unloaded pe.id(c, t) result of the
same depth.

Usage: nice -n 10 engine/.venv/bin/python probe_curve.py <tag> <reps> <trials> <levels comma-sep>
Run under `taskpolicy -b` to confine the whole process tree (probe + load) to the 4 efficiency cores, a rough
stand-in for a small container. Writes out/curve_<tag>.json.
"""
import json, os, sys, time
from collections import Counter, defaultdict
import numpy as np

ARGV = sys.argv[1:]; sys.argv = sys.argv[:1]          # probe_load parses argv at import
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from probe_load import build_states, call, start_load, stop_load, pb, OUT  # noqa: E402

tag, REPS, TRIALS = ARGV[0], int(ARGV[1]), int(ARGV[2])
LEVELS = [int(x) for x in ARGV[3].split(",")]
rows = build_states()
states = [(i, side, r[side]) for i, r in enumerate(rows) for side in ("child", "mirror")]
bo = {(i, s): (-1 if pb.alive(st.side_one) == 0 else (1 if pb.alive(st.side_two) == 0 else 0)) for i, s, st in states}
ref = {(i, s): call(st) for i, s, st in states}
assert all(r["d"] == 1 for r in ref.values())
out, deeper6 = {}, defaultdict(set)
for n in LEVELS:
    out[n] = []
    for trial in range(TRIALS):
        procs = start_load(n, secs=120)
        try:
            time.sleep(0.5)
            per = [(i, s, call(st)) for _ in range(REPS) for i, s, st in states]
        finally:
            stop_load(procs)
        d1_changed = sum(x["d"] == 1 and x["key"] != ref[(i, s)]["key"] for i, s, x in per)
        res = dict(d1_changed=d1_changed)
        for kind in ("1v1", "6v6"):
            for term in (False, True):
                sel = [x for i, s, x in per if rows[i]["kind"] == kind and bool(bo[(i, s)]) == term]
                if sel:
                    res[f"{kind}{'_term' if term else ''}"] = round(sum(x["d"] != 1 for x in sel) / len(sel), 5)
        for i, s, x in per:
            if rows[i]["kind"] == "6v6" and x["d"] >= 2: deeper6[(i, s, x["d"])].add(x["key"])
        out[n].append(res)
        print(tag, "N", n, "trial", trial, res, flush=True)
idx = {(i, s): st for i, s, st in states}
rep = dict(pairs=len(deeper6), reproduced=0, multi=sum(len(k) > 1 for k in deeper6.values()))
for (i, s, d), keys in deeper6.items():
    got = set()
    for t in (1, 1, 2, 2, 3, 5, 10, 20, 50):
        x = call(idx[(i, s)], t)
        if x["d"] == d: got.add(x["key"])
        if keys <= got: break
    rep["reproduced"] += keys <= got
print(tag, "6v6 reproduce", rep, flush=True)
json.dump(dict(levels=out, reproduce_6v6=rep, reps=REPS, trials=TRIALS), open(os.path.join(OUT, f"curve_{tag}.json"), "w"), indent=1)
