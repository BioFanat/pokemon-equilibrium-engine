"""Known-answer test of the i8 wrap path, unloaded: pe.id(c, t>=1) on spin-prone 1v1 states sometimes wraps the
depth counter back to depth_searched == 1. Does that wrapped 'depth 1' give the same matrix bytes / maximin value
as the true depth-1 search (pe.id(c, 0) unloaded)?  Also: how do wrapped negative depths change values?
Usage: nice -n 10 engine/.venv/bin/python probe_wrap.py  -> out/wrap.json"""
import json, os, sys
from collections import Counter
sys.argv = sys.argv[:1]
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from probe_load import build_states, call, pb, OUT  # noqa: E402

rows = build_states()
states = [(i, s, r[s]) for i, r in enumerate(rows) for s in ("child", "mirror") if r["kind"] == "1v1"]
ref = {(i, s): call(st) for i, s, st in states}
spin = [(i, s, st) for i, s, st in states if call(st, 1)["d"] > 3 or call(st, 1)["d"] < 1]
print(len(states), "1v1 states;", len(spin), "spin past depth 3 within 1 ms", flush=True)
res = dict(n_states=len(states), n_spin=len(spin), d1_hits=0, same_key=0, same_value=0, same_s1_order=0,
           same_s1_set=0, examples=[])
dist = Counter()
for i, s, st in spin[:60]:
    for t in (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 15):
        x = call(st, t); dist[x["d"]] += 1
        if x["d"] == 1:
            r = ref[(i, s)]
            res["d1_hits"] += 1
            res["same_key"] += x["key"] == r["key"]
            res["same_value"] += abs(x["v"] - r["v"]) < 1e-6
            res["same_s1_order"] += x["key"][1] == r["key"][1]
            res["same_s1_set"] += sorted(x["key"][1]) == sorted(r["key"][1])
            if len(res["examples"]) < 5:
                res["examples"].append(dict(i=i, side=s, t=t, v=x["v"], ref=r["v"], nan=x["nan"], ref_nan=r["nan"],
                                            s1=list(x["key"][1]), ref_s1=list(r["key"][1])))
res["depth_hist_wrapped_calls"] = dict(sorted(dist.items()))
print({k: v for k, v in res.items() if k not in ("examples", "depth_hist_wrapped_calls")}, flush=True)
for e in res["examples"]: print(e)
json.dump(res, open(os.path.join(OUT, "wrap.json"), "w"), indent=1)
