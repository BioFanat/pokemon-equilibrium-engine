"""Force the i8 wrap back to depth_searched == 1 (unloaded, pe.id(c, 1) on spin-prone 1v1 states) and compare the
wrapped 'depth 1' matrix with the true depth-1 reference: same cell values up to a row permutation? same maximin?
Usage: nice -n 10 engine/.venv/bin/python probe_wrap1.py -> out/wrap1.json"""
import json, os, sys
import numpy as np
sys.argv = sys.argv[:1]
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from probe_load import build_states, call, pb, OUT  # noqa: E402
import poke_engine as pe  # noqa: E402

def full(st, ms):
    r = pe.id(st, ms)
    return r.depth_searched, list(r.s1), list(r.s2), [np.nan if v is None else v for v in r.matrix]

rows = build_states()
states = [(i, s, r[s]) for i, r in enumerate(rows) for s in ("child", "mirror") if r["kind"] == "1v1"]
spin = [(i, s, st) for i, s, st in states if not (1 <= call(st, 1)["d"] <= 3)]
ref = {(i, s): full(st, 0) for i, s, st in spin}
hits, calls = [], 0
while len(hits) < 25 and calls < 9000:
    for i, s, st in spin:
        d, s1, s2, M = full(st, 1); calls += 1
        if d != 1: continue
        _, rs1, rs2, R = ref[(i, s)]
        n2 = len(s2)
        rows_w = {s1[k]: M[k * n2:(k + 1) * n2] for k in range(len(s1))}
        # the depth-1 row-label bug: values are in the ORIGINAL option order while s1 is the re-ordered list, so
        # compare cell multisets of finite values and the maximin, not labelled rows
        mm = lambda A: float(np.nanmax(np.nanmin(np.array(A, float).reshape(-1, n2), axis=1)))
        hits.append(dict(i=i, side=s, terminal=bool(pb.alive(st.side_one) == 0 or pb.alive(st.side_two) == 0),
                         bytes_equal=(s1 == rs1 and [repr(x) for x in M] == [repr(x) for x in R]),
                         s1_same_order=s1 == rs1, nan=int(np.isnan(M).sum()), ref_nan=int(np.isnan(R).sum()),
                         maximin=mm(M), ref_maximin=mm(R)))
        if len(hits) >= 25: break
print("calls", calls, "depth-1 hits", len(hits))
for h in hits: print(h)
json.dump(dict(calls=calls, hits=hits), open(os.path.join(OUT, "wrap1.json"), "w"), indent=1)
