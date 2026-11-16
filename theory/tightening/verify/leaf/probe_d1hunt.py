"""Hunt depth_searched == 1 returns under load whose matrix differs from the unloaded reference; record details.
Usage: nice -n 10 engine/.venv/bin/python probe_d1hunt.py <levels> <trials> <reps> -> out/d1hunt.json"""
import json, os, sys, time
ARGV = sys.argv[1:]; sys.argv = sys.argv[:1]
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from probe_load import build_states, call, start_load, stop_load, pb, OUT  # noqa: E402
LEVELS = [int(x) for x in ARGV[0].split(",")]; TRIALS, REPS = int(ARGV[1]), int(ARGV[2])
rows = build_states()
states = [(i, s, r[s]) for i, r in enumerate(rows) for s in ("child", "mirror")]
ref = {(i, s): call(st) for i, s, st in states}
hits, ncalls, nd1 = [], 0, 0
for n in LEVELS:
    for _ in range(TRIALS):
        procs = start_load(n, secs=120)
        try:
            time.sleep(0.5)
            for _ in range(REPS):
                for i, s, st in states:
                    x = call(st); ncalls += 1
                    if x["d"] == 1:
                        nd1 += 1
                        r = ref[(i, s)]
                        if x["key"] != r["key"]:
                            M, R = x["key"][3], r["key"][3]
                            hits.append(dict(N=n, i=i, side=s, kind=rows[i]["kind"], v=x["v"], ref=r["v"],
                                             nan=x["nan"], ref_nan=r["nan"], same_s1=x["key"][1] == r["key"][1],
                                             same_s2=x["key"][2] == r["key"][2], s1=list(x["key"][1]), ref_s1=list(r["key"][1]),
                                             cells_diff=sum(a != b for a, b in zip(M, R)), n_cells=len(R),
                                             alive=[pb.alive(st.side_one), pb.alive(st.side_two)]))
                            print(hits[-1], flush=True)
        finally:
            stop_load(procs)
print("calls", ncalls, "depth1 returns", nd1, "depth1 mismatches", len(hits))
json.dump(dict(calls=ncalls, d1_returns=nd1, hits=hits), open(os.path.join(OUT, "d1hunt.json"), "w"), indent=1)
