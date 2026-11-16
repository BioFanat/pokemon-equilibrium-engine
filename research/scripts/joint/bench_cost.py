"""Cost of style_dists on the union Abar (as deployed) vs on the K per-world matrices. Usage: bench_cost.py <worlds.jsonl>. all 438 decisions, deployed shapes; reports mean and median per-decision timings + ratio."""
import json, os, sys, time
import numpy as np
ENGINE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "engine")
sys.path[:0] = [ENGINE, os.path.join(ENGINE, "showdown")]
import behavior
W = [json.loads(l) for l in open(sys.argv[1])]
tb, tw = [], []
for w in W:
    Ms = [np.array(M) for M in w["Ms"]]; p = np.full(Ms[0].shape[0], 1 / Ms[0].shape[0])
    sys.path  # Abar as deployed (union, not np.mean)
    labels = w["labels"]; cols = w["cols"]
    Abar = np.zeros((Ms[0].shape[0], len(labels)))
    for j, lab in enumerate(labels):
        ks = [k for k in range(len(Ms)) if lab in cols[k]]
        Abar[:, j] = np.mean([Ms[k][:, cols[k].index(lab)] for k in ks], axis=0)
    t0 = time.perf_counter(); behavior.style_dists(Abar, p); t1 = time.perf_counter()
    for M in Ms: behavior.style_dists(M, p)
    t2 = time.perf_counter(); tb.append(t1 - t0); tw.append(t2 - t1)
tb, tw = np.array(tb) * 1e3, np.array(tw) * 1e3
print(f"n {len(W)}: Abar(union) mean {tb.mean():.2f} med {np.median(tb):.2f} ms | K worlds mean {tw.mean():.2f} med {np.median(tw):.2f} ms | ratio of means {tw.mean()/tb.mean():.2f}")
print("shapes of first 60: Ms[0]", set(tuple(np.array(w['Ms'][0]).shape) for w in W[:60]).__len__(), "distinct; np.mean(Ms) valid only if all worlds same n:",
      all(len(set(len(c) for c in w['cols'])) == 1 for w in W[:60]))
