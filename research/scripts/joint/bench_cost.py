"""CPU cost of the per-world style predictions (K extra nash.lp + 3 softmaxes on each world's matrix) vs the current
single averaged-matrix style_dists, on the logged world matrices. Usage: python bench_cost.py <worlds.jsonl>"""
import json, os, sys, time
import numpy as np
ENGINE = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "engine"))
sys.path[:0] = [ENGINE, os.path.join(ENGINE, "showdown")]
import behavior, hidden_agents as ha
W = [json.loads(l) for l in open(sys.argv[1])][:60]
t_bar = t_world = t_lp = 0.0; n = 0
for w in W:
    Ms = [np.array(M) for M in w["Ms"]]; p = np.full(Ms[0].shape[0], 1 / Ms[0].shape[0])
    Abar = np.mean(Ms, axis=0)
    t0 = time.perf_counter(); behavior.style_dists(Abar, p); t1 = time.perf_counter()
    for M in Ms: behavior.style_dists(M, p)
    t2 = time.perf_counter(); ha.agent_form_lp(Ms, np.full(len(Ms), 1 / len(Ms))); t3 = time.perf_counter()
    t_bar += t1 - t0; t_world += t2 - t1; t_lp += t3 - t2; n += 1
print(f"decisions {n}: style_dists on Abar {1e3*t_bar/n:.1f} ms | on K={len(Ms)} worlds {1e3*t_world/n:.1f} ms | agent-form LP {1e3*t_lp/n:.1f} ms  (shape {Ms[0].shape})")
