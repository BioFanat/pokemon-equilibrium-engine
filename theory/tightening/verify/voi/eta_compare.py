"""Verifier: eta-hat aggregates, stored (scripts/out/eta_hat2.jsonl, two-agree guard) vs det-leaf rerun (out/eta_hat_det.jsonl)."""
import json, sys
import numpy as np
for path in ["../../scripts/out/eta_hat2.jsonl", "out/eta_hat_det.jsonl"]:
    G = [json.loads(l) for l in open(path)]
    D = [r for g in G for r in g["decisions"]]
    es = np.array([r["eta_hat_set"] / r["payoff_range"] for r in D if r["payoff_range"] > 0])
    et = np.array([r["eta_hat_tera"] / r["payoff_range_tera"] for r in D if "eta_hat_tera" in r and r["payoff_range_tera"] > 0])
    pr = np.array([r["payoff_range"] for r in D])
    print(f"{path}: games {len(G)} decisions {len(D)} tera-decisions {len(et)} turns med {np.median([g['turns'] for g in G])}")
    print(f"  set : mean {es.mean():.4f} med {np.median(es):.4f} p90 {np.quantile(es, .9):.4f} max {es.max():.4f} | ==0 exact {np.mean(es <= 0):.3f} <=1e-6 {np.mean(es <= 1e-6):.3f}")
    print(f"  tera: mean {et.mean():.4f} med {np.median(et):.4f} max {et.max():.4f} | ==0 exact {np.mean(et <= 0):.3f} <=1e-6 {np.mean(et <= 1e-6):.3f} >1% {np.mean(et > .01):.3f}")
    print(f"  payoff range med {np.median(pr):.0f} max {pr.max():.0f}; retries max {max(r.get('retries', 0) for r in D)}")
