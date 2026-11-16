"""Verifier: compare the det-leaf recompute (out/recompute46.jsonl) with the stored guarded_all.jsonl, and recompute
the claim-3/4/5 statistics on the reported 30 and on all 46 (30 + 16 dropped). Own code, not summarize_voi.py."""
import json, sys
import numpy as np
OLD = {r["seed"]: r for r in map(json.loads, open("../../scripts/out/guarded_all.jsonl"))}
NEW = {r["seed"]: r for r in map(json.loads, open("out/recompute46.jsonl"))}
rep = [s for s, r in OLD.items() if "error" not in r]; drop = [s for s, r in OLD.items() if "error" in r]
print("new records", len(NEW), "errors in new:", [(s, NEW[s]["error"]) for s in NEW if "error" in NEW[s]])
print("missing:", [s for s in OLD if s not in NEW])
F = ["Delta", "V0", "voi_total", "v_a3", "eta_NL0", "eta_b0_max", "eta_b0_mean", "sandwich_gap", "expl_a3", "e_tv1", "e_eta_play", "bound_C1"]
print("\n== reported 30: max |new - old| per field ==")
for f in F:
    d = [abs(NEW[s][f] - OLD[s][f]) for s in rep if s in NEW and "error" not in NEW[s]]
    print(f"  {f:<13} max {max(d):.3e}  n={len(d)}")
dv = max(abs(NEW[s]["Vfull"][t] - OLD[s]["Vfull"][t]) for s in rep for t in OLD[s]["Vfull"])
print("  Vfull         max %.3e" % dv)
print("  beta_star identical:", sum(NEW[s]["beta_star"] == OLD[s]["beta_star"] for s in rep), "/", len(rep))
print("  leaf: non_d1 first tries total", sum(NEW[s]["leaf_stats"]["non_d1"] for s in NEW), "fallback total", sum(NEW[s]["leaf_stats"]["fallback"] for s in NEW),
      "det mismatches total", sum(NEW[s]["det_mismatch"] for s in NEW), "leaf calls", sum(NEW[s]["leaf_stats"]["calls"] for s in NEW))


def stats(S, tag):
    R = [NEW[s] for s in S if s in NEW and "error" not in NEW[s]]
    D = np.array([r["Delta"] for r in R]); n = len(R)
    print(f"\n== {tag}: n={n}, k dist {dict(zip(*np.unique([r['k'] for r in R], return_counts=True)))} ==")
    print("  Delta q[0,25,50,75,100]:", np.round(np.quantile(D, [0, .25, .5, .75, 1]), 1))
    for f in ["voi_total", "eta_NL0", "eta_b0_max", "e_eta_play", "e_eta_true", "sandwich_gap", "expl_a3"]:
        v = np.array([r[f] for r in R]); rel = v / D
        print(f"  {f:<13} med {np.median(v):7.2f} max {v.max():7.2f} | /D med {np.median(rel):.4f} max {rel.max():.4f} | >0.1%D {int((rel > 1e-3).sum())}/{n} >1%D {int((rel > 1e-2).sum())}/{n} | >1e-6D {int((rel > 1e-6).sum())}/{n}")
    print("  checks: VOI>=0", sum(r["voi_total"] >= -1e-7 * r["Delta"] for r in R), "expl>=0", sum(r["expl_a3"] >= -1e-7 * r["Delta"] for r in R),
          "sandwich", sum(r["sandwich_holds"] for r in R), "C1", sum(r["c1_holds"] for r in R), "expl<=etaNL (no TV)", sum(r["expl_le_etaNL"] for r in R),
          "along_b0", sum(r["along_b0_holds"] for r in R), "along_true", sum(r["along_true_holds"] for r in R),
          "along_true w/o TV", sum(r["expl_a3"] <= r["e_eta_true"] + 1e-7 * r["Delta"] for r in R),
          "along_b0 w/o TV", sum(r["expl_a3"] <= r["e_eta_play"] + 1e-7 * r["Delta"] for r in R), "/", n)
    print("  max |V0 primal - dual| / Delta: %.2e" % max(abs(r["V0_primal_minus_dual"]) / r["Delta"] for r in R))
    print("  E_TV1>0:", sum(r["e_tv1"] > 1e-9 for r in R), [(r["seed"], r["opp"], round(r["e_tv1"], 3), round(r["expl_a3"], 2)) for r in R if r["e_tv1"] > 1e-9])
    rat = sorted(r["expl_a3"] / r["bound_C1"] for r in R if r["expl_a3"] > 1e-9 and r["bound_C1"] > 1e-9)
    print("  expl/bound_C1 (expl>0):", [round(x, 4) for x in rat], "-> loose by >= %.1fx" % (1 / max(rat)))
    top = sorted(R, key=lambda r: -r["expl_a3"] / r["Delta"])[:4]
    for r in top:
        print(f"   top: {r['seed']} {r['opp']}({r['k']}) vs {r['own']} D {r['Delta']:.1f} VOI {r['voi_total']:.2f} etaNL {r['eta_NL0']:.2f} sand {r['sandwich_gap']:.2f} expl {r['expl_a3']:.2f} ({r['expl_a3']/r['Delta']:.4f}) ETV {r['e_tv1']:.3f} Eeta_b0 {r['e_eta_play']:.2f} Eeta_true {r['e_eta_true']:.2f}")
    v = np.array([r["voi_total"] for r in R]); e = np.array([r["eta_NL0"] for r in R]); x = np.array([r["expl_a3"] for r in R]); ep = np.array([r["e_eta_play"] for r in R])
    print("  corr VOI-etaNL %.2f VOI-expl %.2f etaNL-expl %.2f VOI-Eeta_b0 %.2f" % (np.corrcoef(v, e)[0, 1], np.corrcoef(v, x)[0, 1], np.corrcoef(e, x)[0, 1], np.corrcoef(v, ep)[0, 1]))


stats(rep, "reported 30 (det leaf)")
stats([s for s in drop], "dropped 16 (det leaf)")
stats(rep + drop, "all 46 (det leaf)")
