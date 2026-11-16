import json, numpy as np
o = json.load(open("t3_out.json"))
def inv(x): return float("inf") if x >= 1 - 1e-12 else 1 / (1 - x)
rows = []
for tag, per in o.items():
    iid = [sum(inv(v) for v in p["meanBC"].values()) for p in per]       # all turns (forced BC=1 -> inf?) 
    iidnf = [sum(inv(v) for v in p["meanBC_nf"].values()) for p in per]
    supnf = [sum(inv(v) for v in p["supBC_nf"].values()) for p in per]
    supall = [sum(inv(v) for v in p["supBC"].values()) for p in per]
    lhs = [p["lhs"] for p in per]; rp = [p["realprod"] for p in per]
    print(tag, "LHS", np.round(lhs, 1).tolist())
    print("  iid-all", np.round(iid, 1).tolist(), "\n  iid-nf mean", round(np.mean(iidnf), 2), np.round(iidnf, 1).tolist())
    print("  sup-nf mean", round(np.mean(supnf), 1), "sup-all", supall[:2])
    print("  realprod", np.round(rp, 1).tolist(), "mean", round(np.mean(rp), 2))
    print("  holds per stream: iid-nf", sum(l <= b for l, b in zip(lhs, iidnf)), "sup-nf", sum(l <= b for l, b in zip(lhs, supnf)), "realprod", sum(l <= b for l, b in zip(lhs, rp)), "of 10")
    print("  mean-of-bound iid-nf vs mean LHS", round(np.mean(iidnf), 2), round(np.mean(lhs), 2))
