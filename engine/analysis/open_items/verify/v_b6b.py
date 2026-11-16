"""P1 rounding bound and P2(e) E ln n_t (all vs non-forced)."""
import math, numpy as np
from vlib import decs
S = ["eq", "br_unif", "br_us", "safe", "uniform"]
for tag in ("RQeq", "RQsafe", "RQunif"):
    worst = []; lnn = []; lnn_nf = []; excess = []
    for st in range(10):
        seeds = sorted(s for s in range(1, 101) if (s - 1) % 10 == st)
        L, Lq, bound = [], [], []
        for s in seeds:
            for k in (1, 2):
                for r in decs(tag, s, k) or []:
                    fb = r.get("feedback")
                    if not fb or fb.get("skipped"): continue
                    b = fb["bayes"]; L.append([b["p_style"][x] for x in S]); Lq.append(b["p_qhat"])
                    lnn.append(math.log(b["n_cols"]));
                    if not b["forced"]: lnn_nf.append(math.log(b["n_cols"]))
        L = np.array(L); Lq = np.array(Lq)
        reg = np.cumsum(np.log(L), 0) - np.cumsum(np.log(Lq))[:, None]
        bnd = np.cumsum(5e-7 / Lq)[:, None] + np.cumsum(5e-7 / L, 0)   # worst-case 6-dp rounding effect on the regret
        excess.append(float((reg - math.log(5) - bnd).max()))
        worst.append(float((reg - math.log(5)).max()))
    print(tag, "max excess over ln5 %.2e" % max(worst), "max (excess - worst-case rounding bound) %.2e" % max(excess),
          "E ln n all %.4f non-forced %.4f" % (np.mean(lnn), np.mean(lnn_nf)), "frac forced %.3f" % (1 - len(lnn_nf) / len(lnn)))
