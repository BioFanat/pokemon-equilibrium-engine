"""B2: E3a/E3f pre/post scores, paired diff, flip delays, crossing, T5a analog ratio."""
import json, collections
import numpy as np
from vlib import games, pairs, decs, ci, cl_ci
E = ["top", "sampled", "safest", "safex"]
for tag in ("E3a", "E3f"):
    g = games(tag)
    act = collections.Counter((s <= 80, g[(s, k)]["active"] if g[(s, k)]["active"] else None) for (s, k) in g)
    pre = [1.0 * (v["win"] == "A") for (s, k), v in g.items() if not v["err"] and s <= 80]
    post = [1.0 * (v["win"] == "A") for (s, k), v in g.items() if not v["err"] and s > 80]
    print(tag, "active field sample", list(act.items())[:4])
    print(tag, "pre", ci(pre), "post", ci(post))
    flips, ratios, crossings = [], [], []
    for st in range(10):
        seeds = sorted(s for s in range(1, 161) if (s - 1) % 10 == st)
        R, Wt, phase = [], [], []
        for s in seeds:
            for k in (1, 2):
                rs = decs(tag, s, k) or []
                for r in rs:
                    fb = r.get("feedback")
                    if fb and not fb.get("skipped"):
                        R.append(fb["r"]); Wt.append(fb["weights"]); phase.append(s > 80)
        R = np.array(R); Wt = np.array(Wt); phase = np.array(phase)
        t0 = int((~phase).sum())
        a = int(R[~phase].sum(0).argmax()); b = int(R[phase].sum(0).argmax())
        Wp = Wt[phase]
        mx = np.flatnonzero(Wp.argmax(1) == b)
        flip = int(mx[0]) + 1 if len(mx) else None
        ca = np.cumsum(R[:, a]); cb = np.cumsum(R[:, b])
        cr = np.flatnonzero((cb > ca) & phase)
        cross = int(cr[0]) - t0 + 1 if len(cr) else None
        mu = lambda ph, e: R[ph][:, e].mean()
        den = mu(phase, b) - mu(phase, a)
        tau = t0 * (mu(~phase, a) - mu(~phase, b)) / den if den > 0 else None
        flips.append(flip); crossings.append(cross)
        if cross is not None and tau: ratios.append(cross / tau)
        print(f"  st{st} t0 {t0} post {int(phase.sum())} {E[a]}->{E[b]} flip {flip} cross {cross} tau {None if tau is None else round(tau)}")
    f = [x for x in flips if x is not None]
    print(tag, "flips reached", len(f), "/10 median", np.median(f), "range", min(f), max(f), "ratio median", round(float(np.median(ratios)), 3), "over", len(ratios))
pa, pf = pairs("E3a"), pairs("E3f")
c = sorted(set(pa) & set(pf)); d = [pf[s] - pa[s] for s in c]
print("hedge_fs - hedge", ci(d), cl_ci(d, [(s - 1) % 10 for s in c]))
