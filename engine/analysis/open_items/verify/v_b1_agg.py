import json, os, math, collections
import numpy as np
H = os.path.dirname(os.path.abspath(__file__))
res = json.load(open(os.path.join(H, "b1_rows.json")))
lg = lambda v: 2 / (1 + np.exp(-np.asarray(v, float) / 98)) - 1
th = lambda v: np.tanh(np.asarray(v, float) / 400)
chk = collections.Counter(); reasons = collections.Counter()
for g in res:
    chk.update(g["chk"])
    for r in g["rows"]: reasons[r["reason"]] += 1
print("checks", dict(chk)); print("reasons", dict(reasons))
ndec = sum(len(g["rows"]) for g in res); nk = reasons[None]; print("non-forced", ndec, "known", nk, round(nk / ndec, 4))
# agent-form x vs LP value
gap = np.array([r["v"] - r["afv"] for g in res for r in g["rows"]])
gap2 = np.array([r["v"] - r["minxA"] for g in res for r in g["rows"]])
print("v - min_j(Q1 A)_j: min %.4f median %.3f mean %.3f p90 %.3f frac>0.5 %.3f" % (gap.min(), np.median(gap), gap.mean(), np.quantile(gap, .9), (gap > .5).mean()))
print("v - min_j(pfull A)_j (=G^us engine units): median %.3f mean %.3f" % (np.median(gap2), gap2.mean()))
for name, phi in (("logit", lg), ("tanh", th)):
    allD, gameD, gameAbs, clus_g, clus_s, maxclose, wins = [], [], [], [], [], 0.0, []
    for g in res:
        rows = g["rows"]; vs = [r["v"] for r in rows] + [None]
        tot = 0.0; sD = 0.0; sA = 0.0
        for t, r in enumerate(rows):
            pnext = phi(rows[t + 1]["v"]) if t + 1 < len(rows) else g["z"]
            if r["reason"] is None:
                R = phi(r["v"]) - phi(r["xAj"]); S = phi(r["xAj"]) - phi(r["Aij"]); D = pnext - phi(r["Aij"])
                tot += -R - S + D
                allD.append(D); clus_g.append(f'{g["s"]}-{g["k"]}'); clus_s.append((g["s"] - 1) % 10); wins.append(g["z"] > 0)
                sD += D; sA += abs(D)
            else:
                tot += pnext - phi(r["v"])
        if rows:
            maxclose = max(maxclose, abs(g["z"] - phi(rows[0]["v"]) - tot))
        gameD.append(sD); gameAbs.append(sA)
    D = np.array(allD); cg = np.array(clus_g); cs = np.array(clus_s); W = np.array(wins)
    def clse(x, c):
        m = x.mean(); u = np.unique(c); G = len(u)
        s = np.array([(x[c == k] - m).sum() for k in u]); return math.sqrt(G / (G - 1) * (s ** 2).sum()) / len(x)
    print(name, "n", len(D), "E[D] %.4f SE iid %.4f game-cl %.4f stream-cl %.4f" % (D.mean(), D.std(ddof=1) / math.sqrt(len(D)), clse(D, cg), clse(D, cs)),
          "E|D| %.3f" % np.abs(D).mean(), "sum|D|/game %.2f" % np.mean(gameAbs), "sumD/game %.2f" % np.mean(gameD),
          "won %.4f lost %.4f" % (D[W].mean(), D[~W].mean()), "closure %.1e" % maxclose)
z = np.array([g["z"] for g in res]); v1 = np.array([lg(g["rows"][0]["v"]) for g in res if g["rows"]])
print("E z %.3f  E phi(v1) logit %.3f" % (z.mean(), v1.mean()))
