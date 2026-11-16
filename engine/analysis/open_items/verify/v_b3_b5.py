"""B3 gains vs E4 baselines; B4 score split; B5 SX vs AF, logistic slope with CR1 clustered SE."""
import math
import numpy as np
from scipy import stats
from vlib import pairs, games, ci, cl_ci

print("== B3")
for x, e4 in [("XU", "E4u"), ("XUmv", "E4umv"), ("XEmv", "E4emv"), ("XR", None)]:
    px = pairs(x)
    print(x, "score", ci(list(px.values())))
    if e4 is None:
        d = [v - 0.5 for v in px.values()]; print("  vs 0.5", ci(d)); continue
    pb = {s: 1 - v for s, v in pairs(e4, "A").items()}   # RM+ is B in E4
    common = sorted(set(px) & set(pb))
    print("  baseline RM+ mean (all)", round(np.mean(list(pb.values())), 4), len(pb), "common", round(np.mean([pb[s] for s in common]), 4), len(common))
    d = [px[s] - pb[s] for s in common]
    print("  gain paired normal", ci(d), "cluster", cl_ci(d, [(s - 1) % 10 for s in common]))

print("== B4")
p5, pt, ps = pairs("E5"), pairs("E5t"), pairs("E5s")
for n, p in [("E5", p5), ("E5t", pt), ("E5s", ps)]: print(n, ci(list(p.values())))
for a, b, n in [(p5, pt, "E5-E5t"), (p5, ps, "E5-E5s"), (pt, ps, "E5t-E5s")]:
    c = sorted(set(a) & set(b)); d = [a[s] - b[s] for s in c]; print(n, ci(d), cl_ci(d, [(s - 1) % 10 for s in c]))

print("== B5")
af = pairs("AF")
print("AF all", ci(list(af.values())), "AF seeds<=200", ci([v for s, v in af.items() if s <= 200]))
EPS = {"SX1": 0.001, "SX3": 0.003, "SX10": 0.01, "SX30": 0.03}
X, Y, G = [], [], []
for t, e in EPS.items():
    p = pairs(t); c = sorted(set(p) & set(af)); d = [p[s] - af[s] for s in c]
    print(t, ci(list(p.values())), "minus AF paired", ci(d), cl_ci(d, [(s - 1) % 10 for s in c]))
    for (s, k), g in games(t).items():
        if g["err"]: continue
        X.append(math.log(e)); Y.append(1.0 if g["win"] == "A" else 0.0); G.append(f"{t}-{(s - 1) % 10}")
X = np.asarray(X); Y = np.asarray(Y); G = np.asarray(G)
Z = np.column_stack([np.ones_like(X), X]); b = np.zeros(2)
for _ in range(50):
    mu = 1 / (1 + np.exp(-Z @ b)); W = mu * (1 - mu)
    b = b + np.linalg.solve(Z.T @ (Z * W[:, None]), Z.T @ (Y - mu))
mu = 1 / (1 + np.exp(-Z @ b)); W = mu * (1 - mu)
Hinv = np.linalg.inv(Z.T @ (Z * W[:, None]))
ug = np.unique(G); Gn = len(ug); n = len(Y)
meat = sum(np.outer(Z[G == u].T @ (Y - mu)[G == u], Z[G == u].T @ (Y - mu)[G == u]) for u in ug)
c1 = Gn / (Gn - 1) * (n - 1) / (n - 2)
Vcl = c1 * Hinv @ meat @ Hinv
Vcl0 = Gn / (Gn - 1) * Hinv @ meat @ Hinv
se_m = math.sqrt(Hinv[1, 1]); se_c = math.sqrt(Vcl[1, 1]); se_c0 = math.sqrt(Vcl0[1, 1])
print("n", n, "clusters", Gn, "slope", round(b[1], 4), "SE model", round(se_m, 4), "SE CR1(stata)", round(se_c, 4), "SE G/(G-1)", round(se_c0, 4))
for se in (se_c, se_c0):
    z = b[1] / se; print("  z", round(z, 3), "p normal", round(2 * stats.norm.sf(abs(z)), 4), "p t(G-1)", round(2 * stats.t.sf(abs(z), Gn - 1), 4))
# streams-only clustering (10 clusters, seeds share teams across eps)
G2 = np.array([g.split("-")[1] for g in G]); ug2 = np.unique(G2)
meat2 = sum(np.outer(Z[G2 == u].T @ (Y - mu)[G2 == u], Z[G2 == u].T @ (Y - mu)[G2 == u]) for u in ug2)
V2 = 10 / 9 * Hinv @ meat2 @ Hinv; se2 = math.sqrt(V2[1, 1])
print("  cluster by stream only (10):", round(se2, 4), "p t(9)", round(2 * stats.t.sf(abs(b[1] / se2), 9), 4))
