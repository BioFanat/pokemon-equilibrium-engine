"""B5: bayes_safex eps sweep (SX1/SX3/SX10/SX30 = eps 0.001/0.003/0.01/0.03) vs stock. Output out_B5_eps.json"""
import json
import numpy as np
from scipy import stats, optimize
from oi_common import *

def games(tag):
    G = game_jsons(tag); rows = []
    for (s, g), d in sorted(G.items()):
        if d.get("error"): continue
        rows.append((s, g, stream_of(s, 1), score(d)))
    return rows
def pair_scores(tag, hi=200):
    G = game_jsons(tag); out = {}
    for s in range(1, hi + 1):
        if (s, 1) in G and (s, 2) in G and not G[(s, 1)].get("error") and not G[(s, 2)].get("error"):
            out[s] = (score(G[(s, 1)]) + score(G[(s, 2)])) / 2
    return out
res = {"eps": SX_EPS}
allg = []; tab = {}
for tag, eps in SX_EPS.items():
    rows = games(tag); x = np.array([r[3] for r in rows]); st = np.array([r[2] for r in rows])
    ps = pair_scores(tag); pv = np.array(list(ps.values()))
    allg += [(eps, tag, r[0], r[1], r[2], r[3]) for r in rows]
    tab[tag] = dict(eps=eps, n_games=len(x), game_mean=float(x.mean()), game_normal=mean_ci_normal(x)[2:4], game_cluster=cluster_ci(x, st)[2:4],
                    n_pairs=len(pv), pair_mean=float(pv.mean()), pair_ci_normal=mean_ci_normal(pv)[2:4],
                    pair_ci_cluster=cluster_ci(pv, np.array([stream_of(s, 1) for s in ps]))[2:4], wins=int((x == 1).sum()), losses=int((x == 0).sum()))
res["table"] = tab
# comparison with agentform 0.314 (AF/AF_summary.json: agentform vs stock, seeds 1-300) and AF restricted to 1-200
af = pair_scores("AF", 300); af200 = {s: v for s, v in af.items() if s <= 200}
S = summary("AF")
res["agentform_ref"] = dict(source="engine/showdown/results/AF/AF_summary.json", summary_mean=S["pair_score_A"]["mean"], ci95=S["pair_score_A"]["ci95"], n_pairs=S["n_pairs"],
                            mean_seeds_1_200=float(np.mean(list(af200.values()))))
for tag in SX_EPS:
    ps = pair_scores(tag); c = sorted(set(ps) & set(af200)); d = np.array([ps[s] - af200[s] for s in c])
    res["table"][tag]["minus_agentform_paired_seeds1_200"] = dict(mean=float(d.mean()), normal=mean_ci_normal(d)[2:4], cluster=cluster_ci(d, np.array([stream_of(s, 1) for s in c]))[2:4], n=len(d))
    res["table"][tag]["minus_agentform_0.314_unpaired"] = float(res["table"][tag]["pair_mean"] - S["pair_score_A"]["mean"])
# other reference rows for context
for ref in ("BH", "E2", "AF16"):
    s = summary(ref); res.setdefault("context", {})[ref] = dict(A=s["A"], mean=s["pair_score_A"]["mean"], ci95=s["pair_score_A"]["ci95"], n_pairs=s["n_pairs"])
# logistic regression win ~ a + b ln(eps), clustered by (tag,stream)
eps = np.array([a[0] for a in allg]); y = np.array([a[5] for a in allg]); cl = np.array([f"{a[1]}_{a[4]}" for a in allg])
X = np.column_stack([np.ones(len(y)), np.log(eps)])
def nll(b):
    z = X @ b; return np.sum(np.logaddexp(0, z) - y * z)
def fit(X, y):
    b = optimize.minimize(lambda b: np.sum(np.logaddexp(0, X @ b) - y * (X @ b)), np.zeros(X.shape[1]), method="BFGS").x
    p = 1 / (1 + np.exp(-X @ b)); H = X.T @ (X * (p * (1 - p))[:, None]); Hi = np.linalg.inv(H)
    sc = X * (y - p)[:, None]; ucl = np.unique(cl); B = sum(np.outer(sc[cl == u].sum(0), sc[cl == u].sum(0)) for u in ucl)
    G = len(ucl); V = G / (G - 1) * Hi @ B @ Hi
    return b, np.sqrt(np.diag(Hi)), np.sqrt(np.diag(V)), G
b, se_m, se_c, G = fit(X, y)
res["logistic_game_level"] = dict(n=len(y), clusters=int(G), intercept=float(b[0]), slope_per_ln_eps=float(b[1]), se_model=float(se_m[1]), se_cluster=float(se_c[1]),
                                  z_model=float(b[1] / se_m[1]), z_cluster=float(b[1] / se_c[1]), p_cluster_t=float(2 * stats.t.sf(abs(b[1] / se_c[1]), G - 1)),
                                  odds_ratio_per_tenfold_eps=float(np.exp(b[1] * np.log(10))),
                                  ci95_slope_cluster=[float(b[1] - stats.t.ppf(.975, G - 1) * se_c[1]), float(b[1] + stats.t.ppf(.975, G - 1) * se_c[1])],
                                  predicted_score={str(e): float(1 / (1 + np.exp(-(b[0] + b[1] * np.log(e))))) for e in SX_EPS.values()})
# categorical (null) model likelihood ratio for linearity in ln eps
Xc = np.column_stack([np.ones(len(y))] + [(eps == e).astype(float) for e in list(SX_EPS.values())[1:]])
bc = optimize.minimize(lambda b: np.sum(np.logaddexp(0, Xc @ b) - y * (Xc @ b)), np.zeros(4), method="BFGS").x
lr = 2 * (nll(b) - np.sum(np.logaddexp(0, Xc @ bc) - y * (Xc @ bc)))
res["logistic_game_level"]["LR_linear_vs_categorical_chi2_df2"] = float(lr); res["logistic_game_level"]["LR_p"] = float(stats.chi2.sf(lr, 2))
# pair-level linear regression of seed-pair score on ln eps (cluster by tag-stream)
pr = []
for tag, e in SX_EPS.items():
    for s, v in pair_scores(tag).items(): pr.append((np.log(e), v, f"{tag}_{stream_of(s,1)}"))
xl = np.array([p[0] for p in pr]); yl = np.array([p[1] for p in pr]); cl2 = np.array([p[2] for p in pr])
Xl = np.column_stack([np.ones(len(xl)), xl]); bl = np.linalg.lstsq(Xl, yl, rcond=None)[0]; e_ = yl - Xl @ bl
Hi = np.linalg.inv(Xl.T @ Xl); u2 = np.unique(cl2); Bm = sum(np.outer((Xl * e_[:, None])[cl2 == u].sum(0), (Xl * e_[:, None])[cl2 == u].sum(0)) for u in u2)
Vl = len(u2) / (len(u2) - 1) * Hi @ Bm @ Hi; se_l = np.sqrt(Vl[1, 1])
res["linear_pair_level"] = dict(n=len(yl), slope_per_ln_eps=float(bl[1]), se_cluster=float(se_l), t=float(bl[1] / se_l), p=float(2 * stats.t.sf(abs(bl[1] / se_l), len(u2) - 1)),
                                ci95=[float(bl[1] - stats.t.ppf(.975, len(u2) - 1) * se_l), float(bl[1] + stats.t.ppf(.975, len(u2) - 1) * se_l)], per_tenfold=float(bl[1] * np.log(10)))
# paired seed differences between neighbouring eps
pp = {t: pair_scores(t) for t in SX_EPS}
res["paired_adjacent"] = {}
for a, bb in (("SX1", "SX3"), ("SX3", "SX10"), ("SX10", "SX30"), ("SX1", "SX30")):
    c = sorted(set(pp[a]) & set(pp[bb])); d = np.array([pp[bb][s] - pp[a][s] for s in c])
    res["paired_adjacent"][f"{bb}_minus_{a}"] = dict(mean=float(d.mean()), normal=mean_ci_normal(d)[2:4], cluster=cluster_ci(d, np.array([stream_of(s, 1) for s in c]))[2:4], n=len(d))
# Spearman / Jonckheere-like monotone test via Kendall tau of game outcome vs eps
tau, pk = stats.kendalltau(eps, y); res["kendall_tau_game_vs_eps"] = dict(tau=float(tau), p_naive=float(pk))
dump("out_B5_eps.json", res)
for t, v in tab.items(): print(t, v["eps"], round(v["pair_mean"], 4), [round(a, 4) for a in v["pair_ci_normal"]], [round(a, 4) for a in v["pair_ci_cluster"]], v["minus_agentform_paired_seeds1_200"] if "minus_agentform_paired_seeds1_200" in v else res["table"][t]["minus_agentform_paired_seeds1_200"])
print(res["logistic_game_level"]); print(res["linear_pair_level"]); print(res["paired_adjacent"]); print(res["agentform_ref"], res["context"])
