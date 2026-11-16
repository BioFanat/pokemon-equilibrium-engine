"""B4: E5 deficit decomposition. E5 (hedge vs agentform) has the Hedge records; E5t (top) / E5s (agentform self) are not Hedge methods
(no weights/experts logged) so are used only for the score split. Output out_B4_e5.json"""
import json, collections
import numpy as np
from oi_common import *
from common import EXPERTS, streams, records

def pair_scores(tag):
    G = game_jsons(tag); out = {}
    for s in range(1, 301):
        if (s, 1) in G and (s, 2) in G and not G[(s, 1)].get("error") and not G[(s, 2)].get("error"):
            out[s] = (score(G[(s, 1)]) + score(G[(s, 2)])) / 2
    return out
P = {t: pair_scores(t) for t in ("E5", "E5t", "E5s")}
res = {"scores": {}}
for t, p in P.items():
    x = np.array(list(p.values())); st = np.array([stream_of(s, 1) for s in p])
    res["scores"][t] = dict(A=GRID.get(t, ("hedge", "agentform"))[0], n=len(x), mean=float(x.mean()), normal=mean_ci_normal(x)[2:4], cluster=cluster_ci(x, st)[2:4])
def paired(t1, t2):
    c = sorted(set(P[t1]) & set(P[t2])); d = np.array([P[t1][s] - P[t2][s] for s in c]); st = np.array([stream_of(s, 1) for s in c])
    return dict(n=len(d), mean=float(d.mean()), normal=mean_ci_normal(d)[2:4], cluster=cluster_ci(d, st)[2:4])
res["diff_paired_by_seed"] = {"E5_minus_E5t": paired("E5", "E5t"), "E5_minus_E5s": paired("E5", "E5s"), "E5t_minus_E5s": paired("E5t", "E5s")}

# E5 hedge decisions
G = game_jsons("E5")
match = collections.Counter(); n_dec = 0
sub = collections.Counter(); n_sub = 0   # decisions where top != safest argmax
wsum = np.zeros(4); wn = 0
game_rows = []
for (seed, game), meta in sorted(G.items()):
    if meta.get("error"): continue
    R = decisions("E5", seed, game, "A")
    cnt = collections.Counter(); n = 0; wtop = []; frac_top_gt = 0
    for r in R:
        if r.get("final") or "experts" not in r or r["shape"][0] == 1: continue   # single legal move: all experts agree
        mv = r["move"][len("switch "):] if r["move"].startswith("switch ") else r["move"]
        ex = r["experts"]
        n += 1; wt = r["weights"]
        wsum += np.array([wt[e] for e in EXPERTS]); wn += 1
        wtop.append(wt["top"])
        for e in EXPERTS:
            if mv == ex[e]: cnt[e] += 1
        if ex["top"] != ex["safest"]:
            sub["n"] += 1
            sub["top"] += mv == ex["top"]; sub["safest"] += mv == ex["safest"]; sub["neither"] += (mv != ex["top"]) and (mv != ex["safest"])
            sub["sampled=played"] += mv == ex["sampled"]; sub["safex=played"] += mv == ex["safex"]
    game_rows.append(dict(seed=seed, game=game, z=score(meta), n=n, mean_wtop=float(np.mean(wtop)) if wtop else None,
                          **{f"frac_{e}": cnt[e] / n if n else None for e in EXPERTS}))
    for e in EXPERTS: match[e] += cnt[e]
    n_dec += n
res["E5_hedge_decisions"] = dict(n_multi_option_decisions=n_dec, frac_played_equals_expert_argmax={e: match[e] / n_dec for e in EXPERTS},
                                 mean_weights=dict(zip(EXPERTS, (wsum / wn).round(4).tolist())),
                                 top_ne_safest=dict(n=sub["n"], frac_of_decisions=sub["n"] / n_dec, played_eq_top=sub["top"] / sub["n"], played_eq_safest=sub["safest"] / sub["n"],
                                                    played_neither=sub["neither"] / sub["n"], played_eq_sampled_argmax=sub["sampled=played"] / sub["n"], played_eq_safex_argmax=sub["safex=played"] / sub["n"]))
# score split by decision share of top-like play: tercile of mean top weight and frac played == safest
gs = [g for g in game_rows if g["n"] > 0]
for key in ("mean_wtop", "frac_top", "frac_safest"):
    v = np.array([g[key] for g in gs]); z = np.array([g["z"] for g in gs])
    q = np.quantile(v, [1 / 3, 2 / 3])
    res[f"score_by_{key}_tercile"] = [dict(range=[float(lo), float(hi)], n=int(m.sum()), score=float(z[m].mean())) for lo, hi in ((-1, q[0]), (q[0], q[1]), (q[1], 2)) for m in [(v > lo) & (v <= hi)]]
    res[f"corr_score_{key}"] = float(np.corrcoef(v, z)[0, 1])
# weight trajectories per stream from the hedge state (final) and logged weights by phase
trs = []
for k, order in streams("E5"):
    W = []; Rr = []
    for seed, g, path in order:
        pend = None
        for rec in records(path):
            fb = rec.get("feedback")
            if fb and pend is not None and fb.get("for_turn") == pend[0] and not fb.get("skipped") and "r" in fb:
                Rr.append(fb["r"]); W.append(pend[1])
            if fb is not None or rec.get("final"): pend = None
            if isinstance(rec.get("weights"), dict): pend = (rec["turn"], [rec["weights"][e] for e in EXPERTS])
    W = np.array(W); Rr = np.array(Rr); T = len(W)
    seg = [W[int(T * a):int(T * b)].mean(0).round(3).tolist() for a, b in ((0, .1), (.1, .5), (.5, 1))]
    trs.append(dict(stream=k, T=T, mean_w_first10pct=seg[0], mean_w_10_50pct=seg[1], mean_w_last50pct=seg[2], final_w=W[-1].round(3).tolist(),
                    cum_reward=Rr.sum(0).round(1).tolist(), best_cum=EXPERTS[int(Rr.sum(0).argmax())], argmax_weight_share={e: float((W.argmax(1) == i).mean()) for i, e in enumerate(EXPERTS)}))
res["E5_weight_trajectories"] = trs
dump("out_B4_e5.json", res)
print(json.dumps(res["scores"], default=str)[:600]); print(res["diff_paired_by_seed"]); print(res["E5_hedge_decisions"]); print([ (t['best_cum'],t['mean_w_last50pct']) for t in trs])
for key in ("mean_wtop", "frac_top", "frac_safest"): print(key, res[f"corr_score_{key}"], res[f"score_by_{key}_tercile"])
