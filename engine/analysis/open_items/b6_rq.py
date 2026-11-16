"""B6: score pre-registered predictions P1-P5 (theory/tightening/bayes_dynamics.md section 4) on RQeq / RQsafe / RQunif. Output out_B6_rq.json"""
import json, math
import numpy as np
from oi_common import *
from common import STYLES, streams, records
LN5 = math.log(5)
TRUE = {"RQeq": "eq", "RQsafe": "safe", "RQunif": "uniform"}
res = {}
def stream_traj(tag, order):
    pq, ps, post, forced, ncol = [], [], [], [], []
    for seed, g, p in order:
        for rec in records(p):
            fb = rec.get("feedback")
            if not fb or fb.get("skipped") or "bayes" not in fb: continue
            b = fb["bayes"]
            pq.append(b["p_qhat"]); ps.append([b["p_style"][s] for s in STYLES]); post.append([b["posterior"][s] for s in STYLES]); forced.append(b["forced"]); ncol.append(b["n_cols"])
    return np.array(pq), np.array(ps), np.array(post), np.array(forced), np.array(ncol)
for tag, star in TRUE.items():
    si = STYLES.index(star)
    rows = []
    G = game_jsons(tag)
    for k, order in streams(tag):
        pq, ps, post, forced, ncol = stream_traj(tag, order)
        T = len(pq)
        cq = np.cumsum(-np.log(np.maximum(pq, 1e-9))); cs = np.cumsum(-np.log(np.maximum(ps, 1e-9)), axis=0)
        reg = cq[:, None] - cs                       # regret of qhat vs each style
        best = int(cs[-1].argmin())
        meta = json.load(open(os.path.join(RES, tag, "hedge_state", f"{tag}-stream{k}-A.bayes.json")))
        pstar = post[:, si]; prev = np.concatenate([[0.2], pstar[:-1]])
        mapx = post.argmax(1)
        # first turn P(s*) >= thr and stays
        def sustained(thr):
            ok = pstar >= thr
            for t in range(T):
                if ok[t:].all(): return t + 1
            return None
        rows.append(dict(stream=k, T=T, meta_updates=meta["meta"]["updates"], meta_games=meta["meta"]["games"], clean=(T == meta["meta"]["updates"] and meta["meta"]["games"] == 20),
                         final_post=dict(zip(STYLES, post[-1].round(4).tolist())), final_logged_state_post=meta["posterior"],
                         final_map=STYLES[int(mapx[-1])], best_cumloss_style=STYLES[best], P_T_star=float(pstar[-1]), P_T_star_at_25=float(pstar[24]) if T > 24 else None,
                         P_T_star_at_50=float(pstar[49]) if T > 49 else None, P_T_star_at_100=float(pstar[99]) if T > 99 else None,
                         P_T_star_at_600=float(pstar[599]) if T > 599 else None,
                         regret_vs_best=float(cq[-1] - cs[-1].min()), ln5_plus_lnPT_best=float(LN5 + math.log(max(post[-1][best], 1e-9))),
                         anytime_max_regret_vs_any_style=float(reg.max()),
                         cumloss_style={s: float(v) for s, v in zip(STYLES, cs[-1])}, cumloss_qhat=float(cq[-1]),
                         mean_loss_style={s: float(v / T) for s, v in zip(STYLES, cs[-1])}, mean_ln_ncols=float(np.log(ncol).mean()),
                         map_wrong_turns=int((mapx != si).sum()), map_wrong_turns_after50=int((mapx[50:] != si).sum()),
                         sum_1_minus_Pprev=float((1 - prev).sum()), t_sustained_P90=sustained(0.9), t_sustained_P99=sustained(0.99),
                         max_abs_identity_err=float(max(abs((cq[t] - cs[t, i]) - (LN5 + math.log(post[t, i]))) for t in range(T) for i in range(5) if post[t, i] >= 0.01)),
                         runner_up_gap_nats=float(np.sort(cs[-1])[1] - np.sort(cs[-1])[0]),
                         gap_to_best_other=float(np.min(np.delete(cs[-1], si)) - cs[-1][si]),
                         n_forced_updates=int(forced.sum())))
    res[tag] = dict(star=star, streams=rows)
    print(tag, [(r["final_map"], round(r["P_T_star"], 3), r["map_wrong_turns"], round(r["regret_vs_best"], 3), round(r["anytime_max_regret_vs_any_style"], 3)) for r in rows])
dump("out_B6_rq.json", res)
