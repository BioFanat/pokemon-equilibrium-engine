"""B6: rebuild the per-stream posterior from p_style (own Bayes recursion), score P1-P5 against the section 4 text."""
import json, math, os, glob
import numpy as np
from vlib import decs, pairs, ci, RES
S = ["eq", "br_unif", "br_us", "safe", "uniform"]
TRUE = {"RQeq": "eq", "RQsafe": "safe", "RQunif": "uniform"}
out = {}
for tag, true in TRUE.items():
    ti = S.index(true); res = []
    for st in range(10):
        seeds = sorted(s for s in range(1, 101) if (s - 1) % 10 == st)
        L, Lq, Plog, forced, nskip, nfb = [], [], [], [], 0, 0
        for s in seeds:
            for k in (1, 2):
                for r in decs(tag, s, k) or []:
                    fb = r.get("feedback")
                    if not fb: continue
                    nfb += 1
                    if fb.get("skipped"): nskip += 1; continue
                    b = fb["bayes"]
                    L.append([b["p_style"][x] for x in S]); Lq.append(b["p_qhat"]); forced.append(b["forced"])
                    Plog.append([b["posterior"][x] for x in S])
        L = np.log(np.array(L)); Lq = np.log(np.array(Lq)); Plog = np.array(Plog); forced = np.array(forced)
        logp = np.log(0.2) + np.cumsum(L, 0); logp -= logp.max(1, keepdims=True)
        P = np.exp(logp); P /= P.sum(1, keepdims=True)
        st_file = json.load(open(os.path.join(RES, tag, "hedge_state", f"{tag}-stream{st}-A.bayes.json")))
        regret = np.cumsum(L, 0) - np.cumsum(Lq)[:, None]          # loss(qhat) - loss(s), anytime
        T = len(Lq)
        mapwrong = (P.argmax(1) != ti)
        p50 = P[49, ti]; reach50 = bool((P[:50, ti] >= 0.99).any())
        hold = np.flatnonzero(P[:, ti] < 0.9); persist = int(hold[-1]) + 2 if len(hold) else 1
        loss = -L.mean(0)
        gap_su = (L[:, S.index("safe")] - L[:, S.index("br_unif")])
        gap_su_nf = gap_su[~forced]
        finL = -L.sum(0)
        others = [finL[i] for i in range(5) if i != S.index("safe")]
        res.append(dict(st=st, T=T, state_upd=st_file["meta"]["updates"], state_games=st_file["meta"]["games"],
                        maxPdiff=float(np.abs(P - Plog).max()), PT=P[-1].round(4).tolist(), map=S[int(P[-1].argmax())],
                        reg_anytime_max=float(regret.max()), reg_final=regret[-1].round(5).tolist(),
                        reg_final_true=float(regret[-1, ti]), ln5_lnPT=float(math.log(5) + math.log(P[-1, ti]) if P[-1, ti] > 0 else -1e9),
                        p50=float(p50), reach99_by50=reach50, P100=float(P[99, ti]), mapwrong=int(mapwrong.sum()),
                        mapwrong_after50=int(mapwrong[50:].sum()), persist90=persist, sum1mP=float((1 - np.concatenate([[0.2], P[:-1, ti]])).sum()),
                        loss=loss.round(4).tolist(), meanlnn=None, gap_su=float(gap_su.mean()), gap_su_nf=float(gap_su_nf.mean()),
                        gap_su_sum=float(gap_su.sum()), n=int(len(gap_su)), n_nf=int((~forced).sum()), final_gap_safe=float(min(others) - finL[S.index("safe")]),
                        best_style=S[int(finL.argmin())], gap_sus=float((L[:, S.index("safe")] - L[:, S.index("br_us")]).mean())))
    out[tag] = res
    print("====", tag, "score", ci(list(pairs(tag).values())))
    for r in res:
        print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items() if k not in ("loss",)})
    regs = [r["reg_anytime_max"] for r in res]
    print("anytime max regret %.6f (ln5 %.6f)" % (max(regs), math.log(5)),
          "final true-regret 3dp == 1.609:", sum(round(r["reg_final_true"], 3) == 1.609 for r in res if r["PT"][ti] >= 0.999), "of", sum(r["PT"][ti] >= 0.999 for r in res))
    print("best style per stream:", [r["best_style"] for r in res])
    print("mean loss per style:", dict(zip(S, np.mean([r["loss"] for r in res], 0).round(4))))
    if tag == "RQsafe":
        print("gap safe-br_unif: pooled", round(sum(r["gap_su_sum"] for r in res) / sum(r["n"] for r in res), 4),
              "mean of streams", round(np.mean([r["gap_su"] for r in res]), 4),
              "non-forced pooled", round(sum(r["gap_su_nf"] * r["n_nf"] for r in res) / sum(r["n_nf"] for r in res), 4),
              "safe-br_us mean", round(np.mean([r["gap_sus"] for r in res]), 4))
json.dump(out, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "b6_out.json"), "w"), indent=1)
