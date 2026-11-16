"""T3 on RQ runs: BC_t(s) from logged q_{s,t} (decision record bayes.q, 4dp) at non-skipped feedback turns."""
import sys, json, math
import numpy as np
sys.path.insert(0, "/Users/biofanat/Documents/classes/f26/cs535/final/engine/analysis/open_items/verify")
from vlib import decs
S = ["eq", "br_unif", "br_us", "safe", "uniform"]
TRUE = {"RQeq": "eq", "RQsafe": "safe", "RQunif": "uniform"}
out = {}
def stream_turns(tag, st):
    rows = []; bad = 0
    for s in sorted(s for s in range(1, 101) if (s - 1) % 10 == st):
        for k in (1, 2):
            pend = None
            for r in decs(tag, s, k) or []:
                fb = r.get("feedback")
                if fb and not fb.get("skipped") and pend is not None and pend["turn"] == fb["for_turn"]:
                    q = np.array([pend["q"][x] for x in S]); q /= q.sum(1, keepdims=True)
                    j = fb["bayes"]["j"]
                    ps = np.array([fb["bayes"]["p_style"][x] for x in S])
                    if np.abs(q[:, j] - ps).max() > 6e-4: bad += 1
                    rows.append((q, j, ps, fb["bayes"]["forced"], np.array(fb["bayes"]["posterior"] and [fb["bayes"]["posterior"][x] for x in S])))
                elif fb and not fb.get("skipped"): bad += 1000000
                if "bayes" in r and "q" in r["bayes"]: pend = dict(turn=r["turn"], q=r["bayes"]["q"])
    return rows, bad
for tag, true in TRUE.items():
    ti = S.index(true); per = []
    for st in range(10):
        rows, bad = stream_turns(tag, st)
        PS = np.array([r[2] for r in rows]); L = np.log(PS); forced = np.array([r[3] for r in rows])
        logp = np.log(0.2) + np.cumsum(L, 0); logp -= logp.max(1, keepdims=True)
        P = np.exp(logp); P /= P.sum(1, keepdims=True)
        Pprev = np.vstack([np.full(5, 0.2), P[:-1]])
        lhs = float((1 - Pprev[:, ti]).sum())
        BC = np.array([[np.sqrt(q[s] * q[ti]).sum() for s in range(5)] for q, *_ in rows])
        BCc = np.minimum(BC, 1.0)
        prod = np.cumprod(BCc, 0)
        others = [i for i in range(5) if i != ti]
        rp = float(sum(np.minimum(1, prod[:, i]).sum() for i in others))
        nf = ~forced
        ent = np.array([-(q[ti] * np.log(q[ti])).sum() for q, *_ in rows])
        kl = np.array([[(q[ti] * np.log(q[ti] / q[s])).sum() for s in range(5)] for q, *_ in rows])
        per.append(dict(st=st, T=len(rows), nf=int(nf.sum()), bad=bad, lhs=lhs,
            meanBC={S[i]: float(BCc[:, i].mean()) for i in others}, supBC={S[i]: float(BCc[:, i].max()) for i in others},
            meanBC_nf={S[i]: float(BCc[nf, i].mean()) for i in others}, supBC_nf={S[i]: float(BCc[nf, i].max()) for i in others},
            frac99={S[i]: float((BCc[:, i] > .99).mean()) for i in others}, frac99_nf={S[i]: float((BCc[nf, i] > .99).mean()) for i in others},
            realprod=rp,
            ll={S[i]: float(-L[:, i].mean()) for i in range(5)}, ll_nf={S[i]: float(-L[nf, i].mean()) for i in range(5)},
            sumL={S[i]: float(L[:, i].sum()) for i in range(5)},
            ent_true_nf=float(ent[nf].mean()), kl_nf={S[i]: float(kl[nf, i].mean()) for i in others},
            llr_nf={S[i]: float((L[nf, ti] - L[nf, i]).mean()) for i in others},
            llr_all={S[i]: float((L[:, ti] - L[:, i]).mean()) for i in others},
            frac_true_beaten_nf={S[i]: float((L[nf, i] > L[nf, ti]).mean()) for i in others},
            maxBC_over_t=float(BCc[:, others].max())))
    out[tag] = per
    avg = lambda f: float(np.mean([f(p) for p in per]))
    print("=====", tag, "true", true, "bad", sum(p["bad"] for p in per), "T", [p["T"] for p in per])
    print("LHS per stream", [round(p["lhs"], 1) for p in per], "mean", round(avg(lambda p: p["lhs"]), 2))
    for i in others:
        s = S[i]
        mb = avg(lambda p: p["meanBC"][s]); sb = avg(lambda p: p["supBC"][s])
        print(f"{s:8s} meanBC {mb:.4f} supBC(avg over streams) {sb:.4f} frac>.99 {avg(lambda p: p['frac99'][s]):.3f} | nf: meanBC {avg(lambda p: p['meanBC_nf'][s]):.4f} sup {avg(lambda p: p['supBC_nf'][s]):.4f} frac99 {avg(lambda p: p['frac99_nf'][s]):.3f}")
    print("realprod bound mean", round(avg(lambda p: p["realprod"]), 2))
    print("meanll", {s: round(avg(lambda p: p["ll"][s]), 4) for s in S}, " nonforced", {s: round(avg(lambda p: p["ll_nf"][s]), 4) for s in S})
    print("true entropy nf", round(avg(lambda p: p["ent_true_nf"]), 4), "KL(q*||q_s) nf", {S[i]: round(avg(lambda p: p["kl_nf"][S[i]]), 4) for i in others})
    print("LLR true-vs-s per nf turn", {S[i]: round(avg(lambda p: p["llr_nf"][S[i]]), 4) for i in others}, "all", {S[i]: round(avg(lambda p: p["llr_all"][S[i]]), 4) for i in others})
    print("frac nf turns s beats true", {S[i]: round(avg(lambda p: p["frac_true_beaten_nf"][S[i]]), 3) for i in others})
    print("best style by cum loglik per stream", [S[int(np.argmax([p["sumL"][s] for s in S]))] for p in per])
json.dump(out, open("t3_out.json", "w"), indent=1)
