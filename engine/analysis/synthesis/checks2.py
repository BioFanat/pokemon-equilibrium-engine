import json, math, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import RES, game_files, records, streams, STYLES
# 1. E4 gap with the summary's filter (all records with a gap)
for tag in ("E4e","E4u"):
    g_all, g_two = [], []
    for (s,g),p in sorted(game_files(tag,"A").items()):
        for rec in records(p):
            if "gap" in rec and not rec.get("final"):
                g_all.append(rec["gap"]); sh=rec.get("abar_shape") or rec.get("shape")
                if sh and min(sh)>=2: g_two.append(rec["gap"])
    g_all=np.array(g_all); g_two=np.array(g_two)
    print(tag, "all gap records", len(g_all), "median", round(float(np.median(g_all)),4), "mean", round(float(g_all.mean()),4),
          "| gap>0 records", int((g_all>0).sum()), "median of gap>0", round(float(np.median(g_all[g_all>0])),4),
          "| two-sided", len(g_two), round(float(np.median(g_two)),4))
    summ=json.load(open(os.path.join(RES, tag, f"{tag}_summary.json")))
    print("   summary learning:", {k:v for k,v in summ.get("learning",{}).items() if "gap" in k or k in ("decisions",)})
# 2. BH log loss excluding forced turns, clean streams only
tot=dict(q=[],best=[],old=[],unif=[],n=0,nf=0); per=[]
for k,order in streams("BH"):
    if k in (0,6): continue
    lq,ls,lo,lu=[],[],[],[]
    for s,g,p in order:
        for rec in records(p):
            fb=rec.get("feedback")
            if not fb or fb.get("skipped") or "bayes" not in fb: continue
            b=fb["bayes"]; tot["n"]+=1
            if b.get("forced") or b["n_cols"]<=1: tot["nf"]+=1; continue
            lq.append(-math.log(max(b["p_qhat"],1e-9))); ls.append([-math.log(max(b["p_style"][s_],1e-9)) for s_ in STYLES])
            lo.append(-math.log(max(b["p_old"],1e-9))); lu.append(math.log(b["n_cols"]))
    ls=np.array(ls); per.append(dict(k=k,n=len(lq),q=np.mean(lq),best=ls.mean(0).min(),styles=dict(zip(STYLES,ls.mean(0).round(3))),old=np.mean(lo),unif=np.mean(lu),
        acc_q=None))
for r in per: print(r)
print("clean streams, non-forced turns: n=",sum(r["n"] for r in per), "forced excluded", tot["nf"], "of", tot["n"])
print("mean over streams: qhat %.3f best %.3f old %.3f uniform %.3f"%(np.mean([r["q"] for r in per]),np.mean([r["best"] for r in per]),np.mean([r["old"] for r in per]),np.mean([r["unif"] for r in per])))
# accuracy (hit) rates
hq=ho=hs=0; n=0; hst={s:0 for s in STYLES}
for k,order in streams("BH"):
    if k in (0,6): continue
    for s,g,p in order:
        for rec in records(p):
            fb=rec.get("feedback")
            if not fb or fb.get("skipped") or "bayes" not in fb: continue
            b=fb["bayes"]
            if b.get("forced") or b["n_cols"]<=1: continue
            n+=1; hq+=int(b["hit_qhat"]); ho+=int(b["hit_old"])
            for s_ in STYLES: hst[s_]+=int(b["hit_style"][s_])
print("argmax accuracy non-forced: qhat %.3f old %.3f"%(hq/n,ho/n), {s:round(v/n,3) for s,v in hst.items()}, "n",n)
