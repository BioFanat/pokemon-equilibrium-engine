import os, sys, json, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import streams, records, EXPERTS
K=4
def stream_data(order):
    R, W = [], []
    for seed, g, path in order:
        pend = None
        for rec in records(path):
            fb = rec.get("feedback")
            if fb and pend is not None and fb.get("for_turn") == pend[0] and not fb.get("skipped") and "r" in fb:
                R.append(np.array(fb["r"], float)); W.append(pend[1])
            if fb is not None or rec.get("final"): pend = None
            if isinstance(rec.get("weights"), dict): pend = (rec["turn"], np.array([rec["weights"][e] for e in EXPERTS], float))
    return np.array(R), np.array(W)
d = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "out_hedge_dyn.json")))
for tag in ("E2","E2fs","E5","BH"):
    out=[]
    for k, order in streams(tag):
        R, W = stream_data(order); T=len(R); L=np.zeros(K); err=0.0
        for s in range(T):
            z = np.sqrt(8*np.log(K)/s)*L if s>0 else np.zeros(K); e=np.exp(z-z.max()); w=e/e.sum()
            err=max(err, float(np.abs(w-W[s]).max())); L=L+R[s]
        out.append((k,T,round(err,4)))
    bad=[r['stream'] for r in d[tag] if not r['count_match']]
    med_all=np.median([r['ext_per_turn'] for r in d[tag]]); med_cl=np.median([r['ext_per_turn'] for r in d[tag] if r['count_match']])
    print(tag, "max|w_log-w_replay| per stream:", out, "| count-mismatch streams:", bad, f"| ExtR/T median all {med_all:.4f} vs clean {med_cl:.4f}")
