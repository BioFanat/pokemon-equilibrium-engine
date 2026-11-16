import json, os, sys, numpy as np
from multiprocessing import Pool
from scipy.optimize import linprog
RES = "/Users/biofanat/Documents/classes/f26/cs535/final/engine/showdown/results/LOG/replays_headless"
def lpval(A):
    n, m = A.shape
    c = np.zeros(n+1); c[-1] = -1
    r = linprog(c, A_ub=np.hstack([-A.T, np.ones((m,1))]), b_ub=np.zeros(m), A_eq=np.r_[np.ones(n),0][None], b_eq=[1], bounds=[(0,None)]*n+[(None,None)], method="highs")
    return r.x[-1]
def one(a):
    s,k = a
    p = os.path.join(RES, f"LOG-s{s}-g{k}.A.decisions.jsonl")
    if not os.path.exists(p): return []
    g = json.load(open(os.path.join(RES, f"LOG-s{s}-g{k}.json")))
    if g.get("error"): return []
    out=[]
    for l in open(p):
        if not l.strip(): continue
        r = json.loads(l)
        if r.get("final") or "abar" not in r: continue
        A = np.array(r["abar"], float); Q = np.array(r["Q"]); x = np.array(r["pfull"])
        v = lpval(A)
        out.append((v - (Q[1]@A).min(), v - (x@A).min(), np.ptp(A)))
    return out
if __name__ == "__main__":
    jobs=[(s,k) for s in range(1,301) for k in (1,2)]
    with Pool(8) as P: res=P.map(one, jobs, chunksize=4)
    a=np.array([t for r in res for t in r])
    for name,col in (("agent-form x (Q[1])",0),("played Hedge pfull",1)):
        g=a[:,col]; rel=g/a[:,2]
        print(name, "n",len(g),"median %.3f mean %.3f p90 %.3f p99 %.2f max %.1f; rel-to-range median %.5f mean %.4f; frac>1%%range %.3f"%(np.median(g),g.mean(),np.percentile(g,90),np.percentile(g,99),g.max(),np.median(rel),rel.mean(),(rel>0.01).mean()))
    print("median range", np.median(a[:,2]))
