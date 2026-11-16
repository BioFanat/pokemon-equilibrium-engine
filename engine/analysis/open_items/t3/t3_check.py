import sys, numpy as np
sys.path.insert(0,"."); 
import importlib.util
src=open("t3_rq.py").read().split("for tag, true in TRUE.items():")[0]
exec(src)
for tag in TRUE:
    d=[]; 
    for st in range(10):
        rows,bad=stream_turns(tag,st)
        for q,j,ps,f,_ in rows: d.append(np.abs(q[:,j]-ps).max()/max(1e-9,ps.max()) if False else np.abs(np.log(q[:,j]/ps)).max())
    d=np.array(d); print(tag,len(d),"max|dlog|",d.max().round(3),"frac>0.05",(d>0.05).mean().round(4))
