import numpy as np
from scipy.optimize import linprog
def val(A):
    n,m=A.shape; c=np.zeros(n+1); c[-1]=-1
    r=linprog(c,A_ub=np.hstack([-A.T,np.ones((m,1))]),b_ub=np.zeros(m),A_eq=np.r_[np.ones(n),0][None],b_eq=[1],bounds=[(0,None)]*n+[(None,None)],method="highs")
    return r.x[-1], r.x[:n]
for eps in (0.1,1.0,0.37):
    Qh=np.zeros((2,2)); Q=eps*np.array([[1,-1],[1,1.]])
    vQ,_=val(Q); vh,_=val(Qh)
    for x in (np.array([1,0.]),np.array([0,1.])):
        es=vh-(x@Qh).min()            # eps_s of x in Qhat
        loss=vQ-(x@Q).min()           # stage loss in true Q
        print(f"eps={eps} x={x} eps_s={es:.3g} loss={loss:.4g} 2*eps_v={2*np.abs(Q-Qh).max():.4g}")
# Lemma 1' random check
rng=np.random.default_rng(0); viol=0; ratios=[]
for it in range(3000):
    n,m=rng.integers(2,7,2); Qs=rng.normal(size=(n,m)); B=Qs+rng.normal(size=(n,m))*rng.choice([0.01,0.3,1])
    vB,xB=val(B); vQ,xs=val(Qs); vBc,yh=val(-B.T)   # yh minimax for column in B
    x=xB+rng.random(n)*0.0; x=x/x.sum()
    es=vB-(x@B).min()
    lhs=vQ-(x@Qs).min()
    mid=es+xs@(Qs-B)@yh+(x@(B-Qs)).max()
    rhs=es+2*np.abs(B-Qs).max()
    if lhs>mid+1e-7 or mid>rhs+1e-7: viol+=1
    ratios.append((mid-es)/(rhs-es+1e-12))
print("Lemma1' violations",viol,"median budget ratio",np.median(ratios))
