"""Joint-play diagnostic for one DUCT instance (section L4). Usage: lk4_duct_joint.py NAME N [bern].
Prints joint (a,b) counts and each side empirical arm means vs the fair value against the opponent visit shares."""
import sys, math, numpy as np
sys.path.insert(0, "/Users/biofanat/Documents/classes/f26/cs535/final/theory/checks")
import lk4_duct as L
games = L.make_games(); d = [g for g in games if g["name"] == sys.argv[1]][0]; A = d["A"]
print(A.round(3)); val, xs, ys = L.solve(A); print("val", round(val,4), "x*", xs.round(3), "y*", ys.round(3))
import random; rng=random.Random(5); bern=len(sys.argv)>3
n, m = A.shape; SA=[0.]*n; vA=[0]*n; SB=[0.]*m; vB=[0]*m; J=np.zeros((n,m)); N=0
for t in range(int(sys.argv[2])):
    lnN = math.log(N) if N else 0.
    ua=[math.inf if vA[i]==0 else SA[i]/vA[i]+math.sqrt(2*lnN/vA[i]) for i in range(n)]
    ub=[math.inf if vB[j]==0 else SB[j]/vB[j]+math.sqrt(2*lnN/vB[j]) for j in range(m)]
    a=max(range(n), key=lambda i:(ua[i], -i)); b=max(range(m), key=lambda j:(ub[j], -j))
    r=(1.0 if rng.random()<A[a,b] else 0.0) if bern else A[a,b]; SA[a]+=r; vA[a]+=1; SB[b]+=1-r; vB[b]+=1; J[a,b]+=1; N+=1
x=np.array(vA)/N; y=np.array(vB)/N
print("visit x", x.round(4), "y", y.round(4))
print("joint counts\n", J.astype(int))
print("side-two empirical arm means", (np.array(SB)/np.array(vB)).round(4), " vs fair (x^T(1-A))_b", (x@(1-A)).round(4))
print("side-one empirical arm means", (np.array(SA)/np.array(vA)).round(4), " vs fair (A y)_a", (A@y).round(4))
