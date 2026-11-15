"""Repaired Example C.1: a genuine counterexample to the targets' four-term form (both the
sum_t eta_NL(t) reading and the H*max_t eta_NL(t) reading), with eps_s=eps_v=0 and K->infinity.
Theta={1,2,3}, b0 uniform. Stage 0: agent has one action, opponent picks beta in {1,2,3} -> public s_beta.
Stage 1 (last): agent picks risky/safe at s_beta; risky pays -1 to type beta and +c to the other types, safe pays 0."""
import numpy as np
from fractions import Fraction as F
from check_example_c1 import V0_lp
from search_3types import eta_exact, agent

for c in (0.75, 0.6):
    U = []
    for be in range(3):
        risky = np.full(3, c); risky[be] = -1
        U.append(np.vstack([risky, np.zeros(3)]))
    b0 = np.ones(3) / 3
    V0, x = V0_lp(U, b0); pay = agent(U, b0, 1e-9); e = V0 - pay
    eta0 = max(eta_exact(Ub) for Ub in U)
    print(f"c={c}: V0={V0:.6f} agent={pay:.6f} expl={e:.6f}  eta_NL(0)={eta0:.6f} (closed form c/(1+c)={c/(1+c):.6f}), eta_NL(1)=0")
    print(f"   sum_t eta = {eta0:.4f} < expl ;  H*max eta = {2*eta0:.4f} {'<' if 2*eta0 < e else '>='} expl={e:.4f}")
    print(f"   agent's stage-1 margin f(risky)-f(safe) = {(2*c-1)/3:.4f}")
