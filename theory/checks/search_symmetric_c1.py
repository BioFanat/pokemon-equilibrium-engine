"""Exhaustive search over the symmetric Example-C.1 family (b0=(1/2,1/2), u_R = u_L with types swapped,
payoffs on the grid {-1,-1/2,0,1/2,1}) for instances with expl > eta_NL(0) (+ eta_NL(1)=0)."""
import itertools, numpy as np
from check_example_c1 import V0_lp, agent_payoff, eta_stage0
vals = [-1, -0.5, 0, 0.5, 1]; b0 = np.array([0.5, 0.5])
res = []
for nA in (2, 3):
    for ent in itertools.product(vals, repeat=2 * nA):
        uL = np.array(ent, float).reshape(nA, 2); U = [uL, uL[:, ::-1].copy()]
        V0, _ = V0_lp(U, b0); e = V0 - agent_payoff(U, b0)
        if e > 1e-9:
            res.append((e - eta_stage0(U, 4001), e, uL.tolist()))
res.sort(key=lambda r: -r[0])
print("count expl>0:", len(res), " count expl>eta_NL(0):", sum(r[0] > 1e-9 for r in res))
for r in res[:5]: print(f"expl-eta={r[0]:.4f} expl={r[1]:.4f} u_L={r[2]}")

# strict version: agent's stage-1 maximizer must be unique (margin >= 0.25) so no tie-breaking is involved
print("\nstrict (unique agent maximizer, margin>=1/4):")
strict = []
for e_eta, e, uL in res:
    f = np.sort(np.array(uL) @ b0)[::-1]
    if f[0] - f[1] >= 0.25 - 1e-12: strict.append((e_eta, e, uL))
print("count expl>eta_NL(0):", sum(r[0] > 1e-9 for r in strict), "of", len(strict))
for r in strict[:6]: print(f"expl-eta={r[0]:.4f} expl={r[1]:.4f} u_L={r[2]}")
