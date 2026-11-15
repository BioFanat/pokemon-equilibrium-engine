"""Two-stage family of Example C.1 with |Theta| types and nB stage-0 opponent actions (public states).
eta_NL(0) computed EXACTLY by an LP (eta(s,.) is concave in b'). Looks for expl > H*eta_NL with H=2,
i.e. a counterexample to the targets' form H*(... + eta_NL) itself (not just the sum form).
In this family expl <= |Theta| * eta_NL(0) (see CHECK.md), so |Theta|>=3 is needed."""
import itertools, numpy as np
from scipy.optimize import linprog
from check_example_c1 import V0_lp

def eta_exact(Ub):           # max_{b' in simplex} sum b' max_a U - max_a sum b' U
    nA, nT = Ub.shape
    c = np.r_[-Ub.max(0), 1.0]             # minimize -(r.b') + z
    A = np.c_[Ub, -np.ones(nA)]            # U b' - z <= 0
    r = linprog(c, A_ub=A, b_ub=np.zeros(nA), A_eq=np.r_[np.ones(nT), 0][None], b_eq=[1],
                bounds=[(0, None)] * nT + [(None, None)], method="highs")
    return -r.fun

def agent(U, b0, margin):
    pay_choice = []
    for Ub in U:
        f = Ub @ b0; o = np.argsort(-f)
        if f[o[0]] - f[o[1]] < margin: return None
        pay_choice.append(o[0])
    return sum(b0[th] * min(U[be][pay_choice[be], th] for be in range(len(U))) for th in range(len(b0)))

if __name__ == "__main__":
    rng = np.random.default_rng(1)
    best = (0, None)
    for it in range(20000):
        nT = 3; nB = rng.integers(2, 4); nA = rng.integers(2, 4)
        U = [rng.choice([-1, -0.5, 0, 0.5, 1], size=(nA, nT)) for _ in range(nB)]
        b0 = np.ones(nT) / nT
        pay = agent(U, b0, 0.05)
        if pay is None: continue
        V0, _ = V0_lp(U, b0); e = V0 - pay
        if e < 1e-9: continue
        eta = max(eta_exact(Ub) for Ub in U)
        r = e / (2 * eta) if eta > 1e-12 else np.inf
        if r > best[0]: best = (r, (U, b0, e, eta))
    r, (U, b0, e, eta) = best
    print(f"max expl/(H*eta_NL(0)) with H=2: {r:.4f}   expl={e:.4f} eta_NL(0)={eta:.4f}")
    for Ub in U: print("  ", Ub.tolist())
