"""Example C.1 (full_game.md §6.3): brute-force V_0, the hard-evidence agent's payoff, expl,
and the non-locality bias eta_NL(t) at BOTH stages (the note only evaluates t=1).
Also: random search over the same two-stage family for an instance with expl > sum_t eta_NL(t)
(which would be a genuine counterexample to the targets' four-term form)."""
import numpy as np
from scipy.optimize import linprog

def V0_lp(U, b0):
    """U[beta] is (nA x nTheta): stage-1 payoff at public state s_beta. Agent picks x_beta per state;
    each type picks the state (stage-0 opponent action). V0 = max_x sum_th b0 min_beta x_beta^T U[beta][:,th]."""
    nB, (nA, nT) = len(U), U[0].shape
    nv = nB * nA + nT
    c = np.r_[np.zeros(nB * nA), -b0]
    A, r = [], []
    for be in range(nB):
        for th in range(nT):
            row = np.zeros(nv); row[be*nA:(be+1)*nA] = -U[be][:, th]; row[nB*nA + th] = 1
            A.append(row); r.append(0)
    Aeq = np.zeros((nB, nv))
    for be in range(nB): Aeq[be, be*nA:(be+1)*nA] = 1
    res = linprog(c, A_ub=A, b_ub=r, A_eq=Aeq, b_eq=np.ones(nB),
                  bounds=[(0, None)]*(nB*nA) + [(None, None)]*nT, method="highs")
    return -res.fun, res.x[:nB*nA].reshape(nB, nA)

def agent_payoff(U, b0):
    """Hard-evidence agent: belief b0 at every stage-1 state (no legality/chance evidence);
    plays argmax_a of f_b0 (pure; worst tie for the agent is taken)."""
    nB, (nA, nT) = len(U), U[0].shape
    xs = []
    for be in range(nB):
        f = U[be] @ b0
        best = np.flatnonzero(f >= f.max() - 1e-12)
        # adversarial tie-breaking: the maximizer worst for the realized exploitation
        xs.append(best)
    # enumerate tie choices, take the worst (conservative; ties have measure 0 in random search)
    import itertools
    worst = np.inf
    for choice in itertools.product(*xs):
        pay = sum(b0[th] * min(U[be][choice[be], th] for be in range(nB)) for th in range(nT))
        worst = min(worst, pay)
    return worst

def eta_stage0(U, grid=20001):
    """eta_NL(0) = sup_{beta, b'} [ sum_th b'_th max_a U[beta][a,th] - max_a sum_th b'_th U[beta][a,th] ] (2 types)."""
    w = np.linspace(0, 1, grid)
    B = np.c_[w, 1 - w]
    best = 0.0
    for Ub in U:
        rev = B @ Ub.max(0)
        bel = (B @ Ub.T).max(1)
        best = max(best, (rev - bel).max())
    return best

if __name__ == "__main__":
    # --- Example C.1 exactly as in the note
    uL = np.array([[1.0, -1.0], [0.0, 0.5]])      # rows a1,a2 ; cols theta=1,2
    uR = uL[:, ::-1].copy()
    U = [uL, uR]; b0 = np.array([0.5, 0.5])
    V0, x = V0_lp(U, b0)
    pay = agent_payoff(U, b0)
    eta0 = eta_stage0(U)
    print(f"Example C.1: V0={V0:.6f} (note: 1/5)  optimal x_L,x_R={x.round(4).tolist()}")
    print(f"  hard-evidence agent payoff={pay:.6f}  expl={V0-pay:.6f} (note: 1/5)")
    print(f"  eta_NL(0)={eta0:.6f}   eta_NL(1)=0 (last stage, V_2 linear)")
    w = 0.6; print("  eta at (s_L, b'=(3/5,2/5)):", w*1 + (1-w)*0.5 - max(w-(1-w), (1-w)/2))
    print(f"  four-term RHS with eps=0, K=inf: sum_t eta_NL(t) = {eta0:.4f}  vs expl {V0-pay:.4f}"
          f" -> {'VIOLATED' if V0-pay > eta0 + 1e-9 else 'NOT violated'}")

    # --- random search: same family, nA in {2,3}, 2 types, 2 or 3 stage-0 opponent actions, random prior
    rng = np.random.default_rng(0)
    worst_ratio, worst_inst, n = 0.0, None, 0
    for it in range(6000):
        nA, nB = rng.integers(2, 4), rng.integers(2, 4)
        U = [rng.uniform(-1, 1, (nA, 2)) for _ in range(nB)]
        if rng.random() < 0.5:
            U = [np.round(u * 2) / 2 for u in U]   # coarse payoffs -> structured instances
        p = rng.uniform(0.05, 0.95); b0 = np.array([p, 1 - p])
        V0, _ = V0_lp(U, b0); pay = agent_payoff(U, b0); e = V0 - pay
        if e < 1e-9: continue
        n += 1
        eta0 = eta_stage0(U, 2001)
        r = e / max(eta0, 1e-12)
        if r > worst_ratio: worst_ratio, worst_inst = r, (U, b0, e, eta0)
    print(f"\nrandom search: {n} instances with expl>0; max expl/eta_NL(0) = {worst_ratio:.4f}")
    U, b0, e, eta0 = worst_inst
    print(f"  worst: b0={b0.round(3)}, expl={e:.4f}, eta_NL(0)={eta0:.4f}")
    for Ub in U: print("   ", Ub.round(3).tolist())
