"""Checks for search_rl_links.md Theorem L6 (minimax TD residual bound), Remark 1
(incomparability with Theorem A; drift example) and Remark 4 (mis-scored terminals).

Run:  engine/.venv/bin/python theory/checks/lk6_residual.py
"""
import sys, os, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lk_gamelib as L

t0 = time.time()


def make_leaf(rng, g, Vs, mode):
    H = g.H; c = rng.uniform(0.05, 0.6)
    if mode == "noise":
        V = [Vs[t] + rng.uniform(-c, c, g.nlive[t]) for t in range(H)]
    elif mode == "drift":
        V = [Vs[t] + c * (H - t) / H for t in range(H)]
    elif mode == "offset":
        V = [Vs[t] + c for t in range(H)]
    elif mode == "drift+noise":
        V = [Vs[t] + c * (H - t) / H + rng.uniform(-c / 2, c / 2, g.nlive[t]) for t in range(H)]
    else:  # arbitrary leaf, no closeness to V*
        V = [rng.uniform(-1, 1, g.nlive[t]) for t in range(H)]
    return V + [np.zeros(g.nlive[H])]


def max_budget_path(g, X, f):
    """sup over opponents of E_{pi,sigma}[sum_{t<tau} f_t(s_t)] (opponent maximises; DP)."""
    J = [np.zeros(g.nlive[t]) for t in range(g.H + 1)]
    for t in range(g.H - 1, -1, -1):
        nl = g.nlive[t + 1]
        J[t] = np.array([f[t][i] + (X[t][i] @ (g.P[t][i][:, :, :nl] @ J[t + 1])).max()
                         for i in range(g.nlive[t])])
    return J[0][0]


def l6_instance(rng, g, mode, mismatch=0.0):
    """Returns dict of quantities. mismatch > 0: Vhat != u at terminals (Remark 4)."""
    H = g.H
    Vs, Qs, Xs, Ys, _ = L.optimal(g)
    Vh = make_leaf(rng, g, Vs, mode)
    Uh = None
    m = [np.zeros(g.nterm[t]) for t in range(H + 1)]
    if mismatch > 0:
        m = [np.zeros(0)] + [rng.uniform(-mismatch, mismatch, g.nterm[t]) for t in range(1, H + 1)]
        Uh = [g.U[t] + m[t] if t > 0 else np.zeros(0) for t in range(H + 1)]
    delta = []; es = []; dA = []; X = []; Yh = []
    for t in range(H):
        delta.append(np.zeros(g.nlive[t])); es.append(np.zeros(g.nlive[t])); dA.append(np.zeros(g.nlive[t]))
        X.append([]); Yh.append([])
        for i in range(g.nlive[t]):
            M = g.Q(t, i, Vh[t + 1], Uh)
            v, xh, yh = L.solve(M)
            lam = rng.choice([0.0, 0.0, 0.05, 0.3, 1.0])
            x = (1 - lam) * xh + lam * rng.dirichlet(np.ones(len(xh)))
            X[t].append(x); Yh[t].append(yh)
            delta[t][i] = Vh[t][i] - v
            es[t][i] = v - (x @ M).min()
            dA[t][i] = es[t][i] + 2 * np.abs(M - Qs[t][i]).max()
    Ybr, Wbr = L.best_response(g, X, rng)
    expl = Vs[0][0] - Wbr[0][0]
    dB, zB = L.forward(g, X, Ybr)
    dS, zS = L.forward(g, Xs, Yh)
    eps_glob = max(e.max() for e in es)
    T1 = L.path_sum(dB, [delta[t] + es[t] for t in range(H)], H)
    T1g = L.path_sum(dB, [delta[t] + eps_glob for t in range(H)], H)
    T2 = L.path_sum(dS, delta, H)
    supf = sum(2 * np.abs(delta[t]).max() + eps_glob for t in range(H))
    mB = sum(float(zB[t] @ m[t]) for t in range(1, H + 1))
    resid = [np.array([Vh[t][i] - X[t][i] @ g.Q(t, i, Vh[t + 1], Uh) @ Ybr[t][i]
                       for i in range(g.nlive[t])]) for t in range(H)]
    idm = abs((Vh[0][0] - Wbr[0][0]) - (L.path_sum(dB, resid, H) + mB))
    mS = sum(float(zS[t] @ m[t]) for t in range(1, H + 1))
    A_traj = max_budget_path(g, X, dA)
    A_sup = sum(dA[t].max() for t in range(H))
    # Remark-1 auxiliary facts (exact terminals only)
    nV = [np.abs(Vh[t] - Vs[t]).max() if g.nlive[t] else 0.0 for t in range(H + 1)]
    sd = [np.abs(delta[t]).max() if g.nlive[t] else 0.0 for t in range(H)]
    aux1 = min(sum(sd[t + 1:]) - nV[t + 1] for t in range(H))
    aux2 = 2 * max(nV) - max(sd)
    return dict(expl=expl, traj=T1 - T2, trajg=T1g - T2, sup=supf, mB=mB, mS=mS,
                A_traj=A_traj, A_sup=A_sup, aux1=aux1, aux2=aux2, idm=idm)


def chain(H, c, opp_choice=False):
    """Forced deterministic chain without terminals (V* = 0). opp_choice: the opponent may also
    end the game (payoff 0) at every stage, so only the agent is forced."""
    nlive = [1] * (H + 1)
    nterm = [0] + [1 if opp_choice else 0] * H
    U = [np.zeros(0)] + [np.zeros(nterm[t]) for t in range(1, H + 1)]
    P = []
    for t in range(H):
        if opp_choice:
            T = np.zeros((1, 2, 2)); T[0, 0, 0] = 1; T[0, 1, 1] = 1
        else:
            T = np.ones((1, 1, 1))
        P.append([T])
    g = L.Game(H, nlive, nterm, U, [[1]] * H, [[2 if opp_choice else 1]] * H, P)
    return g


def drift_example(H, c, opp_choice=False, br_tie="first"):
    g = chain(H, c, opp_choice)
    Vs, Qs, Xs, Ys, _ = L.optimal(g)
    Vh = [Vs[t] + c * (H - t) / H for t in range(H)] + [np.zeros(1)]
    delta = []; dA = []; X = []; Yh = []
    for t in range(H):
        M = g.Q(t, 0, Vh[t + 1]); v, x, y = L.solve(M)
        delta.append(np.array([Vh[t][0] - v])); dA.append(np.array([2 * np.abs(M - Qs[t][0]).max()]))
        X.append([x]); Yh.append([y])
    if br_tie == "first":
        Ybr, Wbr = L.best_response(g, X)
    elif br_tie == "end":  # a best response that ends at once (payoff 0 = V*)
        Ybr = [[np.eye(g.nB[t][0])[-1]] for t in range(H)]; Wbr = L.evaluate(g, X, Ybr)
    else:  # a best response that always continues (all opponents are best responses here)
        Ybr = [[np.eye(g.nB[t][0])[0]] for t in range(H)]; Wbr = L.evaluate(g, X, Ybr)
    expl = Vs[0][0] - Wbr[0][0]
    T1 = L.path_sum(L.forward(g, X, Ybr)[0], delta, H)
    T2 = L.path_sum(L.forward(g, Xs, Yh)[0], delta, H)
    return expl, delta[0][0], T1 - T2, sum(2 * abs(d[0]) for d in delta), sum(d[0] for d in dA)


def reverse_example(eta=0.01):
    """Trajectory-level instance with L6 trajectory form >> Theorem A trajectory form.
    H=3. s0: agent {L,R}, opponent forced. L -> sL (forced) -> terminal -eta.  R -> sR, a 2x2
    matching-pennies stage whose cells go to forced stage-2 states w_ab -> terminal +-1.
    Leaf: Vhat(sL) = -eta, Vhat(sR) = -2 eta, Vhat(w) = [[1, .9], [-1, -1]] (V*(w) = [[1,-1],[-1,1]]).
    Agent plays L (0-optimal in Qhat_0 = (-eta, -2 eta)); pi* plays R (unique, Q*_0 = (-eta, 0))."""
    H = 3
    nlive = [1, 2, 4, 0]; nterm = [0, 0, 1, 2]
    U = [np.zeros(0), np.zeros(0), np.array([-eta]), np.array([1.0, -1.0])]
    P0 = np.zeros((2, 1, 2)); P0[0, 0, 0] = 1; P0[1, 0, 1] = 1
    PL = np.zeros((1, 1, 5)); PL[0, 0, 4] = 1
    PR = np.zeros((2, 2, 5))
    for a in range(2):
        for b in range(2):
            PR[a, b, 2 * a + b] = 1
    mp = [[1, -1], [-1, 1]]
    Pw = []
    for a in range(2):
        for b in range(2):
            T = np.zeros((1, 1, 2)); T[0, 0, 0 if mp[a][b] == 1 else 1] = 1; Pw.append(T)
    g = L.Game(H, nlive, nterm, U, [[2], [1, 2], [1] * 4], [[1], [1, 2], [1] * 4], [[P0], [PL, PR], Pw])
    Vs, Qs, Xs, Ys, _ = L.optimal(g)
    Vh = [np.array([0.0]), np.array([-eta, -2 * eta]), np.array([1, .9, -1, -1.0]), np.zeros(0)]
    X = []; Yh = []; delta = []; dA = []
    for t in range(H):
        X.append([]); Yh.append([]); delta.append(np.zeros(nlive[t])); dA.append(np.zeros(nlive[t]))
        for i in range(nlive[t]):
            M = g.Q(t, i, Vh[t + 1]); v, x, y = L.solve(M)
            if t == 0:
                x = np.array([1.0, 0.0])
            es = v - (x @ M).min()
            X[t].append(x); Yh[t].append(y); delta[t][i] = Vh[t][i] - v
            dA[t][i] = es + 2 * np.abs(M - Qs[t][i]).max()
    Ybr, Wbr = L.best_response(g, X)
    expl = Vs[0][0] - Wbr[0][0]
    T1 = L.path_sum(L.forward(g, X, Ybr)[0], delta, H)   # eps_s = 0 everywhere
    T2 = L.path_sum(L.forward(g, Xs, Yh)[0], delta, H)
    return expl, T1 - T2, max_budget_path(g, X, dA), Xs[0][0], Yh[1][1]


if __name__ == "__main__":
    modes = ["noise", "drift", "offset", "drift+noise", "arbitrary"]
    print("== Theorem L6 (exact terminals) ==")
    rng = np.random.default_rng(21)
    R = []
    for k in range(1000):
        g = L.random_game(rng, int(rng.integers(2, 7)), max_states=4)
        R.append((modes[k % 5], l6_instance(rng, g, modes[k % 5])))
    s_traj = min(r["traj"] - r["expl"] for _, r in R)
    s_trajg = min(r["trajg"] - r["traj"] for _, r in R)
    s_sup = min(r["sup"] - r["trajg"] for _, r in R)
    print(f"1000 games H=2..6, leaves {modes}:")
    print(f"  min[trajectory form - expl] = {s_traj:.2e} (per-state eps_s); "
          f"min[global-eps_s form - per-state form] = {s_trajg:.2e}; min[sup form - trajectory form] = {s_sup:.2e}")
    print(f"  aux: min[sum_(u>t) sup|delta_u| - ||Vhat_(t+1)-V*_(t+1)||] = {min(r['aux1'] for _, r in R):.2e}; "
          f"min[2||Vhat-V*|| - sup|delta|] = {min(r['aux2'] for _, r in R):.2e}")
    print("  Remark 1 incomparability (count over 1000):")
    for mode in modes:
        rs = [r for md, r in R if md == mode]
        print(f"    {mode:12s}: L6traj<Atraj {sum(r['traj'] < r['A_traj'] - 1e-9 for r in rs):3d}, "
              f"Atraj<L6traj {sum(r['A_traj'] < r['traj'] - 1e-9 for r in rs):3d} | "
              f"L6sup<Asup {sum(r['sup'] < r['A_sup'] - 1e-9 for r in rs):3d}, "
              f"Asup<L6sup {sum(r['A_sup'] < r['sup'] - 1e-9 for r in rs):3d}  (of {len(rs)})")
    ex, l6, at, xs0, yh = reverse_example()
    print(f"  reverse instance (eta=0.01): expl={ex:.3f}, L6 trajectory form={l6:.3f}, Theorem A trajectory "
          f"form={at:.3f}  [pi*(s0)={xs0}, sigma^(sR)={yh}]")
    print(f"  sanity: min[A_traj - expl] = {min(r['A_traj'] - r['expl'] for _, r in R):.2e}")

    print("\n== Remark 1 drift example (forced deterministic chain, V* = 0, Vhat_t = c(H-t)/H) ==")
    for H, c in ((5, 0.3), (10, 0.5)):
        ex, d0, tr, sp, A = drift_example(H, c)
        print(f"H={H}, c={c}: expl={ex:.3f}, delta_t={d0:.4f} (c/H={c/H:.4f}), trajectory form={tr:.2e}, "
              f"sup form={sp:.4f} (2c={2*c}), Theorem A budget={A:.4f} (c(H-1)={c*(H-1):.4f})")
    for tie in ("end", "continue"):
        ex, d0, tr, sp, A = drift_example(5, 0.3, opp_choice=True, br_tie=tie)
        print(f"variant, only the agent forced (opponent may end, payoff 0), BR tie '{tie}': expl={ex:.3f}, "
              f"trajectory form={tr:.4f}, sup form={sp:.4f}, Theorem A budget={A:.4f}")

    print("\n== Remark 4: Vhat != u at terminals (m = Vhat - u on terminals) ==")
    rng = np.random.default_rng(22)
    R = []
    for k in range(1000):
        g = L.random_game(rng, int(rng.integers(2, 7)), max_states=4, p_term=0.5)
        R.append(l6_instance(rng, g, modes[k % 5], mismatch=rng.choice([0.1, 0.5, 1.0])))
    print(f"L0 with mismatched terminals: max |Vhat_0 - V(pi,BR) - (E[sum resid] + E_BR[m])| = "
          f"{max(r['idm'] for r in R):.2e}")
    pm = min(r["traj"] + r["mB"] - r["mS"] - r["expl"] for r in R)
    pp = min(r["traj"] + r["mB"] + r["mS"] - r["expl"] for r in R)
    nn = min(r["traj"] - r["expl"] for r in R)
    abs_ = min(r["traj"] + abs(r["mB"]) + abs(r["mS"]) - r["expl"] for r in R)
    print(f"1000 games: min slack of  traj + E_BR[m] - E_(pi*,sig^)[m]          = {pm:.2e}")
    print(f"            min slack of  traj + E_BR[m] + E_(pi*,sig^)[m]  (+/+)   = {pp:.2e}  "
          f"(violations {sum(r['traj'] + r['mB'] + r['mS'] < r['expl'] - 1e-9 for r in R)})")
    print(f"            min slack of  traj (terms dropped)                     = {nn:.2e}  "
          f"(violations {sum(r['traj'] < r['expl'] - 1e-9 for r in R)})")
    print(f"            min slack of  traj + |E_BR[m]| + |E_(pi*,sig^)[m]|      = {abs_:.2e}")

    print(f"\nLP certificate gap (max hi-lo) = {L.LP_GAP[0]:.2e};  runtime {time.time()-t0:.1f}s")
