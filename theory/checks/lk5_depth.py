"""Checks for search_rl_links.md L5.0 and Theorem L5 (depth-D lookahead with exact terminals),
Remark 1 (D=1 vs R6b), Remark 4 (two-profile sharper factor), and the Theorem A corollary.

Run:  engine/.venv/bin/python theory/checks/lk5_depth.py
"""
import sys, os, time, itertools
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lk_gamelib as L

t0 = time.time()


def lookahead(g, Vhat, D, Vs, Qs, Xs, Ys):
    """Returns per-level lists: V[k][t], Q[k][t][i], X[k][t][i], Y[k][t][i] for k=0..D,
    rho[k][t], rho_plus[k][t], rho_minus[k][t]."""
    H = g.H
    V = [[None] * (H + 1) for _ in range(D + 1)]
    Q = [[[None] * g.nlive[t] for t in range(H)] for _ in range(D + 1)]
    X = [[[None] * g.nlive[t] for t in range(H)] for _ in range(D + 1)]
    Y = [[[None] * g.nlive[t] for t in range(H)] for _ in range(D + 1)]
    rho = [[None] * (H + 1) for _ in range(D + 1)]
    rp = [[None] * (H + 1) for _ in range(D + 1)]
    rm = [[None] * (H + 1) for _ in range(D + 1)]
    for t in range(H + 1):
        V[0][t] = np.array(Vhat[t], float) if t < H else np.zeros(g.nlive[H])
        rho[0][t] = np.full(g.nlive[t], 1.0 if t < H else 0.0)
        rp[0][t] = rho[0][t].copy(); rm[0][t] = rho[0][t].copy()
    for k in range(1, D + 1):
        V[k][H] = np.zeros(g.nlive[H]); rho[k][H] = np.zeros(g.nlive[H])
        rp[k][H] = np.zeros(g.nlive[H]); rm[k][H] = np.zeros(g.nlive[H])
        for t in range(H):
            V[k][t] = np.zeros(g.nlive[t]); rho[k][t] = np.zeros(g.nlive[t])
            rp[k][t] = np.zeros(g.nlive[t]); rm[k][t] = np.zeros(g.nlive[t])
            nl = g.nlive[t + 1]
            for i in range(g.nlive[t]):
                M = g.Q(t, i, V[k - 1][t + 1])
                V[k][t][i], X[k][t][i], Y[k][t][i] = L.solve(M)
                Q[k][t][i] = M
                live = g.P[t][i][:, :, :nl]
                rho[k][t][i] = (live @ rho[k - 1][t + 1]).max()
                rp[k][t][i] = X[k][t][i] @ (live @ rp[k - 1][t + 1]) @ Ys[t][i]
                rm[k][t][i] = Xs[t][i] @ (live @ rm[k - 1][t + 1]) @ Y[k][t][i]
    return V, Q, X, Y, rho, rp, rm


def check_game(rng, g, mode):
    H = g.H
    Vs, Qs, Xs, Ys, _ = L.optimal(g)
    eps = rng.uniform(0.05, 0.5)
    if mode == "noise":
        Vhat = [Vs[t] + rng.uniform(-eps, eps, g.nlive[t]) for t in range(H)]
    elif mode == "offset":
        Vhat = [Vs[t] + eps for t in range(H)]
    else:  # random signs at full size
        Vhat = [Vs[t] + eps * rng.choice([-1, 1], g.nlive[t]) for t in range(H)]
    out = dict(sV=np.inf, sQ=np.inf, ratio=0.0, sP=np.inf, sM=np.inf, sQpm=np.inf, sharper_lt=0, n=0)
    V, Q, X, Y, rho, rp, rm = lookahead(g, Vhat, H, Vs, Qs, Xs, Ys)
    for k in range(H + 1):
        for t in range(H):
            for i in range(g.nlive[t]):
                err = V[k][t][i] - Vs[t][i]
                out["sV"] = min(out["sV"], eps * rho[k][t][i] - abs(err))
                if eps * rho[k][t][i] > 1e-12:
                    out["ratio"] = max(out["ratio"], abs(err) / (eps * rho[k][t][i]))
                out["sP"] = min(out["sP"], eps * rp[k][t][i] - err)
                out["sM"] = min(out["sM"], eps * rm[k][t][i] + err)
                if k >= 1:
                    qerr = np.abs(Q[k][t][i] - Qs[t][i]).max()
                    out["sQ"] = min(out["sQ"], eps * rho[k][t][i] - qerr)
                    nl = g.nlive[t + 1]
                    child = np.maximum(rp[k - 1][t + 1], rm[k - 1][t + 1])
                    cellb = eps * (g.P[t][i][:, :, :nl] @ child)
                    out["sQpm"] = min(out["sQpm"], (cellb - np.abs(Q[k][t][i] - Qs[t][i])).min())
                    out["n"] += 1
                    if max(rp[k][t][i], rm[k][t][i]) < rho[k][t][i] - 1e-9:
                        out["sharper_lt"] += 1
    # rho convention: zero whenever t + k >= H
    conv = all(np.all(rho[k][t] == 0) for k in range(H + 1) for t in range(H + 1) if t + k >= H)
    out["conv"] = conv
    return out


def r6b_compare(rng, g):
    """D = 1: L5 cellwise bound eps_leaf*E[rho_0] vs R6b per-cell pi_live(a,b)*eps_live(s);
    and L5's stated (max-over-cells) bound eps_leaf*rho_1."""
    H = g.H
    Vs, Qs, _, _, _ = L.optimal(g)
    eps = rng.uniform(0.05, 0.5)
    Vhat = [Vs[t] + rng.uniform(-eps, eps, g.nlive[t]) for t in range(H)] + [np.zeros(g.nlive[H])]
    s_r6b = np.inf; s_l5 = np.inf; n_strict = 0; n = 0; n_capdiff = 0
    for t in range(H):
        nl = g.nlive[t + 1]
        for i in range(g.nlive[t]):
            Q1 = g.Q(t, i, Vhat[t + 1]); err = np.abs(Q1 - Qs[t][i])
            pilive = g.P[t][i][:, :, :nl].sum(-1)            # R6b counts every non-terminal child
            eloc = np.abs(Vhat[t + 1] - Vs[t + 1]).max() if (nl and t + 1 < H) else 0.0
            r6b = pilive * eloc
            rho1 = (pilive if t + 1 < H else 0 * pilive).max()
            l5 = eps * rho1
            s_r6b = min(s_r6b, (r6b - err).min()); s_l5 = min(s_l5, l5 - err.max())
            n += 1; n_strict += int(r6b.max() < l5 - 1e-12)
            n_capdiff += int(t + 1 == H and nl > 0 and pilive.max() > 0)
    return s_r6b, s_l5, n_strict, n, n_capdiff


def brute_rho(g, D):
    """max over joint deterministic Markov profiles of P(live at stage D, D < H) from the root."""
    H = g.H
    cells = [(t, i) for t in range(min(D, H)) for i in range(g.nlive[t])]
    choices = [list(itertools.product(range(g.nA[t][i]), range(g.nB[t][i]))) for t, i in cells]
    best = 0.0
    for prof in itertools.product(*choices):
        pol = dict(zip(cells, prof))
        d = np.zeros(g.nlive[0]); d[0] = 1.0
        for t in range(min(D, H)):
            nd = np.zeros(g.nlive[t + 1])
            for i in range(g.nlive[t]):
                a, b = pol[(t, i)]
                nd += d[i] * g.P[t][i][a, b, : g.nlive[t + 1]]
            d = nd
        best = max(best, d.sum() if D < H else 0.0)
    return best


def theoremA_corollary(rng, g, D):
    """Agent: at each (t,s), eps_s-optimal in Q^(D)_t(s). Check e <= eps_s + 2 eps rho_D and
    expl <= E_BR[sum (eps_s + 2 eps rho_D)]."""
    H = g.H
    Vs, Qs, Xs, Ys, _ = L.optimal(g)
    eps = rng.uniform(0.05, 0.5)
    Vhat = [Vs[t] + rng.uniform(-eps, eps, g.nlive[t]) for t in range(H)]
    V, Q, X, Y, rho, _, _ = lookahead(g, Vhat, D, Vs, Qs, Xs, Ys)
    Xa = []; bud = []; s_e = np.inf
    for t in range(H):
        Xa.append([]); bud.append(np.zeros(g.nlive[t]))
        for i in range(g.nlive[t]):
            M = Q[D][t][i]; lam = rng.choice([0.0, 0.1])
            x = (1 - lam) * X[D][t][i] + lam * rng.dirichlet(np.ones(g.nA[t][i]))
            es = V[D][t][i] - (x @ M).min()
            Xa[t].append(x); bud[t][i] = es + 2 * eps * rho[D][t][i]
            s_e = min(s_e, bud[t][i] - (Vs[t][i] - (x @ Qs[t][i]).min()))
    Ybr, Wbr = L.best_response(g, Xa, rng)
    expl = Vs[0][0] - Wbr[0][0]
    d, _ = L.forward(g, Xa, Ybr)
    return s_e, L.path_sum(d, bud, H) - expl


if __name__ == "__main__":
    print("== L5.0 (value difference) ==")
    rng = np.random.default_rng(11)
    worst = np.inf
    for k in range(3000):
        n, m = rng.integers(1, 5, 2)
        if k % 3 == 0:  # integer entries -> ties / degenerate games
            A = rng.integers(-2, 3, (n, m)).astype(float); B = rng.integers(-2, 3, (n, m)).astype(float)
        else:
            A = rng.uniform(-1, 1, (n, m)); B = A + rng.normal(0, 0.3, (n, m)) * (rng.random() < 0.5) \
                if k % 3 == 1 else rng.uniform(-1, 1, (n, m))
        vA, xA, _ = L.solve(A); vB, _, yB = L.solve(B)
        worst = min(worst, xA @ (A - B) @ yB - (vA - vB))
    print(f"3000 pairs (1..4 x 1..4, incl. integer/degenerate): min[x_A^T(A-B)y_B - (valA - valB)] = {worst:.2e}")

    print("\n== Theorem L5 (|V^D - V*| <= eps rho_D, ||Q^(D)-Q*|| <= eps rho_D), all D = 0..H ==")
    rng = np.random.default_rng(12)
    agg = dict(sV=np.inf, sQ=np.inf, ratio=0.0, sP=np.inf, sM=np.inf, sQpm=np.inf, sharper_lt=0, n=0)
    conv_ok = True; per_mode = {}
    for k in range(600):
        mode = ["noise", "offset", "signs"][k % 3]
        g = L.random_game(rng, int(rng.integers(2, 6)), p_term=0.45)
        o = check_game(rng, g, mode)
        conv_ok &= o.pop("conv")
        per_mode.setdefault(mode, []).append(o["ratio"])
        for key in ("sV", "sQ", "sP", "sM", "sQpm"):
            agg[key] = min(agg[key], o[key])
        agg["ratio"] = max(agg["ratio"], o["ratio"]); agg["sharper_lt"] += o["sharper_lt"]; agg["n"] += o["n"]
    print(f"600 games H=2..5: min slack V = {agg['sV']:.2e}, min slack Q = {agg['sQ']:.2e}; "
          f"max |V^D-V*|/(eps rho_D) = {agg['ratio']:.4f} (offset mode max {max(per_mode['offset']):.4f})")
    print(f"rho_D(t,s) = 0 whenever t + D >= H: {conv_ok}")
    print(f"Remark 4 two-profile factors: min[eps rho+ - (V^D-V*)] = {agg['sP']:.2e}, "
          f"min[eps rho- - (V*-V^D)] = {agg['sM']:.2e}, cell Q bound with max(rho+,rho-) children: {agg['sQpm']:.2e}; "
          f"strictly smaller than rho_D at {agg['sharper_lt']}/{agg['n']} nodes")

    print("\n== rho_D recursion vs brute force over joint deterministic Markov profiles ==")
    rng = np.random.default_rng(13)
    md = 0.0
    for _ in range(300):
        g = L.random_game(rng, int(rng.integers(1, 4)), max_states=2, max_act=2, p_term=0.45)
        Vs, Qs, Xs, Ys, _ = L.optimal(g)
        _, _, _, _, rho, _, _ = lookahead(g, [Vs[t] for t in range(g.H)], g.H, Vs, Qs, Xs, Ys)
        for D in range(g.H + 1):
            md = max(md, abs(brute_rho(g, D) - rho[D][0][0]))
    print(f"300 games H=1..3, all D: max |rho_D(0,s0) - brute force| = {md:.2e}")

    print("\n== Remark 1: D = 1 vs R6b exact-terminal cell bound ==")
    rng = np.random.default_rng(14)
    s1 = s2 = np.inf; ns = nn = ncap = 0
    for _ in range(500):
        a, b, c, d_, e = r6b_compare(rng, L.random_game(rng, int(rng.integers(2, 6)), p_term=0.45))
        s1 = min(s1, a); s2 = min(s2, b); ns += c; nn += d_; ncap += e
    print(f"500 games: R6b per-cell bound min slack = {s1:.2e}; L5 D=1 stated bound min slack = {s2:.2e}; "
          f"R6b's max-cell bound strictly below L5's at {ns}/{nn} states (local eps_live < global eps_leaf); "
          f"states at t=H-1 with live (cap) children where R6b charges pi_live but L5 charges 0: {ncap}")

    print("\n== Theorem A with per-state budget eps_s + 2 eps rho_D ==")
    rng = np.random.default_rng(15)
    se = sx = np.inf
    for k in range(500):
        g = L.random_game(rng, int(rng.integers(2, 6)), p_term=0.45)
        a, b = theoremA_corollary(rng, g, int(rng.integers(1, g.H + 1)))
        se = min(se, a); sx = min(sx, b)
    print(f"500 games, D in 1..H: min[(eps_s+2 eps rho_D) - e_t] = {se:.2e}; min[E_BR sum budget - expl] = {sx:.2e}")

    print(f"\nLP certificate gap (max hi-lo) = {L.LP_GAP[0]:.2e};  runtime {time.time()-t0:.1f}s")
