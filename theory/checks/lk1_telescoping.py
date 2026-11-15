"""Checks for search_rl_links.md L0 (potential telescoping) and L1 (sandwich), Example L1b.

Run:  engine/.venv/bin/python theory/checks/lk1_telescoping.py
Deterministic seeds; prints min slacks (should be >= -1e-9) and the LP certificate gap.
"""
import sys, os, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lk_gamelib as L

t0 = time.time()

# ---------------------------------------------------------------- L0, Markov ---------
def l0_markov(rng, g):
    H = g.H
    phi = [rng.uniform(-1, 1, g.nlive[t]) for t in range(H)] + [np.zeros(g.nlive[H])]
    X = L.random_markov(rng, g, "A"); Y = L.random_markov(rng, g, "B")
    W = L.evaluate(g, X, Y)
    d, _ = L.forward(g, X, Y)
    f = [np.array([phi[t][i] - X[t][i] @ g.Q(t, i, phi[t + 1]) @ Y[t][i]
                   for i in range(g.nlive[t])]) for t in range(H)]
    return abs((phi[0][0] - W[0][0]) - L.path_sum(d, f, H))


# ---------------------------------------------------------------- history tree -------
class Tree:
    """Expands the full history tree; strategies and potentials are arbitrary functions of
    the history (memoised draws in deterministic traversal order)."""

    def __init__(self, g, rng):
        self.g, self.rng = g, rng
        self.memo = {}

    def draw(self, key, fn):
        if key not in self.memo:
            self.memo[key] = fn()
        return self.memo[key]

    def children(self, t, i, h, a, b):
        g = self.g
        for k in np.flatnonzero(g.P[t][i][a, b] > 0):
            p = g.P[t][i][a, b, k]
            hk = h + ((a, b, int(k)),)
            if k < g.nlive[t + 1]:
                kind = "cap" if t + 1 == g.H else "live"
                yield p, kind, int(k), hk
            else:
                yield p, "term", g.U[t + 1][k - g.nlive[t + 1]], hk


def l0_tree(rng, g):
    """L0 with a history-dependent potential, agent and opponent; checks the identity at
    every live history. Returns max |error|."""
    T = Tree(g, rng)
    pi = lambda t, i, h: T.draw(("x", h), lambda: rng.dirichlet(np.ones(g.nA[t][i])))
    sg = lambda t, i, h: T.draw(("y", h), lambda: rng.dirichlet(np.ones(g.nB[t][i])))
    phi_live = lambda h: T.draw(("phi", h), lambda: rng.uniform(-1, 1))
    err = [0.0]

    def phi_of(kind, val, h):
        return 0.0 if kind == "cap" else (val if kind == "term" else phi_live(h))

    def rec(t, i, h):  # returns (V, R)
        x, y = pi(t, i, h), sg(t, i, h)
        QP = np.zeros((g.nA[t][i], g.nB[t][i])); V = 0.0; R = 0.0
        for a in range(g.nA[t][i]):
            for b in range(g.nB[t][i]):
                w = x[a] * y[b]
                for p, kind, val, hk in T.children(t, i, h, a, b):
                    QP[a, b] += p * phi_of(kind, val, hk)
                    if kind == "live":
                        v, r = rec(t + 1, val, hk)
                    else:
                        v, r = (0.0 if kind == "cap" else val), 0.0
                    V += w * p * v; R += w * p * r
        R += phi_live(h) - x @ QP @ y
        err[0] = max(err[0], abs((phi_live(h) - V) - R))
        return V, R

    rec(0, 0, ())
    return err[0]


def l1_tree(rng, g, Vs, Qs):
    """L1(b) sandwich for a HISTORY-DEPENDENT agent: exact BR and myopic exploiter on the tree.
    Returns (expl - lower, upper - expl, expl, lower, upper)."""
    T = Tree(g, rng)
    pi = lambda t, i, h: T.draw(("x", h), lambda: rng.dirichlet(np.ones(g.nA[t][i])))

    def e_at(t, i, h):
        return Vs[t][i] - (pi(t, i, h) @ Qs[t][i]).min()

    def br(t, i, h):  # returns (value, E sum e) under BR (random tie break)
        x = pi(t, i, h); nb = g.nB[t][i]
        vals = np.zeros(nb); es = np.zeros(nb)
        for b in range(nb):
            for a in range(g.nA[t][i]):
                for p, kind, val, hk in T.children(t, i, h, a, b):
                    if kind == "live":
                        v, e = br(t + 1, val, hk)
                    else:
                        v, e = (0.0 if kind == "cap" else val), 0.0
                    vals[b] += x[a] * p * v; es[b] += x[a] * p * e
        ties = np.flatnonzero(vals <= vals.min() + 1e-12)
        b = ties[rng.integers(len(ties))]
        return vals[b], es[b] + e_at(t, i, h)

    def greedy(t, i, h):
        x = pi(t, i, h); col = x @ Qs[t][i]
        ties = np.flatnonzero(col <= col.min() + 1e-12)
        b = ties[rng.integers(len(ties))]
        v = 0.0; e = 0.0
        for a in range(g.nA[t][i]):
            for p, kind, val, hk in T.children(t, i, h, a, b):
                if kind == "live":
                    vv, ee = greedy(t + 1, val, hk)
                else:
                    vv, ee = (0.0 if kind == "cap" else val), 0.0
                v += x[a] * p * vv; e += x[a] * p * ee
        return v, e + e_at(t, i, h)

    vbr, upper = br(0, 0, ())
    vg, lower = greedy(0, 0, ())
    expl = Vs[0][0] - vbr
    ident_err = abs((Vs[0][0] - vg) - lower)  # under sigma_g, l_t = e_t exactly
    return expl - lower, upper - expl, expl, lower, upper, ident_err


# ---------------------------------------------------------------- L1, Markov ---------
def e_markov(g, X, Vs, Qs):
    return [np.array([Vs[t][i] - (X[t][i] @ Qs[t][i]).min() for i in range(g.nlive[t])])
            for t in range(g.H)]


def greedy_markov(rng, g, X, Qs):
    Y = []
    for t in range(g.H):
        Y.append([])
        for i in range(g.nlive[t]):
            col = X[t][i] @ Qs[t][i]
            ties = np.flatnonzero(col <= col.min() + 1e-12)
            Y[t].append(np.eye(len(col))[ties[rng.integers(len(ties))]])
    return Y


def l1_markov(rng, g, kind):
    Vs, Qs, Xs, Ys, _ = L.optimal(g)
    H = g.H
    if kind == "random":
        X = L.random_markov(rng, g, "A")
    else:  # perturbed equilibrium
        lam = rng.choice([0.01, 0.1, 0.5])
        X = [[(1 - lam) * Xs[t][i] + lam * rng.dirichlet(np.ones(g.nA[t][i]))
              for i in range(g.nlive[t])] for t in range(H)]
    e = e_markov(g, X, Vs, Qs)
    # (a) identity for a random opponent
    Yr = L.random_markov(rng, g, "B")
    W = L.evaluate(g, X, Yr); d, _ = L.forward(g, X, Yr)
    ell = [np.array([Vs[t][i] - X[t][i] @ Qs[t][i] @ Yr[t][i] for i in range(g.nlive[t])])
           for t in range(H)]
    ida = abs((Vs[0][0] - W[0][0]) - L.path_sum(d, ell, H))
    # (b) sandwich
    Ybr, Wbr = L.best_response(g, X, rng)
    expl = Vs[0][0] - Wbr[0][0]
    dbr, _ = L.forward(g, X, Ybr)
    upper = L.path_sum(dbr, e, H)
    Yg = greedy_markov(rng, g, X, Qs)
    dg, _ = L.forward(g, X, Yg)
    lower = L.path_sum(dg, e, H)
    Wg = L.evaluate(g, X, Yg)
    idg = abs((Vs[0][0] - Wg[0][0]) - lower)
    return ida, idg, expl - lower, upper - expl, expl, lower, upper, min(min(x) for x in e)


def l1c(rng, g):
    """Agent eps_s-optimal in Qhat with ||Qhat - Q*|| <= eps_v(t,s): check e <= eps_s + 2 eps_v
    pointwise and expl <= E_BR[sum budget] <= Hbar * sup budget."""
    Vs, Qs, _, _, _ = L.optimal(g)
    H = g.H
    X = []; bud = []; e_slack = np.inf
    for t in range(H):
        X.append([]); bud.append(np.zeros(g.nlive[t]))
        for i in range(g.nlive[t]):
            ev = rng.choice([0.0, 0.01, 0.05, 0.2]); M = Qs[t][i]
            Qh = M + rng.uniform(-ev, ev, M.shape)
            vh, xh, _ = L.solve(Qh)
            x = (1 - (lam := rng.choice([0, 0.05, 0.3]))) * xh + lam * rng.dirichlet(np.ones(len(xh)))
            es = vh - (x @ Qh).min()
            evv = np.abs(Qh - M).max()
            X[t].append(x); bud[t][i] = es + 2 * evv
            e_slack = min(e_slack, bud[t][i] - (Vs[t][i] - (x @ M).min()))
    Ybr, Wbr = L.best_response(g, X, rng)
    expl = Vs[0][0] - Wbr[0][0]
    dbr, _ = L.forward(g, X, Ybr)
    e = e_markov(g, X, Vs, Qs)
    upper = L.path_sum(dbr, e, H)
    Ebud = L.path_sum(dbr, bud, H)
    Hbar = sum(dbr[t].sum() for t in range(H))  # E[tau] = sum_t P(t < tau)
    supb = max(b.max() for b in bud)
    return e_slack, Ebud - upper, Hbar * supb - Ebud, upper - expl


# ---------------------------------------------------------------- Example L1b --------
def example_l1b(delta, variant=False):
    """variant=False: the note's example. variant=True: s1 has a safe row paying +1 and a bad
    row (+1 / -1); s0: b1 -> terminal 1 - delta.  Agent plays the bad row / pure heads."""
    H = 2
    if not variant:
        U = [np.zeros(0), np.array([-delta]), np.array([1.0, -1.0])]
        P1 = np.zeros((2, 2, 2)); P1[0, 0, 0] = P1[1, 1, 0] = 1; P1[0, 1, 1] = P1[1, 0, 1] = 1
    else:
        U = [np.zeros(0), np.array([1 - delta]), np.array([1.0, -1.0])]
        P1 = np.zeros((2, 2, 2)); P1[0, :, 0] = 1; P1[1, 0, 0] = 1; P1[1, 1, 1] = 1
    P0 = np.zeros((1, 2, 2)); P0[0, 0, 1] = 1; P0[0, 1, 0] = 1  # b1 -> terminal, b2 -> s1
    g = L.Game(H, [1, 1, 0], [0, 1, 2], U, [[1], [2]], [[2], [2]], [[P0], [P1]])
    Vs, Qs, _, _, _ = L.optimal(g)
    X = [[np.ones(1)], [np.array([1.0, 0.0]) if not variant else np.array([0.0, 1.0])]]
    e = e_markov(g, X, Vs, Qs)
    rng = np.random.default_rng(0)
    Ybr, Wbr = L.best_response(g, X, rng)
    expl = Vs[0][0] - Wbr[0][0]
    Yg = greedy_markov(rng, g, X, Qs)
    lower = L.path_sum(L.forward(g, X, Yg)[0], e, H)
    upper = L.path_sum(L.forward(g, X, Ybr)[0], e, H)
    return Vs[0][0], Vs[1][0], Qs[0][0].ravel(), e[0][0], lower, expl, upper


if __name__ == "__main__":
    print("== L0 (potential telescoping) ==")
    rng = np.random.default_rng(1)
    errs = [l0_markov(rng, L.random_game(rng, int(rng.integers(1, 7)))) for _ in range(1000)]
    print(f"Markov Phi, random Markov pi/sigma, 1000 games H=1..6: max |LHS-RHS| = {max(errs):.2e}")
    rng = np.random.default_rng(2)
    errs = []
    for _ in range(300):
        g = L.random_game(rng, int(rng.integers(1, 4)), max_states=2, max_act=2)
        errs.append(l0_tree(rng, g))
    print(f"history-dependent Phi, pi, sigma (full history tree), 300 games H=1..3: "
          f"max |LHS-RHS| over all live histories = {max(errs):.2e}")

    print("\n== L1 (a),(b) Markov agents ==")
    rng = np.random.default_rng(3)
    rows = []
    for k in range(600):
        g = L.random_game(rng, int(rng.integers(2, 6)))
        rows.append(l1_markov(rng, g, "random" if k % 2 == 0 else "perturbed"))
    R = np.array(rows)
    print(f"600 games H=2..5: (a) max identity err = {R[:,0].max():.2e}; sigma_g: max |V*-V - sum e| = {R[:,1].max():.2e}")
    print(f"(b) min (expl - lower) = {R[:,2].min():.2e}, min (upper - expl) = {R[:,3].min():.2e}, "
          f"min e_t = {R[:,7].min():.2e}")
    print(f"    strict lower in {np.mean(R[:,2] > 1e-9):.0%}, strict upper in {np.mean(R[:,3] > 1e-9):.0%}; "
          f"max upper-lower = {(R[:,6]-R[:,5]).max():.3f}")

    print("\n== L1 (b) history-dependent agent (tree BR and tree sigma_g) ==")
    rng = np.random.default_rng(4)
    rows = []
    for _ in range(300):
        g = L.random_game(rng, int(rng.integers(1, 4)), max_states=2, max_act=2)
        Vs, Qs, _, _, _ = L.optimal(g)
        rows.append(l1_tree(rng, g, Vs, Qs))
    R = np.array(rows)
    print(f"300 games H=1..3: min (expl-lower) = {R[:,0].min():.2e}, min (upper-expl) = {R[:,1].min():.2e}, "
          f"sigma_g identity err = {R[:,5].max():.2e}")

    print("\n== L1 (c) Theorem A recovered ==")
    rng = np.random.default_rng(5)
    R = np.array([l1c(rng, L.random_game(rng, int(rng.integers(2, 6)))) for _ in range(500)])
    print(f"500 games: min[(eps_s+2eps_v) - e_t] = {R[:,0].min():.2e}; min[E_BR sum budget - upper] = {R[:,1].min():.2e}; "
          f"min[Hbar*sup - E_BR sum budget] = {R[:,2].min():.2e}; min[upper - expl] = {R[:,3].min():.2e}")

    print("\n== Example L1b ==")
    for dl in (0.1, 0.5, 0.9):
        v0, v1, q0, e0, lo, ex, up = example_l1b(dl)
        print(f"delta={dl}: V*0={v0:+.3f} V*(s1)={v1:+.3f} Q*0={q0} e0={e0:.3f} lower={lo:.3f} "
              f"expl={ex:.3f} (1-delta={1-dl:.3f}) upper={up:.3f}")
    print("variant (safe row +1 / bad row +1,-1 at s1; b1 -> 1-delta):")
    for dl in (0.1, 0.01):
        v0, v1, q0, e0, lo, ex, up = example_l1b(dl, variant=True)
        print(f"delta={dl}: V*0={v0:+.3f} V*(s1)={v1:+.3f} e0={e0:.3f} lower={lo:.3f} expl={ex:.3f} "
              f"(2-delta={2-dl:.3f}) upper={up:.3f}")

    print(f"\nLP certificate gap (max hi-lo) = {L.LP_GAP[0]:.2e};  runtime {time.time()-t0:.1f}s")
