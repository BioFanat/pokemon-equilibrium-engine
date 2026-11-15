"""Shared exact toolkit for the lk*.py checks (search_rl_links.md, L0/L1/L5/L6).

Finite-horizon two-player zero-sum stochastic game, simultaneous moves, chance.
Stages t = 0..H.  Stage t has nlive[t] live states (nlive[0] = 1, the root) and,
for t >= 1, nterm[t] terminal nodes with payoffs U[t] in [-1, 1].  A live state at
stage H is the cap and pays 0; a terminal node at stage H pays its u (terminal is
checked before the cap).  P[t][i] has shape (nA, nB, nlive[t+1] + nterm[t+1]).

Everything is exact up to the LP solver; every LP is certified by its bracket
lo = min_b x^T M e_b <= val <= hi = max_a e_a^T M y, and the largest hi - lo seen
is kept in LP_GAP so slack can be attributed to the solver.
"""
import numpy as np
from scipy.optimize import linprog

LP_GAP = [0.0]
TOL = 1e-12


def solve(M):
    """Exact (LP) value of the matrix game M (row maximises). Returns val, x, y."""
    M = np.asarray(M, float)
    n, m = M.shape
    if n == 1:
        b = np.argmin(M[0]); y = np.zeros(m); y[b] = 1.0
        return M[0, b], np.ones(1), y
    if m == 1:
        a = np.argmax(M[:, 0]); x = np.zeros(n); x[a] = 1.0
        return M[a, 0], x, np.ones(1)
    opts = dict(primal_feasibility_tolerance=1e-10, dual_feasibility_tolerance=1e-10)
    # max v s.t. M^T x >= v
    c = np.zeros(n + 1); c[-1] = -1.0
    A = np.hstack([-M.T, np.ones((m, 1))])
    r = linprog(c, A_ub=A, b_ub=np.zeros(m), A_eq=np.r_[np.ones(n), 0.0][None],
                b_eq=[1.0], bounds=[(0, None)] * n + [(None, None)], method="highs", options=opts)
    x = np.clip(r.x[:n], 0, None); x /= x.sum()
    c = np.zeros(m + 1); c[-1] = 1.0
    A = np.hstack([M, -np.ones((n, 1))])
    r = linprog(c, A_ub=A, b_ub=np.zeros(n), A_eq=np.r_[np.ones(m), 0.0][None],
                b_eq=[1.0], bounds=[(0, None)] * m + [(None, None)], method="highs", options=opts)
    y = np.clip(r.x[:m], 0, None); y /= y.sum()
    lo = (x @ M).min(); hi = (M @ y).max()
    LP_GAP[0] = max(LP_GAP[0], hi - lo)
    return 0.5 * (lo + hi), x, y


class Game:
    def __init__(self, H, nlive, nterm, U, nA, nB, P):
        self.H, self.nlive, self.nterm, self.U = H, nlive, nterm, U
        self.nA, self.nB, self.P = nA, nB, P

    def full(self, t, Vlive, Uterm=None):
        """Value vector over all nodes at stage t (live then terminal)."""
        if t == 0:
            return np.asarray(Vlive, float)
        Ut = self.U[t] if Uterm is None else Uterm[t]
        return np.concatenate([np.asarray(Vlive, float), Ut])

    def Q(self, t, i, Vnext_live, Uterm=None):
        return self.P[t][i] @ self.full(t + 1, Vnext_live, Uterm)


def random_game(rng, H, max_states=4, max_act=3, p_forced=0.15, p_term=0.35):
    nlive = [1] + [int(rng.integers(1, max_states + 1)) for _ in range(H)]
    nterm = [0] + [int(rng.integers(1, 4)) for _ in range(H)]
    U = [np.zeros(0)] + [rng.uniform(-1, 1, nterm[t]) for t in range(1, H + 1)]
    nA, nB, P = [], [], []
    for t in range(H):
        nA.append([]); nB.append([]); P.append([])
        K = nlive[t + 1] + nterm[t + 1]
        for i in range(nlive[t]):
            a = 1 if rng.random() < p_forced else int(rng.integers(2, max_act + 1))
            b = 1 if rng.random() < p_forced else int(rng.integers(2, max_act + 1))
            T = np.zeros((a, b, K))
            for ia in range(a):
                for ib in range(b):
                    k = int(rng.integers(1, min(3, K) + 1))
                    # bias toward terminals with prob p_term
                    w = np.where(np.arange(K) >= nlive[t + 1], p_term, 1 - p_term) + 1e-9
                    sup = rng.choice(K, size=k, replace=False, p=w / w.sum())
                    T[ia, ib, sup] = rng.dirichlet(np.ones(k))
            nA[t].append(a); nB[t].append(b); P[t].append(T)
    return Game(H, nlive, nterm, U, nA, nB, P)


def zeros_like_V(g):
    return [np.zeros(g.nlive[t]) for t in range(g.H + 1)]


def optimal(g, Uterm=None, leafV=None):
    """Backward induction. Returns V (list over t of live values; V[H] = 0), Q, X, Y.
    If leafV is given, V is not computed recursively: Q[t] = Q[leafV]_t (one exact step),
    V = leafV, and X/Y are the stage maximin/minimax of Q[leafV]."""
    H = g.H
    V = zeros_like_V(g); Q = [[None] * g.nlive[t] for t in range(H)]
    X = [[None] * g.nlive[t] for t in range(H)]; Y = [[None] * g.nlive[t] for t in range(H)]
    vals = [np.zeros(g.nlive[t]) for t in range(H + 1)]
    for t in range(H - 1, -1, -1):
        nxt = leafV[t + 1] if leafV is not None else V[t + 1]
        for i in range(g.nlive[t]):
            Q[t][i] = g.Q(t, i, nxt, Uterm)
            vals[t][i], X[t][i], Y[t][i] = solve(Q[t][i])
        V[t] = vals[t] if leafV is None else np.asarray(leafV[t], float)
    if leafV is not None:
        V[H] = np.asarray(leafV[H], float)
    return V, Q, X, Y, vals


def evaluate(g, X, Y, Uterm=None):
    """W[t][i] = value of Markov pair (X, Y) from (t, i)."""
    W = zeros_like_V(g)
    for t in range(g.H - 1, -1, -1):
        W[t] = np.array([X[t][i] @ g.Q(t, i, W[t + 1], Uterm) @ Y[t][i] for i in range(g.nlive[t])])
    return W


def best_response(g, X, rng=None, Uterm=None):
    """Exact best response (minimiser) to Markov agent X. Ties broken at random if rng."""
    W = zeros_like_V(g); Y = [[None] * g.nlive[t] for t in range(g.H)]
    for t in range(g.H - 1, -1, -1):
        for i in range(g.nlive[t]):
            col = X[t][i] @ g.Q(t, i, W[t + 1], Uterm)
            W[t][i] = col.min()
            ties = np.flatnonzero(col <= col.min() + 1e-12)
            b = ties[rng.integers(len(ties))] if rng is not None else ties[0]
            Y[t][i] = np.eye(len(col))[b]
    return Y, W


def forward(g, X, Y):
    """d[t][i] = P(s_t = live i, t < tau) for t < H; d[H] = P(cap at i);
    z[t][k] = P(terminal k at stage t)."""
    d = [np.zeros(g.nlive[t]) for t in range(g.H + 1)]
    z = [np.zeros(g.nterm[t]) for t in range(g.H + 1)]
    d[0][0] = 1.0
    for t in range(g.H):
        for i in range(g.nlive[t]):
            if d[t][i] == 0:
                continue
            p = np.einsum("a,b,abk->k", X[t][i], Y[t][i], g.P[t][i]) * d[t][i]
            d[t + 1] += p[: g.nlive[t + 1]]; z[t + 1] += p[g.nlive[t + 1]:]
    return d, z


def path_sum(d, f, H):
    """E[sum_{t<tau} f_t(s_t)] from the forward distribution."""
    return sum(float(d[t] @ f[t]) for t in range(H))


def mix_eps_opt(rng, x_opt, M, lam):
    """x = (1-lam) x_opt + lam r; returns x and its gap eps = val - min_b x^T M e_b."""
    r = rng.dirichlet(np.ones(len(x_opt)))
    x = (1 - lam) * x_opt + lam * r
    return x


def gap(M, x, val):
    return val - (x @ M).min()


def random_markov(rng, g, side):
    n = g.nA if side == "A" else g.nB
    return [[rng.dirichlet(np.ones(n[t][i])) for i in range(g.nlive[t])] for t in range(g.H)]
