"""Shared helpers for lk2_safe_hedge.py / lk3_rl_hedge.py: random finite-horizon zero-sum stochastic
games with simultaneous moves and chance, exact matrix-game values by LP, exact V*, Q*.

Game: stages t = 0..H-1, stage 0 has one state, later stages S states. At (t, s) the agent (maximiser)
has nA actions (nA = 1 sometimes: forced turn), the opponent nB. A joint action (a, b) ends the game
with probability pt[a,b] and payoff u[a,b] in [-1,1]; otherwise the next state at stage t+1 is drawn
from P[a,b,:]. At t = H-1 the non-terminal mass hits the cap (payoff 0)."""
from __future__ import annotations

import numpy as np
from scipy.optimize import linprog


def maximin(A):
    """Row player's maximin strategy and value of matrix A (row maximises)."""
    n, m = A.shape
    c = np.zeros(n + 1); c[-1] = -1.0
    A_ub = np.hstack([-A.T, np.ones((m, 1))])          # v - x^T A e_b <= 0
    A_eq = np.hstack([np.ones((1, n)), np.zeros((1, 1))])
    res = linprog(c, A_ub=A_ub, b_ub=np.zeros(m), A_eq=A_eq, b_eq=[1.0],
                  bounds=[(0, None)] * n + [(None, None)], method="highs")
    x = np.clip(res.x[:n], 0, None); x /= x.sum()
    return x, float((x @ A).min())


def minimax(A):
    """Column player's minimax strategy (column minimises row's payoff)."""
    y, v = maximin(-A.T)
    return y, -v


class Game:
    def __init__(self, rng, H=5, S=3, amax=4, pforced=0.15, pterm=(0.0, 0.5)):
        self.H, self.S = H, S
        self.nst = [1] + [S] * (H - 1)
        self.cells = {}
        for t in range(H):
            for s in range(self.nst[t]):
                nA = 1 if rng.random() < pforced else int(rng.integers(2, amax + 1))
                nB = int(rng.integers(2, amax + 1))
                pt = rng.uniform(*pterm, size=(nA, nB))
                u = rng.uniform(-1, 1, size=(nA, nB))
                nxt = self.nst[t + 1] if t + 1 < H else 1
                P = rng.dirichlet(np.ones(nxt) * 0.7, size=(nA, nB))
                self.cells[t, s] = dict(nA=nA, nB=nB, pt=pt, u=u, P=P)
        self.solve()

    def solve(self):
        """Exact Q*, V* by backward induction (V*_H = 0 at the cap)."""
        self.V, self.Q = {}, {}
        for t in reversed(range(self.H)):
            Vn = np.array([self.V[t + 1, s] for s in range(self.nst[t + 1])]) if t + 1 < self.H else np.zeros(1)
            for s in range(self.nst[t]):
                c = self.cells[t, s]
                cont = Vn if t + 1 < self.H else np.zeros(1)
                Q = c["pt"] * c["u"] + (1 - c["pt"]) * (c["P"] @ cont)
                self.Q[t, s] = Q
                self.V[t, s] = maximin(Q)[1]

    def step(self, t, s, a, b, rng):
        """Returns (done, payoff_if_done, next_state)."""
        c = self.cells[t, s]
        if rng.random() < c["pt"][a, b]:
            return True, float(c["u"][a, b]), None
        if t + 1 >= self.H:
            return True, 0.0, None                       # cap
        return False, None, int(rng.choice(len(c["P"][a, b]), p=c["P"][a, b]))

    def next_value(self, t, s, a, b, done, pay, s2):
        return pay if done else self.V[t + 1, s2]
