"""Outcome-sampling simultaneous-move MCTS with regret matching at every node (SM-MCTS-RM).

Lisý, Kovařík, Lanctot & Bošanský 2013 (Convergence of MCTS in simultaneous move games); Lanctot, Lisý & Winands
2013 (SM-MCTS in Goofspiel). Each node keeps one regret table per player. A visit samples a ~ σ1, b ~ σ2 with
σ_i = (1 - γ) RM(r_i) + γ / n_i, a chance branch by its percentage, recurses, and updates both tables with the
importance-weighted estimate ũ(a') = u / σ_i(a) if a' = a else 0: r_i(a') += ũ(a') - u (side two uses -u). The
average strategy accumulates the unmixed regret-matching strategy at every visit. Values are side one's, in [-1, 1].
"""
import time

import numpy as np
import poke_engine as pe

import nash
import pkbattle as pb

GAMMA = 0.1
LEAF_TAU = 160.0


class RMNode:
    """Regret tables and average-strategy sums of one simultaneous-move node (pure; no engine)."""

    def __init__(self, n1, n2, gamma=GAMMA, plus=True):
        self.r = [np.zeros(n1), np.zeros(n2)]
        self.avg = [np.zeros(n1), np.zeros(n2)]
        self.gamma, self.plus, self.visits = gamma, plus, 0

    def current(self, i):
        """Unmixed regret-matching strategy of player i (uniform when no positive regret)."""
        pos = np.clip(self.r[i], 0, None)
        total = pos.sum()
        return pos / total if total > 0 else np.full(len(pos), 1.0 / len(pos))

    def strategies(self):
        """(sampling strategies with γ exploration, unmixed RM strategies)."""
        cur = [self.current(0), self.current(1)]
        return [(1 - self.gamma) * c + self.gamma / len(c) for c in cur], cur

    def update(self, a, b, u, sigma, current):
        """One outcome-sampled update from the joint action (a, b) with side-one value u."""
        self.visits += 1
        for i, (act, val) in enumerate(((a, u), (b, -u))):
            est = np.zeros(len(self.r[i]))
            est[act] = val / sigma[i][act]
            self.r[i] += est - val
            if self.plus:
                np.clip(self.r[i], 0, None, out=self.r[i])
            self.avg[i] += current[i]

    def average(self, i):
        total = self.avg[i].sum()
        return self.avg[i] / total if total > 0 else np.full(len(self.avg[i]), 1.0 / len(self.avg[i]))


def solve_matrix(A, iters, rng, gamma=GAMMA, plus=True):
    """SM-MCTS-RM at a single node of a matrix game (rows maximise A); returns the two average strategies."""
    node = RMNode(*A.shape, gamma=gamma, plus=plus)
    for _ in range(iters):
        sigma, cur = node.strategies()
        a, b = rng.choice(len(sigma[0]), p=sigma[0]), rng.choice(len(sigma[1]), p=sigma[1])
        node.update(a, b, A[a, b], sigma, cur)
    return node.average(0), node.average(1)


class _Node:
    def __init__(self, state):
        self.state = state
        self.terminal = nash.terminal_value(state)
        self.rows = self.cols = None
        if self.terminal is None:
            self.rows, self.cols = pb.options(state)
            self.rm = RMNode(len(self.rows), len(self.cols))
        self.branches, self.children = {}, {}


def _leaf_value(state):
    return float(np.tanh(nash._leaf(state, "mid") / LEAF_TAU))


def _terminal(v):
    return float(np.sign(v))


def _sample(p, rng):
    return int(rng.choice(len(p), p=p))


def _simulate(node, depth, rng, stats):
    if node.terminal is not None:
        return _terminal(node.terminal)
    if depth == 0:
        return _leaf_value(node.state)
    sigma, cur = node.rm.strategies()
    a, b = _sample(sigma[0], rng), _sample(sigma[1], rng)
    if (a, b) not in node.branches:
        brs = pe.generate_instructions(node.state, node.rows[a], node.cols[b])
        p = np.array([br.percentage for br in brs], dtype=float)
        node.branches[(a, b)] = (brs, p / p.sum())
    brs, p = node.branches[(a, b)]
    k = _sample(p, rng) if len(p) > 1 else 0
    key = (a, b, k)
    child = node.children.get(key)
    if child is None:
        child = node.children[key] = _Node(node.state.apply_instructions(brs[k]))
        stats["nodes"] += 1
    u = _simulate(child, depth - 1, rng, stats)
    node.rm.update(a, b, u, sigma, cur)
    return u


def search(state, duration_ms, depth, rng):
    """Time-budgeted SM-MCTS-RM from `state` (at least one iteration). Returns (root labels, root average strategy
    of side one, stats)."""
    t0 = time.perf_counter()
    deadline = t0 + duration_ms / 1000.0
    root = _Node(state)
    stats = dict(iterations=0, nodes=1, depth=depth)
    if root.terminal is not None:
        raise ValueError("SM-MCTS-RM root is terminal")
    while True:
        _simulate(root, depth, rng, stats)
        stats["iterations"] += 1
        if time.perf_counter() >= deadline:
            break
    stats["elapsed_secs"] = time.perf_counter() - t0
    return root.rows, root.rm.average(0), stats
