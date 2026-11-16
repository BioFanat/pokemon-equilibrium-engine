"""Hidden-information agents: act(true_state, view, belief, legal_opts, side, rng) -> option.

  random
  oracle:<ms>            LP equilibrium of nash.engine_matrix on the TRUE state (full information)
  mostlikely:<ms>        one MAP determinization, LP, sample
  pimc:<K>:<ms>          K determinizations, LP each, blend x̄ = Σ b_k x_k (b_k = 1/K), sample
  agentform:<K>:<ms>     K determinizations, one LP: max_x Σ_k b_k min_j (M_kᵀ x)_j, sample

<ms> is the TOTAL per-decision engine budget (ms/K per determinization). All matrices are taken
from the acting side's perspective (side 2: -Mᵀ with rows = s2 labels). Rows are aligned by label
to the true legal options; a label missing from some M_k gets that M_k's column-wise minimum.
Columns are per determinization and never aligned.
"""
from __future__ import annotations

import random
import time

import numpy as np
from scipy.optimize import linprog

import hidden as hd
import nash
import pkbattle as pb


def own_matrix(M, s1, s2, side):
    return (M, s1, s2) if side == 1 else (-M.T, s2, s1)


def align(M, rows, legal):
    colmin = M.min(0)
    return np.array([M[rows.index(l)] if l in rows else colmin for l in legal])


def agent_form_lp(Ms, b):
    """max_x Σ_k b_k v_k  s.t. v_k <= (M_kᵀ x)_j  ∀ j, k;  x in simplex. Returns (x, value)."""
    m, K = Ms[0].shape[0], len(Ms)
    A, rhs = [], []
    for k, M in enumerate(Ms):
        blk = np.zeros((M.shape[1], m + K))
        blk[:, :m] = -M.T
        blk[:, m + k] = 1
        A.append(blk)
    r = linprog(np.r_[np.zeros(m), -np.asarray(b, float)], A_ub=np.vstack(A), b_ub=np.zeros(sum(M.shape[1] for M in Ms)),
                A_eq=np.r_[np.ones(m), np.zeros(K)][None], b_eq=[1],
                bounds=[(0, None)] * m + [(None, None)] * K, method="highs")
    x = np.clip(r.x[:m], 0, None)
    return x / x.sum(), -r.fun


def safex_strategy_worlds(Ms, b, qs, eps):
    """Per-world eps-safe exploit (nash.safex_strategy over worlds): max_x Σ_k b_k q_kᵀ M_kᵀ x  s.t. the agent-form
    guarantee Σ_k b_k v_k >= v_AF - eps * r with v_k <= (M_kᵀ x)_j ∀ j, k; x in simplex. q_k: predicted distribution
    over world k's own columns; v_AF = agent_form_lp's value; r = ptp over all M_k (1 if 0), so K = 1 is safex_strategy
    on M_1. eps = 0 gives the best agent-form-optimal x against the q_k. Falls back to agent_form_lp's x if the LP fails."""
    m, K = Ms[0].shape[0], len(Ms)
    b = np.asarray(b, float)
    x_af, v_af = agent_form_lp(Ms, b)
    r = float(np.ptp(np.concatenate([M.ravel() for M in Ms]))) or 1.0
    A = []
    for k, M in enumerate(Ms):
        blk = np.zeros((M.shape[1], m + K))
        blk[:, :m] = -M.T
        blk[:, m + k] = 1
        A.append(blk)
    A.append(np.r_[np.zeros(m), -b][None])
    c = -sum(bk * (M @ np.asarray(q, float)) for bk, M, q in zip(b, Ms, qs))
    res = linprog(np.r_[c, np.zeros(K)], A_ub=np.vstack(A),
                  b_ub=np.r_[np.zeros(sum(M.shape[1] for M in Ms)), -(v_af - eps * r)],
                  A_eq=np.r_[np.ones(m), np.zeros(K)][None], b_eq=[1],
                  bounds=[(0, None)] * m + [(None, None)] * K, method="highs")
    return res.x[:m] if res.status == 0 else x_af


def f_b(Ms, b, x):
    return float(sum(bk * (M.T @ x).min() for bk, M in zip(b, Ms)))


def hausdorff(C1, C2):
    D = np.abs(C1[:, :, None] - C2[:, None, :]).max(axis=0)
    return max(D.min(axis=1).max(), D.min(axis=0).max())


def delta_H(Ms):
    cands = list(Ms) + [np.hstack(Ms)]
    return float(min(max(hausdorff(M, C) for M in Ms) for C in cands))


def lp_x(M):
    x, _, v = nash.lp(M)
    x = np.clip(x, 0, None)
    return x / x.sum(), v


class Agent:
    def __init__(self, spec: str, log: bool = False):
        self.spec = spec
        self.opp = None                           # (opp view, opp belief), set by play_hidden for xlbr
        if spec.startswith("xlbr:"):              # exploiter of the agent spec that follows
            self.kind, self.target, self.argmax = "xlbr", Agent(spec[5:]), False
            self.log, self.records, self.log_s = False, [], 0.0
            self.xrng = random.Random(hash(spec) & 0xFFFF)
            return
        kind, *a = spec.split(":")
        self.argmax = "argmax" in a                # play the most likely action instead of sampling
        a = [x for x in a if x != "argmax"]
        self.kind = kind
        self.K = int(a[0]) if kind in ("pimc", "agentform") else 1
        self.ms = int(a[-1]) if a else 0
        self.log = log and kind in ("pimc", "agentform")
        self.records, self.log_s = [], 0.0

    def _matrices(self, true_state, view, belief, legal, side, rng):
        if self.kind == "oracle":
            states = [true_state]
        elif self.kind == "mostlikely":
            states = [hd.determinize(true_state, view, belief, rng, map_=True)]
        else:
            states = [hd.determinize(true_state, view, belief, rng) for _ in range(self.K)]
        per = max(1, self.ms // len(states))
        return [align(*own_matrix(*nash.engine_matrix(s, per), side)[:2], legal) for s in states]

    def strategy(self, true_state, view, belief, legal, side, rng) -> np.ndarray:
        if self.kind == "random" or len(legal) == 1:
            return np.full(len(legal), 1 / len(legal))
        Ms = self._matrices(true_state, view, belief, legal, side, rng)
        b = np.full(len(Ms), 1 / len(Ms))
        if self.kind in ("oracle", "mostlikely"):
            return lp_x(Ms[0])[0]
        if self.kind == "pimc" and not self.log:
            return sum(bk * lp_x(M)[0] for bk, M in zip(b, Ms))
        if self.kind == "agentform" and not self.log:
            return agent_form_lp(Ms, b)[0]
        t = time.perf_counter()                   # logging path: compute both, time the extra solve
        xbar = sum(bk * lp_x(M)[0] for bk, M in zip(b, Ms))
        t1 = time.perf_counter()
        x_af, v_af = agent_form_lp(Ms, b)
        t2 = time.perf_counter()
        allv = np.concatenate([M.ravel() for M in Ms])
        self.records.append(dict(agent=self.spec, side=side, K=len(Ms), shapes=[list(M.shape) for M in Ms],
                                 v_AF=v_af, v_blend=f_b(Ms, b, xbar), delta_H=delta_H(Ms),
                                 payoff_range=float(allv.max() - allv.min()),
                                 tv_x=float(0.5 * np.abs(x_af - xbar).sum()),
                                 argmax_same=bool(np.argmax(x_af) == np.argmax(xbar))))
        extra = (t2 - t1) if self.kind == "pimc" else (t1 - t)
        self.log_s += extra + (time.perf_counter() - t2)
        return xbar if self.kind == "pimc" else x_af

    def exploit(self, true_state, legal, side):
        """Local best response with full information: knows the target's algorithm and the true
        state. Recomputes the target's strategy q from the target's own view and belief (with its own
        resampled worlds), then plays the best reply to q in the true full matrix."""
        if len(legal) == 1:
            return legal[0]
        ov, ob = self.opp
        o = 3 - side
        o_legal = pb.options(true_state)[o - 1]
        q = self.target.strategy(true_state, ov, ob, o_legal, o, self.xrng)
        if self.target.argmax:
            q = np.eye(len(q))[int(np.argmax(q))]
        M, s1, s2 = nash.full_matrix(true_state)
        return s1[int((M @ q).argmax())] if side == 1 else s2[int((q @ M).argmin())]

    def act(self, true_state, view, belief, legal, side, rng):
        if self.kind == "xlbr":
            return self.exploit(true_state, legal, side)
        p = self.strategy(true_state, view, belief, legal, side, rng)
        if self.argmax and self.kind != "random":
            return legal[int(np.argmax(p))]
        return rng.choices(legal, weights=p)[0]
