"""Per-turn equilibrium play on poke-engine, no learning.

Each turn: an unpruned depth-2 expectiminimax over poke-engine (see full_matrix) gives the joint-action
payoff matrix M (rows = side one's options, cols = side two's; zero-sum, in the engine's
evaluation units). We solve the matrix game three ways:

  * rm_plus  - deterministic no-regret self-play (Regret Matching+, linear averaging)
  * omwu     - deterministic optimistic multiplicative weights
  * closed_form - Shapley-Snow: once the equilibrium supports I, J are known, the solution
                  is closed form: v = 1 / (1^T A^-1 1), x_I = v 1^T A^-1, y_J = v A^-1 1,
                  with A = M[I, J] (after shifting M so v > 0)
  * lp       - exact reference

and play by sampling from the equilibrium mix.
"""
from __future__ import annotations

import os
import time

import numpy as np
import poke_engine as pe
from scipy.optimize import linprog

import pkbattle as pb


# ---------------------------------------------------------------- solvers

def exploitability(M: np.ndarray, x: np.ndarray, y: np.ndarray) -> float:
    """Nash gap: max_a (M y)_a - min_b (x^T M)_b  (0 at equilibrium)."""
    return float((M @ y).max() - (x @ M).min())


def lp(M: np.ndarray):
    m, n = M.shape
    c = np.r_[np.zeros(m), -1.0]
    r = linprog(c, A_ub=np.c_[-M.T, np.ones(n)], b_ub=np.zeros(n),
                A_eq=np.r_[np.ones(m), 0.0][None], b_eq=[1],
                bounds=[(0, None)] * m + [(None, None)], method="highs")
    x = r.x[:m]
    rd = linprog(np.r_[np.zeros(n), 1.0], A_ub=np.c_[M, -np.ones(m)], b_ub=np.zeros(m),
                 A_eq=np.r_[np.ones(n), 0.0][None], b_eq=[1],
                 bounds=[(0, None)] * n + [(None, None)], method="highs")
    return x, rd.x[:n], -r.fun


def rm_plus(M: np.ndarray, iters: int = 1000):
    """Regret Matching+ with alternating updates and linear averaging (deterministic)."""
    m, n = M.shape
    Rx, Ry = np.zeros(m), np.zeros(n)
    sx, sy = np.zeros(m), np.zeros(n)
    x, y = np.full(m, 1 / m), np.full(n, 1 / n)
    for t in range(1, iters + 1):
        u = M @ y
        Rx = np.maximum(Rx + u - x @ u, 0)
        x = Rx / Rx.sum() if Rx.sum() > 0 else np.full(m, 1 / m)
        w = -(x @ M)
        Ry = np.maximum(Ry + w - y @ w, 0)
        y = Ry / Ry.sum() if Ry.sum() > 0 else np.full(n, 1 / n)
        sx += t * x; sy += t * y
    return sx / sx.sum(), sy / sy.sum()


def omwu(M: np.ndarray, iters: int = 1000, eta: float | None = None):
    """Optimistic multiplicative weights (deterministic); last iterate converges in zero-sum games."""
    m, n = M.shape
    scale = np.ptp(M) or 1.0
    eta = eta or 0.1 / scale
    lx, ly = np.zeros(m), np.zeros(n)
    gx_prev, gy_prev = np.zeros(m), np.zeros(n)
    x, y = np.full(m, 1 / m), np.full(n, 1 / n)
    for _ in range(iters):
        gx, gy = M @ y, -(x @ M)
        lx += gx; ly += gy
        ax, ay = eta * (lx + gx - gx_prev), eta * (ly + gy - gy_prev)
        x = np.exp(ax - ax.max()); x /= x.sum()
        y = np.exp(ay - ay.max()); y /= y.sum()
        gx_prev, gy_prev = gx, gy
    return x, y


def closed_form(M: np.ndarray, I, J):
    """Shapley-Snow kernel formula on supports I (rows) and J (cols); requires |I| = |J|."""
    shift = 1.0 - M.min()                  # make all payoffs positive so v > 0
    A = M[np.ix_(I, J)] + shift
    Ainv = np.linalg.inv(A)
    one = np.ones(len(I))
    v = 1.0 / (one @ Ainv @ one)
    x, y = np.zeros(M.shape[0]), np.zeros(M.shape[1])
    x[I], y[J] = v * (one @ Ainv), v * (Ainv @ one)
    return x, y, v - shift


# ---------------------------------------------------------------- engine glue

# Opt-in leaf controls (read once at import; with neither set the leaf path is unchanged).
# Under CPU load pe.id(child, 0) sometimes returns a complete deeper search (depth_searched >= 2, or a wrapped
# negative i8); OURBOT_DET_LEAF=1 retries until depth_searched == 1 (up to DET_LEAF_TRIES, else keeps the last
# result and counts a fallback). OURBOT_LEAF_STATS=1 (or OURBOT_DET_LEAF=1) counts leaf calls in LEAF_STATS:
# non_d1 / depth_hist are over first-try returns (depth_hist keyed by str(depth_searched)).
DET_LEAF = os.environ.get("OURBOT_DET_LEAF") == "1"
LEAF_STATS_ON = DET_LEAF or os.environ.get("OURBOT_LEAF_STATS") == "1"
DET_LEAF_TRIES = 100
LEAF_STATS = {"calls": 0, "non_d1": 0, "retries": 0, "fallback": 0, "depth_hist": {}}


def _leaf_id(child: pe.State):
    """pe.id(child, 0) with LEAF_STATS counting and, under DET_LEAF, the depth_searched == 1 retry."""
    r = pe.id(child, 0)
    d = int(r.depth_searched)
    LEAF_STATS["calls"] += 1
    LEAF_STATS["depth_hist"][str(d)] = LEAF_STATS["depth_hist"].get(str(d), 0) + 1
    if d == 1:
        return r
    LEAF_STATS["non_d1"] += 1
    if DET_LEAF:
        tries = 1
        while r.depth_searched != 1 and tries < DET_LEAF_TRIES:
            r = pe.id(child, 0)
            tries += 1
            LEAF_STATS["retries"] += 1
        if r.depth_searched != 1:
            LEAF_STATS["fallback"] += 1
    return r


def _child_value(child: pe.State) -> float:
    """Side-one maximin of the child's depth-1 matrix = the engine's own leaf value one ply down.
    Exact even though pe.id prunes (a pruned row already has a cell <= the row it lost to, and
    Rust's comparisons skip NaN like nanmin), and independent of the depth-1 row-label bug."""
    r = _leaf_id(child) if LEAF_STATS_ON else pe.id(child, 0)
    M = np.array([np.nan if v is None else v for v in r.matrix], dtype=float).reshape(len(r.s1), len(r.s2))
    with np.errstate(all="ignore"):
        return float(np.nanmax(np.nanmin(M, axis=1)))


def _mirror(st: pe.State) -> pe.State:
    """Same position with the sides swapped (field-level fields copied)."""
    from hidden import GF
    return pe.State(side_one=st.side_two, side_two=st.side_one, **{f: getattr(st, f) for f in GF})


def _leaf(child: pe.State, leaf: str) -> float:
    """'mid' (default): midpoint of side one's maximin and side two's maximin (via the mirrored
    position) -- both bracket the child's matrix-game value, the midpoint is exact when the child
    has a pure saddle and makes the matrix (nearly) antisymmetric under swapping sides.
    'maximin': side one's maximin only (the engine's own pick_safest leaf; biased toward side two:
    swapping sides changes cells by a median 32% of the payoff range on test positions)."""
    # pe.id on an already finished battle returns its material heuristic, not
    # the search's win/loss reward. A guaranteed win must outrank healing or
    # setup in a position that still has an opponent left.
    a1, a2 = pb.alive(child.side_one), pb.alive(child.side_two)
    if a1 == 0 or a2 == 0:
        return 0.0 if a1 == a2 else (10000.0 if a1 else -10000.0)
    lo = _child_value(child)
    return lo if leaf == "maximin" else 0.5 * (lo - _child_value(_mirror(child)))


def full_matrix(state: pe.State, leaf: str = "mid", side_one_options=None):
    """Unpruned depth-2 joint-action matrix: M[i, j] = sum_b p_b * leaf(apply(branch_b)).
    Every cell is computed (pe.id's root alpha-beta leaves 55-80% of cells NaN). Note the Python
    generate_instructions branches on damage rolls while the engine's search does not, so cells
    differ slightly (~2% of payoff range at worst position) from the engine's own depth-2 values.
    side_one_options optionally intersects root rows with the live server's request.
    """
    s1, s2 = pb.options(state)
    if side_one_options is not None:
        allowed = set(side_one_options)
        s1 = [a for a in s1 if a in allowed]
        if not s1:
            raise ValueError("No engine options match the server's legal actions")
    M = np.empty((len(s1), len(s2)))
    for i, a in enumerate(s1):
        for j, b in enumerate(s2):
            M[i, j] = sum(br.percentage / 100.0 * _leaf(state.apply_instructions(br), leaf)
                          for br in pe.generate_instructions(state, a, b))
    return M, s1, s2


def squash_matrix(state: pe.State, tau: float):
    """full_matrix with a squashed leaf: a child where a side has no mon left scores exactly +1 / -1
    from side one's view (0 if both), every other child tanh(_leaf(child, 'mid') / tau). Entries in
    [-1, 1]. Ported from the leaf prototype (scratchpad leaf/mini_match.py)."""
    s1, s2 = pb.options(state)
    M = np.zeros((len(s1), len(s2)))
    for i, a in enumerate(s1):
        for j, b in enumerate(s2):
            for br in pe.generate_instructions(state, a, b):
                c, p = state.apply_instructions(br), br.percentage / 100.0
                a1, a2 = pb.alive(c.side_one), pb.alive(c.side_two)
                if a1 == 0 or a2 == 0:
                    M[i, j] += p * (0.0 if a1 == a2 else (1.0 if a1 else -1.0))
                else:
                    M[i, j] += p * np.tanh(_leaf(c, "mid") / tau)
    return M, s1, s2


def engine_matrix(state: pe.State, ms: int = 50, legacy: bool = False):
    """Payoff matrix (rows = side one's options, cols = side two's; side-one perspective).

    Default: `full_matrix` (fixed depth 2, midpoint leaves, `ms` is ignored; ~2x the ~65 ms
    median of maximin leaves).
    legacy=True: the original pe.id(state, ms) matrix with pruned (NaN) cells set to 0 -- WRONG
    (it invents values for most cells); kept only to reproduce E1-E3 in results/experiments/.
    """
    if not legacy:
        return full_matrix(state)
    r = pe.id(state, ms)
    s1, s2 = [pb._canon(a) for a in r.s1], [pb._canon(b) for b in r.s2]
    M = np.array([np.nan if v is None else v for v in r.matrix], dtype=float).reshape(len(s1), len(s2))
    finite = M[np.isfinite(M)]
    big = 2 * max(1.0, np.abs(finite).max()) if finite.size else 1.0
    M = np.nan_to_num(M, nan=0.0, posinf=big, neginf=-big)
    return M, s1, s2


def nash_policy(solver: str = "rm_plus", ms: int = 50, iters: int = 1000,
                pick: str = "sample", legacy: bool = False, leaf_tau: float | None = None):
    """pick='sample' plays the equilibrium mix; 'argmax' its most likely action (better against
    non-adaptive opponents, exploitable by adaptive ones). leaf_tau: solve squash_matrix(state, leaf_tau)
    (exact +-1 terminals, tanh-squashed leaf) instead of engine_matrix."""
    def pol(state, opts, side, rng):
        if len(opts) == 1:
            return opts[0]
        M, s1, s2 = engine_matrix(state, ms, legacy) if leaf_tau is None else squash_matrix(state, leaf_tau)
        if solver == "lp":
            x, y, _ = lp(M)
        elif solver == "omwu":
            x, y = omwu(M, iters)
        else:
            x, y = rm_plus(M, iters)
        acts, p = (s1, x) if side == 1 else (s2, y)
        p = np.clip(p, 0, None)
        if pick == "argmax":
            return acts[int(p.argmax())]
        return rng.choices(acts, weights=p / p.sum())[0]
    return pol


def vnet_matrix(state: pe.State):
    """Depth-1 joint-action matrix with learned leaves: M[i, j] = sum_b p_b * V(apply(branch_b)),
    V = value_net (side one's expected outcome in [-1, 1]); a child where a side has no mon left
    is scored exactly +1 / -1 (0 if both). All non-terminal children go through one value_many call."""
    import value_net
    s1, s2 = pb.options(state)
    M = np.zeros((len(s1), len(s2)))
    cells, probs, kids = [], [], []
    for i, a in enumerate(s1):
        for j, b in enumerate(s2):
            for br in pe.generate_instructions(state, a, b):
                child, p = state.apply_instructions(br), br.percentage / 100.0
                a1, a2 = pb.alive(child.side_one), pb.alive(child.side_two)
                if a1 == 0 or a2 == 0:
                    M[i, j] += p * (0.0 if a1 == a2 else (1.0 if a1 else -1.0))
                else:
                    cells.append((i, j)); probs.append(p); kids.append(child)
    if kids:
        w = np.array(probs) * value_net.value_many(kids)
        np.add.at(M, tuple(np.array(cells).T), w)
    return M, s1, s2


def vnet_policy(iters: int = 1000, pick: str = "argmax"):
    """RM+ equilibrium of vnet_matrix; 'argmax' plays its most likely action, 'sample' the mix."""
    def pol(state, opts, side, rng):
        if len(opts) == 1:
            return opts[0]
        M, s1, s2 = vnet_matrix(state)
        x, y = rm_plus(M, iters)
        acts, p = (s1, x) if side == 1 else (s2, y)
        p = np.clip(p, 0, None)
        if pick == "argmax":
            return acts[int(p.argmax())]
        return rng.choices(acts, weights=p / p.sum())[0]
    return pol


def safest_policy(ms: int = 50, legacy: bool = False):
    """Pure maximin (what poke-engine's get_safest_move does): deterministic, exploitable."""
    def pol(state, opts, side, rng):
        if len(opts) == 1:
            return opts[0]
        M, s1, s2 = engine_matrix(state, ms, legacy)
        return s1[int(M.min(1).argmax())] if side == 1 else s2[int(M.max(0).argmin())]
    return pol


def safex_strategy(A: np.ndarray, eps: float, q: np.ndarray | None = None) -> np.ndarray:
    """eps-safe exploitation (the safe-exploitation idea of Ganzfried & Sandholm 2015, one stage game).
    A is the deciding player's own-perspective matrix (rows = own actions, maximiser). Predict the
    opponent's move as q = one-hot on argmin_j mean_i A[i, j] (its best reply to our uniform mix), then
    solve  max_x x.A q  s.t.  (x^T A)_j >= v - eps * r for every column j, x in the simplex,
    with v = the exact game value (lp) and r = ptp(A) (1 if 0).
    Guarantee: against ANY opponent move the stage payoff is at least v - eps * r, i.e. the worst-case
    loss relative to the equilibrium value is at most eps times the payoff range; eps = 0 gives the best
    equilibrium strategy against q. Falls back to lp's equilibrium x if the LP fails.
    q: optional predicted distribution over the opponent's columns (maximise x.A q under the same constraints);
    None keeps the one-hot prediction above."""
    m, n = A.shape
    x_eq, _, v = lp(A)
    r = float(np.ptp(A)) or 1.0
    if q is None:
        q = np.zeros(n); q[int(A.mean(axis=0).argmin())] = 1
    res = linprog(-(A @ q), A_ub=-A.T, b_ub=-(v - eps * r) * np.ones(n), A_eq=np.ones((1, m)), b_eq=[1],
                  bounds=[(0, None)] * m, method="highs")
    return res.x if res.status == 0 else x_eq


def safex_policy(eps: float, pick: str = "sample"):
    """Each turn: safex_strategy on full_matrix from our own perspective (side two: -M.T); sample the
    eps-safe mix, or play its most likely action with pick='argmax'."""
    def pol(state, opts, side, rng):
        if len(opts) == 1:
            return opts[0]
        M, s1, s2 = full_matrix(state)
        A, acts = (M, s1) if side == 1 else (-M.T, s2)
        p = np.clip(safex_strategy(A, eps), 0, None)
        if pick == "argmax":
            return acts[int(p.argmax())]
        return rng.choices(acts, weights=p / p.sum())[0]
    return pol


def target_strategy(M: np.ndarray, side: int, target: str, iters: int = 1000) -> np.ndarray:
    """The mixed strategy a full-information matrix agent plays on M from `side`
    ('argmax' / 'sample' = nash_policy picks, 'safest' = safest_policy, 'safex:<eps>' = safex_policy
    sampled). Deterministic given M."""
    if target.startswith("safex:"):
        p = np.clip(safex_strategy(M if side == 1 else -M.T, float(target.split(":")[1])), 0, None)
        return p / p.sum()
    if target == "safest":
        k = int(M.min(1).argmax()) if side == 1 else int(M.max(0).argmin())
        q = np.zeros(M.shape[side - 1]); q[k] = 1
        return q
    x, y = rm_plus(M, iters)
    p = np.clip(x if side == 1 else y, 0, None)
    if target == "argmax":
        q = np.zeros_like(p); q[int(p.argmax())] = 1
        return q
    return p / p.sum()


def foulplay_strategy(state: pe.State, side: int, ms: int = 0, iters: int = 2000):
    """foul-play's move distribution (fp/search/main.py select_move_from_mcts_results) under full
    information: one sample with chance 1, so policy = root visit fractions of `side`'s options; keep
    the options with fraction >= 0.75 x the top one and renormalise (foul-play then random.choices
    among them with these weights). Returns (own options in pb.options order, q). One MCTS run
    (ms > 0: time budget, else `iters` iterations), so q is stochastic."""
    opts = pb.options(state)[side - 1]
    if len(opts) == 1:
        return opts, np.ones(1)
    r = pe.mcts(state, ms, iters, 1)
    res = r.s1 if side == 1 else r.s2
    frac = np.zeros(len(opts))
    for o in res:                                  # options MCTS never lists keep 0
        frac[opts.index(pb._canon(o.move_choice))] += o.visits
    frac /= frac.sum()
    q = np.where(frac >= 0.75 * frac.max(), frac, 0.0)
    return opts, q / q.sum()


def lbr_policy(target: str, iters: int = 1000):
    """Local best response (Lisý & Bowling 2017 style, one turn deep): knows the opponent is a
    full-matrix agent of type `target`, recomputes the same matrix M and the opponent's strategy q,
    and plays the best reply to q in M. A lower bound on that agent's exploitability.
    MCTS targets: 'foulplay_it:N' / 'foulplay_ms:N' (q = foulplay_strategy from the target's side) and
    'mcts' (q = one-hot on the most-visited move of a 2000-iteration run). MCTS is stochastic, so here q
    is the exploiter's own estimate from a fresh run, not the target's actual distribution this turn."""
    if target.startswith("foulplay_") or target == "mcts":
        def pol(state, opts, side, rng):
            if len(opts) == 1:
                return opts[0]
            M, s1, s2 = full_matrix(state)
            tside = 3 - side
            if target == "mcts":
                tops = (s1, s2)[tside - 1]
                r = pe.mcts(state, 0, 2000, 1)
                best = max(r.s1 if tside == 1 else r.s2, key=lambda x: x.visits)
                q = np.zeros(len(tops)); q[tops.index(pb._canon(best.move_choice))] = 1
            else:
                kind, n = target.split(":")
                _, q = foulplay_strategy(state, tside, *((0, int(n)) if kind == "foulplay_it" else (int(n), 0)))
            return s1[int((M @ q).argmax())] if side == 1 else s2[int((q @ M).argmin())]
        return pol

    def pol(state, opts, side, rng):
        if len(opts) == 1:
            return opts[0]
        M, s1, s2 = full_matrix(state)
        if side == 1:                              # opponent is side two: q over columns
            return s1[int((M @ target_strategy(M, 2, target, iters)).argmax())]
        return s2[int((target_strategy(M, 1, target, iters) @ M).argmin())]
    return pol


def wb2_policy(target: str, K: int = 5, mass: float = 0.95, supp_mass: float = 0.95, iters: int = 1000):
    """White-box depth-2 exploiter of a full-matrix agent of type `target` ('argmax' / 'sample' /
    'safex:<eps>', the sampled safex_policy).
    Root: recompute the target's matrix M and its exact strategy q = target_strategy(M) (the agent's
    own decision rule, so its argmax move or its mix). Candidates: the top-K own actions by depth-1
    value against q. For each candidate a and each target action b in q's support (top actions up to
    `supp_mass` of q, q renormalised over them), expand the chance branches of (a, b) (most likely
    first, truncated at `mass` and renormalised). Each child is valued as the best-response value
    against the target's exact strategy at that child, max_a' (M_c q_c) from the exploiter's side, with
    a win / loss / draw scored +-1e4 / 0 when a side has no mon left. Plays the candidate with the
    highest expected value. The leaf is full_matrix's midpoint leaf, so heuristic depth 3 along the
    exploited line. Ported from the white-box prototype (scratchpad rlbr/wb_exploiter.py), without its
    matrix cache shared with the target (full_matrix is deterministic, so the same M; wb2's ms/move then
    includes its own root matrix).
    Against a fixed target any fixed policy's expected score is at most the best response's, so wb2's
    score against the target is a valid lower bound on the target's exploitability (score - 0.5)."""
    TERM = 1e4

    def terminal_value(child, side):
        a1, a2 = pb.alive(child.side_one), pb.alive(child.side_two)
        if a1 and a2:
            return None
        if a1 == a2:
            return 0.0
        return TERM if (a1 > 0) == (side == 1) else -TERM

    def branches(state, own, opp, side):
        brs = pe.generate_instructions(state, own, opp) if side == 1 else pe.generate_instructions(state, opp, own)
        brs = sorted(brs, key=lambda b: -b.percentage)
        out, m = [], 0.0
        for b in brs:
            out.append(b); m += b.percentage / 100.0
            if m >= mass:
                break
        tot = sum(b.percentage for b in out) / 100.0
        return [(b.percentage / 100.0 / tot, b) for b in out]

    def child_value(child, side, tside):
        tv = terminal_value(child, side)
        if tv is not None:
            return tv
        Mc, _, _ = full_matrix(child)
        qc = target_strategy(Mc, tside, target, iters)
        return float((Mc @ qc).max()) if side == 1 else float(-(qc @ Mc).min())

    def pol(state, opts, side, rng):
        if len(opts) == 1:
            return opts[0]
        M, s1, s2 = full_matrix(state)
        tside = 3 - side
        q = target_strategy(M, tside, target, iters)
        own, topp = (s1, s2) if side == 1 else (s2, s1)
        d1 = M @ q if side == 1 else -(q @ M)                 # depth-1 value of each own action vs q
        cand = list(np.argsort(-d1)[:K])
        order = np.argsort(-q); cum = np.cumsum(q[order])
        supp = order[: int(np.searchsorted(cum, supp_mass) + 1)]
        q = q / q[supp].sum()
        best, bestv = None, -np.inf
        for i in cand:
            v = 0.0
            for j in supp:
                vj = 0.0
                for p, br in branches(state, own[i], topp[j], side):
                    vj += p * child_value(state.apply_instructions(br), side, tside)
                v += q[j] * vj
            if v > bestv:
                best, bestv = i, v
        return own[best]
    return pol


# ---------------------------------------------------------------- experiments

def solver_report(n_positions: int = 60, seed: int = 0):
    import random
    rng = random.Random(seed)
    rows, mixed = [], 0
    while len(rows) < n_positions:
        st = pb.random_state(rng)
        for _ in range(rng.randrange(0, 15)):          # advance to a random mid-game position
            o1, o2 = pb.options(st)
            st = pb.step(st, rng.choice(o1), rng.choice(o2), rng)
            if not pb.alive(st.side_one) or not pb.alive(st.side_two):
                break
        if not pb.alive(st.side_one) or not pb.alive(st.side_two):
            continue
        M, _, _ = engine_matrix(st, 50)
        if min(M.shape) < 2:
            continue
        scale = np.ptp(M) or 1.0
        x, y, v = lp(M)
        I, J = np.flatnonzero(x > 1e-7), np.flatnonzero(y > 1e-7)
        mixed += len(I) > 1
        rec = {"shape": M.shape, "support": (len(I), len(J)),
               "pure_gap": (v - M.min(1).max()) / scale}      # value of randomising vs safest pure move
        for name, f in [("rm+ 100", lambda: rm_plus(M, 100)), ("rm+ 1000", lambda: rm_plus(M, 1000)),
                        ("omwu 1000", lambda: omwu(M, 1000))]:
            t = time.perf_counter(); xs, ys = f(); dt = time.perf_counter() - t
            rec[name] = (exploitability(M, xs, ys) / scale, dt)
        if len(I) == len(J):
            xc, yc, vc = closed_form(M, I, J)
            rec["closed"] = (abs(vc - v) / scale, exploitability(M, xc, yc) / scale)
        rows.append(rec)
    print(f"{len(rows)} real mid-game positions, matrices {min(r['shape'] for r in rows)}..{max(r['shape'] for r in rows)}")
    print(f"equilibrium is mixed (support > 1) in {mixed}/{len(rows)} positions")
    pg = np.array([r["pure_gap"] for r in rows])
    print(f"value of randomising over the safest pure move: median {np.median(pg):.3f}, max {pg.max():.3f} (fraction of payoff range)")
    for name in ["rm+ 100", "rm+ 1000", "omwu 1000"]:
        e = np.array([r[name][0] for r in rows]); t = np.array([r[name][1] for r in rows])
        print(f"{name:10s}: Nash gap median {np.median(e):.2e}, max {e.max():.2e} | {np.median(t)*1e3:.1f} ms")
    cf = [r["closed"] for r in rows if "closed" in r]
    print(f"closed form (Shapley-Snow on LP supports): applicable {len(cf)}/{len(rows)}, "
          f"max |v - v_LP| {max(c[0] for c in cf):.1e}, max Nash gap {max(c[1] for c in cf):.1e}")


def match(p1, p2, n: int, seed: int):
    import random
    rng = random.Random(seed)
    w = [0, 0, 0]
    for g in range(n):
        if g % 2 == 0:
            r, _ = pb.play(p1, p2, rng)
            w[r] += 1
        else:                                            # swap sides every other game
            r, _ = pb.play(p2, p1, rng)
            w[{0: 0, 1: 2, 2: 1}[r]] += 1
    return w[1], w[2], w[0]


if __name__ == "__main__":
    solver_report()
    print()
    for name, opp in [("random", pb.random_policy), ("MCTS argmax (2000 it)", pb.mcts_policy(2000)),
                      ("safest pure move", safest_policy(50))]:
        t = time.perf_counter()
        a, b, d = match(nash_policy("rm_plus", 50), opp, 40, seed=1)
        print(f"RM+ equilibrium bot vs {name:22s}: {a}-{b} (draws {d})  [{time.perf_counter()-t:.0f}s]")
