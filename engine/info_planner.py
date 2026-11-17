"""S1: information-set-correct depth-2 planner (public-observation grouping at the children).

Design: research/info_gathering_concealment.md §3 (option S1). Engine simulation only; nothing here is called by
nash.py / hidden_agents.py / run_ourbot.py.

Today's agent (A0, hidden_agents 'agentform:K') solves one agent-form LP over K sampled worlds at the root, but values
every child world by world (nash._leaf, the midpoint of two pure maximins of the child's depth-1 matrix), i.e. at the
children it plays as if it knew the world (strategy fusion, assumption A3). S1 replaces the children's values:

  nodes      for every world k, root cell (a, beta) and chance branch: child c, p = P(branch | k, a, beta), and the
             PUBLIC observation key o (obs_key) the decider sees after the turn.
  groups     nodes of all worlds with the same key (the key contains a and beta) form one information set G_o; one
             continuation strategy x_o is shared by every node of G_o. The opponent sees the true node, so the group
             LP has one block per NODE: x_o = argmax_x sum_n pi_n min_j (M_n^T x)_j (hidden_agents.agent_form_lp).
             M_n = the child's UNPRUNED depth-1 matrix from the decider's perspective (full_d1), rows aligned by label
             (a label missing at some node gets that node's column-wise minimum, the hidden_agents.align convention).
  A1         live posterior: pi_n ∝ w_k p_n  (b'_k ∝ w_k P(o | k, a, beta)).
  A2         frozen posterior (no-information control): pi~_n ∝ w_k p_n / P(o | k, a, beta), i.e. every world present
             in G_o keeps its prior weight. Worlds with NO node in G_o cannot enter the LP (there is no matrix to put
             there), so they drop out in both arms: hard eliminations (a move world k lacks, disjoint HP outcomes) are
             credited to BOTH arms and V_A1 - V_A2 measures likelihood-ratio ("soft") updating only. It is a lower
             bound on the learning value; per decision we log the probability mass in groups whose world support is
             a strict subset of the worlds that have the cell (strict_support_mass).
  values     every arm's x_o is EVALUATED under the true model: V(o) = sum_n pi_n min_j (M_n^T x_o)_j with the A1
             posterior pi_n. Then V_A1(o) >= V_A2(o) by LP optimality.
  cells      report formula (§3.2, primary): R_k[a, beta] = sum_o P(o | k, a, beta) V(o). Elementwise R^A1 >= R^A2,
             so V(a) = sum_k w_k min_beta R_k[a, beta] satisfies V_A1(a) >= V_A2(a) for every root action (E1).
             Variant 'pw' (per-world, computed from the same x_o): R_k[a, beta] = sum_{n in cell, world k} p_n
             min_j (M_n^T x_o)_j, world k's own payoff under the shared continuation. Same w-expectation per cell,
             but the root's per-world minimum sees a different spread; Jensen is NOT a theorem for 'pw'.
             Each formula satisfies one of the two natural bounds (research/scripts/voi/s1/tests/test_c_jensen.py):
             'rep' gives Jensen V_A1(a) >= V_A2(a) but not the clairvoyance bound V_A0L(a) >= V_A1(a) (world k's cell
             takes the cross-world group average); 'pw' gives the clairvoyance bound but not Jensen.
  root       today's agent-form LP over the K cell-value matrices (rows = the true legal options; a row missing in
             world k gets that world's column-wise minimum).
  A0L        reference: every node its own group (clairvoyant at the children) with the same LP leaf. With all
             nodes distinguishable, A1 == A2 == A0L == agent-form LP over full_matrix_lpleaf(world) (gate a).

The leaf is the LP value of the child's depth-1 matrix, NOT A0's midpoint leaf (nash._leaf): a singleton group must
solve an LP, and the two differ when the child has no pure saddle. A0 itself is still today's agent (stock
nash.full_matrix + agent_form_lp) and is computed on the same K worlds for reference.

Unpruned depth-1 matrices (full_d1). pe.id(c, 0) alpha-beta prunes each row after its first cell <= alpha (row 0 and
column 0 are always complete), and a depth-1 return carries poke-engine 0.0.48's row-label bug: the matrix is in
get_all_options order while r.s1 is that list stably sorted by row minimum (search.rs
re_order_moves_for_iterative_deepening; s2 is not re-ordered). _d1 inverts the stable sort, which is exact. A row r
with pruned cells is recomputed from a copy of the child in which r is the FIRST option of side one (first rows are
never pruned): a move row keeps only that move enabled and sets force_trapped; a '-tera' row additionally
pre-terastallizes the active mon (tera is then unavailable, so the move is the only option); a switch row disables
every move and swaps the target into the first alive bench slot (slot-indexed side fields remapped). evaluate() reads
neither `disabled` nor `force_trapped` (genx/evaluate.rs), and the cells agree with every unpruned cell of the
original matrix (research/scripts/voi/s1/tests). The mirror's depth-1 matrix equals -F^T on ~99.3 % of cells, so no
midpoint symmetrisation is used.

Keys (obs_key). mode 'public' (default): our action a, the opponent's option label beta (move id incl. '-tera' or the
switch-in's species), our whole side after the turn (exact: we know it), the opponent's PUBLIC fields (active slot and
species, hp_public, status, tera flag and type once terastallized, types, item-gone flag, boosts, volatile statuses,
side conditions with durations reduced to presence (layers kept), wish / future-sight presence; per revealed slot:
species, hp_public, status) and the field (weather / terrain / trick room, no turn counters). Hidden attributes (item,
ability, unrevealed moves and their slot order, tera type before use, stats, exact HP, durations,
switch_out_move_second_saved_move, unrevealed slots) never enter. mode 'report': the literal §3.1 key (a, beta, faint
flags of both actives, hp_public of both actives), kept for E4 statistics.
"""
from __future__ import annotations

import time
from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np
import poke_engine as pe

import hidden as hd
import hidden_agents as ha
import nash
import pkbattle as pb

ARMS = ("A1", "A2")
SC_FIELDS = ["aurora_veil", "crafty_shield", "healing_wish", "light_screen", "lucky_chant", "lunar_dance",
             "mat_block", "mist", "protect", "quick_guard", "reflect", "safeguard", "spikes", "stealth_rock",
             "sticky_web", "tailwind", "toxic_count", "toxic_spikes", "wide_guard"]
LAYERS = {"spikes", "toxic_spikes", "stealth_rock", "sticky_web"}
BOOSTS = ["attack_boost", "defense_boost", "special_attack_boost", "special_defense_boost", "speed_boost",
          "accuracy_boost", "evasion_boost"]
OWN_SF = ["active_index", "force_switch", "force_trapped", "baton_passing", "shed_tailing", "slow_uturn_move",
          "substitute_health", "wish", "future_sight", "last_used_move", "switch_out_move_second_saved_move"] + BOOSTS
OWN_PF = ["id", "hp", "maxhp", "status", "item", "ability", "types", "terastallized", "tera_type", "rest_turns",
          "sleep_turns"]


# ---------------------------------------------------------------- unpruned depth-1 matrices

def _d1(state: pe.State):
    """pe.id(state, 0) (det-leaf guarded) with the depth-1 row labels repaired. Returns (M with NaN, rows, cols)."""
    r = nash._leaf_id(state)
    M = np.array([np.nan if v is None else v for v in r.matrix], dtype=float).reshape(len(r.s1), len(r.s2))
    s1, s2 = [pb._canon(x) for x in r.s1], [pb._canon(x) for x in r.s2]
    with np.errstate(all="ignore"):
        mins = np.nanmin(M, axis=1)
    rows = [None] * len(s1)
    for t, i in enumerate(np.argsort(-mins, kind="stable")):    # s1[t] is the label of original row i
        rows[i] = s1[t]
    return M, rows, s2


def _with_side_one(st: pe.State, new: pe.Side) -> pe.State:
    return pe.State(side_one=new, side_two=st.side_two, **{f: getattr(st, f) for f in hd.GF})


def _isolate(st: pe.State, label: str) -> pe.State:
    """Copy of st in which side one's option `label` is its first option (so its row is never pruned)."""
    sd = st.side_one
    a = int(sd.active_index)
    act = sd.pokemon[a]
    ids = [m.id.lower() for m in act.moves]
    mons = [hd.copy_pokemon(p) for p in sd.pokemon]
    tera = label.endswith("-tera")
    mv = label[:-5] if tera else label
    if mv in ids:
        keep = ids.index(mv)
        mons[a] = hd.copy_pokemon(act, moves=[pe.Move(id=m.id, pp=m.pp, disabled=(i != keep) or m.disabled)
                                              for i, m in enumerate(act.moves)],
                                  **({"terastallized": True} if tera else {}))
        return _with_side_one(st, hd.copy_side(sd, pokemon=mons, force_trapped=True))
    tgt = [i for i, p in enumerate(sd.pokemon) if i != a and p.hp > 0 and p.id.lower() == label][0]
    first = [i for i, p in enumerate(sd.pokemon) if i != a and p.hp > 0][0]
    mons[a] = hd.copy_pokemon(act, moves=[pe.Move(id=m.id, pp=m.pp, disabled=True) for m in act.moves])
    over = {}
    if tgt != first:
        mons[tgt], mons[first] = mons[first], mons[tgt]
        sw = {str(tgt): str(first), str(first): str(tgt)}
        lum = sd.last_used_move
        if lum.startswith("switch:") and lum[7:] in sw:
            over["last_used_move"] = "switch:" + sw[lum[7:]]
        fs = sd.future_sight
        if str(fs[1]) in sw:
            over["future_sight"] = (fs[0], sw[str(fs[1])])
    return _with_side_one(st, hd.copy_side(sd, pokemon=mons, **over))


FULL_D1_STATS = {"children": 0, "calls": 0, "cache_hits": 0}


def full_d1(state: pe.State, cache: dict | None = None):
    """Unpruned depth-1 matrix of `state` from side one's view: (F, rows, cols), no NaN. Optional cache keyed by
    state.to_string()."""
    key = None
    if cache is not None:
        key = state.to_string()
        if key in cache:
            FULL_D1_STATS["cache_hits"] += 1
            return cache[key]
    M, rows, cols = _d1(state)
    FULL_D1_STATS["children"] += 1
    FULL_D1_STATS["calls"] += 1
    F = M.copy()
    for i, lab in enumerate(rows):
        if not np.isnan(M[i]).any():
            continue
        R, r1, r2 = _d1(_isolate(state, lab))
        FULL_D1_STATS["calls"] += 1
        if r2 != cols:
            raise RuntimeError(f"isolated columns differ: {r2} vs {cols}")
        F[i] = R[r1.index(lab[:-5] if lab.endswith("-tera") else lab)]
    if np.isnan(F).any():
        raise RuntimeError("unfilled cells")
    out = (F, rows, cols)
    if cache is not None:
        cache[key] = out
    return out


def own_child_matrix(child: pe.State, side: int, cache: dict | None = None):
    # full_d1 is deliberately a faithful native-matrix extractor. Finished
    # battles need the same exact reward as the live planner, not its material
    # heuristic. A finished game has no continuation choice for either side.
    terminal = nash.terminal_value(child)
    if terminal is not None:
        value = terminal if side == 1 else -terminal
        return np.array([[value]]), ["none"], ["none"]
    F, r1, r2 = full_d1(child, cache)
    return (F, r1, r2) if side == 1 else (-F.T, r2, r1)


def value_of(M: np.ndarray):
    """(x, value) of the zero-sum matrix game M (row player maximises); pure saddle shortcut, else nash.lp."""
    lo, hi = M.min(1).max(), M.max(0).min()
    if hi - lo <= 1e-9 * max(1.0, abs(lo)):
        x = np.zeros(M.shape[0]); x[int(M.min(1).argmax())] = 1.0
        return x, float(lo)
    x, v = ha.lp_x(M)
    return x, float((M.T @ x).min())


def full_matrix_lpleaf(state: pe.State, cache: dict | None = None):
    """Depth-2 matrix with LP continuation values and exact terminal rewards (side one's view)."""
    s1, s2 = pb.options(state)
    M = np.empty((len(s1), len(s2)))
    for i, a in enumerate(s1):
        for j, b in enumerate(s2):
            M[i, j] = sum(br.percentage / 100.0 * value_of(own_child_matrix(state.apply_instructions(br), 1, cache)[0])[1]
                          for br in pe.generate_instructions(state, a, b))
    return M, s1, s2


# ---------------------------------------------------------------- public observation keys

def _sc(side, exact: bool):
    sc = side.side_conditions
    return tuple(int(getattr(sc, f)) if exact or f in LAYERS else int(getattr(sc, f) > 0) for f in SC_FIELDS)


def _own_key(sd):
    mons = tuple(tuple(str(getattr(p, f)) for f in OWN_PF) for p in sd.pokemon)
    return (mons, tuple(str(getattr(sd, f)) for f in OWN_SF), tuple(sorted(str(v) for v in sd.volatile_statuses)),
            _sc(sd, True))


def _opp_key(sd, revealed):
    a = int(sd.active_index)
    p = sd.pokemon[a]
    act = (a, p.id, hd.hp_public(p.hp, p.maxhp), p.status, bool(p.terastallized),
           p.tera_type if p.terastallized else None, tuple(p.types), p.item.lower() == "none")
    slots = tuple((i, sd.pokemon[i].id, hd.hp_public(sd.pokemon[i].hp, sd.pokemon[i].maxhp), sd.pokemon[i].status)
                  for i in sorted(set(revealed) | {a}))
    return (act, slots, tuple(int(getattr(sd, f)) for f in BOOSTS), tuple(sorted(str(v) for v in sd.volatile_statuses)),
            _sc(sd, False), bool(sd.force_switch), int(sd.wish[0]) > 0, int(sd.future_sight[0]) > 0)


def obs_key(child: pe.State, side: int, a: str, b: str, revealed, mode: str = "public"):
    """What `side` observes after playing a against the opponent's b, reaching `child`. `revealed`: opponent slots
    revealed in the decider's view before the turn."""
    own, opp = hd._sides(child, side)
    if mode == "report":
        po, pp = own.pokemon[int(own.active_index)], opp.pokemon[int(opp.active_index)]
        return (a, b, po.hp <= 0, pp.hp <= 0, hd.hp_public(po.hp, po.maxhp), hd.hp_public(pp.hp, pp.maxhp))
    return (a, b, _own_key(own), _opp_key(opp, revealed),
            (str(child.weather), str(child.terrain), int(child.trick_room)))


# ---------------------------------------------------------------- nodes, groups, root

@dataclass
class Node:
    world: int
    a: str                     # our root option
    b: str                     # opponent root option (world-specific label set)
    p: float                   # P(branch | world, a, b)
    key: tuple                 # public observation (contains a and b)
    M: np.ndarray              # continuation matrix, decider's perspective (rows = our child options)
    rows: list
    r: float = 0.0             # immediate payoff (toy only)
    key_report: tuple | None = None
    b_switch: bool = False     # the opponent's root option is a switch


@dataclass
class Expansion:
    nodes: list
    cols: list                 # per world: the opponent's root options
    own: list                  # per world: our root options
    secs_cpu: float = 0.0
    secs_wall: float = 0.0
    stats: dict = field(default_factory=dict)


def expand(worlds, side: int, legal, revealed, cache: dict | None = None, key_report: bool = True) -> Expansion:
    """All (world, a, b, branch) children of the K worlds with keys and unpruned continuation matrices."""
    t0, c0 = time.perf_counter(), time.process_time()
    calls0 = FULL_D1_STATS["calls"]
    nodes, cols, owns = [], [], []
    for k, w in enumerate(worlds):
        s1, s2 = pb.options(w)
        own, opp = (s1, s2) if side == 1 else (s2, s1)
        owns.append(own); cols.append(opp)
        opp_ids = {p.id.lower() for p in hd._sides(w, side)[1].pokemon}
        for a in own:
            if a not in legal:
                continue
            for b in opp:
                for br in pe.generate_instructions(w, a, b) if side == 1 else pe.generate_instructions(w, b, a):
                    c = w.apply_instructions(br)
                    M, rows, _ = own_child_matrix(c, side, cache)
                    nodes.append(Node(k, a, b, br.percentage / 100.0, obs_key(c, side, a, b, revealed), M, rows,
                                      key_report=obs_key(c, side, a, b, revealed, "report") if key_report else None,
                                      b_switch=b in opp_ids))
    return Expansion(nodes, cols, owns, time.process_time() - c0, time.perf_counter() - t0,
                     {"d1_calls": FULL_D1_STATS["calls"] - calls0})


def _node_vals(Ms, x):
    return np.array([float((M.T @ x).min()) for M in Ms])


def solve_groups(nodes, w, want_a0l: bool = True, grouping: str = "key"):
    """Per-node continuation values for each arm. grouping: 'key' (S1) or 'node' (every node alone = A0L).
    Returns dict arm -> dict(rep=..., pw=...) of per-node value arrays, plus group records and timing."""
    w = np.asarray(w, float)
    n = len(nodes)
    groups = defaultdict(list)
    for i, nd in enumerate(nodes):
        groups[nd.key if grouping == "key" else i].append(i)
    out = {arm: {"rep": np.zeros(n), "pw": np.zeros(n)} for arm in ARMS}
    a0l = np.zeros(n)
    recs, secs = [], {"shared": 0.0, "A1": 0.0, "A2": 0.0, "A0L": 0.0}   # shared: groups with one world
    for g, idx in groups.items():
        rows = []
        for i in idx:
            rows += [r for r in nodes[i].rows if r not in rows]
        Ms = [nodes[i].M if nodes[i].rows == rows else ha.align(nodes[i].M, nodes[i].rows, rows) for i in idx]
        aligned = sum(nodes[i].rows != rows for i in idx)
        r = np.array([nodes[i].r for i in idx])
        ks = [nodes[i].world for i in idx]
        p = np.array([nodes[i].p for i in idx])
        pk = defaultdict(float)
        for k, pi in zip(ks, p):
            pk[k] += pi                                      # P(o | k, a, b)
        pi1 = np.array([w[k] * pi for k, pi in zip(ks, p)])
        pi2 = np.array([w[k] * pi / pk[k] for k, pi in zip(ks, p)])
        pi1, pi2 = pi1 / pi1.sum(), pi2 / pi2.sum()
        t = time.process_time()
        if len(idx) == 1:
            x1 = value_of(Ms[0])[0]
        else:
            x1 = ha.agent_form_lp(Ms, pi1)[0]
        v1 = _node_vals(Ms, x1)
        secs["shared" if len(pk) == 1 else "A1"] += time.process_time() - t
        t = time.process_time()
        if len(pk) == 1:                                     # one world: frozen == live
            x2, v2 = x1, v1
        else:
            x2 = ha.agent_form_lp(Ms, pi2)[0]
            v2 = _node_vals(Ms, x2)
            secs["A2"] += time.process_time() - t
        for arm, v in (("A1", v1), ("A2", v2)):
            V = float(pi1 @ v)                               # evaluated under the live (true) posterior
            out[arm]["pw"][idx] = r + v
            out[arm]["rep"][idx] = r + V
        if want_a0l:
            t = time.process_time()
            if len(idx) == 1:
                a0l[idx] = r + v1
            else:
                a0l[idx] = r + np.array([value_of(M)[1] for M in Ms])
            secs["A0L"] += time.process_time() - t
        recs.append(dict(n=len(idx), worlds=sorted(pk), aligned=int(aligned),
                         V1=float(pi1 @ v1), V2=float(pi1 @ v2), V2_self=float(pi2 @ v2)))
    res = {"arms": out, "groups": recs, "secs": secs}
    if want_a0l:
        res["A0L"] = a0l
    return res


def root_matrices(nodes, vals, cols, own, legal, K):
    """R_k[i, j] = sum over world k's nodes of cell (legal[i], cols[k][j]) of p_n * vals[n]; rows missing in world k
    get that world's column-wise minimum (hidden_agents.align)."""
    Rs = []
    for k in range(K):
        R = np.zeros((len(own[k]), len(cols[k])))
        ri = {a: i for i, a in enumerate(own[k])}
        ci = {b: j for j, b in enumerate(cols[k])}
        for nd, v in zip(nodes, vals):
            if nd.world == k:
                R[ri[nd.a], ci[nd.b]] += nd.p * v
        Rs.append(ha.align(R, own[k], legal))
    return Rs


def root_solve(Rs, w):
    x, v = ha.agent_form_lp(Rs, w)
    Va = sum(wk * R.min(1) for wk, R in zip(w, Rs))
    return x, float(v), np.asarray(Va, float)


def plan(worlds, side: int, legal, revealed, w=None, cache: dict | None = None, want_a0l: bool = True,
         key_report: bool = True):
    """Full S1 analysis on the K worlds. Returns dict with, per arm in {A1, A2} x {rep, pw} (and A0L): root strategy x,
    root value v, per-root-action values V (aligned to legal); plus the expansion and group records."""
    K = len(worlds)
    w = np.full(K, 1 / K) if w is None else np.asarray(w, float)
    ex = expand(worlds, side, legal, revealed, cache, key_report)
    sg = solve_groups(ex.nodes, w, want_a0l)
    res = {"ex": ex, "groups": sg["groups"], "lp_secs": sg["secs"]}
    res["root_secs"] = {}
    for arm in ARMS:
        for form in ("rep", "pw"):
            t = time.process_time()
            Rs = root_matrices(ex.nodes, sg["arms"][arm][form], ex.cols, ex.own, legal, K)
            x, v, Va = root_solve(Rs, w)
            res[f"{arm}_{form}"] = dict(x=x, v=v, V=Va, R=Rs)
            res["root_secs"][f"{arm}_{form}"] = time.process_time() - t
    if want_a0l:
        t = time.process_time()
        Rs = root_matrices(ex.nodes, sg["A0L"], ex.cols, ex.own, legal, K)
        x, v, Va = root_solve(Rs, w)
        res["A0L"] = dict(x=x, v=v, V=Va, R=Rs)
        res["root_secs"]["A0L"] = time.process_time() - t
    return res


# ---------------------------------------------------------------- E4 statistics

def key_stats(nodes, K, which: str = "key"):
    """Keys per world-cell (split by whether the opponent's option is a switch), and the fraction of children merged
    across worlds (a node is merged if its group holds a node of another world), by count and by probability mass
    (mean over world-cells of the merged branch probability). Also the strict-support mass (groups whose worlds are a
    strict subset of the worlds that have the cell) and within-world merges."""
    kf = (lambda nd: nd.key) if which == "key" else (lambda nd: nd.key_report)
    gw = defaultdict(set)
    gn = defaultdict(lambda: defaultdict(int))
    cellw = defaultdict(set)
    for nd in nodes:
        gw[kf(nd)].add(nd.world)
        gn[kf(nd)][nd.world] += 1
        cellw[(nd.a, nd.b)].add(nd.world)
    cells = defaultdict(list)
    for nd in nodes:
        cells[(nd.world, nd.a, nd.b)].append(nd)
    kpc_sw, kpc_mv, merged_mass, strict_mass = [], [], [], []
    merged_n = within_n = 0
    for (k, a, b), nds in cells.items():
        keys = {kf(nd) for nd in nds}
        (kpc_sw if nds[0].b_switch else kpc_mv).append(len(keys))
        mm = sm = 0.0
        for nd in nds:
            g = kf(nd)
            if len(gw[g]) > 1:
                merged_n += 1; mm += nd.p
            if gn[g][nd.world] > 1:
                within_n += 1
            if gw[g] < cellw[(a, b)]:
                sm += nd.p
        merged_mass.append(mm); strict_mass.append(sm)
    elim = [len(ws) < K for ws in cellw.values()]
    return dict(n_nodes=len(nodes), n_groups=len(gw), n_cells=len(cells),
                keys_per_cell_switch=kpc_sw, keys_per_cell_move=kpc_mv,
                merged_frac_count=merged_n / max(1, len(nodes)), merged_frac_mass=float(np.mean(merged_mass)),
                strict_support_mass=float(np.mean(strict_mass)), within_world_frac=within_n / max(1, len(nodes)),
                cells_missing_worlds=float(np.mean(elim)) if elim else 0.0)
