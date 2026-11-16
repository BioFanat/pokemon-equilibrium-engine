"""Per-decision information value and concealment cost along real 6v6 self-play (engine simulation, no new leaf).

At every `--every`-th decision of an `agentform:4` agent in `play_hidden`-style self-play (both Trackers available),
with OURBOT_DET_LEAF=1:

 (1) DECISION REGIME.  K=4 opponent worlds with our TRUE team -> M_k (today's agent).  AF strategy x_AF; per-world LP
     strategies x_k and values val_k.  Logged: do the per-world argmaxes disagree?  Is the AF argmax outside the set of
     per-world argmaxes?  regret_k = val_k - min_j (M_k^T x_AF)_j  (what a type-informed agent gains in world k).
 (2) CONCEALMENT (honest opponent model).  Our own team is sampled from the OPPONENT's public view of us
     (hd.determinize on the opponent's Tracker): L = {true set} U 3 samples, uniform weights p_l (stated assumption; the
     opponent's prior over our sets is not modelled).  Matrices M_{l,k} for all 16 pairs.  One-stage two-sided Bayesian
     LP:  max_{x_l, v_k} sum_k q_k v_k  s.t.  v_k <= sum_l p_l (M_{lk}^T x_l)_j  for all j, k.   Play x_{true}.
     Logged: ex-ante concealment value  V2 - sum_l p_l v_AF(l)  (>= 0 exactly; asserted),
             regret of today's x_AF for our true type against the two-sided opponent y_k (dual),
             regret of x_true in the omniscient-opponent game (sum_k q_k min_j (M^T x_true)_j vs v_AF),
             TV(x_AF, x_true), argmax change and whether the changed action is a tera / an unrevealed move / a switch.
     Two conditions: 'pub' = the opponent's actual public view of us (back-row species unknown) and 'spk' = species of
     all our six mons known (isolates set-level concealment: moves / item / ability / tera).
 (3) ATTRIBUTE SPLIT (VOI by hidden variable) on the active opponent's MAP set via FixedBelief, worlds varying ONLY
     tera type / item / ability / unrevealed moves (3 completions):  eta_hat_attr = mean_k val_k - v_AF(attr worlds),
     argmax disagreement across the attribute worlds, and max_k regret of the AF action.

PRE-REGISTERED PREDICTIONS (written 2026-11-16 12:05 -05:00, before the first run; see
research/info_gathering_concealment.md §4 for the same text):
 P1  per-world argmax disagreement in 40-60% of decisions; AF argmax outside the per-world argmax set in < 15%.
 P2  attribute order by eta_hat/range and by argmax-disagreement frequency: moves > item > tera; eta_hat_tera median 0
     (voi.md: 87% zero); eta_hat_item median < 1% of range; eta_hat_moves median < 2% of range.
 P3  concealment value / range (condition spk): median 0, > 1% in < 10% of decisions; both per-type regrets median 0.
 P4  argmax(x_AF) != argmax(x_true) in < 15% of decisions (spk); tera-involving changes < 3%.
 P5  condition pub shows a larger concealment value than spk (median ratio >= 2) because species uncertainty dominates.

Usage: OURBOT_DET_LEAF=1 python two_sided_live.py --games 6 --workers 2 --every 4 --out out/two_sided_live.jsonl
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import random
import sys
import time
from dataclasses import replace

import numpy as np
from scipy.optimize import linprog

HERE = os.path.dirname(os.path.abspath(__file__))
ENGINE = os.path.abspath(os.path.join(HERE, "..", "..", "..", "engine"))
sys.path.insert(0, ENGINE)
os.environ.setdefault("OURBOT_DET_LEAF", "1")

import poke_engine as pe  # noqa: E402
import hidden as hd  # noqa: E402
import hidden_agents as ha  # noqa: E402
import nash  # noqa: E402
import pkbattle as pb  # noqa: E402
import randbats as rb  # noqa: E402

TOL = 1e-7


def stable_matrix(st, side, legal=None):
    """full_matrix until two consecutive evaluations agree (det leaf on, but guard anyway). Returns (M_own, rows, cols)."""
    prev = nash.full_matrix(st)
    for _ in range(6):
        cur = nash.full_matrix(st)
        if cur[1:] == prev[1:] and np.array_equal(cur[0], prev[0]):
            M, rows, cols = ha.own_matrix(*cur, side)
            if legal is not None:
                M, rows = ha.align(M, rows, legal), list(legal)
            return M, list(rows), list(cols)
        prev = cur
    raise RuntimeError("matrix did not stabilise")


def lp_rows(M):
    x, v = ha.lp_x(M)
    return x, float(v)


def regret_of(M, x):
    """Value of M minus what x guarantees in M (>= 0)."""
    return float(lp_rows(M)[1] - (M.T @ x).min())


def two_sided_lp(Ms, p, q):
    """Ms[l][k] = payoff matrix (rows = type-l actions, cols = type-k opponent actions, our perspective).
    max_{x_l, v_k} sum_k q_k v_k  s.t.  v_k <= sum_l p_l (M_{lk}^T x_l)_j.  Returns (xs, V, ys)."""
    L, K = len(Ms), len(Ms[0])
    ms = [Ms[l][0].shape[0] for l in range(L)]
    off = np.cumsum([0] + ms)
    nv = off[-1] + K
    c = np.zeros(nv); c[off[-1]:] = -np.asarray(q, float)
    A, rhs = [], []
    for k in range(K):
        n = Ms[0][k].shape[1]
        blk = np.zeros((n, nv))
        for l in range(L):
            assert Ms[l][k].shape[1] == n
            blk[:, off[l]:off[l + 1]] = -p[l] * Ms[l][k].T
        blk[:, off[-1] + k] = 1
        A.append(blk); rhs.append(np.zeros(n))
    Aeq = np.zeros((L, nv))
    for l in range(L):
        Aeq[l, off[l]:off[l + 1]] = 1
    r = linprog(c, A_ub=np.vstack(A), b_ub=np.concatenate(rhs), A_eq=Aeq, b_eq=np.ones(L),
                bounds=[(0, None)] * off[-1] + [(None, None)] * K, method="highs")
    assert r.status == 0, r.message
    xs = []
    for l in range(L):
        x = np.clip(r.x[off[l]:off[l + 1]], 0, None); xs.append(x / x.sum())
    marg = -np.asarray(r.ineqlin.marginals)
    ys, pos = [], 0
    for k in range(K):
        n = Ms[0][k].shape[1]
        y = np.clip(marg[pos:pos + n], 0, None); pos += n
        ys.append(y / y.sum() if y.sum() > 0 else np.full(n, 1 / n))
    return xs, float(-r.fun), ys


def align_cols(M, cols, ref):
    """Restrict/reorder columns to `ref` (opponent legality should not depend on our hidden set). Returns None if a
    reference column is missing."""
    if cols == ref:
        return M
    idx = []
    for c in ref:
        if c not in cols:
            return None
        idx.append(cols.index(c))
    return M[:, idx]


def classify(label, revealed_moves, own_active):
    if label.endswith("-tera"):
        return "tera"
    if label.startswith("switch"):
        return "switch"
    return "move_revealed" if label in revealed_moves else "move_unrevealed"


def opp_view_of_us(state, tr_opp, species_known):
    tr = copy.deepcopy(tr_opp)
    if species_known:
        _, us = hd._sides(state, tr.side)
        for i in range(6):
            tr._reveal(us, i)
        tr._see(state)
    v = hd.make_view(state, tr)
    return v, hd.Belief(v)


def analyse_decision(state, side, view, belief, tr_opp, legal, rng, K=4, L_extra=3, moves_n=3):
    rec = dict(side=side, n_legal=len(legal))
    t0 = time.perf_counter()
    # (1) today's worlds
    worlds = [hd.determinize(state, view, belief, rng) for _ in range(K)]
    Mk = [stable_matrix(w, side, legal) for w in worlds]
    Ms = [m[0] for m in Mk]
    q = np.full(K, 1 / K)
    x_af, v_af = ha.agent_form_lp(Ms, q)
    per = [lp_rows(M) for M in Ms]
    arg_w = [int(np.argmax(x)) for x, _ in per]
    allv = np.concatenate([M.ravel() for M in Ms])
    rng_ = float(allv.max() - allv.min())
    rec.update(v_AF=float(v_af), v_worlds=[v for _, v in per], payoff_range=rng_,
               eta_hat_set=float(np.mean([v for _, v in per]) - v_af),
               world_argmax=[legal[i] for i in arg_w], af_argmax=legal[int(np.argmax(x_af))],
               worlds_disagree=len(set(arg_w)) > 1, af_outside=int(np.argmax(x_af)) not in set(arg_w),
               regret_af_in_world=[regret_of(M, x_af) for M in Ms])
    # revealed info about us (from the opponent's tracker)
    own_side, _ = hd._sides(state, side)
    act = int(own_side.active_index)
    sr = tr_opp.slots.get(act)
    revealed = set(sr.moves.values()) if sr else set()
    rec["our_revealed_moves"] = sorted(revealed)
    rec["our_tera_used"] = bool(own_side.pokemon[act].terastallized)
    # (2) concealment, two conditions
    for cond, spk in (("pub", False), ("spk", True)):
        ov, ob = opp_view_of_us(state, tr_opp, spk)
        ours = [None] + [hd.determinize(state, ov, ob, rng) for _ in range(L_extra)]
        Mlk, rows_l, ok = [], [], True
        for l, w_us in enumerate(ours):
            row = []
            for k, w in enumerate(worlds):
                if l == 0:
                    row.append(Ms[k]); continue
                if side == 1:
                    st = hd.make_state(w_us.side_one, w.side_two, view.field)
                else:
                    st = hd.make_state(w.side_one, w_us.side_two, view.field)
                M, rows, cols = stable_matrix(st, side)
                M = align_cols(M, cols, Mk[k][2])
                if M is None:
                    ok = False; break
                row.append(M)
            if not ok:
                break
            Mlk.append(row); rows_l.append(legal if l == 0 else rows)
        if not ok:
            rec[cond] = dict(skipped="column_mismatch"); continue
        L = len(Mlk)
        p = np.full(L, 1 / L)
        v_af_l = [ha.agent_form_lp(Mlk[l], q)[1] for l in range(L)]
        xs, V2, ys = two_sided_lp(Mlk, p, q)
        x_true = xs[0]
        conc = V2 - float(np.dot(p, v_af_l))
        assert conc >= -1e-6 * max(1.0, rng_), (conc, V2, v_af_l)
        # true type's payoff vs the two-sided opponent, for x_true and for today's x_AF
        pay = lambda x: float(sum(q[k] * x @ Mlk[0][k] @ ys[k] for k in range(K)))
        u_true, u_af = pay(x_true), pay(x_af)
        # best response of the true type to ys (check: x_true should be a BR)
        br = max(float(sum(q[k] * Mlk[0][k][i] @ ys[k] for k in range(K))) for i in range(len(legal)))
        regret_af_vs_honest = br - u_af                       # cost of the omniscient model vs an honest opponent
        regret_true_vs_omni = float(v_af - sum(q[k] * (Mlk[0][k].T @ x_true).min() for k in range(K)))
        ia, it = int(np.argmax(x_af)), int(np.argmax(x_true))
        rec[cond] = dict(L=L, V2=float(V2), v_af_types=[float(v) for v in v_af_l], conc_value=float(conc),
                         u_true=u_true, u_af=u_af, br_true=br, regret_af_vs_honest=float(regret_af_vs_honest),
                         regret_true_vs_omni=regret_true_vs_omni,
                         tv=float(0.5 * np.abs(x_af - x_true).sum()), argmax_change=ia != it,
                         from_=legal[ia], to=legal[it], from_class=classify(legal[ia], revealed, act),
                         to_class=classify(legal[it], revealed, act),
                         sampled_species=[[pk.id for pk in (w.side_one if side == 1 else w.side_two).pokemon] for w in ours[1:]])
    # (3) attribute split on the active opponent's MAP set
    opp_act = int(view.opp_fields["active_index"])
    pub = view.opp_public[opp_act]
    s = belief.map_set(opp_act)
    srv = view.reveals[opp_act]
    fixed = {i: belief.map_set(i) for i in view.reveals}
    fixed.update(belief.sample_unrevealed(random.Random(0)))
    pool, abil, items, teras = hd._role_lists(s.species, s.role)
    variants = {}
    if not pub["terastallized"] and len(set(teras)) >= 2:
        variants["tera"] = [replace(s, tera=t) for t in sorted(set(teras))]
    if not pub["item_gone"] and len(set(items)) >= 2:
        variants["item"] = [replace(s, item=it) for it in sorted(set(items))]
    if len(set(abil)) >= 2:
        variants["ability"] = [replace(s, ability=a) for a in sorted(set(abil))]
    known = [m for m in pool if rb.to_id(m) in srv.moves.values()]
    rest = [m for m in pool if rb.to_id(m) not in srv.moves.values()]
    need = max(0, min(4, len(pool)) - len(known))
    if need >= 1 and len(rest) > need:
        r0 = random.Random(1)
        seen, comps = set(), []
        for _ in range(40):
            c = tuple(sorted(r0.sample(rest, need)))
            if c not in seen:
                seen.add(c); comps.append(c)
            if len(comps) == moves_n:
                break
        if len(comps) >= 2:
            variants["moves"] = [replace(s, moves=tuple(known) + c) for c in comps]
    attr = {}
    for name, sets_ in variants.items():
        Mt = []
        for s2 in sets_:
            fb = hd.FixedBelief(view, {**fixed, opp_act: s2})
            st = hd.determinize(state, view, fb, rng)
            Mt.append(stable_matrix(st, side, legal)[0])
        pv = [lp_rows(M) for M in Mt]
        xa, va = ha.agent_form_lp(Mt, np.full(len(Mt), 1 / len(Mt)))
        allt = np.concatenate([M.ravel() for M in Mt])
        args = [int(np.argmax(x)) for x, _ in pv]
        attr[name] = dict(n=len(Mt), eta_hat=float(np.mean([v for _, v in pv]) - va),
                          range=float(allt.max() - allt.min()), disagree=len(set(args)) > 1,
                          argmaxes=[legal[i] for i in args],
                          regret_af=[regret_of(M, xa) for M in Mt],
                          values=[v for _, v in pv])
    rec.update(attr=attr, opp_species=s.species, opp_role=s.role, opp_revealed_moves=sorted(srv.moves.values()),
               leaf_stats=dict(nash.LEAF_STATS), secs=time.perf_counter() - t0)
    return rec, x_af


def one_game(args):
    seed, every, K = args
    rng = random.Random(seed)
    state = pb.random_state(rng)
    tr = {1: hd.Tracker(1, state), 2: hd.Tracker(2, state)}
    recs, n = [], 0
    t = time.perf_counter()
    for turn in range(1, 301):
        opts = dict(zip((1, 2), pb.options(state)))
        ch = {}
        for s in (1, 2):
            view = hd.make_view(state, tr[s]); belief = hd.Belief(view)
            legal = opts[s]
            if len(legal) == 1:
                ch[s] = legal[0]; continue
            n += 1
            if n % every == 0:
                rec, x = analyse_decision(state, s, view, belief, tr[3 - s], legal, rng, K=K)
                recs.append(dict(seed=seed, turn=turn, **rec))
            else:
                worlds = [hd.determinize(state, view, belief, rng) for _ in range(K)]
                Ms = [stable_matrix(w, s, legal)[0] for w in worlds]
                x = ha.agent_form_lp(Ms, np.full(K, 1 / K))[0]
            ch[s] = rng.choices(legal, weights=x)[0]
        branches = pe.generate_instructions(state, ch[1], ch[2])
        pick = rng.choices(branches, weights=[b.percentage for b in branches])[0]
        new = state.apply_instructions(pick)
        for s in (1, 2):
            tr[s].update(state, ch[1], ch[2], pick, new)
        state = new
        a1, a2 = pb.alive(state.side_one), pb.alive(state.side_two)
        if a1 == 0 or a2 == 0:
            break
    return dict(seed=seed, turns=turn, wall_s=time.perf_counter() - t, decisions=recs)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=6)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--every", type=int, default=4)
    ap.add_argument("--K", type=int, default=4)
    ap.add_argument("--seed0", type=int, default=20_011)
    ap.add_argument("--out", default=os.path.join(HERE, "out", "two_sided_live.jsonl"))
    a = ap.parse_args()
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    from multiprocessing import get_context
    tasks = [(a.seed0 * (i + 1), a.every, a.K) for i in range(a.games)]
    with get_context("spawn").Pool(a.workers) as pool, open(a.out, "a") as f:
        for g in pool.imap_unordered(one_game, tasks):
            f.write(json.dumps(g) + "\n"); f.flush()
            d = g["decisions"]
            cv = [r["spk"]["conc_value"] / r["payoff_range"] for r in d if "conc_value" in r.get("spk", {}) and r["payoff_range"] > 0]
            ch = [r["spk"]["argmax_change"] for r in d if "argmax_change" in r.get("spk", {})]
            print(f"seed {g['seed']} turns {g['turns']} sampled {len(d)} conc(spk)/range mean {np.mean(cv) if cv else float('nan'):.4f} "
                  f"max {np.max(cv) if cv else float('nan'):.4f} argmax_change {np.mean(ch) if ch else float('nan'):.2f} | {g['wall_s']:.0f}s", flush=True)
