"""Exact one-sided hidden-information 1v1 subgame: the opponent's tera type is hidden.

Game (two stages, agent = side one, opponent = side two, poke-engine 0.0.48 transitions):
  * Theta = the opponent role's `teraTypes` (uniform prior b0 = the generator's rng.choice, so A4 holds);
    moves / item / ability are fixed and known (one-sided information, our team conceded).
  * Stage 0 at s0: simultaneous (a, beta); chance = engine branches. Pre-tera, the hidden type has no
    transition footprint, so every non-tera beta leads to public children shared by all types (checked).
    A `-tera` beta reveals theta (public `terastallized` flag + `tera_type`).
  * Stage 1 at each public child I = (a, beta, pub(child)): simultaneous (a', beta'), then the terminal
    payoff u(c'', theta) = nash._leaf(c'', 'mid') in engine units (the agent's own leaf; terminal states
    included). So the stage-1 matrices U_theta^I are exact `full_matrix` calls and epsilon_v = 0.

Computed per matchup:
  V0(b0)          sequence-form LP value of the whole game (agent vs type-informed opponent)
  V0(delta_theta) per-type full-information values -> VOI_total = sum b0 V0(delta) - V0(b0) (oracle gap)
  eta(0,I,b)      sum_theta b(theta) val(U_theta^I) - AF_LP(U^I, b) at every pre-tera child; eta_NL(0) = sup on a grid
  v_A3            agent-form stage value with world-only leaves; sandwich v_A3 - V0 in [0, eta_NL(0)]
  expl(A3)        exact exploitability of the A3 agent (AF-LP at stage 0 with world-only continuation
                  values, AF-LP over b_hard at each stage-1 info set) against each type's best response
  TV_1            E_BR[TV(p_h, b_hard)] at stage 1 under that best response; check of Theorem C (C.1):
                  expl <= eta_NL(0) + 2*Delta*E_BR[TV_1]  (eps_s = eps_v = S = 0, eta_NL(1) = 0)
  tera behaviour  equilibrium P(tera at stage 0 | theta) in Gamma(b0) vs Gamma(delta_theta) (from LP duals)

Usage: tera_subgame.py --n 12 --seed 0 --workers 6 --out out.jsonl
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
import os
import random
import sys
import time
from dataclasses import replace

import numpy as np

ENGINE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "engine")
sys.path.insert(0, os.path.abspath(ENGINE))

import poke_engine as pe  # noqa: E402
import nash  # noqa: E402
import pkbattle as pb  # noqa: E402
import randbats as rb  # noqa: E402
import hidden as hd  # noqa: E402
import hidden_agents as ha  # noqa: E402
from scipy.optimize import linprog  # noqa: E402


# ---------------------------------------------------------------- deterministic leaf guard

RETRIES = [0]


def _stable(fn, key, tries=12, need=3):
    """pe.id(child, 0) is time-budgeted: under CPU contention a fraction of leaf calls return truncated-search
    values (7% of children with spreads up to 8550 units at 12 busy processes; 0/480 unloaded). Evaluate until
    `need` evaluations agree exactly (majority vote), counting the extra evaluations in RETRIES."""
    seen = {}
    for i in range(tries):
        v = fn()
        k = key(v)
        seen.setdefault(k, []).append(v)
        if len(seen[k]) >= need:
            RETRIES[0] += i + 1 - need
            return v
    raise RuntimeError(f"leaf did not stabilise ({len(seen)} distinct values in {tries} evaluations)")


def stable_full_matrix(c):
    return _stable(lambda: nash.full_matrix(c), lambda r: (r[0].tobytes(), tuple(r[1]), tuple(r[2])))


def stable_leaf(c):
    return _stable(lambda: nash._leaf(c, "mid"), lambda v: v)


# ---------------------------------------------------------------- public keys

def _ser(x):
    """Stable serialisation (poke_engine helper objects have no __str__)."""
    if isinstance(x, (str, int, float, bool)) or x is None:
        return str(x)
    if isinstance(x, (list, tuple)):
        return "(" + ",".join(_ser(v) for v in x) + ")"
    return "{" + ",".join(f"{a}={_ser(getattr(x, a))}" for a in dir(x) if not a.startswith("_") and not callable(getattr(x, a))) + "}"


def _mon_key(p, mask_tera):
    f = [getattr(p, x) for x in hd.PF if x != "tera_type"]
    f.append(None if (mask_tera and not p.terastallized) else p.tera_type)
    f.append(tuple((m.id, m.disabled, m.pp) for m in p.moves))
    return tuple(_ser(x) for x in f)


def pub_key(c: pe.State):
    """Everything except the opponent's (side two) tera type while it is not terastallized."""
    own = tuple(_mon_key(p, False) for p in c.side_one.pokemon)
    opp = tuple(_mon_key(p, True) for p in c.side_two.pokemon)
    sides = tuple(_ser(getattr(s, f)) for s in (c.side_one, c.side_two) for f in hd.SF)
    glob = tuple(_ser(getattr(c, f)) for f in hd.GF)
    return (own, opp, sides, glob)


def terminal(c):
    return pb.alive(c.side_one) == 0 or pb.alive(c.side_two) == 0


# ---------------------------------------------------------------- game construction

def build(opp_set: rb.PokeSet, own_set: rb.PokeSet, teras):
    states = {th: pe.State(side_one=pe.Side(pokemon=[rb.to_pokemon(own_set)]),
                           side_two=pe.Side(pokemon=[rb.to_pokemon(replace(opp_set, tera=th))])) for th in teras}
    A0, B0 = (sorted(x) for x in pb.options(states[teras[0]]))
    for th in teras:
        o = pb.options(states[th])
        assert (sorted(o[0]), sorted(o[1])) == (A0, B0), ("options differ across types", o, A0, B0)
    # children: info set key -> dict(a, beta, prob per theta, child state per theta)
    info = {}
    for a in A0:
        for beta in B0:
            per = {}
            for th in teras:
                acc = {}
                for br in pe.generate_instructions(states[th], a, beta):
                    c = states[th].apply_instructions(br)
                    k = pub_key(c)
                    if k in acc:
                        acc[k][0] += br.percentage / 100.0
                    else:
                        acc[k] = [br.percentage / 100.0, c]
                per[th] = acc
            if not beta.endswith("-tera"):
                ref = {k: round(v[0], 9) for k, v in per[teras[0]].items()}
                for th in teras[1:]:
                    if {k: round(v[0], 9) for k, v in per[th].items()} != ref:
                        raise RuntimeError(f"type-variant pre-tera child at ({a},{beta})")
            for th in teras:
                for k, (p, c) in per[th].items():
                    I = (a, beta, k)
                    d = info.setdefault(I, dict(a=a, beta=beta, P={}, child={}))
                    d["P"][th] = p
                    d["child"][th] = c
    return states, A0, B0, info


def stage1(info):
    """Stage-1 matrices U_theta^I (full_matrix of each child; terminal -> 1x1 leaf)."""
    for I, d in info.items():
        d["U"], d["rows"], d["cols"] = {}, None, None
        for th, c in d["child"].items():
            if terminal(c):
                M, s1, s2 = np.array([[stable_leaf(c)]]), ["none"], ["none"]
            else:
                M, s1, s2 = stable_full_matrix(c)
                ri, ci = sorted(range(len(s1)), key=lambda i: s1[i]), sorted(range(len(s2)), key=lambda j: s2[j])
                M, s1, s2 = M[np.ix_(ri, ci)], sorted(s1), sorted(s2)
            if d["rows"] is None:
                d["rows"], d["cols"] = s1, s2
            elif (s1, s2) != (d["rows"], d["cols"]):
                raise RuntimeError("row/col labels differ across types at a shared child")
            d["U"][th] = M
            d.setdefault("val", {})[th] = ha.lp_x(M)[1] if M.size > 1 else float(M[0, 0])


# ---------------------------------------------------------------- sequence-form LP

def seqform(info, A0, B0, teras, b0):
    """max_r sum_theta b0 lambda_theta; returns (value, r0, y_tera_prob_per_type)."""
    keys = list(info)
    nA = len(A0)
    idx_r = {a: i for i, a in enumerate(A0)}
    off = nA
    idx_rI = {}
    for I in keys:
        idx_rI[I] = off
        off += len(info[I]["rows"])
    idx_lam = {th: off + i for i, th in enumerate(teras)}
    off += len(teras)
    idx_mu = {}
    for th in teras:
        for I in keys:
            if th in info[I]["U"]:
                idx_mu[(th, I)] = off
                off += 1
    nv = off
    c = np.zeros(nv)
    for th in teras:
        c[idx_lam[th]] = -b0[th]
    # equalities
    Aeq, beq = [], []
    row = np.zeros(nv); row[:nA] = 1; Aeq.append(row); beq.append(1.0)
    for I in keys:
        row = np.zeros(nv)
        row[idx_rI[I]: idx_rI[I] + len(info[I]["rows"])] = 1
        row[idx_r[info[I]["a"]]] = -1
        Aeq.append(row); beq.append(0.0)
    # inequalities (sparse build)
    from scipy.sparse import lil_matrix
    n_ub = sum(len(B0) for _ in teras) + sum(len(info[I]["cols"]) for (th, I) in idx_mu)
    Aub = lil_matrix((n_ub, nv)); bub = np.zeros(n_ub)
    r = 0
    cons3 = {}
    for th in teras:
        for beta in B0:
            Aub[r, idx_lam[th]] = 1
            for I in keys:
                if info[I]["beta"] == beta and (th, I) in idx_mu:
                    Aub[r, idx_mu[(th, I)]] = -1
            cons3[(th, beta)] = r
            r += 1
    for (th, I), j in idx_mu.items():
        d = info[I]
        P, U = d["P"][th], d["U"][th]
        for jb in range(len(d["cols"])):
            Aub[r, j] = 1
            for ia in range(len(d["rows"])):
                Aub[r, idx_rI[I] + ia] = -P * U[ia, jb]
            r += 1
    bounds = [(0, None)] * (nA + sum(len(info[I]["rows"]) for I in keys)) + [(None, None)] * (nv - nA - sum(len(info[I]["rows"]) for I in keys))
    res = linprog(c, A_ub=Aub.tocsr(), b_ub=bub, A_eq=np.array(Aeq), b_eq=beq, bounds=bounds, method="highs")
    if res.status != 0:
        raise RuntimeError(res.message)
    r0 = np.clip(res.x[:nA], 0, None); r0 /= r0.sum()
    # opponent's stage-0 realization weights from the duals of constraints (3): |marginal| = b0(theta) y_theta(beta)
    marg = np.abs(res.ineqlin.marginals)
    y = {}
    for th in teras:
        w = np.array([marg[cons3[(th, beta)]] for beta in B0])
        y[th] = (w / w.sum()) if w.sum() > 1e-12 else None
    return -res.fun, r0, y


def backward_single(info, A0, B0, th):
    """Full-information value of the 2-stage game for one type (backward induction), as a check."""
    M = np.zeros((len(A0), len(B0)))
    for I, d in info.items():
        if th in d["U"]:
            M[A0.index(d["a"]), B0.index(d["beta"])] += d["P"][th] * d["val"][th]
    return ha.lp_x(M)[1], M


# ---------------------------------------------------------------- eta and the A3 agent

def simplex_grid(k, step):
    n = int(round(1 / step))
    for comp in itertools.product(range(n + 1), repeat=k - 1):
        if sum(comp) <= n:
            yield np.array(list(comp) + [n - sum(comp)]) / n


def eta_at(d, teras, b):
    Us = [d["U"][th] for th in teras]
    if Us[0].size == 1:
        return 0.0
    lin = sum(bk * d["val"][th] for bk, th in zip(b, teras))
    return float(lin - ha.agent_form_lp(Us, b)[1])


def analyse(opp_set, own_set, teras, grid_step=0.1):
    t0 = time.perf_counter()
    RETRIES[0] = 0
    states, A0, B0, info = build(opp_set, own_set, teras)
    stage1(info)
    t_build = time.perf_counter() - t0
    k = len(teras)
    b0 = {th: 1.0 / k for th in teras}
    allu = np.concatenate([d["U"][th].ravel() for d in info.values() for th in d["U"]])
    Delta = float(allu.max() - allu.min())

    V0, r0, y_eq = seqform(info, A0, B0, teras, b0)
    Vfull, Mw = {}, {}
    y_full = {}
    for th in teras:
        v_bi, Mw[th] = backward_single(info, A0, B0, th)
        v_sf, _, y_th = seqform(info, A0, B0, teras, {t: float(t == th) for t in teras})
        assert abs(v_bi - v_sf) < 1e-6 * max(1.0, Delta), (v_bi, v_sf)
        Vfull[th] = v_bi
        y_full[th] = y_th[th]
    voi_total = sum(b0[th] * Vfull[th] for th in teras) - V0

    # eta at pre-tera children (shared by all types) and its sup over a belief grid
    pre = [I for I, d in info.items() if len(d["U"]) == k]
    eta_b0 = {I: eta_at(info[I], teras, np.full(k, 1 / k)) for I in pre}
    eta_sup, eta_arg = 0.0, None
    grid = list(simplex_grid(k, grid_step))
    for I in pre:
        if info[I]["U"][teras[0]].size == 1:
            continue
        for b in grid:
            e = eta_at(info[I], teras, b)
            if e > eta_sup:
                eta_sup, eta_arg = e, (I[0], I[1], [round(float(x), 2) for x in b])

    # A3 agent: stage 0 with world-only continuation (exact per-type values), stage 1 AF over b_hard
    Ms = [Mw[th] for th in teras]
    x0, v_a3 = ha.agent_form_lp(Ms, [b0[th] for th in teras])
    x1 = {}
    for I, d in info.items():
        ths = [th for th in teras if th in d["U"]]
        if d["U"][ths[0]].size == 1:
            x1[I] = np.array([1.0]); continue
        if len(ths) == 1:
            x1[I] = ha.lp_x(d["U"][ths[0]])[0]
        else:
            w = np.array([b0[th] * d["P"][th] for th in ths]); w /= w.sum()
            x1[I] = ha.agent_form_lp([d["U"][th] for th in ths], w)[0]
    # best response per type
    lam, beta_star, mu = {}, {}, {}
    for th in teras:
        val = {}
        for beta in B0:
            tot = 0.0
            for I, d in info.items():
                if d["beta"] == beta and th in d["U"]:
                    m = float((d["U"][th].T @ x1[I]).min())
                    mu[(th, I)] = m
                    tot += x0[A0.index(d["a"])] * d["P"][th] * m
            val[beta] = tot
        beta_star[th] = min(val, key=val.get)
        lam[th] = val[beta_star[th]]
    payoff_br = sum(b0[th] * lam[th] for th in teras)
    expl = V0 - payoff_br
    # TV at stage 1 under the pure best response, and eta along that play
    e_tv, e_eta, p_reach = 0.0, 0.0, 0.0
    for I, d in info.items():
        ths = [th for th in teras if th in d["U"] and beta_star[th] == d["beta"]]
        if not ths:
            continue
        pr = sum(b0[th] * x0[A0.index(d["a"])] * d["P"][th] for th in ths)
        if pr <= 0:
            continue
        true_post = np.array([b0[th] * d["P"][th] if th in ths else 0.0 for th in teras if th in d["U"]])
        true_post /= true_post.sum()
        hard = np.array([b0[th] * d["P"][th] for th in teras if th in d["U"]]); hard /= hard.sum()
        tv = 0.5 * np.abs(true_post - hard).sum()
        e_tv += pr * tv
        if len(d["U"]) == k:
            e_eta += pr * eta_b0[I]
        p_reach += pr
    bound_C1 = eta_sup + 2 * Delta * e_tv
    tera0 = lambda y: None if y is None else float(sum(p for p, beta in zip(y, B0) if beta.endswith("-tera")))
    return dict(
        opp=opp_set.species, opp_role=opp_set.role, own=own_set.species, own_role=own_set.role, teras=list(teras), k=k,
        n_info=len(info), n_pre=len(pre), Delta=Delta, leaf_retries=RETRIES[0], build_s=round(t_build, 1), total_s=round(time.perf_counter() - t0, 1),
        V0=V0, Vfull=Vfull, voi_total=voi_total, v_a3=v_a3, sandwich_gap=v_a3 - V0,
        eta_b0_max=max(eta_b0.values()) if eta_b0 else 0.0, eta_b0_mean=float(np.mean(list(eta_b0.values()))) if eta_b0 else 0.0,
        frac_eta_pos=float(np.mean([e > 1e-6 * Delta for e in eta_b0.values()])) if eta_b0 else 0.0,
        eta_NL0=eta_sup, eta_NL0_arg=eta_arg, expl_a3=expl, e_tv1=e_tv, e_eta_play=e_eta, bound_C1=bound_C1,
        c1_holds=bool(expl <= bound_C1 + 1e-7 * max(1, Delta)), sandwich_holds=bool(-1e-7 * max(1, Delta) <= v_a3 - V0 <= eta_sup + 1e-7 * max(1, Delta)),
        x0_eq={a: round(float(p), 3) for a, p in zip(A0, r0) if p > 1e-3}, x0_a3={a: round(float(p), 3) for a, p in zip(A0, x0) if p > 1e-3},
        beta_star=beta_star, tera0_eq={th: tera0(y_eq[th]) for th in teras}, tera0_full={th: tera0(y_full[th]) for th in teras},
    )


def pick_matchup(rng):
    while True:
        t = rb.sample_team(rng, n=2)
        own, opp = t[0], t[1]
        teras = rb.sets()[opp.species]["roles"][opp.role]["teraTypes"]
        if len(teras) >= 2:
            return own, opp, list(teras)


def worker(args):
    seed, step = args
    rng = random.Random(seed)
    own, opp, teras = pick_matchup(rng)
    try:
        out = analyse(opp, own, teras, step)
        out["seed"] = seed
    except Exception as e:  # noqa: BLE001
        out = dict(seed=seed, opp=opp.species, own=own.species, teras=teras, error=repr(e)[:300])
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--step", type=float, default=0.1)
    ap.add_argument("--out", default="tera_subgame_out.jsonl")
    a = ap.parse_args()
    from multiprocessing import get_context
    tasks = [(a.seed * 1000 + i, a.step) for i in range(a.n)]
    with get_context("spawn").Pool(a.workers) as pool, open(a.out, "a") as f:
        for out in pool.imap_unordered(worker, tasks):
            f.write(json.dumps(out) + "\n"); f.flush()
            if "error" in out:
                print("ERR", out["seed"], out["opp"], out["error"], flush=True)
            else:
                print(f"seed {out['seed']:>5} {out['opp']:>16} vs {out['own']:>16} k={out['k']} info={out['n_info']:>4} Delta={out['Delta']:7.1f} "
                      f"V0={out['V0']:7.2f} VOI={out['voi_total']:6.2f} etaNL0={out['eta_NL0']:6.2f} sand={out['sandwich_gap']:6.2f} "
                      f"expl={out['expl_a3']:6.2f} ETV={out['e_tv1']:.3f} C1={out['c1_holds']} S={out['sandwich_holds']} {out['total_s']}s", flush=True)
