"""Two-sided 1v1 tera subgame: both players' tera types hidden (joint types (theta, phi), independent uniform priors).

Same two-stage game as tera_subgame.py. Computes
  V_real          sequence-form value with both types hidden (agent's plan indexed by its own type phi)
  V_conc(phi)     value when the agent's type is conceded to the opponent (q = delta_phi), per phi
  conceal_value   V_real - sum_phi q(phi) V_conc(phi)   = the Remark's "value of concealing our team", here our tera
  voi_opp         sum_theta b(theta) V_real(delta_theta) - V_real   (the opponent's concealment value, two-sided game)
All in engine leaf units; Delta = payoff range of the subgame.

Usage: tera_two_sided.py --n 8 --seed 0 --workers 4 --out two_sided.jsonl
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import random
import sys
import time
from dataclasses import replace

import numpy as np
from scipy.optimize import linprog
from scipy.sparse import lil_matrix

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tera_subgame as ts  # noqa: E402
import poke_engine as pe  # noqa: E402
import nash  # noqa: E402
import pkbattle as pb  # noqa: E402
import randbats as rb  # noqa: E402


def pub_key2(c):
    own = tuple(ts._mon_key(p, True) for p in c.side_one.pokemon)
    opp = tuple(ts._mon_key(p, True) for p in c.side_two.pokemon)
    sides = tuple(ts._ser(getattr(s, f)) for s in (c.side_one, c.side_two) for f in ts.hd.SF)
    glob = tuple(ts._ser(getattr(c, f)) for f in ts.hd.GF)
    return (own, opp, sides, glob)


def build2(own_set, opp_set, phis, thetas):
    states = {(th, ph): pe.State(side_one=pe.Side(pokemon=[rb.to_pokemon(replace(own_set, tera=ph))]),
                                 side_two=pe.Side(pokemon=[rb.to_pokemon(replace(opp_set, tera=th))]))
              for th in thetas for ph in phis}
    A0, B0 = (sorted(x) for x in pb.options(next(iter(states.values()))))
    for st in states.values():
        o = pb.options(st)
        assert (sorted(o[0]), sorted(o[1])) == (A0, B0), ("options differ across joint types", o, A0, B0)
    info = {}
    for a in A0:
        for beta in B0:
            for jt, st in states.items():
                acc = {}
                for br in pe.generate_instructions(st, a, beta):
                    c = st.apply_instructions(br)
                    k = pub_key2(c)
                    if k in acc:
                        acc[k][0] += br.percentage / 100.0
                    else:
                        acc[k] = [br.percentage / 100.0, c]
                for k, (p, c) in acc.items():
                    d = info.setdefault((a, beta, k), dict(a=a, beta=beta, P={}, child={}))
                    d["P"][jt] = p
                    d["child"][jt] = c
    for d in info.values():
        d["U"], d["rows"], d["cols"] = {}, None, None
        for jt, c in d["child"].items():
            if ts.terminal(c):
                M, s1, s2 = np.array([[ts.stable_leaf(c)]]), ["none"], ["none"]
            else:
                M, s1, s2 = ts.stable_full_matrix(c)
                ri, ci = sorted(range(len(s1)), key=lambda i: s1[i]), sorted(range(len(s2)), key=lambda j: s2[j])
                M, s1, s2 = M[np.ix_(ri, ci)], sorted(s1), sorted(s2)
            if d["rows"] is None:
                d["rows"], d["cols"] = s1, s2
            elif (s1, s2) != (d["rows"], d["cols"]):
                raise RuntimeError("labels differ at shared child")
            d["U"][jt] = M
    return A0, B0, info


def seqform2(info, A0, B0, thetas, phis, b, q):
    keys = list(info)
    nA = len(A0)
    off = 0
    idx_r = {(ph, a): pi * nA + i for pi, ph in enumerate(phis) for i, a in enumerate(A0)}   # one stage-0 plan per own type
    off += len(phis) * nA
    idx_rI = {}
    for ph in phis:
        for I in keys:
            if any((th, ph) in info[I]["U"] for th in thetas):
                idx_rI[(ph, I)] = off
                off += len(info[I]["rows"])
    n_nonneg = off
    idx_lam = {th: off + i for i, th in enumerate(thetas)}
    off += len(thetas)
    idx_mu = {}
    for th in thetas:
        for I in keys:
            if any((th, ph) in info[I]["U"] for ph in phis):
                idx_mu[(th, I)] = off
                off += 1
    nv = off
    c = np.zeros(nv)
    for th in thetas:
        c[idx_lam[th]] = -b[th]
    Aeq, beq = [], []
    for ph in phis:
        row = np.zeros(nv)
        for a in A0:
            row[idx_r[(ph, a)]] = 1
        Aeq.append(row); beq.append(1.0)
    for (ph, I), j in idx_rI.items():
        row = np.zeros(nv)
        row[j: j + len(info[I]["rows"])] = 1
        row[idx_r[(ph, info[I]["a"])]] = -1
        Aeq.append(row); beq.append(0.0)
    n_ub = len(thetas) * len(B0) + sum(len(info[I]["cols"]) for (th, I) in idx_mu)
    Aub = lil_matrix((n_ub, nv)); bub = np.zeros(n_ub)
    r = 0
    for th in thetas:
        for beta in B0:
            Aub[r, idx_lam[th]] = 1
            for I in keys:
                if info[I]["beta"] == beta and (th, I) in idx_mu:
                    Aub[r, idx_mu[(th, I)]] = -1
            r += 1
    for (th, I), j in idx_mu.items():
        d = info[I]
        for jb in range(len(d["cols"])):
            Aub[r, j] = 1
            for ph in phis:
                if (th, ph) in d["U"] and q[ph] > 0:
                    P, U = d["P"][(th, ph)], d["U"][(th, ph)]
                    base = idx_rI[(ph, I)]
                    for ia in range(len(d["rows"])):
                        Aub[r, base + ia] += -q[ph] * P * U[ia, jb]
            r += 1
    bounds = [(0, None)] * n_nonneg + [(None, None)] * (nv - n_nonneg)
    res = linprog(c, A_ub=Aub.tocsr(), b_ub=bub, A_eq=np.array(Aeq), b_eq=beq, bounds=bounds, method="highs")
    if res.status != 0:
        raise RuntimeError(res.message)
    return -res.fun


def analyse2(own_set, opp_set, phis, thetas):
    t0 = time.perf_counter()
    ts.RETRIES[0] = 0
    A0, B0, info = build2(own_set, opp_set, phis, thetas)
    bu = {th: 1 / len(thetas) for th in thetas}
    qu = {ph: 1 / len(phis) for ph in phis}
    allu = np.concatenate([U.ravel() for d in info.values() for U in d["U"].values()])
    Delta = float(allu.max() - allu.min())
    V_real = seqform2(info, A0, B0, thetas, phis, bu, qu)
    V_conc = {ph: seqform2(info, A0, B0, thetas, phis, bu, {p: float(p == ph) for p in phis}) for ph in phis}
    V_real_th = {th: seqform2(info, A0, B0, thetas, phis, {t: float(t == th) for t in thetas}, qu) for th in thetas}
    V_both = {(th, ph): seqform2(info, A0, B0, thetas, phis, {t: float(t == th) for t in thetas}, {p: float(p == ph) for p in phis})
              for th in thetas for ph in phis}
    conceal = V_real - sum(qu[ph] * V_conc[ph] for ph in phis)
    voi_opp = sum(bu[th] * V_real_th[th] for th in thetas) - V_real
    voi_opp_conc = sum(qu[ph] * (sum(bu[th] * V_both[(th, ph)] for th in thetas) - V_conc[ph]) for ph in phis)
    return dict(own=own_set.species, own_role=own_set.role, opp=opp_set.species, opp_role=opp_set.role, phis=phis, thetas=thetas,
                n_info=len(info), Delta=Delta, leaf_retries=ts.RETRIES[0], V_real=V_real, V_conc=V_conc, conceal_value=conceal, conceal_ge0=bool(conceal >= -1e-7 * max(1, Delta)),
                voi_opp_two_sided=voi_opp, voi_opp_one_sided_avg=voi_opp_conc, total_s=round(time.perf_counter() - t0, 1))


def worker(seed):
    rng = random.Random(seed)
    while True:
        t = rb.sample_team(rng, n=2)
        own, opp = t[0], t[1]
        phis = list(rb.sets()[own.species]["roles"][own.role]["teraTypes"])
        thetas = list(rb.sets()[opp.species]["roles"][opp.role]["teraTypes"])
        if 2 <= len(phis) <= 3 and 2 <= len(thetas) <= 3:
            break
    try:
        out = analyse2(own, opp, phis, thetas); out["seed"] = seed
    except Exception as e:  # noqa: BLE001
        out = dict(seed=seed, own=own.species, opp=opp.species, error=repr(e)[:300])
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", default="two_sided.jsonl")
    a = ap.parse_args()
    from multiprocessing import get_context
    with get_context("spawn").Pool(a.workers) as pool, open(a.out, "a") as f:
        for out in pool.imap_unordered(worker, [a.seed * 1000 + 500 + i for i in range(a.n)]):
            f.write(json.dumps(out) + "\n"); f.flush()
            if "error" in out:
                print("ERR", out, flush=True)
            else:
                print(f"seed {out['seed']} {out['own']}({len(out['phis'])}) vs {out['opp']}({len(out['thetas'])}) info={out['n_info']} Delta={out['Delta']:.0f} "
                      f"V_real={out['V_real']:.2f} conceal={out['conceal_value']:.2f} ({out['conceal_value'] / out['Delta']:.3%}) "
                      f"voi_opp2={out['voi_opp_two_sided']:.2f} voi_opp1={out['voi_opp_one_sided_avg']:.2f} {out['total_s']}s", flush=True)
