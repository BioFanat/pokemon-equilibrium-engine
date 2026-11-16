"""Verifier: recompute the 46 one-sided tera subgames (30 reported + 16 dropped) with OURBOT_DET_LEAF=1.

Uses tera_subgame.analyse unchanged for the headline fields; the leaf guard ts._stable is replaced by
"evaluate twice, must agree exactly" (det leaf => any disagreement is recorded, not voted away).
Extras computed from the same info structure (stashed by wrapping ts.stage1):
  V0_dual        opponent-side (minimiser) sequence-form LP value: independent formulation of V0(b0)
  expl_le_etaNL  expl <= eta_NL0 alone (no TV term)
  e_eta_true     E_BR[eta(0, I, p_I)] at the TRUE posterior p_I (the provable along-play term), and its check
  leaf stats     nash.LEAF_STATS (non_d1 first tries, retries, fallback)
Usage: OURBOT_DET_LEAF=1 python recompute_subgames.py --seeds 2001,2006,... --workers 2 --out out/x.jsonl
"""
from __future__ import annotations
import argparse, json, os, random, sys, time
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "scripts"))
assert os.environ.get("OURBOT_DET_LEAF") == "1", "set OURBOT_DET_LEAF=1"
import tera_subgame as ts  # noqa: E402
import nash  # noqa: E402
import hidden_agents as ha  # noqa: E402
from scipy.optimize import linprog  # noqa: E402
from scipy.sparse import lil_matrix  # noqa: E402

MISMATCH = [0]
STASH = {}


def _stable2(fn, key, tries=12, need=3):
    a = fn(); b = fn()
    if key(a) != key(b):
        MISMATCH[0] += 1
        c = fn()  # record and take the majority of three
        return c if key(c) == key(b) else a
    return a


ts._stable = _stable2
_orig_stage1 = ts.stage1


def _stage1(info):
    _orig_stage1(info)
    STASH["info"] = info


ts.stage1 = _stage1


def dual_value(info, A0, B0, teras, b0):
    """min z s.t. z >= sum_{I:a} nu_I (all a); nu_I >= sum_th b0 P U[row,:] . y_{th,I} (all rows); y realisation plans."""
    keys = list(info)
    off = 0
    iy0 = {}
    for th in teras:
        for beta in B0:
            iy0[(th, beta)] = off; off += 1
    iyI = {}
    for I in keys:
        for th in info[I]["U"]:
            iyI[(th, I)] = off; off += len(info[I]["cols"])
    n_nonneg = off
    inu = {I: off + i for i, I in enumerate(keys)}; off += len(keys)
    iz = off; off += 1
    c = np.zeros(off); c[iz] = 1
    Aeq, beq = [], []
    for th in teras:
        row = np.zeros(off)
        for beta in B0:
            row[iy0[(th, beta)]] = 1
        Aeq.append(row); beq.append(1.0)
    for (th, I), j in iyI.items():
        row = np.zeros(off); row[j:j + len(info[I]["cols"])] = 1; row[iy0[(th, info[I]["beta"])]] = -1
        Aeq.append(row); beq.append(0.0)
    nub = len(B0) * 0 + len(set(info[I]["a"] for I in keys)) + sum(len(info[I]["rows"]) for I in keys)
    Aub = lil_matrix((nub + len(A0), off)); bub = np.zeros(nub + len(A0)); r = 0
    for a in A0:
        Aub[r, iz] = -1
        for I in keys:
            if info[I]["a"] == a:
                Aub[r, inu[I]] = 1
        r += 1
    for I in keys:
        d = info[I]
        for ia in range(len(d["rows"])):
            Aub[r, inu[I]] = -1
            for th in d["U"]:
                w = b0[th] * d["P"][th]
                for jb in range(len(d["cols"])):
                    Aub[r, iyI[(th, I)] + jb] += w * d["U"][th][ia, jb]
            r += 1
    Aub = Aub[:r]; bub = bub[:r]
    bounds = [(0, None)] * n_nonneg + [(None, None)] * (off - n_nonneg)
    res = linprog(c, A_ub=Aub.tocsr(), b_ub=bub, A_eq=np.array(Aeq), b_eq=beq, bounds=bounds, method="highs")
    assert res.status == 0, res.message
    return res.fun


def extras(out, teras):
    info = STASH["info"]
    k = len(teras)
    A0 = sorted({d["a"] for d in info.values()}); B0 = sorted({d["beta"] for d in info.values()})
    b0 = {th: 1.0 / k for th in teras}
    vd = dual_value(info, A0, B0, teras, b0)
    Mw = {th: ts.backward_single(info, A0, B0, th)[1] for th in teras}
    x0, _ = ha.agent_form_lp([Mw[th] for th in teras], [b0[th] for th in teras])
    bs = out["beta_star"]
    e_true, e_b0 = 0.0, 0.0
    for I, d in info.items():
        if len(d["U"]) != k:
            continue
        ths = [th for th in teras if bs[th] == d["beta"]]
        if not ths:
            continue
        pr = sum(b0[th] * x0[A0.index(d["a"])] * d["P"][th] for th in ths)
        if pr <= 0:
            continue
        post = np.array([b0[th] * d["P"][th] if th in ths else 0.0 for th in teras]); post /= post.sum()
        e_true += pr * ts.eta_at(d, teras, post)
        e_b0 += pr * ts.eta_at(d, teras, np.full(k, 1 / k))
    D = out["Delta"]
    return dict(V0_dual=vd, V0_primal_minus_dual=out["V0"] - vd, e_eta_true=e_true, e_eta_b0_recheck=e_b0,
                along_true_holds=bool(out["expl_a3"] <= e_true + 2 * D * out["e_tv1"] + 1e-7 * max(1, D)),
                along_b0_holds=bool(out["expl_a3"] <= out["e_eta_play"] + 2 * D * out["e_tv1"] + 1e-7 * max(1, D)),
                expl_le_etaNL=bool(out["expl_a3"] <= out["eta_NL0"] + 1e-7 * max(1, D)),
                eta_true_le_etaNL=bool(e_true <= out["eta_NL0"] + 1e-7 * max(1, D)))


def worker(seed):
    MISMATCH[0] = 0
    for kk in nash.LEAF_STATS:
        nash.LEAF_STATS[kk] = {} if kk == "depth_hist" else 0
    rng = random.Random(seed)
    own, opp, teras = ts.pick_matchup(rng)
    t0 = time.perf_counter()
    try:
        out = ts.analyse(opp, own, teras, 0.1)
        out["seed"] = seed
        out.update(extras(out, teras))
    except Exception as e:  # noqa: BLE001
        import traceback
        out = dict(seed=seed, opp=opp.species, own=own.species, teras=teras, error=repr(e)[:300], tb=traceback.format_exc()[-800:])
    out["det_mismatch"] = MISMATCH[0]
    out["leaf_stats"] = {kk: (dict(v) if isinstance(v, dict) else v) for kk, v in nash.LEAF_STATS.items()}
    out["wall_s"] = round(time.perf_counter() - t0, 1)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", required=True)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    seeds = [int(s) for s in a.seeds.split(",")]
    from multiprocessing import get_context
    with get_context("spawn").Pool(a.workers) as pool, open(a.out, "a") as f:
        for out in pool.imap_unordered(worker, seeds):
            f.write(json.dumps(out) + "\n"); f.flush()
            if "error" in out:
                print("ERR", out["seed"], out["error"], flush=True)
            else:
                print(f"seed {out['seed']} {out['opp']} vs {out['own']} D={out['Delta']:.1f} VOI={out['voi_total']:.2f} etaNL={out['eta_NL0']:.2f} "
                      f"expl={out['expl_a3']:.2f} dual_diff={out['V0_primal_minus_dual']:.2e} mism={out['det_mismatch']} "
                      f"nond1={out['leaf_stats']['non_d1']} fb={out['leaf_stats']['fallback']} {out['wall_s']}s", flush=True)
