"""Two-sided no-regret / minimax / CCE quantities along real trajectories, from OURBOT_LOG_ABAR=1 logs.

Per non-skipped decision t (Abar_t = A, our mixture p_t, realised opponent column j_t, experts Q_t, styles qs_t):
  g_t        = p_t' A e_j                         realised stage payoff (our mixture, their realised column)
  e_{k,t}    = Q_t[k]' A e_j                      what fixed expert k would have earned against the realised column
  brR_t      = max_i A[i, j]                      our per-turn best response (dynamic comparator)
  s_{k,t}    = p_t' A qs_t[k]                     what fixed style k would have cost the opponent against our p_t
  brC_t      = min_j' p_t' A e_j'                 the opponent's per-turn best response to our mixture
  v_t        = val(A)                             stage minimax value (LP)
  expl_t     = v_t - brC_t  >= 0                  our exploitability at the stage (b1_gap's quantity)
Regrets (per turn, averaged over the T non-skipped turns; units: Hedge's DELTA = 400, so comparable to E2's 0.004):
  eps_R(E)   = max_k mean e_k - mean g            our external regret vs the 4 fixed experts
  eps_R^dyn  = mean brR - mean g                  our dynamic regret
  eps_C(S)   = mean g - min_k mean s_k            the opponent's external regret vs the 5 fixed styles (in OUR model)
  eps_C^dyn  = mean g - mean brC                  the opponent's dynamic regret
Identities (arithmetic; the closure residual checks the implementation):
  (I1)  max_k mean e_k - min_k mean s_k = eps_R(E) + eps_C(S)        restricted empirical CCE gap
  (I2)  mean g - mean v = eps_C^dyn - expl_mean                         dynamic / stage-value form
Also: the posterior learner's own expert Q[3] (safex on q-hat) as a player; calibration of q-hat; Hedge log consistency;
a categorical correlation diagnostic; and for a logged opponent side (CEQ's style_eq) the model-mismatch check.
Usage: python two_sided.py <results_dir> <tag> [--out out.json] [--procs 4]
"""
import argparse
import json
import os
import sys
from multiprocessing import Pool

import numpy as np
from scipy.optimize import linprog

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cce_load import load_tag, DELTA

EXPERTS = ["top", "sampled", "safest", "safex"]
STYLES = ["eq", "br_unif", "br_us", "safe", "uniform"]


def lpval(A):
    """max_x min_y x'Ay by LP (highs)."""
    n, m = A.shape
    if n == 1:
        return float(A.min())
    if m == 1:
        return float(A.max())
    c = np.zeros(n + 1); c[-1] = -1
    r = linprog(c, A_ub=np.hstack([-A.T, np.ones((m, 1))]), b_ub=np.zeros(m),
                A_eq=np.r_[np.ones(n), 0][None], b_eq=[1], bounds=[(0, None)] * n + [(None, None)], method="highs")
    return float(r.x[-1])


def per_row(row):
    A, p, j = row["A"], row["p"], row["j"]
    out = dict(t=row["t"], stream=row["stream"], seed=row["seed"], game=row["game"], n=A.shape[0], m=A.shape[1],
               rng=float(np.ptp(A)), how=row["how"], v=lpval(A))
    pa = p @ A
    out.update(g=float(pa[j]), brR=float(A[:, j].max()), brC=float(pa.min()), both=bool(A.shape[0] > 1 and A.shape[1] > 1))
    out["expl"] = out["v"] - out["brC"]
    if "Q" in row:
        Q = row["Q"]
        out["e"] = (Q @ A[:, j]).tolist()
        out["expl_e"] = [out["v"] - float((q @ A).min()) for q in Q]
        out["brC_e"] = [float((q @ A).min()) for q in Q]          # the opponent's BR value against each expert
        out["w_dec"] = [row["w_hedge"][k] for k in EXPERTS]
        if row["r_log"] is not None:
            out["r_log"] = row["r_log"].tolist()
    if "qs" in row:
        qs, qh = row["qs"], row["qhat"]
        out["s"] = (pa @ qs.T).tolist()
        out["qhat_j"] = float(qh[j]); out["qs_j"] = qs[:, j].tolist()
        out["qhat_all"] = qh.tolist()
        out["s_e3"] = [float(row["Q"][3] @ A @ q) for q in qs] if "Q" in row else None
        # categorical coding for the correlation diagnostic
        if "Q" in row:
            Q = row["Q"]
            top, safest = int(Q[0].argmax()), int(Q[2].argmax())
            cat_us = np.zeros(4)                                    # top-only, safest-only, both, other
            for i, pi in enumerate(p):
                k = 2 if (i == top and i == safest) else 0 if i == top else 1 if i == safest else 3
                cat_us[k] += pi
            out["cat_us"] = cat_us.tolist()
            am = [int(q.argmax()) for q in qs[:4]]                   # eq, br_unif, br_us, safe
            out["cat_them"] = next((k for k, a in enumerate(am) if a == j), 4)
            out["cat_them_model"] = [float(sum(qh[a] for a in set(am[:k + 1]) - set(am[:k]))) for k in range(4)]
    return out


def mean_se_games(x, games):
    """Mean and game-clustered SE of a per-decision quantity."""
    x = np.asarray(x, float); games = np.asarray(games)
    ug, inv = np.unique(games, return_inverse=True)
    sums = np.bincount(inv, x); cnt = np.bincount(inv)
    G = len(ug); mu = x.mean()
    resid = sums - mu * cnt
    se = np.sqrt((resid ** 2).sum() * G / (G - 1)) / len(x)
    return float(mu), float(se)


def summarise(rows, scale=DELTA, label=""):
    """Aggregate the per-row quantities (all in units of `scale` per turn)."""
    T = len(rows)
    games = [(r["seed"], r["game"]) for r in rows]
    gid = np.array([s * 10 + g for s, g in games])
    g = np.array([r["g"] for r in rows]) / scale
    v = np.array([r["v"] for r in rows]) / scale
    brR = np.array([r["brR"] for r in rows]) / scale
    brC = np.array([r["brC"] for r in rows]) / scale
    expl = np.array([r["expl"] for r in rows]) / scale
    res = dict(label=label, T=T, games=len(set(games)), rng_median=float(np.median([r["rng"] for r in rows])),
               rng_mean=float(np.mean([r["rng"] for r in rows])), scale=scale,
               g_mean=mean_se_games(g, gid), v_mean=mean_se_games(v, gid),
               eps_R_dyn=mean_se_games(brR - g, gid), eps_C_dyn=mean_se_games(g - brC, gid),
               expl_mean=mean_se_games(expl, gid))
    res["I2_lhs"] = g.mean() - v.mean(); res["I2_rhs"] = (g - brC).mean() - expl.mean()
    res["I2_residual"] = res["I2_lhs"] - res["I2_rhs"]
    if "e" in rows[0]:
        E = np.array([r["e"] for r in rows]) / scale
        res["expert_mean"] = dict(zip(EXPERTS, E.mean(0).tolist()))
        res["expert_expl"] = dict(zip(EXPERTS, (np.array([r["expl_e"] for r in rows]) / scale).mean(0).tolist()))
        k = int(E.mean(0).argmax()); res["best_expert"] = EXPERTS[k]
        res["eps_R_E"] = mean_se_games(E[:, k] - g, gid)
        res["eps_R_E_all"] = dict(zip(EXPERTS, (E.mean(0) - g.mean()).tolist()))
        # the posterior learner's expert (safex on q-hat) as a player of its own
        e3 = E[:, 3]
        res["post_player"] = dict(payoff=mean_se_games(e3, gid), eps_R_E=float(E.mean(0).max() - e3.mean()),
                                  eps_R_dyn=mean_se_games(brR - e3, gid),
                                  expl=mean_se_games(np.array([r["expl_e"][3] for r in rows]) / scale, gid))
        if rows[0].get("s_e3") is not None:
            S3 = np.array([r["s_e3"] for r in rows]) / scale
            res["post_player"]["eps_C_S_against_it"] = float(e3.mean() - S3.mean(0).min())
            res["post_player"]["eps_C_dyn_against_it"] = float(e3.mean() - (np.array([r["brC_e"][3] for r in rows]) / scale).mean())
            res["post_player"]["style_cost_against_it"] = dict(zip(STYLES, S3.mean(0).tolist()))
        if "r_log" in rows[0]:
            R = np.array([r["r_log"] for r in rows if "r_log" in r]); W = np.array([r["w_dec"] for r in rows if "r_log" in r])
            res["hedge_log"] = dict(T=len(R), eps_R_E_logged=float(R.mean(0).max() - (W * R).sum(1).mean()),
                                    best_expert_logged=EXPERTS[int(R.mean(0).argmax())],
                                    expert_mean_logged=dict(zip(EXPERTS, R.mean(0).tolist())),
                                    mixture_mean_logged=float((W * R).sum(1).mean()))
    if "s" in rows[0]:
        S = np.array([r["s"] for r in rows]) / scale
        res["style_cost"] = dict(zip(STYLES, S.mean(0).tolist()))
        k = int(S.mean(0).argmin()); res["best_style"] = STYLES[k]
        res["eps_C_S"] = mean_se_games(g - S[:, k], gid)
        res["eps_C_S_all"] = dict(zip(STYLES, (g.mean() - S.mean(0)).tolist()))
        if "e" in rows[0]:
            res["I1_lhs"] = float(E.mean(0).max() - S.mean(0).min())
            res["I1_rhs"] = res["eps_R_E"][0] + res["eps_C_S"][0]
            res["I1_residual"] = res["I1_lhs"] - res["I1_rhs"]
        # calibration of q-hat: reliability over all (turn, column) pairs, 10 bins; ECE; log loss
        res["calibration"] = calibration(rows)
        res["logloss_qhat"] = float(np.mean([-np.log(r["qhat_j"]) for r in rows]))
        res["logloss_styles"] = dict(zip(STYLES, (-np.log(np.array([r["qs_j"] for r in rows]))).mean(0).tolist()))
        # correlation diagnostic (categorical)
        if "cat_us" in rows[0]:
            res["correlation"] = correlation(rows)
    return res


def calibration(rows, bins=10):
    pred = np.concatenate([r["qhat_all"] for r in rows])
    hit = np.concatenate([(np.arange(len(r["qhat_all"])) == r["j_idx"]).astype(float) for r in rows])
    edges = np.array([0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 1.0001])
    out = []; ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (pred >= lo) & (pred < hi)
        if m.sum():
            out.append(dict(lo=float(lo), hi=float(hi), n=int(m.sum()), pred=float(pred[m].mean()), freq=float(hit[m].mean())))
            ece += abs(pred[m].mean() - hit[m].mean()) * m.sum() / len(pred)
    # per-turn "probability of the realised column" vs its frequency is the same thing; also the top-1 reliability
    top = np.array([r["qhat_all"][int(np.argmax(r["qhat_all"]))] for r in rows])
    tophit = np.array([int(np.argmax(r["qhat_all"])) == r["j_idx"] for r in rows], float)
    return dict(ece=float(ece), bins=out, top1_pred=float(top.mean()), top1_freq=float(tophit.mean()), n_pairs=int(len(pred)))


def correlation(rows, n_perm=200, seed=0):
    U = np.array([r["cat_us"] for r in rows]); th = np.array([r["cat_them"] for r in rows])
    K = 5
    def joint(th_):
        J = np.zeros((4, K))
        for u, c in zip(U, th_):
            J[:, c] += u
        return J / len(th_)
    J = joint(th); tv = 0.5 * np.abs(J - np.outer(J.sum(1), J.sum(0))).sum()
    rng = np.random.default_rng(seed)
    null = []
    for _ in range(n_perm):
        Jp = joint(rng.permutation(th)); null.append(0.5 * np.abs(Jp - np.outer(Jp.sum(1), Jp.sum(0))).sum())
    null = np.array(null)
    # model product: our category distribution x the model's predicted category distribution, averaged over turns
    M = np.array([r["cat_them_model"] + [1 - sum(r["cat_them_model"])] for r in rows])
    Jm = (U[:, :, None] * M[:, None, :]).mean(0)
    return dict(tv_from_product=float(tv), perm_null_mean=float(null.mean()), perm_null_p95=float(np.percentile(null, 95)),
                perm_p=float((null >= tv).mean()), tv_from_model_product=float(0.5 * np.abs(J - Jm).sum()),
                marg_us=J.sum(1).tolist(), marg_them=J.sum(0).tolist(), joint=J.tolist())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("res"); ap.add_argument("tag"); ap.add_argument("--out", default=None); ap.add_argument("--procs", type=int, default=4)
    ap.add_argument("--rows-out", default=None, help="write the per-row quantities (jsonl.gz) here")
    ap.add_argument("--side", default="A")
    a = ap.parse_args()
    rows = load_tag(a.res, a.tag, a.side)
    used = [r for r in rows if not r["skipped"] and r["j"] is not None]
    for r in used:
        r["j_idx"] = r["j"]
    print(f"{a.tag}: {len(rows)} decisions with feedback, {len(used)} usable ({len(rows) - len(used)} skipped), "
          f"{len(set((r['seed'], r['game']) for r in used))} games, {len(set(r['stream'] for r in used))} streams")
    with Pool(a.procs) as P:
        per = P.map(per_row, used, chunksize=64)
    for r, q in zip(used, per):
        q["j_idx"] = r["j"]
    res = dict(all=summarise(per, label="all usable turns"),
               both_sided=summarise([q for q in per if q["both"]], label="both sides >= 2 options"),
               nonforced=summarise([q for q in per if q["how"] != "forced"], label="opponent not forced"))
    res["per_stream"] = {str(s): summarise([q for q in per if q["stream"] == s], label=f"stream {s}")
                         for s in sorted(set(q["stream"] for q in per))}
    if a.rows_out:
        import gzip
        with gzip.open(a.rows_out, "wt") as f:
            for q in per:
                f.write(json.dumps(q) + "\n")
    if a.out:
        json.dump(res, open(a.out, "w"), indent=1)
    s = res["all"]
    print(f"T={s['T']} games={s['games']} range median {s['rng_median']:.1f} (units: 1 = {DELTA} engine units per turn)")
    print(f"  g_mean {s['g_mean'][0]:.4f} (SE {s['g_mean'][1]:.4f})  v_mean {s['v_mean'][0]:.4f}")
    if "eps_R_E" in s:
        print(f"  ours:  eps_R(E) {s['eps_R_E'][0]:.5f} best {s['best_expert']}", end=" | ")
    print(f"  ours: eps_R^dyn {s['eps_R_dyn'][0]:.4f} | expl {s['expl_mean'][0]:.5f}")
    if "eps_C_S" in s:
        print(f"  opp:   eps_C(S) {s['eps_C_S'][0]:.5f} best style {s['best_style']}", end=" | ")
    print(f"  opp: eps_C^dyn {s['eps_C_dyn'][0]:.4f}")
    print(f"  I1 residual {s.get('I1_residual', float('nan')):.2e}   I2 residual {s['I2_residual']:.2e}")
    if "hedge_log" in s:
        print(f"  hedge logged-r regret {s['hedge_log']['eps_R_E_logged']:.5f} (best {s['hedge_log']['best_expert_logged']})")
    if "calibration" in s:
        print(f"  calibration ECE {s['calibration']['ece']:.4f}; top1 pred {s['calibration']['top1_pred']:.3f} vs freq {s['calibration']['top1_freq']:.3f}")
    if "correlation" in s:
        c = s["correlation"]; print(f"  correlation TV {c['tv_from_product']:.4f} (perm null mean {c['perm_null_mean']:.4f}, p {c['perm_p']:.2f}); TV from model product {c['tv_from_model_product']:.4f}")


if __name__ == "__main__":
    main()
