"""Model-mismatch check with a logged opponent side (CEQ: hedge_bayes (A) vs style_eq (B), both with OURBOT_LOG_ABAR=1).

B plays y_t = q_eq(A'_t) on ITS OWN world-averaged matrix A'_t (its worlds, its perspective); our side predicts the
opponent with q_eq(A_t) on OUR matrix. Per turn with exactly one decision on each side:
  tv_eq      TV(y_t^B, q_eq,t^A) on the union of label strings (B's legal options vs our column labels)
  tv_qhat    TV(y_t^B, qhat_t^A)
  mass_out   the mass of y_t^B on options absent from our labels (what map_column would skip)
  val_sum    val(A_t) + val(A'_t): in a shared exact game this is 0 (B's value is the negative of ours)
  cost_true  p_t' A_t y_t^B (the opponent's TRUE expected cost in our model) vs cost_eq = p_t' A_t q_eq,t^A:
             the realisability gap in payoff space; eps_C(S=eq) averages g_t - cost_eq, whose expectation over b_t
             is cost_true - cost_eq (sampling noise averages out), so this isolates the mismatch term.
Usage: python opp_side_check.py <results_dir> <tag>
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cce_load import load_tag, DELTA
from two_sided import lpval, mean_se_games
ENGINE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "engine")
sys.path.insert(0, ENGINE)
import nash                                   # nash.safex_strategy: the eps-safe best response (BAYES_SAFEX_EPS = 0.003)
EPS = 0.003


def main():
    res, tag = sys.argv[1], sys.argv[2]
    A = load_tag(res, tag, "A")
    B = load_tag(res, tag, "B", require_feedback=False)
    byB = {}
    for r in B:
        byB.setdefault((r["seed"], r["game"], r["turn"]), []).append(r)
    byA = {}
    for r in A:
        byA.setdefault((r["seed"], r["game"], r["turn"]), []).append(r)
    out = []
    n_pairs = n_multi = 0
    for key, ra in byA.items():
        rb = byB.get(key)
        if rb is None or len(ra) != 1 or len(rb) != 1:
            n_multi += 1
            continue
        a, b = ra[0], rb[0]
        if a["skipped"] or a["j"] is None or "qs" not in a:
            continue
        n_pairs += 1
        labs = a["labels"]; pos = {l: i for i, l in enumerate(labs)}
        yB = np.zeros(len(labs)); mass_out = 0.0
        for l, m in zip(b["legal"], b["p"]):
            if l in pos:
                yB[pos[l]] += m
            else:
                mass_out += m
        qeq, qh, p, Am = a["qs"][0], a["qhat"], a["p"], a["A"]
        yB_in = yB / yB.sum() if yB.sum() > 0 else None
        d = dict(seed=a["seed"], game=a["game"], turn=a["turn"], mass_out=mass_out, nB=len(b["legal"]), mA=len(labs),
                 realised_in=bool(a["b"] in pos), tv_eq=None, tv_qhat=None, cost_true=None, cost_eq=None, cost_qhat=None,
                 g=float(p @ Am[:, a["j"]]), val_A=lpval(Am), val_B=lpval(b["A"]))
        if yB_in is not None:
            # Prop. 3's oracle: the eps-safe best response to the opponent's TRUE mixture y_t^B, vs the posterior player's
            # eps-safe best response to q-hat (Q[3] for hedge_bayes, the played p for bayes_safex); expected payoffs under y^B
            xhat = a["Q"][3] if "Q" in a else p
            xstar = np.clip(nash.safex_strategy(Am, EPS, yB_in), 0, None); xstar /= xstar.sum()
            d.update(oracle_gap=float(xstar @ Am @ yB_in - xhat @ Am @ yB_in), l1_qhat=float(np.abs(qh - yB_in).sum()),
                     l1_bound=float(np.ptp(Am) * np.abs(qh - yB_in).sum()))
            d.update(tv_eq=0.5 * float(np.abs(yB_in - qeq).sum()), tv_qhat=0.5 * float(np.abs(yB_in - qh).sum()),
                     cost_true=float(p @ Am @ yB_in), cost_eq=float(p @ Am @ qeq), cost_qhat=float(p @ Am @ qh),
                     tv_eq_unif=0.5 * float(np.abs(yB_in - np.full(len(labs), 1 / len(labs))).sum()))
        out.append(d)
    gid = np.array([d["seed"] * 10 + d["game"] for d in out])
    ok = [d for d in out if d["tv_eq"] is not None]
    gid_ok = np.array([d["seed"] * 10 + d["game"] for d in ok])
    rep = dict(tag=tag, n_turns_paired=n_pairs, n_turns_unpaired_or_multi=n_multi,
               mass_out_mean=float(np.mean([d["mass_out"] for d in out])),
               frac_turns_mass_out_gt_half=float(np.mean([d["mass_out"] > 0.5 for d in out])),
               tv_eq=mean_se_games([d["tv_eq"] for d in ok], gid_ok), tv_qhat=mean_se_games([d["tv_qhat"] for d in ok], gid_ok),
               tv_eq_median=float(np.median([d["tv_eq"] for d in ok])), tv_eq_vs_uniform=float(np.mean([d["tv_eq_unif"] for d in ok])),
               val_sum=mean_se_games([(d["val_A"] + d["val_B"]) / DELTA for d in out], gid),
               val_A=float(np.mean([d["val_A"] for d in out]) / DELTA), val_B=float(np.mean([d["val_B"] for d in out]) / DELTA),
               val_abs_sum_median=float(np.median([abs(d["val_A"] + d["val_B"]) for d in out])),
               mismatch_eq=mean_se_games([(d["cost_true"] - d["cost_eq"]) / DELTA for d in ok], gid_ok),
               mismatch_qhat=mean_se_games([(d["cost_true"] - d["cost_qhat"]) / DELTA for d in ok], gid_ok),
               sampling_term=mean_se_games([(d["g"] - d["cost_true"]) / DELTA for d in ok], gid_ok),
               eps_C_eq_on_paired=mean_se_games([(d["g"] - d["cost_eq"]) / DELTA for d in ok], gid_ok),
               oracle_gap=mean_se_games([d["oracle_gap"] / DELTA for d in ok], gid_ok),
               oracle_gap_median=float(np.median([d["oracle_gap"] for d in ok]) / DELTA),
               oracle_gap_frac_neg=float(np.mean([d["oracle_gap"] < -1e-9 for d in ok])),
               l1_qhat_mean=float(np.mean([d["l1_qhat"] for d in ok])),
               lipschitz_bound_mean=float(np.mean([d["l1_bound"] for d in ok]) / DELTA),
               bound_holds_frac=float(np.mean([d["oracle_gap"] <= d["l1_bound"] + 1e-6 for d in ok])))
    print(json.dumps(rep, indent=1))
    json.dump(rep, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "out", f"{tag}_oppside.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
