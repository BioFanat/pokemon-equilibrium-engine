"""B6 scoring of P1-P5 from out_B6_rq.json + A-audit scores. Output out_B6_scores.json"""
import json, math
import numpy as np
from oi_common import *
d = json.load(open(os.path.join(OUT, "out_B6_rq.json"))); au = json.load(open(os.path.join(OUT, "out_A_audit.json")))
LN5 = math.log(5)
sc = {}
def S(t): return au[t]["summary"]["mean"]
out = {}
# P1
p1 = {}
for tag in d:
    rows = d[tag]["streams"]
    allmax = max(r["anytime_max_regret_vs_any_style"] for r in rows)
    ends = [r["regret_vs_best"] for r in rows]
    exact = [abs(r["regret_vs_best"] - r["ln5_plus_lnPT_best"]) for r in rows if r["final_post"][r["best_cumloss_style"]] >= 0.999]
    p1[tag] = dict(anytime_max=allmax, ln5=LN5, excess_over_ln5=allmax - LN5, end_regret_range=[min(ends), max(ends)], n_clean=sum(r["clean"] for r in rows),
                   max_identity_err_all_turns=max(r["max_abs_identity_err"] for r in rows),
                   end_regret_eq_ln5_when_PT_ge_0999=dict(n=len(exact), max_abs_dev=max(exact) if exact else None))
out["P1"] = dict(per_tag=p1, verdict="PASS" if all(v["excess_over_ln5"] < 5e-4 for v in p1.values()) else "see numbers")
# P2
r = d["RQunif"]["streams"]
out["P2"] = dict(P_T_uniform_at_50_min=min(x["P_T_star_at_50"] for x in r), n_streams_P50_ge_099=sum(x["P_T_star_at_50"] >= 0.99 for x in r),
                 P_T_end=[x["P_T_star"] for x in r], n_final_MAP_uniform=sum(x["final_map"] == "uniform" for x in r),
                 map_wrong_turns=[x["map_wrong_turns"] for x in r], map_wrong_after50=[x["map_wrong_turns_after50"] for x in r],
                 uniform_mean_logloss=float(np.mean([x["mean_loss_style"]["uniform"] for x in r])), mean_ln_ncols=float(np.mean([x["mean_ln_ncols"] for x in r])),
                 score=S("RQunif"), score_ci=au["RQunif"]["pair_level_normal"][2:4],
                 sum_1_minus_Pprev=[x["sum_1_minus_Pprev"] for x in r])
out["P2"]["verdict_parts"] = dict(by_turn50_P_ge_099_all=out["P2"]["n_streams_P50_ge_099"] == 10, end_equals_1_000=all(x >= 0.9995 for x in out["P2"]["P_T_end"]),
                                  final_MAP_10_of_10=out["P2"]["n_final_MAP_uniform"] == 10, wrong_turns_le_10=sum(w <= 10 for w in out["P2"]["map_wrong_turns"]),
                                  score_gt_0_8=S("RQunif") > 0.8)
# P3
r = d["RQsafe"]["streams"]
gap = [x["mean_loss_style"]["br_unif"] - x["mean_loss_style"]["safe"] for x in r]
gap_br_us = [x["mean_loss_style"]["br_us"] - x["mean_loss_style"]["safe"] for x in r]
out["P3"] = dict(decision_rule_mean_loss_gap_br_unif_minus_safe_nats_per_turn=dict(per_stream=gap, mean=float(np.mean(gap))),
                 branch="low-mismatch (>=0.1 nats/turn)" if np.mean(gap) >= 0.1 else "high-mismatch (<0.1)",
                 gap_br_us_minus_safe=float(np.mean(gap_br_us)),
                 final_MAP_safe=sum(x["final_map"] == "safe" for x in r), P_T_end=[x["P_T_star"] for x in r], n_P_T_ge_0999=sum(x["P_T_star"] >= 0.9995 for x in r),
                 t_sustained_P90=[x["t_sustained_P90"] for x in r], P_at_100=[x["P_T_star_at_100"] for x in r],
                 map_wrong_turns=[x["map_wrong_turns"] for x in r], n_wrong_turns_in_10_50=sum(10 <= w <= 50 for w in [x["map_wrong_turns"] for x in r]),
                 final_gap_to_runner_up_nats=[x["gap_to_best_other"] for x in r], n_gap_ge_60=sum(x["gap_to_best_other"] >= 60 for x in r),
                 score=S("RQsafe"), score_ci=au["RQsafe"]["pair_level_normal"][2:4], predicted_score_range=[0.6, 0.75])
# P4
r = d["RQeq"]["streams"]
out["P4"] = dict(final_MAP=[x["final_map"] for x in r], n_MAP_in_safe_br_us_br_unif=sum(x["final_map"] in ("safe", "br_us", "br_unif") for x in r),
                 P_T_eq=[x["P_T_star"] for x in r], n_P_T_eq_lt_0_1=sum(x["P_T_star"] < 0.1 for x in r), n_P_T_eq_ge_0_9=sum(x["P_T_star"] >= 0.9 for x in r),
                 mean_logloss_eq=float(np.mean([x["mean_loss_style"]["eq"] for x in r])), mean_logloss_safe=float(np.mean([x["mean_loss_style"]["safe"] for x in r])),
                 n_streams_eq_loss_gt_safe=sum(x["mean_loss_style"]["eq"] > x["mean_loss_style"]["safe"] for x in r),
                 mean_loss_all_styles={s: float(np.mean([x["mean_loss_style"][s] for x in r])) for s in STYLES} if False else {s: float(np.mean([x["mean_loss_style"][s] for x in r])) for s in ["eq", "br_unif", "br_us", "safe", "uniform"]},
                 P_eq_at_25=float(np.mean([x["P_T_star_at_25"] for x in r])), P_eq_at_100=float(np.mean([x["P_T_star_at_100"] for x in r])),
                 score=S("RQeq"), score_ci=au["RQeq"]["pair_level_normal"][2:4], predicted_score_range=[0.5, 0.6])
# P5
ident = {t: dict(mean_sum_1_minus_Pprev=float(np.mean([x["sum_1_minus_Pprev"] for x in d[t]["streams"]])), mean_final_P=float(np.mean([x["P_T_star"] for x in d[t]["streams"]])),
                 median_t_sustained_P90=None, n_never_P90=sum(x["t_sustained_P90"] is None for x in d[t]["streams"]),
                 median_t_P90=float(np.median([x["t_sustained_P90"] for x in d[t]["streams"] if x["t_sustained_P90"] is not None])) if any(x["t_sustained_P90"] is not None for x in d[t]["streams"]) else None)
         for t in d}
out["P5"] = dict(identification=ident, scores={t: S(t) for t in d}, order_scores_unif_gt_safe_gt_eq=S("RQunif") > S("RQsafe") > S("RQeq"),
                 order_final_P=ident["RQunif"]["mean_final_P"] >= ident["RQsafe"]["mean_final_P"] >= ident["RQeq"]["mean_final_P"],
                 order_sum_1_minus_P=ident["RQunif"]["mean_sum_1_minus_Pprev"] < ident["RQsafe"]["mean_sum_1_minus_Pprev"] < ident["RQeq"]["mean_sum_1_minus_Pprev"])
dump("out_B6_scores.json", out)
print(json.dumps(out, indent=1, default=str)[:5500])
