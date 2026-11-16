"""Hedge learner dynamics on the real-Showdown logs (E2, E2fs, E5, BH; player A), per carried-weight stream.

Per stream (feedback turns only, non-skipped; reward r in [0,1] per expert, order top/sampled/safest/safex; weights =
the mixture played at that turn, from the decision record = feedback.weights):
  - T, cumulative reward per expert, best expert in hindsight, final weights, time-average weights w_bar.
  - External regret R_T = max_e sum r_e - sum w.r  (Lecture 6 comparator: fixed expert), per turn R_T/T, R2a bound.
  - Dynamic regret D_T = sum_t max_e r_{t,e} - sum w.r   (per-turn best response among experts: dynamic regret, a stronger comparator than the fixed-expert (CCE)
    one; no equilibrium meaning; always >= R_T).
  - Last-iterate vs time-average: sd of the top-expert weight over the last half of the stream vs the drift of w_bar.
  - Spread of expert cumulative rewards: (max_e - min_e)/T in reward units, i.e. how much the experts differ at all.
Also E5 B side (agentform): fraction of decisions where agent form and PIMC argmax agree, TV between mixes.
"""
import json
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import EXPERTS, RES, records, streams  # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out_hedge_dyn.json")
K = 4


def r2a_bound(T):
    return 1.5 * math.sqrt(T * math.log(K) / 2) + math.sqrt(math.log(K) / 8)


def stream_data(order):
    R, W = [], []
    for seed, g, path in order:
        pend = None
        for rec in records(path):
            fb = rec.get("feedback")
            if fb and pend is not None and fb.get("for_turn") == pend[0] and not fb.get("skipped") and "r" in fb:
                R.append(np.array(fb["r"], float)); W.append(pend[1])
            if fb is not None or rec.get("final"):
                pend = None
            if isinstance(rec.get("weights"), dict):
                pend = (rec["turn"], np.array([rec["weights"][e] for e in EXPERTS], float))
    return np.array(R), np.array(W)


results = {}
for tag in ("E2", "E2fs", "E5", "BH"):
    rows = []
    for k, order in streams(tag):
        R, W = stream_data(order)
        T = len(R)
        cum = R.cumsum(0)
        earned = (W * R).sum(1).cumsum()
        ext = cum.max(1) - earned
        dyn = R.max(1).cumsum() - earned
        wbar = W.cumsum(0) / np.arange(1, T + 1)[:, None]
        half = T // 2
        st = json.load(open(os.path.join(RES, tag, "hedge_state", f"{tag}-stream{k}-A.json")))
        rows.append(dict(
            stream=k, T=T, T_state=st["t"], count_match=(T == st["t"]),
            cum_reward={e: float(v) for e, v in zip(EXPERTS, cum[-1])},
            cum_state={e: float(v) for e, v in zip(EXPERTS, st["cum"])},
            best_expert=EXPERTS[int(cum[-1].argmax())],
            spread_per_turn=float((cum[-1].max() - cum[-1].min()) / T),
            ext_regret=float(ext[-1]), ext_per_turn=float(ext[-1] / T), r2a_per_turn=r2a_bound(T) / T,
            ext_anytime_max_over_bound=float(max(ext[t] / r2a_bound(t + 1) for t in range(T))),
            dyn_regret=float(dyn[-1]), dyn_per_turn=float(dyn[-1] / T),
            final_w={e: float(v) for e, v in zip(EXPERTS, W[-1])},
            wbar={e: float(v) for e, v in zip(EXPERTS, wbar[-1])},
            top_w_sd_last_half=float(W[half:, 0].std()), top_w_range_last_half=[float(W[half:, 0].min()), float(W[half:, 0].max())],
            top_wbar_drift_last_half=float(abs(wbar[-1, 0] - wbar[half, 0])),
            n_turns_top_leader=int((W.argmax(1) == 0).sum()),
            leader_changes=int((np.diff(W.argmax(1)) != 0).sum()),
        ))
    results[tag] = rows
    print(f"\n=== {tag}: {len(rows)} streams ===")
    print(f"{'k':>2} {'T':>5} {'match':>5} {'best':>7} {'spread/T':>8} {'ExtR':>7} {'ExtR/T':>7} {'R2a/T':>7} {'DynR/T':>7} "
          f"{'w_T(top)':>8} {'wbar(top)':>9} {'sd_top_2ndhalf':>14} {'leaderchg':>9}")
    for r in rows:
        print(f"{r['stream']:>2} {r['T']:>5} {str(r['count_match']):>5} {r['best_expert']:>7} {r['spread_per_turn']:>8.4f} "
              f"{r['ext_regret']:>7.2f} {r['ext_per_turn']:>7.4f} {r['r2a_per_turn']:>7.4f} {r['dyn_per_turn']:>7.4f} "
              f"{r['final_w']['top']:>8.3f} {r['wbar']['top']:>9.3f} {r['top_w_sd_last_half']:>14.4f} {r['leader_changes']:>9}")
    med = lambda key: float(np.median([r[key] for r in rows]))
    print(f"medians: ExtR/T {med('ext_per_turn'):.4f}, R2a/T {med('r2a_per_turn'):.4f}, DynR/T {med('dyn_per_turn'):.4f}, "
          f"spread/T {med('spread_per_turn'):.4f}; best expert counts "
          f"{ {e: sum(r['best_expert'] == e for r in rows) for e in EXPERTS} }; "
          f"max over streams of anytime ExtR/bound {max(r['ext_anytime_max_over_bound'] for r in rows):.3f}")
    mean_cum = {e: float(np.mean([r['cum_reward'][e] / r['T'] for r in rows])) for e in EXPERTS}
    print("mean reward per turn by expert (avg over streams):", {e: round(v, 4) for e, v in mean_cum.items()})
    print("mean reward per turn earned by Hedge:", round(float(np.mean([(r['cum_reward'][r['best_expert']] - r['ext_regret']) / r['T'] for r in rows])), 4))

# E5 B side: agent form vs its PIMC alternative
agree, tvs, n = 0, [], 0
for k, order in streams("E5", role="B"):
    for seed, g, path in order:
        for rec in records(path):
            if "argmax_same" in rec:
                n += 1; agree += int(bool(rec["argmax_same"])); tvs.append(float(rec.get("tv", float("nan"))))
tvs = np.array(tvs)
print(f"\nE5 B (agentform): {n} decisions with PIMC alternative; argmax agree {agree / n:.3f}; "
      f"TV(agentform, pimc) median {np.nanmedian(tvs):.3f}, mean {np.nanmean(tvs):.3f}")
results["E5_B"] = dict(n=n, argmax_agree=agree / n, tv_median=float(np.nanmedian(tvs)), tv_mean=float(np.nanmean(tvs)))
json.dump(results, open(OUT, "w"), indent=1)
print("wrote", OUT)
