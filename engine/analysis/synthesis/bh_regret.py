"""BH: pathwise log-loss regret of the style posterior q_hat against the best fixed style, per stream, vs ln 5 (B1(ii)).

Log loss from feedback.bayes: p_qhat, p_style[s] (values assigned to the realised column BEFORE the update, 6 dp).
Regret_T = sum_{t<=T} -ln p_qhat_t - min_s sum_{t<=T} -ln p_style_t[s]. Anytime max over T and final value per stream.
Consistency: log update counts vs hedge_state/BH-stream<k>-A.bayes.json meta.updates, and the per-style cumulative
log-likelihood differences vs the state file's logp (normalised to max 0).
Also: identity check p_qhat == sum_s P_pre(s) p_style[s]; mean log loss of q_hat / best style / uniform / old predictor
(the 24-pair pilot Table 5 of the report, now at the full run); time-average posterior vs last posterior.
"""
import json
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import RES, STYLES, records, streams  # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out_bh_regret.json")
LN5 = math.log(5)
rows = []
curves = {}
for k, order in streams("BH"):
    lq, ls, lold, lunif = [], [], [], []
    ident_err, forced, n_games = 0.0, 0, 0
    post_pre = []
    for seed, g, path in order:
        n_games += 1
        for rec in records(path):
            fb = rec.get("feedback")
            if not fb or fb.get("skipped") or "bayes" not in fb:
                continue
            b = fb["bayes"]
            pq = max(b["p_qhat"], 1e-9)
            ps = [max(b["p_style"][s], 1e-9) for s in STYLES]
            lq.append(-math.log(pq))
            ls.append([-math.log(p) for p in ps])
            lold.append(-math.log(max(b["p_old"], 1e-9)))
            lunif.append(math.log(b["n_cols"]))
            forced += int(bool(b.get("forced")))
            # pre-update posterior = decision record's posterior for this turn; the feedback's posterior is post-update.
            # Reconstruct pre-update from post-update: P_pre(s) ∝ P_post(s) / p_style(s).
            post = np.array([b["posterior"][s] for s in STYLES])
            pre = post / np.array(ps)
            if pre.sum() > 0:
                pre = pre / pre.sum()
                ident_err = max(ident_err, abs(float(pre @ np.array(ps)) - b["p_qhat"]))
                post_pre.append(pre)
    lq = np.array(lq); ls = np.array(ls); lold = np.array(lold); lunif = np.array(lunif)
    T = len(lq)
    cum_q = np.cumsum(lq)
    cum_s = np.cumsum(ls, 0)
    reg = cum_q - cum_s.min(1)
    reg_vs_each = cum_q[-1] - cum_s[-1]
    st = json.load(open(os.path.join(RES, "BH", "hedge_state", f"BH-stream{k}-A.bayes.json")))
    logp = np.array(st["logp"])  # cum log-lik per style, max = 0  =>  L_s - L_best = -logp
    L_from_logs = cum_s[-1] - cum_s[-1].min()
    state_match = float(np.abs(L_from_logs - (-logp)).max())
    best = STYLES[int(cum_s[-1].argmin())]
    P = np.array(post_pre)
    rows.append(dict(
        stream=k, games_logged=n_games, games_state=st["meta"]["games"], updates_logs=T, updates_state=st["meta"]["updates"],
        count_match=(T == st["meta"]["updates"]), forced_turns=forced,
        regret_final=float(reg[-1]), regret_anytime_max=float(reg.max()), regret_T_at_max=int(reg.argmax() + 1),
        regret_le_ln5=bool(reg.max() <= LN5 + 1e-6),
        best_style=best, regret_vs_each_style={s: float(v) for s, v in zip(STYLES, reg_vs_each)},
        state_logp_gap={s: float(-v) for s, v in zip(STYLES, logp)}, state_vs_logs_max_abs=state_match,
        identity_max_err=ident_err,
        mean_logloss=dict(qhat=float(lq.mean()), best_style=float(cum_s[-1].min() / T), uniform_pred=float(lunif.mean()),
                          old=float(lold.mean()), **{s: float(ls[:, i].mean()) for i, s in enumerate(STYLES)}),
        final_posterior=st["posterior"], time_avg_posterior={s: float(v) for s, v in zip(STYLES, P.mean(0))},
    ))
    curves[k] = reg.tolist()

clean = [r for r in rows if r["stream"] not in (0, 6)]
print(f"{'k':>2} {'games':>5} {'upd_log':>7} {'upd_state':>9} {'match':>5} {'R_final':>8} {'R_max':>7} {'T@max':>6} {'<=ln5':>5} {'best':>8} {'state-log':>9} {'ident':>8}")
for r in rows:
    print(f"{r['stream']:>2} {r['games_logged']:>5} {r['updates_logs']:>7} {r['updates_state']:>9} {str(r['count_match']):>5} "
          f"{r['regret_final']:>8.3f} {r['regret_anytime_max']:>7.3f} {r['regret_T_at_max']:>6} {str(r['regret_le_ln5']):>5} "
          f"{r['best_style']:>8} {r['state_vs_logs_max_abs']:>9.3f} {r['identity_max_err']:>8.1e}")
print(f"\nln5 = {LN5:.4f}; clean streams (not 0, 6): n={len(clean)}, "
      f"final regret median {np.median([r['regret_final'] for r in clean]):.3f} "
      f"[{min(r['regret_final'] for r in clean):.3f}, {max(r['regret_final'] for r in clean):.3f}], "
      f"anytime max over clean streams {max(r['regret_anytime_max'] for r in clean):.4f}; "
      f"all clean <= ln5: {all(r['regret_le_ln5'] for r in clean)}")
print("per-stream regret vs each style (final), clean streams:")
for r in clean:
    print(" ", r["stream"], {s: round(v, 2) for s, v in r["regret_vs_each_style"].items()}, "final post", r["final_posterior"])
print("\nmean log loss per feedback turn (nats), all 10 streams pooled by stream mean:")
keys = ["qhat", "best_style", "old", "uniform_pred"] + STYLES
for key in keys:
    v = [r["mean_logloss"][key] for r in rows]
    print(f"  {key:>10}: mean over streams {np.mean(v):.3f}  (min {min(v):.3f}, max {max(v):.3f})")
print("\ntime-average posterior vs final posterior per stream:")
for r in rows:
    print(" ", r["stream"], {s: round(v, 3) for s, v in r["time_avg_posterior"].items()}, "->", r["final_posterior"])
json.dump(dict(rows=rows, curves=curves, ln5=LN5), open(OUT, "w"), indent=1)
print("wrote", OUT)
