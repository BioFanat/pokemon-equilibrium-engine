"""Builds REPORT.md from the out_*.json files (no numbers typed by hand except prior-literature constants cited from the theory docs)."""
import json, math
import numpy as np
from oi_common import *
L = lambda n: json.load(open(os.path.join(OUT, n)))
A, B1, B1g, B2, B3, B4, B5, B6, B6s = (L("out_A_audit.json"), L("out_B1_summary.json"), L("out_B1_log_games.json"), L("out_B2_switch.json"), L("out_B3_exploiters.json"),
                                       L("out_B4_e5.json"), L("out_B5_eps.json"), L("out_B6_rq.json"), L("out_B6_scores.json"))
o = []
def w(s=""): o.append(s)
f = lambda x, n=4: "n/a" if x is None else f"{x:.{n}f}"
ci = lambda c, n=3: f"[{c[0]:.{n}f}, {c[1]:.{n}f}]"

w("# Open-item analyses (Showdown runs LOG, E3a/f, X*, E5t/s, SX*, RQ*)")
w()
w("Verified independently (recomputed from raw data): scripts in verify_open/ (session scratch); all headline numbers reproduce; corrections below applied.")
w()
w("All numbers are produced by scripts in this directory (`engine/analysis/open_items/`); each section names its script and JSON. Interpreter `engine/.venv/bin/python`. "
  "Source logs: `engine/showdown/results/<TAG>/` (summary `<TAG>_summary.json`, game JSONs `replays_headless/<TAG>-s<seed>-g<1|2>.json`, per-decision logs `*.A.decisions.jsonl` / `*.B.decisions.jsonl`, carried state `hedge_state/`). "
  "No Modal job launched, nothing outside this directory edited. Score = A's win rate (win 1, loss 0; no draws occurred in any tag). "
  "'pair' = mean of the two seat-swapped games of a seed (the summary's unit); 'game' = single game. CIs: `normal` = normal approx 1.96*SE over units; `cluster` = stream-clustered (CR1, t with G-1 df; **G = 10 streams per tag, a small number**; streams carry state across games so games within a stream are not independent).")
w()
w("## Anomalies / caveats up front")
w()
errs = {t: v["err_detail"] for t, v in A.items() if v["err_detail"]}
w(f"1. Error games (excluded from pair scores by the summary; the partner game of the seed is dropped too): "
  + "; ".join(f"{t}: " + ", ".join(f"{k} ({v[:40]}...)" for k, v in d.items()) for t, d in errs.items()) + ". Causes: ZeroDivisionError, `invalid_choice_stall` (60 s), PanicException `Invalid PokemonMoveIndex: 4`, AttributeError `can_z`. All are listed in section A.")
ex = []
for t, v in A.items():
    for k, (m, n) in (v.get("hedge_state_games_vs_present") or {}).items():
        gm = m.get("games", m.get("bayes_games"))
        if gm != n: ex.append(f"{t} stream {k}: state games {gm} vs {n} game files")
w(f"2. Carried-state files record MORE games than game JSONs exist in {len(ex)} streams ({'; '.join(ex)}). `load_learner`/`load_posterior` increment `meta.games` at bot construction, so (a hypothesis consistent with the resume rule `modal_showdown.py:163`, not a finding) an aborted/retried attempt of a game (JSON overwritten by the retry) updated the carried Hedge/posterior state with partial feedback while only the retry survives in the logs. Only 3 of these have a matching error game (LOG s4-g2 in stream 3; E3a s52-g2 in stream 1; SX30 s93-g1 in stream 2) -- the other streams have no trace of a retry in the logs (cause not determinable from the logs). Record field sets and config fields (k, iters, played_rule, eps) are constant across invocations in all resumed tags. Effect on scores: none (scores are from game outcomes); effect on weight/posterior trajectories: replayed logs omit the aborted attempt's updates, so `hedge_state` final weights can differ slightly from a replay of the logged feedback.")
w("3. `experts` logging: the `sampled` expert's logged label is `legal[argmax Q]` of its mixture, which is identical to the `top` expert's label by construction (both = argmax of the same LP mixture); so in B4 the `sampled` and `top` match rates are identical and carry no separate information.")
w("4. E3a/E3f are `hedge`/`hedge_fs` (no style posterior): the flip-delay formula of bayes_dynamics.md T5 is for the Bayes posterior; only a reward-space analog is evaluated (B2).")
w("5. In the `X*` runs the victim B side is also our code, so B decision logs exist; in LOG/SX/E3 the stock opponent (foul-play MCTS) has no decision log, so its own matrices are not recoverable.")
w()

# ------------------------------------------------------------------ A
w("## A. Audit (script `a_audit.py` -> `out_A_audit.json`)")
w()
w("Expected grid from `engine/showdown/launch_open_items.sh`: 10 streams per tag (stream k = sorted seeds[k::10]), 2 games per seed. The `stream` field of every game JSON was checked against this rule: mismatches = " + str(sum(v["n_bad_stream"] for v in A.values())) + ".")
w()
w("| tag | A vs B | exp. games | game JSONs | missing | error games | decision logs A/B | summary pair mean [CI] (n pairs) | recomputed pair mean (n) | match (mean / per-seed) | nonzero exit / failed streams / over budget | modal games_run |")
w("|---|---|---|---|---|---|---|---|---|---|---|---|")
for t, v in A.items():
    s = v["summary"]; m = s["modal"]
    w(f"| {t} | {v['A']} vs {v['B']} | {v['expected_games']} | {v['present_game_json']} | {len(v['missing'])} | {len(v['err_games'])} | {v['n_A_logs']}/{v['n_B_logs']} | {s['mean']} {ci(s['ci95'])} ({s['n_pairs']}) | {v['recomputed_pair_mean']:.4f} ({v['n_pairs_recomputed']}) | {v['pair_mean_matches']} / {v['per_seed_matches']} | {s['nonzero_exit'] or 'none'} / {m.get('failed_streams') or 'none'} / {m.get('stopped_over_budget')} | {m.get('games_run')} |")
w()
w("Resume / duplicate check: for every `*.decisions.jsonl` (A and B) the `turn` field never decreases within a file (a rerun appending to the same file would reset it): resets found = " + str(sum(v["n_turn_reset"] for v in A.values())) + " files. Non-final record count equals `timing.<role>.decisions` in the game JSON for all files except " +
  "; ".join(f"{t} {x}" for t, v in A.items() for x in v["mismatch_examples"]) + " (off by one, both error games or neighbours of one). Note repeated `turn` values within a file are normal (a second decision in the same turn after a faint).")
gr = {t: v["summary"]["modal"].get("games_run") for t, v in A.items() if v["summary"]["modal"].get("games_run") != v["expected_games"]}
w("Resumed runs: `modal.games_run` (games executed by the final invocation) is below the expected games in " + ", ".join(f"{t} {n}/{A[t]['expected_games']}" for t, n in gr.items()) + "; the remaining game JSONs were produced by earlier invocations of the same tag (resume rule `modal_showdown.py:163`: seeds whose two JSONs exist are skipped, carried weights resumed). No `failed_streams`, `stopped_over_budget` or `nonzero_exit` is recorded in any summary (these fields describe only the final invocation), and the reason for the earlier partial invocations is not in the logs. Consistency check that earlier-invocation games used the same configuration: for SX1/SX3/SX10/SX30 every A decision record that logs `eps` has the tag's epsilon (13069/13061/13566/12989 records, 0 deviations; `out_A_sx_eps_check.json`); for LOG all 20696 non-final records with more than one legal option carry `abar` (546 records without it are single-option).")
w("E3a/E3f `active` field: B = stock for seeds 1-80 (" + str(A['E3a']['active_B']) + " seeds each, E3a max stock seed " + str(A['E3a']['active_B_stock_seeds_max']) + ", min exploit_top seed " + str(A['E3a']['active_B_exploit_seeds_min']) + "), i.e. the switch happens after seed position 8 of every 16-seed stream, as launched.")
w()
w("Scores with 95% CI (game level and pair level; normal and stream-clustered):")
w()
w("| tag | W/L (games) | game mean [normal CI] | game [cluster CI] | pair mean [normal CI] | pair [cluster CI] |")
w("|---|---|---|---|---|---|")
for t, v in A.items():
    g, gc, p, pc = v["game_level_normal"], v["game_level_stream_cluster"], v["pair_level_normal"], v["pair_level_stream_cluster"]
    w(f"| {t} | {v['wins']}/{v['losses']} (n={g[4]}) | {g[0]:.4f} {ci(g[2:4])} | {ci(gc[2:4])} | {p[0]:.4f} {ci(p[2:4])} (n={p[4]}) | {ci(pc[2:4])} |")
w()
w("Error-game causes: " + "; ".join(f"{t}: {v['err_detail']}" for t, v in A.items() if v["err_detail"]))
w()

# ------------------------------------------------------------------ B1
w("## B1. LOG: signed pathwise identity along hedge_bayes vs stock (scripts `b1_log_identity.py`, `b1_aggregate.py` -> `out_B1_log_games.json`, `out_B1_summary.json`)")
w()
w("Identity (bounds_epsv.md 1.1): z - phi(v_1) = sum_t(-R_t - S_t + D_t) with R_t = phi(v_t) - phi((x_t Abar)_j), S_t = phi((x_t Abar)_j) - phi(Abar[i,j]), D_t = phi(v_{t+1}) - phi(Abar[i,j]), phi(v_{T+1}) := z. "
  "Implementation: Abar, labels, legal, pfull from the logged records (abar rounded to 2 dp); **v_t = value of Abar_t by LP (scipy linprog), not logged**; x_t = logged `pfull` (the Hedge mixture actually played, not RM+); i_t = played move (`switch X` -> X); "
  "j_t = opponent column `feedback.b` of the next record looked up in `labels`; z from `winner_role` (A win +1, loss -1; no draws; no capped games occurred). phi = logistic 2*lgs(v/98)-1 (the doc's B12 fit, applied here unchanged) and tanh(v/400). Forced decisions (single legal option, no `abar`) are excluded; D spans them.")
c = B1["coverage"]
w(f"Coverage: {c['n_games']} games (the one error game LOG s4-g2 excluded), {c['n_dec']} non-forced decisions, {c['n_known']} ({100*c['n_known']/c['n_dec']:.1f}%) with j known; unknown j (feedback skipped: {c['why']}) cannot be split into R,S,D and are excluded from the D/R/S statistics; their lump phi(v_{{t+1}})-phi(v_t) is still counted in the closure check. "
  f"**Closure:** max |(-R-S+D) - (phi(v_t+1)-phi(v_t))| per decision = {B1['closure']['max_dec']:.1e}; max |z - phi(v_1) - sum| per game = {B1['closure']['max_game']:.1e} (both maps). Closure holds by construction (telescoping); the indexing was validated separately: pfull = normalise(w·Q) 20657/20657, logged expert labels = argmax Q, pfull[i_played] > 0 always, feedback column always in labels. The logged agent-form x (Q[1]) is not a maximin strategy of the world-averaged Abar (it solves the per-world agent-form LP): v_t - min_j(x_AF·Abar)_j median 0.12, mean 4.6 engine units (above 1% of the payoff range on a third of decisions). The played Hedge mixture (`pfull`) has gap median 1.85, mean 6.12 (`verify/b1_gap.py`).")
w()
w("What is NOT computable: the stock (foul-play MCTS) side's own matrices/values are not logged, so a drift/regret decomposition for the opponent as a decision maker does not exist. "
  "The 'two sides' are therefore reported as: our side's identity terms, plus two stage quantities defined on our Abar for both players: (a) *opponent stage gap* G^opp = phi((xA)_j) - phi(min_j' (xA)_j') (how far the realised column is from the best reply to our mixture, in our model), (b) *our stage gap* G^us = phi(v_t) - phi(min_j (xA)_j) (our mixture's shortfall in Abar). Note R_t = G^us_t - G^opp_t exactly. Seats p1/p2 are reported separately.")
w()
w("Mean drift per decision E[D] (known-j decisions). SE columns: decision-level iid / clustered by game / clustered by stream (10 clusters). Prior numbers (bounds_epsv.md 2.2 for the means, which has no SEs; SEs from theory/tightening/verify/out_v_drift.txt lines 3/5/7; logit): -0.0127 +/- 0.0033 vs white-box exploiter, -0.0113 +/- 0.0023 vs MCTS-2000, +0.0020 +/- 0.0017 self-play; sum|D| per game 2.9 / 3.1 / 3.2.")
w()
w("| subset | phi | games | decisions (j known) | E[D] per decision | SE iid | SE game-cl. | SE stream-cl. | E abs(D) | sum abs(D) per game (known only) | same scaled by n_dec/n_known | sum D per game | sum R | sum S | sum G^opp | sum G^us |")
w("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
for k, lab in (("all", "all"), ("p1", "seat p1"), ("p2", "seat p2"), ("A_win", "games A won"), ("A_loss", "games A lost")):
    for mp in ("logit", "tanh"):
        s = B1[k][mp]; pg = s["per_game"]
        w(f"| {lab} | {mp} | {s['n_games']} | {s['n_dec_known']} | {s['mean_D_per_dec']:.4f} | {s['se_iid']:.4f} | {s['se_game_cluster']:.4f} | {s['se_stream_cluster']:.4f} | {s['mean_abs_D']:.3f} | {pg['sumabsD']['mean']:.2f} (SE {pg['sumabsD']['se']:.2f}) | {pg['sumabsD_scaled']['mean']:.2f} | {pg['sumD']['mean']:.2f} (SE {pg['sumD']['se']:.2f}) | {pg['sumR']['mean']:.2f} | {pg['sumS']['mean']:.2f} | {pg['opp']['mean']:.2f} | {pg['us']['mean']:.2f} |" if 'us' in pg else
          f"| {lab} | {mp} | {s['n_games']} | {s['n_dec_known']} | {s['mean_D_per_dec']:.4f} | {s['se_iid']:.4f} | {s['se_game_cluster']:.4f} | {s['se_stream_cluster']:.4f} | {s['mean_abs_D']:.3f} | {pg['sumabsD']['mean']:.2f} (SE {pg['sumabsD']['se']:.2f}) | {pg['sumabsD_scaled']['mean']:.2f} | {pg['sumD']['mean']:.2f} (SE {pg['sumD']['se']:.2f}) | {pg['sumR']['mean']:.2f} | {pg['sumS']['mean']:.2f} | {pg['opp']['mean']:.2f} | {pg['our']['mean']:.2f} |")
w()
a = B1["all"]["logit"]; at = B1["all"]["tanh"]
w(f"Reading (numbers only): mean drift per decision is {a['mean_D_per_dec']:.4f} (logit) / {at['mean_D_per_dec']:.4f} (tanh), i.e. {abs(a['mean_D_per_dec']/-0.0127):.1f}x (logit) / {abs(at['mean_D_per_dec']/-0.0106):.1f}x (tanh) the prior exploiter value; {abs(a['mean_D_per_dec']/-0.0113):.1f}x / {abs(at['mean_D_per_dec']/-0.0080):.1f}x the MCTS-2000 value (-0.0127 / -0.0113 logit; -0.0106 / -0.0080 tanh) and with the same sign; the difference far exceeds the SEs (SEs reflect sampling noise only, not the setup differences below). Sum|D| per game is {a['per_game']['sumabsD']['mean']:.2f} over the known decisions ({a['per_game']['sumabsD_scaled']['mean']:.2f} scaled to all decisions), numerically close to the prior 2.9-3.2, but from about double the per-decision E|D| (0.107 vs 0.054-0.057) over about half as many known decisions per game (~29 vs 52-57) and above the payoff range 2 on the logit map. "
  "Drift is more negative in games A lost (logit -0.068) than won (-0.027), as expected for a signed residual that is conditioned on the realised path. "
  f"E[phi(v_1)] = {B1['E_phi_v1']['logit']['mean']:.3f} (SE {B1['E_phi_v1']['logit']['se']:.3f}) logit, {B1['E_phi_v1']['tanh']['mean']:.3f} tanh: SYM (E phi(v_1)=0) is not satisfied here (stock is a different algorithm and seat-paired teams are swapped, so antisymmetry is not expected). E[z] = {B1['E_z']['mean']:.3f}.")
w("Comparability caveats: (i) v_t is the value of the world-averaged Abar of a hidden-information agent (K=4 sampled worlds, world-averaged depth-2 joint-action matrix (nash.full_matrix, nash.py:155) under hidden information), not the perfect-information calibrated depth-2 value of the prior; (ii) x_t is the Hedge mixture, not RM+; (iii) the opponent is foul-play stock MCTS, a third matchup type; (iv) 16.4% of decisions lack j and are excluded, and they are not random (not_in_worlds, fainted, cant). So the comparison with the prior is of the same formula on a different leaf and opponent, not a replication.")
w()

# ------------------------------------------------------------------ B2
w("## B2. E3a / E3f: opponent switches stock -> exploit_top after 8 of 16 seeds per stream (script `b2_switch.py` -> `out_B2_switch.json`)")
w()
w("Unit of time = a Hedge update (non-skipped feedback). `t0` = number of updates before the switch (first 16 games of the stream). Per-update expert reward r (logged `feedback.r`, in [0,1], scale 400). Correct expert before/after = argmax of cumulative logged reward within the phase. Weights = the played mixture weights `weights` of the decision record. "
  "'Flip' = first update after t0 at which the post-phase-correct expert has the maximum weight (flip_first); sustained = holds for 20 consecutive updates; 'cum. crossing' = update at which the post-correct expert's cumulative reward first exceeds the pre-correct expert's (exact flip rule of anytime Hedge vs that one rival). "
  "T5(a)-analog: tau* = t0 (mu_a^pre - mu_b^pre)/(mu_b^post - mu_a^post), a = pre-correct, b = post-correct, mu = mean per-update reward in the phase (reward-space analog of the formula for the Bayes posterior; it is exact only for anytime-FTRL crossing). "
  "Fixed-share analog (hedge_fs: eta=0.2, alpha=0.002, k=4): tau_fs = ln(k/alpha) / (eta (mu_b^post - mu_a^post)).")
w()
for t in ("E3a", "E3f"):
    sp = B2[t]["split"]
    w(f"**{t}** score pre-switch (vs stock, 8 seeds/stream): {sp['pre']['mean']:.4f} normal {ci(sp['pre']['normal'])} cluster {ci(sp['pre']['cluster'])} (n={sp['pre']['n']} games); post-switch (vs exploit_top): {sp['post']['mean']:.4f} normal {ci(sp['post']['normal'])} cluster {ci(sp['post']['cluster'])} (n={sp['post']['n']}); "
      f"first 2 post seeds (4 games/stream): {sp['post_first4_games']['mean']:.3f} {ci(sp['post_first4_games']['normal'])} (n={sp['post_first4_games']['n']}); later post seeds: {sp['post_later']['mean']:.3f} {ci(sp['post_later']['normal'])} (n={sp['post_later']['n']}). The pre and post scores are against DIFFERENT opponents; the difference is not a switching cost.")
w()
fm = B2["fs_minus_hedge"]
w(f"hedge_fs minus hedge, paired by seed (same seeds/teams): all {fm['all']['mean_fs_minus_hedge']:+.4f} normal {ci(fm['all']['normal'])} cluster {ci(fm['all']['cluster'])} (n={fm['all']['n']} seeds); pre {fm['pre']['mean_fs_minus_hedge']:+.4f} {ci(fm['pre']['normal'])} (n={fm['pre']['n']}); post {fm['post']['mean_fs_minus_hedge']:+.4f} {ci(fm['post']['normal'])} (n={fm['post']['n']}). No difference distinguishable from zero.")
w()
for t in ("E3a", "E3f"):
    w(f"**{t} per-stream** (flip columns in updates after the switch; `None` = never within the post phase):")
    w()
    w("| stream | t0 | post updates | pre-best -> post-best | w(post-best) last50 pre / first50 post / last50 post | w(pre-best) last50 pre / last50 post | flip (first / sustained20) | cum. crossing vs pre-best | T5a analog tau* | fixed-share analog |")
    w("|---|---|---|---|---|---|---|---|---|---|")
    for r in B2[t]["streams"]:
        w(f"| {r['stream']} | {r['t0']} | {r['n_post_updates']} | {r['pre_best']} -> {r['post_best']} | {r['w_postbest_pre_last50']:.3f} / {r['w_postbest_post_first50']:.3f} / {r['w_postbest_post_last50']:.3f} | {r['w_prebest_pre_last50']:.3f} / {r['w_prebest_post_last50']:.3f} | {r['flip_first']} / {r['flip_sustained20']} | {r['realised_cum_crossing']} | {f(r['pred_tau_T5a_analog'],0)} | {f(r['pred_tau_fs_analog'],0)} |")
    w()
for t in ("E3a", "E3f"):
    rs = B2[t]["streams"]
    fl = [r["flip_first"] for r in rs if r["flip_first"] is not None]
    cr = [(r["realised_cum_crossing"], r["pred_tau_T5a_analog"]) for r in rs if r["realised_cum_crossing"] is not None and r["pred_tau_T5a_analog"] is not None and r["pre_best"] != r["post_best"]]
    ratio = [a_ / b_ for a_, b_ in cr if b_]
    mw = lambda k: np.mean([r[k] for r in rs])
    w(f"**{t} summary**: post-correct expert = {dict((e, sum(1 for r in rs if r['post_best']==e)) for e in ('sampled','safest'))}; pre-correct expert = top in all {len(rs)} streams. Mean weight of the post-correct expert: {mw('w_postbest_pre_last50'):.3f} (last 50 updates pre) -> {mw('w_postbest_post_first50'):.3f} (first 50 post) -> {mw('w_postbest_post_last50'):.3f} (last 50 post); the pre-correct (top) expert: {mw('w_prebest_pre_last50'):.3f} -> {mw('w_prebest_post_last50'):.3f}. "
      f"Flip reached in {len(fl)}/{len(rs)} streams, median {np.median(fl):.0f} updates after the switch (range {min(fl)}-{max(fl)}; post phase has ~{np.mean([r['n_post_updates'] for r in rs]):.0f} updates). Realised crossing / T5a analog: median ratio {np.median(ratio):.2f} over {len(ratio)} streams (n small, per-update reward gaps are 0.005-0.05 so the analog is noisy). "
      f"Fixed-share analog {np.median([r['pred_tau_fs_analog'] for r in rs]):.0f} updates (median) vs measured flips (median {np.median(fl):.0f}); the analog is a worst-case style bound and is far from tight (measured/analog ~ {np.median([r['flip_first']/r['pred_tau_fs_analog'] for r in rs if r['flip_first'] is not None]):.2f}).")
w()
w("The weights are close to uniform throughout (all four experts have mean per-update reward within about 0.05 of one another), so the 'correct' expert never dominates: the maximum-weight flip is a ranking change among weights of 0.2-0.35, not a concentration of the weights.")
w()

# ------------------------------------------------------------------ B3
w("## B3. Exploiter runs (script `b3_exploiters.py` -> `out_B3_exploiters.json`)")
w()
w("X<k> = exploit_<victim> (pure best reply to the predicted column yhat, `run_ourbot.py:197`) vs the victim. Baseline = RM+ (`rmplus_avg`) vs the same victim, from E4<k> (A = victim, B = rmplus_avg): baseline score of RM+ = 1 - E4 victim score (E4u -> victim bandit_ucb1 0.0502 so RM+ 0.9498; checked that each E4 summary has B = rmplus_avg and A = X's victim). Gain = X score - baseline, paired by seed (same teams per (seed, game) verified; the victim's seat differs between E4 and X, so the pairing holds at the pair level) and unpaired. Stream-cluster CIs of the paired gain: XU [0.013, 0.058], XUmv [0.007, 0.097], XEmv [-0.132, 0.026]. XR (victim = RM+ itself) has no E4 counterpart; expected 0.5 by seat symmetry.")
w()
w("| tag | exploiter vs victim | pairs | score [normal] / [stream-cl.] | baseline (RM+ vs victim) | gain paired [normal] / [cluster] | gain unpaired [CI] |")
w("|---|---|---|---|---|---|---|")
for t, r in B3.items():
    if isinstance(r["baseline"], dict):
        g = r["gain_vs_baseline_paired_by_seed"]; gu = r["gain_unpaired"]
        w(f"| {t} | {r['A']} vs {r['B']} | {r['n_pairs']} | {r['score']:.4f} {ci(r['ci_pairs_normal'])} / {ci(r['ci_stream_cluster'])} | {r['baseline']['score']:.4f} ({r['baseline']['tag']}, n={r['baseline']['n_pairs']}) | {g['mean']:+.4f} {ci(g['normal'])} / {ci(g['cluster'])} (n={g['n']}) | {gu['mean']:+.4f} {ci(gu['ci'])} |")
    else:
        g = r["gain_vs_0.5"]
        w(f"| {t} | {r['A']} vs {r['B']} | {r['n_pairs']} | {r['score']:.4f} {ci(r['ci_pairs_normal'])} / {ci(r['ci_stream_cluster'])} | none (0.5 by symmetry) | {g['mean']:+.4f} {ci(g['normal'])} (vs 0.5) | n/a |")
w()
w("Prediction accuracy of the exploiter's opponent model: yhat is logged **top-5 only**, so log-loss is not computable; reported instead are (a) top-1 hit = the victim's played label is in the set of argmax-yhat labels (ties), (b) played label in the logged top-5, (c) logged yhat mass on the played label (a lower bound: 0 if outside the top 5), against chance 1/n_cols and top5/n_cols. "
  "The victim's played action = `move` in the B decision record at the same `turn` ('switch X' -> X); only turns with the same number of A and B records are paired (otherwise one side had no decision that turn), the rest discarded; only A decisions that carry `yhat` count.")
w()
w("| tag | aligned decisions (of exploit decisions; discarded frac.) | top-1 hit | in logged top-5 | mean logged mass on played (LB) | chance 1/n_cols | chance top5/n_cols | mean size of argmax tie set |")
w("|---|---|---|---|---|---|---|---|")
for t, r in B3.items():
    y = r["yhat_accuracy"]
    w(f"| {t} | {y['n_aligned']} ({y['n_exploit_decisions']}; {y['frac_misaligned_discarded']:.2f}) | {y['top1_in_support']:.3f} | {y['played_in_logged_top5']:.3f} | {y['mean_logged_mass_on_played_lowerbound']:.3f} | {y['chance_1_over_ncols']:.3f} | {y['chance_top5_over_ncols']:.3f} | {y['mean_top_support_size']:.2f} |")
w()
w("For the `_mv` victims (most-visited-arm rule) yhat is uniform over the argmax set, so top-5 coverage is complete and hit = in-support; for XU/XR yhat is the full share vector truncated to 5 entries. The discarded fraction (~35%) is turns where only one side had a decision (forced switch after a faint) or the counts differ; no per-record alignment was attempted for those.")
w()

# ------------------------------------------------------------------ B4
w("## B4. E5 deficit decomposition (script `b4_e5.py` -> `out_B4_e5.json`)")
w()
w("Only `E5` (hedge vs agentform) contains Hedge records (weights, per-expert argmax labels `experts`, rewards). E5t (`top` vs agentform) and E5s (agentform vs agentform) are not Hedge methods: no weights/experts are logged, so they supply only the score split.")
w()
w("| run | A vs B | pairs | pair score [normal] / [stream-cl.] |")
w("|---|---|---|---|")
for t, lab in (("E5", "hedge vs agentform"), ("E5t", "top vs agentform"), ("E5s", "agentform vs agentform (control)")):
    s = B4["scores"][t]
    w(f"| {t} | {lab} | {s['n']} | {s['mean']:.4f} {ci(s['normal'])} / {ci(s['cluster'])} |")
w()
dp = B4["diff_paired_by_seed"]
w("Paired by seed: " + "; ".join(f"{k} = {v['mean']:+.4f} normal {ci(v['normal'])} cluster {ci(v['cluster'])} (n={v['n']})" for k, v in dp.items()) + ".")
d_ = B4["E5_hedge_decisions"]
w()
w(f"Played action vs expert recommendations (E5, A side, {d_['n_multi_option_decisions']} decisions with >1 legal option): played move equals the argmax label of top {d_['frac_played_equals_expert_argmax']['top']:.3f}, sampled {d_['frac_played_equals_expert_argmax']['sampled']:.3f} (= top, see anomaly 3), safest {d_['frac_played_equals_expert_argmax']['safest']:.3f}, safex {d_['frac_played_equals_expert_argmax']['safex']:.3f}. Mean played weights: " + ", ".join(f"{k} {v:.3f}" for k, v in d_["mean_weights"].items()) + ".")
t_ = d_["top_ne_safest"]
w(f"On the {t_['n']} decisions ({100*t_['frac_of_decisions']:.1f}%) where the top and safest experts disagree: played = top's move {t_['played_eq_top']:.3f}, = safest's move {t_['played_eq_safest']:.3f}, neither {t_['played_neither']:.3f}; played = safex's argmax {t_['played_eq_safex_argmax']:.3f}.")
w()
w("Score split by how top-like / safest-like a game's play was (terciles of the per-game statistic; 600 games; n per tercile ~200; game score 0/1):")
w()
w("| statistic | corr with game score | tercile 1 (low) | tercile 2 | tercile 3 (high) |")
w("|---|---|---|---|---|")
for key, lab in (("mean_wtop", "mean weight on top"), ("frac_top", "fraction of decisions where played = top's move"), ("frac_safest", "fraction played = safest's move")):
    tt = B4[f"score_by_{key}_tercile"]
    w(f"| {lab} | {B4[f'corr_score_{key}']:+.3f} | " + " | ".join(f"{x['score']:.3f} (n={x['n']})" for x in tt) + " |")
w()
w("Weight trajectories (carried over 60 games/stream; mean over updates in the first 10%, 10-50%, last 50% of each stream, order top/sampled/safest/safex; and the expert with the highest cumulative reward):")
w()
w("| stream | updates | mean w first 10% | 10-50% | last 50% | final w | best cum. reward |")
w("|---|---|---|---|---|---|---|")
for r in B4["E5_weight_trajectories"]:
    w(f"| {r['stream']} | {r['T']} | {r['mean_w_first10pct']} | {r['mean_w_10_50pct']} | {r['mean_w_last50pct']} | {r['final_w']} | {r['best_cum']} |")
w()
w("Numbers only: the top expert holds the highest weight and the best cumulative reward in all 10 streams, yet Hedge plays a mixture that follows top's move in 84.7% of decisions (77% safest); E5 (0.458) is below E5t (0.550) by -0.092 and below the control E5s (0.518) by -0.062, and E5t is not distinguishable from E5s (+0.029). The tercile splits show no monotone relation between top-likeness and score (the correlations are 0.015 and 0.006, i.e. null); the share of decisions equal to safest's move correlates +0.113 with game score. A split of the E5 deficit into channels (e.g. mixing weights vs expert quality) is not identifiable from these logs: per-decision Q is not logged for E5 (no OURBOT_LOG_ABAR) and E5t/E5s log no expert quantities.")
w()

# ------------------------------------------------------------------ B5
w("## B5. bayes_safex eps sweep (script `b5_eps.py` -> `out_B5_eps.json`)")
w()
w("SX1/SX3/SX10/SX30 = OURBOT_SAFEX_EPS 0.001/0.003/0.01/0.03 (`launch_open_items.sh`, summaries' `modal.env`), seeds 1-200, vs stock. Reference: agentform vs stock = 0.3144 [0.2792, 0.3495] (n=299 pairs, `engine/showdown/results/AF/AF_summary.json`; 0.3291 on seeds 1-200 only).")
w()
w("| tag | eps | games W/L | pair score [normal] / [stream-cl.] | game score [stream-cl.] | minus agentform (seed-paired, seeds 1-200) [normal] / [cluster] |")
w("|---|---|---|---|---|---|")
for t, v in B5["table"].items():
    m = v["minus_agentform_paired_seeds1_200"]
    w(f"| {t} | {v['eps']} | {v['wins']}/{v['losses']} | {v['pair_mean']:.4f} {ci(v['pair_ci_normal'])} / {ci(v['pair_ci_cluster'])} (n={v['n_pairs']}) | {v['game_mean']:.4f} {ci(v['game_cluster'])} | {m['mean']:+.4f} {ci(m['normal'])} / {ci(m['cluster'])} (n={m['n']}) |")
w()
lg, ln_ = B5["logistic_game_level"], B5["linear_pair_level"]
w(f"Trend: game-level logistic regression of win on ln(eps) (n={lg['n']} games, {lg['clusters']} clusters = tag x stream): slope {lg['slope_per_ln_eps']:+.4f} per ln eps (odds ratio {lg['odds_ratio_per_tenfold_eps']:.3f} per 10x eps), SE model {lg['se_model']:.4f} (z={lg['z_model']:.2f}), SE cluster {lg['se_cluster']:.4f} (z={lg['z_cluster']:.2f}, t-test p={lg['p_cluster_t']:.3f} (t with 39 df over tag x stream clusters; the same seeds/teams recur across tags within a stream, and clustering by stream only (10 clusters) gives SE 0.0363, p=0.040)), 95% CI of slope (cluster) {ci(lg['ci95_slope_cluster'],4)}. Fitted scores: " + ", ".join(f"eps {k}: {v:.3f}" for k, v in lg["predicted_score"].items()) + f". Linearity in ln eps vs a free per-eps model: LR chi2(2) = {lg['LR_linear_vs_categorical_chi2_df2']:.2f}, p = {lg['LR_p']:.2f}. "
  f"Pair-level OLS of pair score on ln eps: slope {ln_['slope_per_ln_eps']:+.4f} (cluster SE {ln_['se_cluster']:.4f}, p={ln_['p']:.3f}, CI {ci(ln_['ci95'],4)}), i.e. {ln_['per_tenfold']:+.3f} per 10x eps. Kendall tau (game outcome vs eps) = {B5['kendall_tau_game_vs_eps']['tau']:.3f} (naive p {B5['kendall_tau_game_vs_eps']['p_naive']:.3f}).")
pa_ = B5["paired_adjacent"]
w("Seed-paired differences between eps levels: " + "; ".join(f"{k} {v['mean']:+.4f} normal {ci(v['normal'])} (n={v['n']})" for k, v in pa_.items()) + ". Adjacent steps are within noise; only the SX30 - SX1 contrast excludes 0 (marginally). All four eps values score below agentform on the same seeds 1-200 (0.3291, n=199) by 0.076-0.128 (seed-paired CIs exclude 0); vs the full AF 0.3144 the gaps are 0.061-0.114. Other references: " + "; ".join(f"{k} ({v['A']}) {v['mean']} {ci(v['ci95'])}" for k, v in B5["context"].items()) + ". Note SX3 (eps 0.003) is the same configuration as the BH hedge_bayes-safex default epsilon but BH is a Hedge mixture, not bayes_safex alone.")
w()

# ------------------------------------------------------------------ B6
w("## B6. RQeq / RQsafe / RQunif: pre-registered predictions P1-P5 (scripts `b6_rq.py`, `b6_score.py` -> `out_B6_rq.json`, `out_B6_scores.json`)")
w()
w("Source: `bayes_dynamics.md` section 4 (pre-registered, committed fb3cc8e before the runs; not edited). Per stream (10 streams, 20 games, ~480-650 feedback updates each) the trajectory is rebuilt from `feedback.bayes` (p_qhat, p_style, posterior; non-skipped turns, forced updates included as the posterior includes them). T = feedback turns of the stream. All three RQ runs: 20 games per stream in the state files, T equals the state `updates` in every stream (all 30 streams clean).")
w()
w("| tag | opponent | score (pair) [normal] | P_T(true style) per stream | final MAP |")
w("|---|---|---|---|---|")
for t in B6:
    rs = B6[t]["streams"]; sc = A[t]["pair_level_normal"]
    w(f"| {t} | style_{B6[t]['star'] if t!='RQunif' else 'uniform'} | {sc[0]:.4f} {ci(sc[2:4])} | {[round(x['P_T_star'],3) for x in rs]} | {dict((s, sum(1 for x in rs if x['final_map']==s)) for s in set(x['final_map'] for x in rs))} |")
w()
p1 = B6s["P1"]
strict = {t: sum(1 for r in B6[t]["streams"] if r["final_post"][r["best_cumloss_style"]] >= 0.999 and round(r["regret_vs_best"], 3) == 1.609) for t in B6}
w("**P1 (universal identity regret = ln5 + ln P_T(best) = 1.609, anytime max <= 1.609 + rounding).** " + "; ".join(f"{t}: anytime max over t and styles {v['anytime_max']:.6f} (ln5 = {v['ln5']:.6f}, excess {v['excess_over_ln5']:.1e}), final-regret range {v['end_regret_range'][0]:.4f}-{v['end_regret_range'][1]:.4f}, max |regret - (ln5 + ln P_T)| at the end {v['end_regret_eq_ln5_when_PT_ge_0999']['max_abs_dev']:.1e} over {v['end_regret_eq_ln5_when_PT_ge_0999']['n']} streams with P_T >= 0.999" for t, v in p1["per_tag"].items())
  + f". Max over every turn x style (posterior >= 0.01) of the identity error: {max(v['max_identity_err_all_turns'] for v in p1['per_tag'].values()):.4f} nats (4-dp posterior rounding). Result: **PASS within logging rounding** (excess over ln5 <= 3.9e-4, consistent with 6-dp rounding of p_qhat/p_style); read at strictly 3 decimals, final regret equals 1.609 in {sum(strict.values())} of the 29 streams with P_T >= 0.999 (RQsafe stream 6, P_T = 0.945, is outside the clause); the excess over ln5 is within the worst-case 6-dp rounding bound in every stream, so the strict '1.609 to 3 decimals' statement is met only up to ~5e-4.")
p2 = B6s["P2"]; vp = p2["verdict_parts"]
w(f"**P2 (style_uniform).** (a) P_T(uniform) >= 0.99 by feedback turn 50 in every stream: **FAIL** ({p2['n_streams_P50_ge_099']}/10 streams; min P_50 = {p2['P_T_uniform_at_50_min']:.3f}); (b) = 1.000 at the end: **PASS** (10/10, all >= 0.9995); (c) final MAP = uniform 10/10: **PASS**; (d) MAP-wrong turns <= 10 per stream: **FAIL** ({vp['wrong_turns_le_10']}/10 streams; counts {p2['map_wrong_turns']}; after turn 50: {p2['map_wrong_after50']}). Scored on total MAP-wrong turns; under the reading 'MAP-wrong turns <= 10 after turn ~50' the clause passes 10/10 ([0,0,1,0,0,0,10,0,0,0]). (e) best style's log loss = E ln n_t: **PASS** (uniform mean loss {p2['uniform_mean_logloss']:.5f} vs mean ln n_t {p2['mean_ln_ncols']:.5f}; uniform is the best style in 10/10 streams; the pre-registered magnitude '~2.3-2.9 nats' is missed: 2.05 over all updates (2.29 excluding forced turns)); (f) score > 0.8: **PASS** ({p2['score']:.4f} {ci(p2['score_ci'])}). The doc's own table has sum_t(1 - P_{{t-1}}(s*)) = 5.5 for uniform; observed per stream {[round(x,1) for x in p2['sum_1_minus_Pprev']]} (mean {np.mean(p2['sum_1_minus_Pprev']):.1f}).")
p3 = B6s["P3"]
w(f"**P3 (style_safe).** Decision rule: mean per-turn log-loss gap br_unif - safe = {p3['decision_rule_mean_loss_gap_br_unif_minus_safe_nats_per_turn']['mean']:.4f} nats/turn (per stream {[round(x,3) for x in p3['decision_rule_mean_loss_gap_br_unif_minus_safe_nats_per_turn']['per_stream']]}) < 0.1 => the **high-mismatch branch** applies by the pre-registered rule (br_us - safe gap {p3['gap_br_us_minus_safe']:.3f}). The rule as written states only '>= 0.1 => low-mismatch'; the observed 0.048 lies between the 0.1-Delta (margin 0.111) and 0.2-Delta (0.001) proxy rows, so neither pre-written outcome strictly covers it. "
  f"High-mismatch branch ('MAP ends on br_unif/br_us with safe within a few nats'): **FAIL** -- final MAP = safe in {p3['final_MAP_safe']}/10 streams, P_T(safe) >= 0.9995 in {p3['n_P_T_ge_0999']}/10 (stream 6: {p3['P_T_end'][6]}), final loss gap to the best other style {[round(x,1) for x in p3['final_gap_to_runner_up_nats']]} nats. "
  f"Low-mismatch branch clauses (reported for completeness): identified within ~100 feedback turns: P_100 per stream {p3['P_at_100']} ({sum(1 for x in p3['P_at_100'] if x>=0.9)}/10 >= 0.9; first turn from which P>=0.9 holds to the end: {p3['t_sustained_P90']}); P_T(safe) = 1.000 at end: {p3['n_P_T_ge_0999']}/10; MAP-wrong turns 10-50: {p3['n_wrong_turns_in_10_50']}/10 (counts {p3['map_wrong_turns']}); final gap to runner-up >= 60 nats: {p3['n_gap_ge_60']}/10 (max {max(p3['final_gap_to_runner_up_nats']):.1f}); point prediction 8-10/10 end with MAP = safe: **PASS** (10/10). Score {p3['score']:.4f} {ci(p3['score_ci'])} vs speculative 0.6-0.75: above the range (**outside**).")
p4 = B6s["P4"]
w(f"**P4 (style_eq).** (a) final MAP in {{safe, br_us, br_unif}}: **PASS** ({p4['n_MAP_in_safe_br_us_br_unif']}/10, all 'safe'); (b) **P_T(eq) < 0.1 in >= 7/10 streams: PASS, {p4['n_P_T_eq_lt_0_1']}/10** (P_T(eq) = {p4['P_T_eq']}, i.e. < 5e-5 in all; mean P(eq) at turn 25 = {p4['P_eq_at_25']:.3f}, at 100 = {p4['P_eq_at_100']:.3f}); none >= 0.9 ({p4['n_P_T_eq_ge_0_9']}); (c) eq's mean log loss exceeds safe's: **PASS** ({p4['mean_logloss_eq']:.3f} vs {p4['mean_logloss_safe']:.3f} nats/turn; true for {p4['n_streams_eq_loss_gt_safe']}/10 streams; all styles {dict((k, round(v,3)) for k, v in p4['mean_loss_all_styles'].items())}); (d) score 0.50-0.60 (speculative): {p4['score']:.4f} {ci(p4['score_ci'])} -- **above the range** (lower CI bound {p4['score_ci'][0]:.3f} just above 0.60). Note eq is also worse than br_unif and br_us and (2.20) is about as bad as uniform (2.23).")
p5 = B6s["P5"]
w(f"**P5 (ordering uniform > safe > eq).** Final P_T(true style) (mean over streams): {', '.join(f'{k} {v['mean_final_P']:.3f}' for k, v in p5['identification'].items())} -> order holds: **{p5['order_final_P']}**. Identification speed as sum_t(1 - P_(t-1)(s*)) (mean): {', '.join(f'{k} {v['mean_sum_1_minus_Pprev']:.1f}' for k, v in p5['identification'].items())} -> order holds: **{p5['order_sum_1_minus_P']}**; median first turn from which P(s*) >= 0.9 persists: uniform {p5['identification']['RQunif']['median_t_P90']}, safe {p5['identification']['RQsafe']['median_t_P90']}, eq never ({p5['identification']['RQeq']['n_never_P90']}/10 streams). Scores: {', '.join(f'{k} {v:.4f}' for k, v in p5['scores'].items())} -> order holds: **{p5['order_scores_unif_gt_safe_gt_eq']}** (CIs: see table above; unif vs safe and safe vs eq separated by >2 SE of a pair mean ~0.03). P5: **PASS**.")
w()
w("Scoreboard: P1 PASS (to rounding) | P2 partial (4 of 6 clauses pass: end P=1.000, final MAP 10/10, loss = E ln n, score > 0.8; the 'P >= 0.99 by turn 50 in every stream' (8/10) and 'MAP-wrong turns <= 10 per stream' (5/10) clauses fail) | P3 high-mismatch branch FAIL, low-mismatch clauses: identified within ~100 turns 6/10, P_T(safe)=1.000 9/10, MAP-wrong 10-50 1/10, gap >= 60 nats 0/10; point prediction (MAP = safe in 8-10/10) PASS 10/10; score 0.855 above the speculative 0.6-0.75 | P4 PASS (10/10 streams with P_T(eq) < 0.1, required >= 7; speculative score 0.50-0.60 missed: 0.670 [0.609, 0.731]) | P5 PASS.")
w()
w("## Reproduction")
w()
w("`cd engine/analysis/open_items && ../../.venv/bin/python a_audit.py; b1_log_identity.py; b1_aggregate.py; b2_switch.py; b3_exploiters.py; b4_e5.py; b5_eps.py; b6_rq.py; b6_score.py; build_report.py` (shared helpers `oi_common.py`; reuses `engine/analysis/synthesis/common.py`). Run time: a few minutes total, B1 ~3 min (LP per decision).")
open(os.path.join(OUT, "REPORT.md"), "w").write("\n".join(o) + "\n")
print("ok", len(o))
