# Open-item analyses (Showdown runs LOG, E3a/f, X*, E5t/s, SX*, RQ*)

Verified independently (recomputed from raw data): scripts in verify_open/ (session scratch); all headline numbers reproduce; corrections below applied.

All numbers are produced by scripts in this directory (`engine/analysis/open_items/`); each section names its script and JSON. Interpreter `engine/.venv/bin/python`. Source logs: `engine/showdown/results/<TAG>/` (summary `<TAG>_summary.json`, game JSONs `replays_headless/<TAG>-s<seed>-g<1|2>.json`, per-decision logs `*.A.decisions.jsonl` / `*.B.decisions.jsonl`, carried state `hedge_state/`). No Modal job launched, nothing outside this directory edited. Score = A's win rate (win 1, loss 0; no draws occurred in any tag). 'pair' = mean of the two seat-swapped games of a seed (the summary's unit); 'game' = single game. CIs: `normal` = normal approx 1.96*SE over units; `cluster` = stream-clustered (CR1, t with G-1 df; **G = 10 streams per tag, a small number**; streams carry state across games so games within a stream are not independent).

## Anomalies / caveats up front

1. Error games (excluded from pair scores by the summary; the partner game of the seed is dropped too): LOG: s4-g2 (ZeroDivisionError('float division by zer...); E3a: s52-g2 (RuntimeError('invalid_choice_stall: p1 g...), s93-g2 (RuntimeError('invalid_choice_stall: p2 g...); E3f: s4-g2 (ZeroDivisionError('float division by zer...); XUmv: s245-g1 (PanicException('Invalid PokemonMoveIndex...); XU: s205-g1 (AttributeError("'NoneType' object has no...), s93-g1 (RuntimeError('invalid_choice_stall: p2 g...); XEmv: s22-g1 (AttributeError("'NoneType' object has no...); E5s: s245-g2 (PanicException('Invalid PokemonMoveIndex...), s291-g2 (RuntimeError('invalid_choice_stall: p2 g...); SX30: s93-g1 (ZeroDivisionError('float division by zer...); RQunif: s3-g1 (AttributeError("'NoneType' object has no...), s52-g2 (RuntimeError('invalid_choice_stall: p1 g...), s93-g1 (RuntimeError('invalid_choice_stall: p2 g...). Causes: ZeroDivisionError, `invalid_choice_stall` (60 s), PanicException `Invalid PokemonMoveIndex: 4`, AttributeError `can_z`. All are listed in section A.
2. Carried-state files record MORE games than game JSONs exist in 12 streams (LOG stream 3: state games 61 vs 60 game files; LOG stream 9: state games 61 vs 60 game files; E3a stream 1: state games 33 vs 32 game files; E3a stream 3: state games 33 vs 32 game files; E3a stream 7: state games 33 vs 32 game files; E3f stream 2: state games 35 vs 32 game files; SX1 stream 9: state games 42 vs 40 game files; SX10 stream 5: state games 41 vs 40 game files; SX10 stream 9: state games 42 vs 40 game files; SX30 stream 2: state games 41 vs 40 game files; SX30 stream 4: state games 41 vs 40 game files; SX30 stream 7: state games 43 vs 40 game files). `load_learner`/`load_posterior` increment `meta.games` at bot construction, so (a hypothesis consistent with the resume rule `modal_showdown.py:163`, not a finding) an aborted/retried attempt of a game (JSON overwritten by the retry) updated the carried Hedge/posterior state with partial feedback while only the retry survives in the logs. Only 3 of these have a matching error game (LOG s4-g2 in stream 3; E3a s52-g2 in stream 1; SX30 s93-g1 in stream 2) -- the other streams have no trace of a retry in the logs (cause not determinable from the logs). Record field sets and config fields (k, iters, played_rule, eps) are constant across invocations in all resumed tags. Effect on scores: none (scores are from game outcomes); effect on weight/posterior trajectories: replayed logs omit the aborted attempt's updates, so `hedge_state` final weights can differ slightly from a replay of the logged feedback.
3. `experts` logging: the `sampled` expert's logged label is `legal[argmax Q]` of its mixture, which is identical to the `top` expert's label by construction (both = argmax of the same LP mixture); so in B4 the `sampled` and `top` match rates are identical and carry no separate information.
4. E3a/E3f are `hedge`/`hedge_fs` (no style posterior): the flip-delay formula of bayes_dynamics.md T5 is for the Bayes posterior; only a reward-space analog is evaluated (B2).
5. In the `X*` runs the victim B side is also our code, so B decision logs exist; in LOG/SX/E3 the stock opponent (foul-play MCTS) has no decision log, so its own matrices are not recoverable.

## A. Audit (script `a_audit.py` -> `out_A_audit.json`)

Expected grid from `engine/showdown/launch_open_items.sh`: 10 streams per tag (stream k = sorted seeds[k::10]), 2 games per seed. The `stream` field of every game JSON was checked against this rule: mismatches = 0.

| tag | A vs B | exp. games | game JSONs | missing | error games | decision logs A/B | summary pair mean [CI] (n pairs) | recomputed pair mean (n) | match (mean / per-seed) | nonzero exit / failed streams / over budget | modal games_run |
|---|---|---|---|---|---|---|---|---|---|---|---|
| LOG | hedge_bayes vs stock | 600 | 600 | 0 | 1 | 600/0 | 0.3077 [0.272, 0.343] (299) | 0.3077 (299) | True / True | none / none / False | 524 |
| E3a | hedge vs switch:stock@8:exploit_top | 320 | 320 | 0 | 2 | 320/160 | 0.4873 [0.433, 0.542] (158) | 0.4873 (158) | True / True | none / none / False | 320 |
| E3f | hedge_fs vs switch:stock@8:exploit_top | 320 | 320 | 0 | 1 | 320/160 | 0.4969 [0.442, 0.552] (159) | 0.4969 (159) | True / True | none / none / False | 290 |
| XUmv | exploit_bandit_ucb1_mv vs bandit_ucb1_mv | 600 | 600 | 0 | 1 | 600/600 | 0.4582 [0.418, 0.498] (299) | 0.4582 (299) | True / True | none / none / False | 592 |
| XU | exploit_bandit_ucb1 vs bandit_ucb1 | 600 | 600 | 0 | 2 | 600/600 | 0.9849 [0.975, 0.995] (298) | 0.9849 (298) | True / True | none / none / False | 600 |
| XR | exploit_rmplus_avg vs rmplus_avg | 600 | 600 | 0 | 0 | 600/600 | 0.4683 [0.431, 0.505] (300) | 0.4683 (300) | True / True | none / none / False | 542 |
| XEmv | exploit_bandit_exp3_mv vs bandit_exp3_mv | 600 | 600 | 0 | 1 | 600/600 | 0.4548 [0.415, 0.494] (299) | 0.4548 (299) | True / True | none / none / False | 458 |
| E5t | top vs agentform | 600 | 600 | 0 | 0 | 600/600 | 0.55 [0.513, 0.587] (300) | 0.5500 (300) | True / True | none / none / False | 542 |
| E5s | agentform vs agentform | 600 | 600 | 0 | 2 | 600/600 | 0.5185 [0.480, 0.557] (298) | 0.5185 (298) | True / True | none / none / False | 530 |
| SX1 | bayes_safex vs stock | 400 | 400 | 0 | 0 | 400/0 | 0.2 [0.163, 0.237] (200) | 0.2000 (200) | True / True | none / none / False | 390 |
| SX3 | bayes_safex vs stock | 400 | 400 | 0 | 0 | 400/0 | 0.2275 [0.189, 0.266] (200) | 0.2275 (200) | True / True | none / none / False | 400 |
| SX10 | bayes_safex vs stock | 400 | 400 | 0 | 0 | 400/0 | 0.245 [0.203, 0.287] (200) | 0.2450 (200) | True / True | none / none / False | 328 |
| SX30 | bayes_safex vs stock | 400 | 400 | 0 | 1 | 400/0 | 0.2538 [0.213, 0.294] (199) | 0.2538 (199) | True / True | none / none / False | 320 |
| RQeq | hedge_bayes vs style_eq | 200 | 200 | 0 | 0 | 200/200 | 0.67 [0.608, 0.732] (100) | 0.6700 (100) | True / True | none / none / False | 200 |
| RQsafe | hedge_bayes vs style_safe | 200 | 200 | 0 | 0 | 200/200 | 0.855 [0.808, 0.902] (100) | 0.8550 (100) | True / True | none / none / False | 200 |
| RQunif | hedge_bayes vs style_uniform | 200 | 200 | 0 | 3 | 200/200 | 0.9948 [0.985, 1.005] (97) | 0.9948 (97) | True / True | none / none / False | 200 |

Resume / duplicate check: for every `*.decisions.jsonl` (A and B) the `turn` field never decreases within a file (a rerun appending to the same file would reset it): resets found = 0 files. Non-final record count equals `timing.<role>.decisions` in the game JSON for all files except XUmv [245, 1, 'B', 27, 28]; E5s [245, 2, 'A', 24, 25] (off by one, both error games or neighbours of one). Note repeated `turn` values within a file are normal (a second decision in the same turn after a faint).
Resumed runs: `modal.games_run` (games executed by the final invocation) is below the expected games in LOG 524/600, E3f 290/320, XUmv 592/600, XR 542/600, XEmv 458/600, E5t 542/600, E5s 530/600, SX1 390/400, SX10 328/400, SX30 320/400; the remaining game JSONs were produced by earlier invocations of the same tag (resume rule `modal_showdown.py:163`: seeds whose two JSONs exist are skipped, carried weights resumed). No `failed_streams`, `stopped_over_budget` or `nonzero_exit` is recorded in any summary (these fields describe only the final invocation), and the reason for the earlier partial invocations is not in the logs. Consistency check that earlier-invocation games used the same configuration: for SX1/SX3/SX10/SX30 every A decision record that logs `eps` has the tag's epsilon (13069/13061/13566/12989 records, 0 deviations; `out_A_sx_eps_check.json`); for LOG all 20696 non-final records with more than one legal option carry `abar` (546 records without it are single-option).
E3a/E3f `active` field: B = stock for seeds 1-80 ({'stock': 80, 'exploit_top': 80} seeds each, E3a max stock seed 80, min exploit_top seed 81), i.e. the switch happens after seed position 8 of every 16-seed stream, as launched.

Scores with 95% CI (game level and pair level; normal and stream-clustered):

| tag | W/L (games) | game mean [normal CI] | game [cluster CI] | pair mean [normal CI] | pair [cluster CI] |
|---|---|---|---|---|---|
| LOG | 184/415 (n=599) | 0.3072 [0.270, 0.344] | [0.275, 0.340] | 0.3077 [0.273, 0.343] (n=299) | [0.276, 0.340] |
| E3a | 155/163 (n=318) | 0.4874 [0.432, 0.542] | [0.434, 0.541] | 0.4873 [0.433, 0.542] (n=158) | [0.432, 0.543] |
| E3f | 158/161 (n=319) | 0.4953 [0.440, 0.550] | [0.447, 0.543] | 0.4969 [0.442, 0.552] (n=159) | [0.449, 0.545] |
| XUmv | 274/325 (n=599) | 0.4574 [0.417, 0.497] | [0.410, 0.505] | 0.4582 [0.418, 0.498] (n=299) | [0.411, 0.505] |
| XU | 589/9 (n=598) | 0.9849 [0.975, 0.995] | [0.975, 0.995] | 0.9849 [0.975, 0.995] (n=298) | [0.974, 0.995] |
| XR | 281/319 (n=600) | 0.4683 [0.428, 0.508] | [0.424, 0.513] | 0.4683 [0.431, 0.505] (n=300) | [0.424, 0.513] |
| XEmv | 273/326 (n=599) | 0.4558 [0.416, 0.496] | [0.401, 0.510] | 0.4548 [0.415, 0.494] (n=299) | [0.399, 0.510] |
| E5t | 330/270 (n=600) | 0.5500 [0.510, 0.590] | [0.523, 0.577] | 0.5500 [0.513, 0.587] (n=300) | [0.523, 0.577] |
| E5s | 309/289 (n=598) | 0.5167 [0.477, 0.557] | [0.463, 0.571] | 0.5185 [0.480, 0.557] (n=298) | [0.465, 0.572] |
| SX1 | 80/320 (n=400) | 0.2000 [0.161, 0.239] | [0.166, 0.234] | 0.2000 [0.163, 0.237] (n=200) | [0.166, 0.234] |
| SX3 | 91/309 (n=400) | 0.2275 [0.186, 0.269] | [0.195, 0.260] | 0.2275 [0.189, 0.266] (n=200) | [0.195, 0.260] |
| SX10 | 98/302 (n=400) | 0.2450 [0.203, 0.287] | [0.204, 0.286] | 0.2450 [0.203, 0.287] (n=200) | [0.204, 0.286] |
| SX30 | 101/298 (n=399) | 0.2531 [0.210, 0.296] | [0.212, 0.294] | 0.2538 [0.214, 0.294] (n=199) | [0.213, 0.294] |
| RQeq | 134/66 (n=200) | 0.6700 [0.605, 0.735] | [0.602, 0.738] | 0.6700 [0.609, 0.731] (n=100) | [0.602, 0.738] |
| RQsafe | 171/29 (n=200) | 0.8550 [0.806, 0.904] | [0.803, 0.907] | 0.8550 [0.808, 0.902] (n=100) | [0.803, 0.907] |
| RQunif | 196/1 (n=197) | 0.9949 [0.985, 1.005] | [0.983, 1.006] | 0.9948 [0.985, 1.005] (n=97) | [0.983, 1.006] |

Error-game causes: LOG: {'s4-g2': "ZeroDivisionError('float division by zero')"}; E3a: {'s52-g2': "RuntimeError('invalid_choice_stall: p1 got [Invalid choice] and no new request within 60s'", 's93-g2': "RuntimeError('invalid_choice_stall: p2 got [Invalid choice] and no new request within 60s'"}; E3f: {'s4-g2': "ZeroDivisionError('float division by zero')"}; XUmv: {'s245-g1': "PanicException('Invalid PokemonMoveIndex: 4')"}; XU: {'s205-g1': 'AttributeError("\'NoneType\' object has no attribute \'can_z\'")', 's93-g1': "RuntimeError('invalid_choice_stall: p2 got [Invalid choice] and no new request within 60s'"}; XEmv: {'s22-g1': 'AttributeError("\'NoneType\' object has no attribute \'can_z\'")'}; E5s: {'s245-g2': "PanicException('Invalid PokemonMoveIndex: 4')", 's291-g2': "RuntimeError('invalid_choice_stall: p2 got [Invalid choice] and no new request within 60s'"}; SX30: {'s93-g1': "ZeroDivisionError('float division by zero')"}; RQunif: {'s3-g1': 'AttributeError("\'NoneType\' object has no attribute \'can_z\'")', 's52-g2': "RuntimeError('invalid_choice_stall: p1 got [Invalid choice] and no new request within 60s'", 's93-g1': "RuntimeError('invalid_choice_stall: p2 got [Invalid choice] and no new request within 60s'"}

## B1. LOG: signed pathwise identity along hedge_bayes vs stock (scripts `b1_log_identity.py`, `b1_aggregate.py` -> `out_B1_log_games.json`, `out_B1_summary.json`)

Identity (bounds_epsv.md 1.1): z - phi(v_1) = sum_t(-R_t - S_t + D_t) with R_t = phi(v_t) - phi((x_t Abar)_j), S_t = phi((x_t Abar)_j) - phi(Abar[i,j]), D_t = phi(v_{t+1}) - phi(Abar[i,j]), phi(v_{T+1}) := z. Implementation: Abar, labels, legal, pfull from the logged records (abar rounded to 2 dp); **v_t = value of Abar_t by LP (scipy linprog), not logged**; x_t = logged `pfull` (the Hedge mixture actually played, not RM+); i_t = played move (`switch X` -> X); j_t = opponent column `feedback.b` of the next record looked up in `labels`; z from `winner_role` (A win +1, loss -1; no draws; no capped games occurred). phi = logistic 2*lgs(v/98)-1 (the doc's B12 fit, applied here unchanged) and tanh(v/400). Forced decisions (single legal option, no `abar`) are excluded; D spans them.
Coverage: 599 games (the one error game LOG s4-g2 excluded), 20657 non-forced decisions, 17271 (83.6%) with j known; unknown j (feedback skipped: {'not_in_worlds': 1533, 'fainted': 1003, 'no_action': 431, 'cant': 419}) cannot be split into R,S,D and are excluded from the D/R/S statistics; their lump phi(v_{t+1})-phi(v_t) is still counted in the closure check. **Closure:** max |(-R-S+D) - (phi(v_t+1)-phi(v_t))| per decision = 2.2e-16; max |z - phi(v_1) - sum| per game = 4.4e-16 (both maps). Closure holds by construction (telescoping); the indexing was validated separately: pfull = normalise(w·Q) 20657/20657, logged expert labels = argmax Q, pfull[i_played] > 0 always, feedback column always in labels. The logged agent-form x (Q[1]) is not a maximin strategy of the world-averaged Abar (it solves the per-world agent-form LP): v_t - min_j(x_AF·Abar)_j median 0.12, mean 4.6 engine units (above 1% of the payoff range on a third of decisions). The played Hedge mixture (`pfull`) has gap median 1.85, mean 6.12 (`verify/b1_gap.py`).

What is NOT computable: the stock (foul-play MCTS) side's own matrices/values are not logged, so a drift/regret decomposition for the opponent as a decision maker does not exist. The 'two sides' are therefore reported as: our side's identity terms, plus two stage quantities defined on our Abar for both players: (a) *opponent stage gap* G^opp = phi((xA)_j) - phi(min_j' (xA)_j') (how far the realised column is from the best reply to our mixture, in our model), (b) *our stage gap* G^us = phi(v_t) - phi(min_j (xA)_j) (our mixture's shortfall in Abar). Note R_t = G^us_t - G^opp_t exactly. Seats p1/p2 are reported separately.

Mean drift per decision E[D] (known-j decisions). SE columns: decision-level iid / clustered by game / clustered by stream (10 clusters). Prior numbers (bounds_epsv.md 2.2 for the means, which has no SEs; SEs from theory/tightening/verify/out_v_drift.txt lines 3/5/7; logit): -0.0127 +/- 0.0033 vs white-box exploiter, -0.0113 +/- 0.0023 vs MCTS-2000, +0.0020 +/- 0.0017 self-play; sum|D| per game 2.9 / 3.1 / 3.2.

| subset | phi | games | decisions (j known) | E[D] per decision | SE iid | SE game-cl. | SE stream-cl. | E abs(D) | sum abs(D) per game (known only) | same scaled by n_dec/n_known | sum D per game | sum R | sum S | sum G^opp | sum G^us |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| all | logit | 599 | 17271 | -0.0548 | 0.0012 | 0.0015 | 0.0016 | 0.107 | 3.10 (SE 0.06) | 3.72 | -1.58 (SE 0.04) | -1.07 | 0.05 | 1.66 | 0.60 |
| all | tanh | 599 | 17271 | -0.0410 | 0.0009 | 0.0012 | 0.0010 | 0.077 | 2.21 (SE 0.04) | 2.65 | -1.18 (SE 0.03) | -0.66 | 0.03 | 1.02 | 0.36 |
| seat p1 | logit | 300 | 8677 | -0.0573 | 0.0017 | 0.0020 | 0.0018 | 0.109 | 3.14 (SE 0.09) | 3.76 | -1.66 (SE 0.06) | -1.10 | 0.05 | 1.70 | 0.60 |
| seat p1 | tanh | 300 | 8677 | -0.0432 | 0.0013 | 0.0016 | 0.0014 | 0.078 | 2.24 (SE 0.05) | 2.68 | -1.25 (SE 0.04) | -0.68 | 0.03 | 1.04 | 0.36 |
| seat p2 | logit | 299 | 8594 | -0.0522 | 0.0018 | 0.0022 | 0.0018 | 0.106 | 3.06 (SE 0.09) | 3.67 | -1.50 (SE 0.06) | -1.04 | 0.05 | 1.63 | 0.59 |
| seat p2 | tanh | 299 | 8594 | -0.0388 | 0.0013 | 0.0018 | 0.0013 | 0.076 | 2.17 (SE 0.05) | 2.61 | -1.12 (SE 0.04) | -0.64 | 0.03 | 0.99 | 0.36 |
| games A won | logit | 184 | 5584 | -0.0265 | 0.0021 | 0.0022 | 0.0021 | 0.100 | 3.04 (SE 0.12) | 3.80 | -0.80 (SE 0.07) | -1.22 | -0.04 | 1.87 | 0.65 |
| games A won | tanh | 184 | 5584 | -0.0113 | 0.0014 | 0.0013 | 0.0010 | 0.066 | 1.99 (SE 0.07) | 2.48 | -0.34 (SE 0.04) | -0.76 | -0.03 | 1.14 | 0.38 |
| games A lost | logit | 415 | 11687 | -0.0683 | 0.0015 | 0.0015 | 0.0016 | 0.111 | 3.12 (SE 0.08) | 3.68 | -1.92 (SE 0.04) | -1.00 | 0.09 | 1.57 | 0.57 |
| games A lost | tanh | 415 | 11687 | -0.0552 | 0.0012 | 0.0010 | 0.0011 | 0.082 | 2.31 (SE 0.04) | 2.72 | -1.56 (SE 0.03) | -0.62 | 0.06 | 0.97 | 0.35 |

Reading (numbers only): mean drift per decision is -0.0548 (logit) / -0.0410 (tanh), i.e. 4.3x (logit) / 3.9x (tanh) the prior exploiter value; 4.8x / 5.1x the MCTS-2000 value (-0.0127 / -0.0113 logit; -0.0106 / -0.0080 tanh) and with the same sign; the difference far exceeds the SEs (SEs reflect sampling noise only, not the setup differences below). Sum|D| per game is 3.10 over the known decisions (3.72 scaled to all decisions), numerically close to the prior 2.9-3.2, but from about double the per-decision E|D| (0.107 vs 0.054-0.057) over about half as many known decisions per game (~29 vs 52-57) and above the payoff range 2 on the logit map. Drift is more negative in games A lost (logit -0.068) than won (-0.027), as expected for a signed residual that is conditioned on the realised path. E[phi(v_1)] = -0.071 (SE 0.005) logit, -0.035 tanh: SYM (E phi(v_1)=0) is not satisfied here (stock is a different algorithm and seat-paired teams are swapped, so antisymmetry is not expected). E[z] = -0.386.
Comparability caveats: (i) v_t is the value of the world-averaged Abar of a hidden-information agent (K=4 sampled worlds, world-averaged depth-2 joint-action matrix (nash.full_matrix, nash.py:155) under hidden information), not the perfect-information calibrated depth-2 value of the prior; (ii) x_t is the Hedge mixture, not RM+; (iii) the opponent is foul-play stock MCTS, a third matchup type; (iv) 16.4% of decisions lack j and are excluded, and they are not random (not_in_worlds, fainted, cant). So the comparison with the prior is of the same formula on a different leaf and opponent, not a replication.

## B2. E3a / E3f: opponent switches stock -> exploit_top after 8 of 16 seeds per stream (script `b2_switch.py` -> `out_B2_switch.json`)

Unit of time = a Hedge update (non-skipped feedback). `t0` = number of updates before the switch (first 16 games of the stream). Per-update expert reward r (logged `feedback.r`, in [0,1], scale 400). Correct expert before/after = argmax of cumulative logged reward within the phase. Weights = the played mixture weights `weights` of the decision record. 'Flip' = first update after t0 at which the post-phase-correct expert has the maximum weight (flip_first); sustained = holds for 20 consecutive updates; 'cum. crossing' = update at which the post-correct expert's cumulative reward first exceeds the pre-correct expert's (exact flip rule of anytime Hedge vs that one rival). T5(a)-analog: tau* = t0 (mu_a^pre - mu_b^pre)/(mu_b^post - mu_a^post), a = pre-correct, b = post-correct, mu = mean per-update reward in the phase (reward-space analog of the formula for the Bayes posterior; it is exact only for anytime-FTRL crossing). Fixed-share analog (hedge_fs: eta=0.2, alpha=0.002, k=4): tau_fs = ln(k/alpha) / (eta (mu_b^post - mu_a^post)).

**E3a** score pre-switch (vs stock, 8 seeds/stream): 0.3208 normal [0.248, 0.394] cluster [0.223, 0.418] (n=159 games); post-switch (vs exploit_top): 0.6541 normal [0.580, 0.728] cluster [0.572, 0.737] (n=159); first 2 post seeds (4 games/stream): 0.590 [0.433, 0.746] (n=39); later post seeds: 0.675 [0.591, 0.759] (n=120). The pre and post scores are against DIFFERENT opponents; the difference is not a switching cost.
**E3f** score pre-switch (vs stock, 8 seeds/stream): 0.3459 normal [0.272, 0.420] cluster [0.266, 0.426] (n=159 games); post-switch (vs exploit_top): 0.6438 normal [0.569, 0.718] cluster [0.565, 0.723] (n=160); first 2 post seeds (4 games/stream): 0.525 [0.368, 0.682] (n=40); later post seeds: 0.683 [0.600, 0.767] (n=120). The pre and post scores are against DIFFERENT opponents; the difference is not a switching cost.

hedge_fs minus hedge, paired by seed (same seeds/teams): all +0.0032 normal [-0.066, 0.072] cluster [-0.086, 0.092] (n=157 seeds); pre +0.0128 [-0.085, 0.111] (n=78); post -0.0063 [-0.104, 0.091] (n=79). No difference distinguishable from zero.

**E3a per-stream** (flip columns in updates after the switch; `None` = never within the post phase):

| stream | t0 | post updates | pre-best -> post-best | w(post-best) last50 pre / first50 post / last50 post | w(pre-best) last50 pre / last50 post | flip (first / sustained20) | cum. crossing vs pre-best | T5a analog tau* | fixed-share analog |
|---|---|---|---|---|---|---|---|---|---|
| 0 | 490 | 496 | top -> sampled | 0.286 / 0.293 / 0.337 | 0.354 / 0.273 | 192 / 192 | 191 | 223 | 5006 |
| 1 | 473 | 385 | top -> sampled | 0.270 / 0.275 / 0.326 | 0.321 / 0.313 | 333 / 333 | 332 | 272 | 9458 |
| 2 | 426 | 517 | top -> sampled | 0.279 / 0.280 / 0.321 | 0.405 / 0.299 | 431 / 431 | 430 | 372 | 5724 |
| 3 | 460 | 405 | top -> sampled | 0.250 / 0.240 / 0.308 | 0.342 / 0.255 | 228 / 246 | 228 | 231 | 3855 |
| 4 | 380 | 404 | top -> sampled | 0.286 / 0.296 / 0.314 | 0.364 / 0.293 | 312 / 312 | 311 | 246 | 6990 |
| 5 | 488 | 577 | top -> sampled | 0.265 / 0.267 / 0.312 | 0.402 / 0.289 | 389 / 404 | 388 | 427 | 5553 |
| 6 | 511 | 471 | top -> sampled | 0.283 / 0.279 / 0.309 | 0.356 / 0.328 | None / None | None | 787 | 18718 |
| 7 | 509 | 486 | top -> sampled | 0.287 / 0.285 / 0.329 | 0.400 / 0.313 | 234 / 324 | 233 | 388 | 6658 |
| 8 | 461 | 471 | top -> safest | 0.188 / 0.186 / 0.283 | 0.323 / 0.222 | None / None | 102 | 293 | 3088 |
| 9 | 449 | 539 | top -> safest | 0.246 / 0.239 / 0.290 | 0.314 / 0.220 | 168 / 168 | 167 | 213 | 4329 |

**E3f per-stream** (flip columns in updates after the switch; `None` = never within the post phase):

| stream | t0 | post updates | pre-best -> post-best | w(post-best) last50 pre / first50 post / last50 post | w(pre-best) last50 pre / last50 post | flip (first / sustained20) | cum. crossing vs pre-best | T5a analog tau* | fixed-share analog |
|---|---|---|---|---|---|---|---|---|---|
| 0 | 461 | 514 | top -> safest | 0.183 / 0.193 / 0.309 | 0.354 / 0.182 | 412 / 412 | 417 | 462 | 3321 |
| 1 | 530 | 369 | top -> safest | 0.198 / 0.232 / 0.262 | 0.374 / 0.271 | None / None | None | 607 | 4986 |
| 2 | 516 | 402 | top -> safest | 0.222 / 0.222 / 0.348 | 0.364 / 0.241 | 205 / 250 | 385 | 369 | 3162 |
| 3 | 444 | 444 | top -> safest | 0.219 / 0.208 / 0.286 | 0.316 / 0.204 | None / None | 388 | 367 | 4689 |
| 4 | 437 | 423 | top -> safest | 0.196 / 0.208 / 0.279 | 0.377 / 0.250 | 145 / 145 | None | 789 | 5827 |
| 5 | 495 | 419 | top -> safest | 0.191 / 0.192 / 0.254 | 0.352 / 0.256 | 216 / None | None | 772 | 5545 |
| 6 | 412 | 505 | top -> sampled | 0.264 / 0.263 / 0.303 | 0.431 / 0.257 | 370 / 370 | None | 616 | 7188 |
| 7 | 464 | 482 | top -> safest | 0.167 / 0.190 / 0.287 | 0.374 / 0.252 | 419 / 445 | None | 738 | 4563 |
| 8 | 617 | 365 | top -> sampled | 0.227 / 0.239 / 0.311 | 0.382 / 0.310 | 274 / 274 | None | 685 | 7053 |
| 9 | 495 | 500 | top -> safest | 0.214 / 0.216 / 0.284 | 0.326 / 0.229 | 129 / 129 | 179 | 464 | 4804 |

**E3a summary**: post-correct expert = {'sampled': 8, 'safest': 2}; pre-correct expert = top in all 10 streams. Mean weight of the post-correct expert: 0.264 (last 50 updates pre) -> 0.264 (first 50 post) -> 0.313 (last 50 post); the pre-correct (top) expert: 0.358 -> 0.280. Flip reached in 8/10 streams, median 273 updates after the switch (range 168-431; post phase has ~475 updates). Realised crossing / T5a analog: median ratio 0.91 over 9 streams (n small, per-update reward gaps are 0.005-0.05 so the analog is noisy). Fixed-share analog 5639 updates (median) vs measured flips (median 273); the analog is a worst-case style bound and is far from tight (measured/analog ~ 0.04).
**E3f summary**: post-correct expert = {'sampled': 2, 'safest': 8}; pre-correct expert = top in all 10 streams. Mean weight of the post-correct expert: 0.208 (last 50 updates pre) -> 0.216 (first 50 post) -> 0.292 (last 50 post); the pre-correct (top) expert: 0.365 -> 0.245. Flip reached in 8/10 streams, median 245 updates after the switch (range 129-419; post phase has ~442 updates). Realised crossing / T5a analog: median ratio 0.97 over 4 streams (n small, per-update reward gaps are 0.005-0.05 so the analog is noisy). Fixed-share analog 4895 updates (median) vs measured flips (median 245); the analog is a worst-case style bound and is far from tight (measured/analog ~ 0.05).

The weights are close to uniform throughout (all four experts have mean per-update reward within about 0.05 of one another), so the 'correct' expert never dominates: the maximum-weight flip is a ranking change among weights of 0.2-0.35, not a concentration of the weights.

## B3. Exploiter runs (script `b3_exploiters.py` -> `out_B3_exploiters.json`)

X<k> = exploit_<victim> (pure best reply to the predicted column yhat, `run_ourbot.py:197`) vs the victim. Baseline = RM+ (`rmplus_avg`) vs the same victim, from E4<k> (A = victim, B = rmplus_avg): baseline score of RM+ = 1 - E4 victim score (E4u -> victim bandit_ucb1 0.0502 so RM+ 0.9498; checked that each E4 summary has B = rmplus_avg and A = X's victim). Gain = X score - baseline, paired by seed (same teams per (seed, game) verified; the victim's seat differs between E4 and X, so the pairing holds at the pair level) and unpaired. Stream-cluster CIs of the paired gain: XU [0.013, 0.058], XUmv [0.007, 0.097], XEmv [-0.132, 0.026]. XR (victim = RM+ itself) has no E4 counterpart; expected 0.5 by seat symmetry.

| tag | exploiter vs victim | pairs | score [normal] / [stream-cl.] | baseline (RM+ vs victim) | gain paired [normal] / [cluster] | gain unpaired [CI] |
|---|---|---|---|---|---|---|
| XUmv | exploit_bandit_ucb1_mv vs bandit_ucb1_mv | 299 | 0.4582 [0.418, 0.498] / [0.411, 0.505] | 0.4064 (E4umv, n=299) | +0.0518 [-0.002, 0.106] / [0.007, 0.097] (n=299) | +0.0518 [-0.004, 0.107] |
| XU | exploit_bandit_ucb1 vs bandit_ucb1 | 298 | 0.9849 [0.975, 0.995] / [0.974, 0.995] | 0.9495 (E4u, n=297) | +0.0354 [0.014, 0.057] / [0.013, 0.058] (n=297) | +0.0354 [0.014, 0.057] |
| XR | exploit_rmplus_avg vs rmplus_avg | 300 | 0.4683 [0.431, 0.505] / [0.424, 0.513] | none (0.5 by symmetry) | -0.0317 [-0.069, 0.005] (vs 0.5) | n/a |
| XEmv | exploit_bandit_exp3_mv vs bandit_exp3_mv | 299 | 0.4548 [0.415, 0.494] / [0.399, 0.510] | 0.5068 (E4emv, n=294) | -0.0527 [-0.107, 0.002] / [-0.132, 0.026] (n=294) | -0.0527 [-0.109, 0.004] |

Prediction accuracy of the exploiter's opponent model: yhat is logged **top-5 only**, so log-loss is not computable; reported instead are (a) top-1 hit = the victim's played label is in the set of argmax-yhat labels (ties), (b) played label in the logged top-5, (c) logged yhat mass on the played label (a lower bound: 0 if outside the top 5), against chance 1/n_cols and top5/n_cols. The victim's played action = `move` in the B decision record at the same `turn` ('switch X' -> X); only turns with the same number of A and B records are paired (otherwise one side had no decision that turn), the rest discarded; only A decisions that carry `yhat` count.

| tag | aligned decisions (of exploit decisions; discarded frac.) | top-1 hit | in logged top-5 | mean logged mass on played (LB) | chance 1/n_cols | chance top5/n_cols | mean size of argmax tie set |
|---|---|---|---|---|---|---|---|
| XUmv | 13157 (20190; 0.35) | 0.373 | 0.373 | 0.340 | 0.099 | 0.393 | 1.22 |
| XU | 8080 (12695; 0.36) | 0.234 | 0.550 | 0.123 | 0.157 | 0.546 | 1.38 |
| XR | 14394 (21924; 0.34) | 0.420 | 0.750 | 0.392 | 0.101 | 0.411 | 1.02 |
| XEmv | 13116 (20110; 0.35) | 0.290 | 0.290 | 0.289 | 0.101 | 0.408 | 1.01 |

For the `_mv` victims (most-visited-arm rule) yhat is uniform over the argmax set, so top-5 coverage is complete and hit = in-support; for XU/XR yhat is the full share vector truncated to 5 entries. The discarded fraction (~35%) is turns where only one side had a decision (forced switch after a faint) or the counts differ; no per-record alignment was attempted for those.

## B4. E5 deficit decomposition (script `b4_e5.py` -> `out_B4_e5.json`)

Only `E5` (hedge vs agentform) contains Hedge records (weights, per-expert argmax labels `experts`, rewards). E5t (`top` vs agentform) and E5s (agentform vs agentform) are not Hedge methods: no weights/experts are logged, so they supply only the score split.

| run | A vs B | pairs | pair score [normal] / [stream-cl.] |
|---|---|---|---|
| E5 | hedge vs agentform | 300 | 0.4583 [0.421, 0.495] / [0.416, 0.501] |
| E5t | top vs agentform | 300 | 0.5500 [0.513, 0.587] / [0.523, 0.577] |
| E5s | agentform vs agentform (control) | 298 | 0.5185 [0.480, 0.557] / [0.465, 0.572] |

Paired by seed: E5_minus_E5t = -0.0917 normal [-0.145, -0.038] cluster [-0.138, -0.046] (n=300); E5_minus_E5s = -0.0621 normal [-0.117, -0.008] cluster [-0.116, -0.008] (n=298); E5t_minus_E5s = +0.0285 normal [-0.026, 0.083] cluster [-0.016, 0.073] (n=298).

Played action vs expert recommendations (E5, A side, 21914 decisions with >1 legal option): played move equals the argmax label of top 0.847, sampled 0.847 (= top, see anomaly 3), safest 0.774, safex 0.786. Mean played weights: top 0.345, sampled 0.266, safest 0.214, safex 0.175.
On the 5685 decisions (25.9%) where the top and safest experts disagree: played = top's move 0.597, = safest's move 0.313, neither 0.090; played = safex's argmax 0.496.

Score split by how top-like / safest-like a game's play was (terciles of the per-game statistic; 600 games; n per tercile ~200; game score 0/1):

| statistic | corr with game score | tercile 1 (low) | tercile 2 | tercile 3 (high) |
|---|---|---|---|---|
| mean weight on top | +0.015 | 0.455 (n=200) | 0.425 (n=200) | 0.495 (n=200) |
| fraction of decisions where played = top's move | +0.006 | 0.470 (n=200) | 0.431 (n=202) | 0.475 (n=198) |
| fraction played = safest's move | +0.113 | 0.415 (n=205) | 0.436 (n=195) | 0.525 (n=200) |

Weight trajectories (carried over 60 games/stream; mean over updates in the first 10%, 10-50%, last 50% of each stream, order top/sampled/safest/safex; and the expert with the highest cumulative reward):

| stream | updates | mean w first 10% | 10-50% | last 50% | final w | best cum. reward |
|---|---|---|---|---|---|---|
| 0 | 1781 | [0.27, 0.24, 0.264, 0.226] | [0.323, 0.258, 0.216, 0.203] | [0.358, 0.282, 0.198, 0.162] | [0.398, 0.279, 0.186, 0.137] | top |
| 1 | 1779 | [0.291, 0.259, 0.234, 0.215] | [0.344, 0.27, 0.214, 0.173] | [0.414, 0.265, 0.181, 0.14] | [0.429, 0.263, 0.179, 0.129] | top |
| 2 | 1896 | [0.322, 0.247, 0.25, 0.181] | [0.371, 0.262, 0.199, 0.168] | [0.447, 0.253, 0.168, 0.132] | [0.448, 0.254, 0.173, 0.125] | top |
| 3 | 1859 | [0.256, 0.256, 0.251, 0.238] | [0.34, 0.268, 0.216, 0.175] | [0.372, 0.271, 0.204, 0.154] | [0.385, 0.272, 0.192, 0.15] | top |
| 4 | 1685 | [0.269, 0.253, 0.252, 0.227] | [0.3, 0.276, 0.248, 0.176] | [0.372, 0.263, 0.228, 0.137] | [0.4, 0.262, 0.212, 0.126] | top |
| 5 | 1792 | [0.242, 0.22, 0.306, 0.232] | [0.3, 0.254, 0.253, 0.192] | [0.357, 0.269, 0.205, 0.169] | [0.406, 0.271, 0.182, 0.14] | top |
| 6 | 1776 | [0.305, 0.263, 0.228, 0.203] | [0.335, 0.279, 0.214, 0.172] | [0.353, 0.288, 0.198, 0.161] | [0.383, 0.286, 0.194, 0.137] | top |
| 7 | 1793 | [0.268, 0.269, 0.227, 0.236] | [0.329, 0.273, 0.202, 0.197] | [0.385, 0.269, 0.182, 0.164] | [0.428, 0.27, 0.15, 0.152] | top |
| 8 | 1882 | [0.264, 0.252, 0.247, 0.237] | [0.278, 0.264, 0.238, 0.22] | [0.356, 0.265, 0.203, 0.176] | [0.372, 0.27, 0.199, 0.159] | top |
| 9 | 1876 | [0.281, 0.257, 0.22, 0.242] | [0.3, 0.261, 0.24, 0.198] | [0.348, 0.262, 0.225, 0.166] | [0.376, 0.264, 0.204, 0.156] | top |

Numbers only: the top expert holds the highest weight and the best cumulative reward in all 10 streams, yet Hedge plays a mixture that follows top's move in 84.7% of decisions (77% safest); E5 (0.458) is below E5t (0.550) by -0.092 and below the control E5s (0.518) by -0.062, and E5t is not distinguishable from E5s (+0.029). The tercile splits show no monotone relation between top-likeness and score (the correlations are 0.015 and 0.006, i.e. null); the share of decisions equal to safest's move correlates +0.113 with game score. A split of the E5 deficit into channels (e.g. mixing weights vs expert quality) is not identifiable from these logs: per-decision Q is not logged for E5 (no OURBOT_LOG_ABAR) and E5t/E5s log no expert quantities.

## B5. bayes_safex eps sweep (script `b5_eps.py` -> `out_B5_eps.json`)

SX1/SX3/SX10/SX30 = OURBOT_SAFEX_EPS 0.001/0.003/0.01/0.03 (`launch_open_items.sh`, summaries' `modal.env`), seeds 1-200, vs stock. Reference: agentform vs stock = 0.3144 [0.2792, 0.3495] (n=299 pairs, `engine/showdown/results/AF/AF_summary.json`; 0.3291 on seeds 1-200 only).

| tag | eps | games W/L | pair score [normal] / [stream-cl.] | game score [stream-cl.] | minus agentform (seed-paired, seeds 1-200) [normal] / [cluster] |
|---|---|---|---|---|---|
| SX1 | 0.001 | 80/320 | 0.2000 [0.163, 0.237] / [0.166, 0.234] (n=200) | 0.2000 [0.166, 0.234] | -0.1281 [-0.183, -0.073] / [-0.190, -0.066] (n=199) |
| SX3 | 0.003 | 91/309 | 0.2275 [0.189, 0.266] / [0.195, 0.260] (n=200) | 0.2275 [0.195, 0.260] | -0.1030 [-0.157, -0.049] / [-0.164, -0.042] (n=199) |
| SX10 | 0.01 | 98/302 | 0.2450 [0.203, 0.287] / [0.204, 0.286] (n=200) | 0.2450 [0.204, 0.286] | -0.0829 [-0.137, -0.028] / [-0.144, -0.022] (n=199) |
| SX30 | 0.03 | 101/298 | 0.2538 [0.214, 0.294] / [0.213, 0.294] (n=199) | 0.2531 [0.212, 0.294] | -0.0758 [-0.136, -0.016] / [-0.124, -0.027] (n=198) |

Trend: game-level logistic regression of win on ln(eps) (n=1599 games, 40 clusters = tag x stream): slope +0.0873 per ln eps (odds ratio 1.223 per 10x eps), SE model 0.0466 (z=1.87), SE cluster 0.0352 (z=2.48, t-test p=0.018 (t with 39 df over tag x stream clusters; the same seeds/teams recur across tags within a stream, and clustering by stream only (10 clusters) gives SE 0.0363, p=0.040)), 95% CI of slope (cluster) [0.0161, 0.1586]. Fitted scores: eps 0.001: 0.206, eps 0.003: 0.222, eps 0.01: 0.240, eps 0.03: 0.258. Linearity in ln eps vs a free per-eps model: LR chi2(2) = 0.26, p = 0.88. Pair-level OLS of pair score on ln eps: slope +0.0157 (cluster SE 0.0062, p=0.016, CI [0.0031, 0.0283]), i.e. +0.036 per 10x eps. Kendall tau (game outcome vs eps) = 0.043 (naive p 0.061).
Seed-paired differences between eps levels: SX3_minus_SX1 +0.0275 normal [-0.026, 0.081] (n=200); SX10_minus_SX3 +0.0175 normal [-0.040, 0.075] (n=200); SX30_minus_SX10 +0.0075 normal [-0.051, 0.067] (n=199); SX30_minus_SX1 +0.0528 normal [0.002, 0.103] (n=199). Adjacent steps are within noise; only the SX30 - SX1 contrast excludes 0 (marginally). All four eps values score below agentform on the same seeds 1-200 (0.3291, n=199) by 0.076-0.128 (seed-paired CIs exclude 0); vs the full AF 0.3144 the gaps are 0.061-0.114. Other references: BH (hedge_bayes) 0.3329 [0.302, 0.364]; E2 (hedge) 0.3108 [0.280, 0.341]; AF16 (agentform) 0.3278 [0.292, 0.363]. Note SX3 (eps 0.003) is the same configuration as the BH hedge_bayes-safex default epsilon but BH is a Hedge mixture, not bayes_safex alone.

## B6. RQeq / RQsafe / RQunif: pre-registered predictions P1-P5 (scripts `b6_rq.py`, `b6_score.py` -> `out_B6_rq.json`, `out_B6_scores.json`)

Source: `bayes_dynamics.md` section 4 (pre-registered, committed fb3cc8e before the runs; not edited). Per stream (10 streams, 20 games, ~480-650 feedback updates each) the trajectory is rebuilt from `feedback.bayes` (p_qhat, p_style, posterior; non-skipped turns, forced updates included as the posterior includes them). T = feedback turns of the stream. All three RQ runs: 20 games per stream in the state files, T equals the state `updates` in every stream (all 30 streams clean).

| tag | opponent | score (pair) [normal] | P_T(true style) per stream | final MAP |
|---|---|---|---|---|
| RQeq | style_eq | 0.6700 [0.609, 0.731] | [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0] | {'safe': 10} |
| RQsafe | style_safe | 0.8550 [0.808, 0.902] | [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.945, 1.0, 1.0, 1.0] | {'safe': 10} |
| RQunif | style_uniform | 0.9948 [0.985, 1.005] | [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0] | {'uniform': 10} |

**P1 (universal identity regret = ln5 + ln P_T(best) = 1.609, anytime max <= 1.609 + rounding).** RQeq: anytime max over t and styles 1.609825 (ln5 = 1.609438, excess 3.9e-04), final-regret range 1.6085-1.6098, max |regret - (ln5 + ln P_T)| at the end 3.7e-04 over 10 streams with P_T >= 0.999; RQsafe: anytime max over t and styles 1.609594 (ln5 = 1.609438, excess 1.6e-04), final-regret range 1.5535-1.6095, max |regret - (ln5 + ln P_T)| at the end 2.4e-04 over 9 streams with P_T >= 0.999; RQunif: anytime max over t and styles 1.609522 (ln5 = 1.609438, excess 8.4e-05), final-regret range 1.6094-1.6095, max |regret - (ln5 + ln P_T)| at the end 8.4e-05 over 10 streams with P_T >= 0.999. Max over every turn x style (posterior >= 0.01) of the identity error: 0.0049 nats (4-dp posterior rounding). Result: **PASS within logging rounding** (excess over ln5 <= 3.9e-4, consistent with 6-dp rounding of p_qhat/p_style); read at strictly 3 decimals, final regret equals 1.609 in 18 of the 29 streams with P_T >= 0.999 (RQsafe stream 6, P_T = 0.945, is outside the clause); the excess over ln5 is within the worst-case 6-dp rounding bound in every stream, so the strict '1.609 to 3 decimals' statement is met only up to ~5e-4.
**P2 (style_uniform).** (a) P_T(uniform) >= 0.99 by feedback turn 50 in every stream: **FAIL** (8/10 streams; min P_50 = 0.513); (b) = 1.000 at the end: **PASS** (10/10, all >= 0.9995); (c) final MAP = uniform 10/10: **PASS**; (d) MAP-wrong turns <= 10 per stream: **FAIL** (5/10 streams; counts [0, 1, 6, 12, 13, 3, 42, 21, 0, 14]; after turn 50: [0, 0, 1, 0, 0, 0, 10, 0, 0, 0]). Scored on total MAP-wrong turns; under the reading 'MAP-wrong turns <= 10 after turn ~50' the clause passes 10/10 ([0,0,1,0,0,0,10,0,0,0]). (e) best style's log loss = E ln n_t: **PASS** (uniform mean loss 2.05306 vs mean ln n_t 2.05306; uniform is the best style in 10/10 streams; the pre-registered magnitude '~2.3-2.9 nats' is missed: 2.05 over all updates (2.29 excluding forced turns)); (f) score > 0.8: **PASS** (0.9948 [0.985, 1.005]). The doc's own table has sum_t(1 - P_{t-1}(s*)) = 5.5 for uniform; observed per stream [3.2, 7.1, 29.3, 14.2, 14.2, 8.5, 41.2, 23.1, 4.0, 17.7] (mean 16.2).
**P3 (style_safe).** Decision rule: mean per-turn log-loss gap br_unif - safe = 0.0480 nats/turn (per stream [0.046, 0.041, 0.062, 0.041, 0.073, 0.051, 0.006, 0.09, 0.028, 0.042]) < 0.1 => the **high-mismatch branch** applies by the pre-registered rule (br_us - safe gap 0.117). The rule as written states only '>= 0.1 => low-mismatch'; the observed 0.048 lies between the 0.1-Delta (margin 0.111) and 0.2-Delta (0.001) proxy rows, so neither pre-written outcome strictly covers it. High-mismatch branch ('MAP ends on br_unif/br_us with safe within a few nats'): **FAIL** -- final MAP = safe in 10/10 streams, P_T(safe) >= 0.9995 in 9/10 (stream 6: 0.9451), final loss gap to the best other style [21.8, 19.5, 36.2, 20.7, 42.2, 23.8, 2.8, 54.5, 11.4, 20.0] nats. Low-mismatch branch clauses (reported for completeness): identified within ~100 feedback turns: P_100 per stream [0.3864, 0.922, 0.3838, 0.904, 0.9998, 0.0004, 0.9717, 0.9993, 0.2632, 0.9986] (6/10 >= 0.9; first turn from which P>=0.9 holds to the end: [168, 109, 133, 85, 12, 291, 399, 80, 299, 42]); P_T(safe) = 1.000 at end: 9/10; MAP-wrong turns 10-50: 1/10 (counts [111, 10, 80, 53, 1, 273, 94, 4, 87, 2]); final gap to runner-up >= 60 nats: 0/10 (max 54.5); point prediction 8-10/10 end with MAP = safe: **PASS** (10/10). Score 0.8550 [0.808, 0.902] vs speculative 0.6-0.75: above the range (**outside**).
**P4 (style_eq).** (a) final MAP in {safe, br_us, br_unif}: **PASS** (10/10, all 'safe'); (b) **P_T(eq) < 0.1 in >= 7/10 streams: PASS, 10/10** (P_T(eq) = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], i.e. < 5e-5 in all; mean P(eq) at turn 25 = 0.000, at 100 = 0.000); none >= 0.9 (0); (c) eq's mean log loss exceeds safe's: **PASS** (2.202 vs 1.616 nats/turn; true for 10/10 streams; all styles {'eq': 2.202, 'br_unif': 1.684, 'br_us': 1.695, 'safe': 1.616, 'uniform': 2.23}); (d) score 0.50-0.60 (speculative): 0.6700 [0.609, 0.731] -- **above the range** (lower CI bound 0.609 just above 0.60). Note eq is also worse than br_unif and br_us and (2.20) is about as bad as uniform (2.23).
**P5 (ordering uniform > safe > eq).** Final P_T(true style) (mean over streams): RQeq 0.000, RQsafe 0.995, RQunif 1.000 -> order holds: **True**. Identification speed as sum_t(1 - P_(t-1)(s*)) (mean): RQeq 589.0, RQsafe 78.5, RQunif 16.2 -> order holds: **True**; median first turn from which P(s*) >= 0.9 persists: uniform 24.5, safe 121.0, eq never (10/10 streams). Scores: RQeq 0.6700, RQsafe 0.8550, RQunif 0.9948 -> order holds: **True** (CIs: see table above; unif vs safe and safe vs eq separated by >2 SE of a pair mean ~0.03). P5: **PASS**.

Scoreboard: P1 PASS (to rounding) | P2 partial (4 of 6 clauses pass: end P=1.000, final MAP 10/10, loss = E ln n, score > 0.8; the 'P >= 0.99 by turn 50 in every stream' (8/10) and 'MAP-wrong turns <= 10 per stream' (5/10) clauses fail) | P3 high-mismatch branch FAIL, low-mismatch clauses: identified within ~100 turns 6/10, P_T(safe)=1.000 9/10, MAP-wrong 10-50 1/10, gap >= 60 nats 0/10; point prediction (MAP = safe in 8-10/10) PASS 10/10; score 0.855 above the speculative 0.6-0.75 | P4 PASS (10/10 streams with P_T(eq) < 0.1, required >= 7; speculative score 0.50-0.60 missed: 0.670 [0.609, 0.731]) | P5 PASS.

## Reproduction

`cd engine/analysis/open_items && ../../.venv/bin/python a_audit.py; b1_log_identity.py; b1_aggregate.py; b2_switch.py; b3_exploiters.py; b4_e5.py; b5_eps.py; b6_rq.py; b6_score.py; build_report.py` (shared helpers `oi_common.py`; reuses `engine/analysis/synthesis/common.py`). Run time: a few minutes total, B1 ~3 min (LP per decision).
