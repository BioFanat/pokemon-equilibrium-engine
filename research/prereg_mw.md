# Pre-registration: membership-aware style posterior (`hedge_bayes_mw`) in live Showdown play

Written: 2026-11-16, before any `*_mw` game other than three single-game smoke tests (deleted, not analysed beyond
"no errors"). The predictions in §3 are not edited after launch; results go in an appended "Results" section.

## 1. What is tested

`hedge_bayes_mw` (commit on `integration/joint-belief`) differs from `hedge_bayes` in two places:
- the style posterior's likelihood and prediction use each sampled world's own matrix and columns
  (`behavior.predict_worlds`, `lik_worlds`, `WorldPosterior`) instead of the world-averaged matrix over the union of
  labels;
- its exploit expert solves the per-world safe-exploit LP (`hidden_agents.safex_strategy_worlds`) instead of the
  averaged-matrix `nash.safex_strategy`.

Evidence before the run (`research/joint_posterior.md`, independently verified 2026-11-16): on 18 logged games, the
membership-aware predictor's next-action log loss is 0.324 nats lower than the averaged predictor's (1.623 → 1.299,
± 0.031 game-clustered, 372 turns vs stock). The averaged predictor puts a mean 0.25 of its mass on labels absent from a
given sampled world.

## 2. Runs (local, after the G1 gauntlet; `batch.py --k 4 --search-time-ms 100 --streams 10`)

| tag | --a | --b | seeds | control |
|---|---|---|---|---|
| MWs | hedge_bayes_mw | stock | 1-200 | BHs (same seeds, same machine) |
| BHs | hedge_bayes | stock | 1-200 | — |
| MWeq | hedge_bayes_mw | style_eq | 1-100 | RQeq (same seeds and stream layout; earlier run) |

Fresh `OURBOT_WORK` per arm (no carried state files). Scored turns: feedback records that are not skipped and have
`feedback.bayes`, paired with their decision by record order (never by turn number).

## 3. Predictions (with refutation criteria)

- **L1 (within-run, primary).** In MWs, the paired per-turn difference
  $-\ln p_{\text{avg}} - (-\ln p_{\text{mw}})$ (both logged pre-update for the same realised label, under the MW
  posterior) has mean in [0.25, 0.40] nats, with a game-clustered 95% CI excluding 0.
  Refuted if the mean is < 0.15 or the CI includes 0.
- **L2 (across runs).** The MW predictor's mean log loss in MWs is lower than the averaged predictor's in BHs by
  [0.20, 0.45] nats (different trajectories; unpaired, game-clustered). Refuted if < 0.10.
- **L3 (vs eq).** In MWeq, L1's paired difference has mean in [0.25, 0.45]. Refuted if < 0.15.
- **R1 (realisability, vs eq).** The research found that, per world, the eq style still trails safe against the live
  eq bot (log loss 1.355 vs 1.265, 18 games), while one smoke game ended with posterior eq = 1.0. Prediction from the
  larger evidence: the final MAP style is eq in at most 4 of 10 MWeq streams (RQeq: 0 of 10).
  Refuted if eq is the final MAP in 8 or more streams.
- **W1 (win rate, descriptive).** Paired score difference MWs − BHs on the same seeds in [−0.03, +0.03]. At 200 pairs the
  95% half-width is about ±0.044, so no significant difference is expected. Refuted only if the CI excludes 0 in the
  negative direction (MW worse).
- **W2 (vs eq, descriptive).** MWeq vs RQeq paired score difference in [−0.05, +0.05].
- **C1 (cost).** CPU per decision of hedge_bayes_mw within 10% of hedge_bayes on the same machine (microbench: +7 ms on
  ≈ 0.4 s).

## 4. Analysis

Script to be written before the runs finish; it reads only the fields named above. Game-clustered SEs (games as
clusters), stream bootstrap for stream-level quantities. Error games are reported and dropped by seed pair, as in
batch.py.

## Addendum A (2026-11-16, written after MWs/BHs were scored, before any replication game)

Observed on seeds 1-200 (research/out_mw.md): W1 = +0.0625 [0.0032, 0.1218], outside the predicted [-0.03, +0.03]
(no detectable effect was predicted); L1, L2 passed. Because the win-rate gain was not predicted and its CI only just
excludes 0, it is treated as a hypothesis, tested on fresh seeds:

| tag | --a | --b | seeds | notes |
|---|---|---|---|---|
| MWs2 | hedge_bayes_mw | stock | 201-400 | run concurrently with BHs2 |
| BHs2 | hedge_bayes | stock | 201-400 | |

- **W1-rep.** Paired MWs2 - BHs2 on seeds 201-400: point estimate > 0. Supported if the one-sided 95% interval
  (mean - 1.645 SE) is > 0; refuted if the point estimate is <= 0. The pooled 400-seed estimate is reported alongside
  but is not the test (seeds 1-200 selected the hypothesis).
- **L1-rep.** L1 on MWs2 in [0.25, 0.40] (should replicate; refuted if < 0.15).

## Results (appended 2026-11-16; predictions above unchanged)

Runs MWs, BHs (seeds 1-200), MWeq (1-100), all local, 0 error games. Full table: `research/out_mw.md` (+ `.json`), from
`research/scripts/joint/analyze_mw.py`.

| P | observed | predicted | verdict |
|---|---|---|---|
| L1 | 0.297 nats/turn [0.281, 0.314] | [0.25, 0.40] | pass |
| L2 | 0.270 [0.233, 0.307] | [0.20, 0.45] | pass |
| L3 | 0.2499 [0.224, 0.276] | [0.25, 0.45] | not refuted (< 0.15 refutes); point estimate 1e-4 below the interval |
| R1 | eq is the final MAP in 0/10 MWeq streams | at most 4/10 | pass: realisability still fails with per-world styles |
| W1 | +0.0625 [0.003, 0.122] (iid); stream-clustered t(9) [-0.011, 0.136] | [-0.03, 0.03] | not refuted, interval missed; not robust (`mw_w1_verification.md`); replication pending (Addendum A) |
| W2 | +0.095 [0.014, 0.176] vs RQeq | [-0.05, 0.05] | descriptive; interval missed; confounded: RQeq ran on Modal, MWeq locally (same confound as BH vs BHs) |
| C1 | +2.0 % CPU per decision | within 10 % | as predicted |

## Replication results (Addendum A; appended 2026-11-16)

MWs2 / BHs2, seeds 201-400, run concurrently, 0 error games.

| P | observed | criterion | verdict |
|---|---|---|---|
| W1-rep | MWs2 0.3425 vs BHs2 0.3350: +0.0075, iid SE 0.0316, one-sided lower bound -0.045; stream-clustered SE 0.0358 (lower -0.058); stream sign-flip one-sided p = 0.45; 5 of 10 stream means positive | supported if one-sided lower > 0; refuted if point <= 0 | **not supported** (point estimate positive, so not formally refuted; indistinguishable from 0) |
| L1-rep | 0.311 nats/turn [0.291, 0.331] (game-clustered), 11,529 turns | [0.25, 0.40]; refuted if < 0.15 | pass (replicates) |

Pooled over seeds 1-400 (reported alongside, not the test): +0.035, iid 95% [-0.008, +0.078], stream-clustered
t(9) [-0.010, +0.080], 400 seed pairs.

Reading: the prediction gain replicates (0.297 then 0.311 nats/turn); the win-rate gain does not. BHs2 (0.335) is in
line with the earlier Modal runs of the same method (0.318-0.348), so BHs (0.2825) on seeds 1-200 was a low draw, as
the verification suspected. Conclusion: membership-aware style prediction improves next-action prediction by about
0.3 nats per decision and changes the hedge's behaviour, with no detectable effect on win rate against stock
foul-play, matching the original pre-registered W1 and the earlier pattern (better prediction, same win rate).
