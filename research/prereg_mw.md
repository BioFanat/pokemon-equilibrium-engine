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
