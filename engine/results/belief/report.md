# Hidden-set belief: accuracy and calibration (2026-11-15)

Two beliefs over the opponent's hidden sets are scored against the truth. This is **not** a paired comparison:
simulator, generator and opponent all differ. Each row measures one belief in its own setting.

| | foul-play (real Showdown) | ours (`hidden.py`, engine) |
|---|---|---|
| prior | empirical full-set counts from pkmn.github.io/randbats (5,802 full sets); sharp, close to Showdown's generator | `randbats.sample_set`: role uniform, then 4 moves uniform from the pool, then ability/item/tera uniform. In-engine this IS the true generator |
| evidence | revealed moves/item/ability/tera, speed ranges, damage/choice-item inference, item removal (`get_all_remaining_sets`) | legality only: revealed moves (pool membership) and revealed tera. Item and ability are never revealed in-engine; Belief does not use `item_gone` |
| bench | uniform species with soft team-limit rejection | uniform over unused bases (`sample_unrevealed`) |

In-engine our belief is the exact posterior of the generator that drew the truth (up to the Tracker's approximations and
A4), so its calibration checks the implementation; its accuracy is limited by the generator's spread, not by inference.

## Data
- Showdown: 40 paired seeds (1001–1040), 80 games, agent form (K=4) vs stock foul-play, 0 errors; 3,064 of our decisions,
  8,028 decision × revealed-alive-mon pairs. CIs resample the 80 games.
- Engine: 20 paired seeds, 40 games (agentform:4:200 vs pimc:4:200), 0 errors, 3,700 decision nodes; both sides' beliefs
  scored. CIs resample the 20 team draws (g1/g2 share both teams).
- Event = one decision × one revealed, alive opponent mon × one attribute still unknown (moves: one event per true
  unrevealed move).

## Main table
Accuracy = argmax equals truth (moves: true move among the 4−k most probable unrevealed). Log loss = −ln max(P(truth),
1e-3). ECE: 10 bins; "distinct" counts each distinct belief state once per team draw; "null" = ECE a perfectly
calibrated belief with the same states would get (mean/p95; treats states as independent).

| attribute / belief | events | mean P(truth) | accuracy [95% CI] | log loss [95% CI] | Brier | ECE (dec.) | ECE (distinct) | null mean/p95 | P(truth)=0 |
|---|---|---|---|---|---|---|---|---|---|
| item: foul-play exact | 4630 | 0.839 | 0.838 [0.780, 0.891] | 0.257 [0.193, 0.324] | 0.170 | 0.038 | 0.042 | 0.029/0.042 | 0.002 |
| item: foul-play K=4 worlds | 4630 | 0.816 | 0.823 [0.774, 0.869] | 0.614 [0.396, 0.893] | 0.253 | 0.125 | 0.164 | 0.012/0.022 | 0.062 |
| item: ours (engine) | 9920 | 0.739 | 0.740 [0.680, 0.794] | 0.427 [0.361, 0.502] | 0.273 | 0.044 | 0.055 | 0.024/0.038 | 0 |
| ability: foul-play exact | 6602 | 0.969 | 0.974 [0.956, 0.990] | 0.045 [0.023, 0.072] | 0.032 | 0.016 | 0.020 | 0.022/0.037 | 0 |
| ability: foul-play K=4 | 6602 | 0.970 | 0.971 [0.953, 0.986] | 0.077 [0.037, 0.128] | 0.038 | 0.020 | 0.067 | 0.009/0.019 | 0.006 |
| ability: ours (engine) | 9920 | 0.956 | 0.948 [0.922, 0.970] | 0.062 [0.034, 0.093] | 0.045 | 0.007 | 0.019 | 0.020/0.031 | 0 |
| tera: foul-play exact | 7173 | 0.610 | 0.633 [0.578, 0.686] | 0.627 [0.569, 0.687] | 0.390 | 0.025 | 0.025 | 0.020/0.033 | 0.001 |
| tera: foul-play K=4 | 7173 | 0.613 | 0.620 [0.586, 0.652] | 1.127 [1.014, 1.249] | 0.480 | 0.158 | 0.229 | 0.007/0.018 | 0.096 |
| tera: ours (engine) | 9407 | 0.586 | 0.590 [0.503, 0.662] | 0.677 [0.568, 0.798] | 0.423 | 0.029 | 0.027 | 0.019/0.029 | 0 |
| move: foul-play exact | 19083 (7395 mon-dec) | 0.815 | 0.847 [0.821, 0.870] | 0.273 [0.245, 0.299] | 0.109 | 0.013 | 0.021 | 0.011/0.015 | 0.001 |
| move: foul-play K=4 | 19083 | 0.816 | 0.820 [0.803, 0.839] | 0.447 [0.399, 0.496] | 0.141 | 0.089 | 0.129 | 0.004/0.007 | 0.031 |
| move: ours (engine) | 24786 (9219 mon-dec) | 0.670 | 0.683 [0.630, 0.727] | 0.480 [0.414, 0.548] | 0.181 | 0.031 | 0.028 | 0.015/0.021 | 0 |

"foul-play exact" = count-weighted distribution over foul-play's remaining sets (what its worlds are drawn from);
"K=4" = the 4 worlds our agent actually solves over, each weight 1/K.

## Findings
1. **foul-play's posterior is accurate and well calibrated on real Showdown**: item 0.84, ability 0.97, unrevealed moves
   0.85, tera 0.63; ECE 0.01–0.04, near the null; truth outside its support ≤ 0.2%.
2. **The K=4 worlds our Showdown agent solves over are poorly calibrated**: ECE 0.09–0.16 vs 0.01–0.04 for the posterior
   they are sampled from; the true value is in no world 3–10% of the time (tera 9.6%, item 6.2%); log loss 1.6–2.4×
   higher. Mean P(truth) is unchanged (unbiased samples); the 1/4 steps make them overconfident.
3. **Our tracker is calibrated within noise in-engine** (ECE 0.007–0.055; closed-form marginals match Monte Carlo from
   `Belief.sample` within 0.009). Lower accuracy than foul-play (item 0.74, moves 0.68, tera 0.59) reflects the engine
   generator's spread, not worse inference. Per-decision weighting carries a survival selection effect neither belief
   models.
4. **Sharpening**: foul-play item accuracy 0.77 → 0.87 and log loss 0.40 → 0.18 from turns 1–5 to 31+ (k = 0 → 2 moves
   revealed: 0.77 → 0.93); ours item 0.71 → 0.88, log loss 0.50 → 0.22 from k = 0 to 3. Ours, remaining-move accuracy
   drops 0.70 → 0.51 from k = 0 to 3 (the last move is one draw from n−3 pool moves: a harder target, not lost
   information). Per-cell n in `by_turn`/`by_k`.
5. **Bench belief is near-uninformative for both:**

| bench | events | mean P(true species present) | uniform | ratio | P = 0 | log loss |
|---|---|---|---|---|---|---|
| foul-play, 32 extra worlds | 5727 | 0.0077 | 0.0064 | 1.20 | 0.787 | 6.15 |
| foul-play, K=4 worlds | 5727 | 0.0081 | 0.0064 | 1.27 | 0.968 | 6.73 |
| ours (`sample_unrevealed`) | 5964 | 0.0063 | 0.0063 | 1.00 | 0 | 5.17 |
| exact generator conditional | 5964 | 0.0063 | 0.0063 | 1.00 | 0 | 5.17 |

## A4 at scale (game_model.md §6 item 2)
`sample_team` fills slot i uniformly over keys whose base is unused in earlier slots; the tracker's sampler fills
unrevealed slots in index order over unused bases. Both conditionals are exchangeable within forme-count classes, so the
TV between their bench-species marginals is computed exactly (toy case reproduced: (0.375, 0.375, 0.25) vs uniform;
exact vs Monte Carlo agree within 3.6e-5 on 40 configs). Universe: 509 keys, 421 bases (54 with >1 forme). Over the 287
revealed configurations seen in play: prefix configurations TV = 0 exactly; non-prefix mean 0.0020; all: mean 0.0015,
median 0.0013, p95 0.0045, max 0.0061. **The A4 species error is real but negligible.**

## Caveats
Not paired; few independent units (80 games, 20 team draws: use the cluster CIs); truth is the start-of-game set;
already-public attributes excluded; dead mons excluded; Showdown species matching uses a forme-prefix fallback
(0 unmatched; could confuse mew/mewtwo, unseen); logging adds CPU per decision, so BELIEF timings and its pair score
(0.275) are not performance numbers; log-loss clipping at 1e-3 matters only for K=4 and bench rows.

## Files
`showdown/run_ourbot.py` (`belief_log`, `OURBOT_LOG_WORLDS=1`), `showdown/score_belief.py` → `showdown.json`;
`score_belief_engine.py` (`play`/`score`/`a4`/`check`) → `engine.json`, `a4_tv.json`, `engine_*.jsonl`;
`comparison.json`; `figures/fig_belief_calibration.{png,pdf}`.
