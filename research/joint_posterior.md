# Joint posterior over opponent sets and decision style: is it worth building?

Ideation sprint, 2026-11-16. Code read in `/Users/biofanat/Documents/classes/f26/cs535/final-integration`
(branch `integration/joint-belief`). Scripts in `research/scripts/joint/`. Nothing under `engine/` was edited.

## Corrections (independent verification, 2026-11-16)

An independent verifier re-derived sections 3.1-3.5 from the raw logs. The prediction numbers (items 1-5) are
recomputed with the corrected `research/scripts/joint/analyze_worlds.py` (row dumps
`research/analyze_worlds_{stock,styleeq}.json`; `--pool` for the pooled rows). Item 6 was rerun with the verifier's
`rq_support.py` and item 7 with its `bench2.py` (session scratchpad `verify_joint/`).

1. **Tera label mismatch (bug).** The averaged-matrix predictors were scored on the tera-stripped label while the
   per-world predictors were scored on "x-tera". Every predictor is now scored on the one label the deployed
   `map_column` returns (`engine/showdown/run_ourbot.py:121-128`); the script asserts it equals the logged
   `feedback.bayes.col` on every row.
2. **Turn-keyed feedback join (bug).** Feedback was keyed by `for_turn`, and turn numbers repeat. It is now joined by
   record order (the feedback in record i+1 belongs to decision i), and decisions are matched to world lines on labels,
   row count and the reproduced Abar. Check (printed by the script): the deployed posterior replayed over this join
   from the logged `qs` matches the logged posterior to mean TV 0.0001 on both opponents. Scored turns: 372 vs stock (was 336), 219 vs
   style_eq (was 192).
3. **Support fix, corrected.** Current 1.623 vs membership-aware 1.299 nats: +0.324 ± 0.031 game-clustered
   (± 0.036 seed-clustered), 372 turns vs stock. The old "current" baseline (1.564) was 0.04 below the logged loss
   because of the two bugs above (not separated), not because of the reference mixture: on the corrected rows the
   recomputed q-hat equals the logged one to 4 dp.
4. **"Correlation term" was not zero, and was not a correlation term.** The per-world style vs averaged-matrix style
   gain is +0.026 ± 0.011 vs stock (G = 10 games), +0.032 ± 0.023 vs style_eq (G = 8), pooled +0.028 ± 0.011
   (G = 18), favouring per-world. The predictor called "per-world joint" is a factorised posterior with per-world
   likelihoods; it is renamed **per-world style likelihood**. A genuinely joint (active set × style) predictor carried
   across turns is now measured (3.1, "true joint"): it adds +0.001 ± 0.001 pooled, nothing measurable.
5. **Game count.** 18 games, not 19: 10 vs stock and 8 vs style_eq.
6. **Claim 4 denominator.** 36.9 % of the eq bot's actions outside the eq support is out of 5,218 multi-option turns;
   it is 32.6 % of all 5,904 scored turns.
7. **CPU cost.** Per-world `style_dists` costs about 4× the averaged-matrix call (1.0 vs 3.9 ms per decision on the
   union Abar, ratio 3.8; the verifier measured 1.2 vs 4.7 ms under heavier load), not 2.2 vs 9.8 ms (the old
   benchmark used `np.mean` of the world matrices, not the union Abar).
8. **HBA.** The link to HBA (Albrecht & Ramamoorthy 2013) is structural only. The toy example (3.3) shows
   explaining-away between set and style, not HBA's result on correlated beliefs across players.
9. **§3.3 wording.** The "product" tracker is not "exactly what the current pipeline does": foul-play keeps no soft set
   belief at all, only hard set filters (section 2). The product tracker is a hypothetical two-filter design.

## 1. Question

The planner samples K = 4 complete opponent teams ("worlds") with foul-play's sampler
(`engine/showdown/run_ourbot.py:416`), builds one depth-2 payoff matrix M_k per world and solves the agent-form LP
with equal world weights (`engine/hidden_agents.py:36`). Separately, `engine/showdown/behavior.py` keeps a Bayesian
posterior over 5 opponent styles computed on the world-AVERAGED matrix Abar (`run_ourbot.py:132 avg_matrix`).
Should the two beliefs be one joint posterior w[k, σ] over (team, style), as the external plan (workstreams C/D/E of
`final/notes/pokemon_remaining_implementation_plan.txt`) proposes? Does correlation matter (a) for predicting the
opponent's next action and (b) for our strategy? What is the cheapest faithful version, and can anything make the
style model realisable against the equilibrium opponent?

Objective metrics: prequential log loss of the opponent's next action (nats per scored turn), and the paired
Showdown pair score. Constraints: ≤ 2 local cores, no project-code edits, exact where it is cheap.

## 2. Facts established by reading the code (not the summary)

* **foul-play's sampler is stateless and uses hard filters only.** `prepare_random_battles`
  (`~/sandbox/foul-play/fp/search/random_battles.py:34-63`) draws each revealed mon's set independently per call,
  with replacement, weighted by the randbats usage count, and returns `chance = 1 / num_battles` for every world
  (`random_battles.py:61`) — a constant, not a likelihood. `get_all_remaining_sets`
  (`fp/data/sets/randbats.py:95-124`) filters on revealed moves, item, ability, speed range, level, tera; the
  damage-based reverse calc is computed on a copy and discarded for randbats (`fp/battle/inference.py:465-545`,
  `fp/modes/random_battle.py:214-218`). There is no soft per-set weight anywhere. Consequence: a style-conditioned
  action likelihood cannot double count anything foul-play does; and "importance weights on foul-play's samples"
  cannot be indexed by world — worlds do not persist across turns. They can be indexed by set: each decision's
  `belief.worlds[k].opp[0]` (`OURBOT_LOG_WORLDS=1`, same order as the matrices) names the active's sampled set
  (species, item, ability, tera type, moves), which is what the true-joint predictor in 3.1 keys on.
* **Unrevealed teammates are sampled uniformly over species and uniformly over that species' sets**
  (`random_battles.py:86-87`), not count-weighted. That is a prior mismatch of its own, independent of this question.
* **The style predictor normalises over the label union, not the opponent's options.** `avg_matrix` builds Abar over
  the union of the K worlds' column labels (first-seen order). Measured on 372 scored turns vs stock: the union has
  13.8 labels on average, a world has 8.1 columns. Every style distribution q_σ is a distribution over 13.8 labels of
  which the opponent actually has 8.1. This "dilution" turns out to be the dominant effect below.
* **Feedback skips 16-28 % of turns** (`run_ourbot.py:364-381`): BH 17.4 % (not_in_worlds 2168 of 4715 skips),
  RQeq 18.7 %, RQsafe 22.7 %, RQunif 27.9 %. All logged log losses are over the remaining turns.
* **The RQ/LOG/BH decision logs are not under `~/sandbox/ourbot/replays_headless/`** (that directory holds E1, demo,
  smoke and the running G1_ gauntlet). They are under
  `/Users/biofanat/Documents/classes/f26/cs535/final/engine/showdown/results/<TAG>/replays_headless/`.
  No run logs per-world matrices or per-world column sets: RQ records carry `bayes.{posterior, labels, q, qhat}`
  and `feedback.{b, how, world, bayes.{j, n_cols, p_qhat, p_style, ...}}`; LOG adds `abar, labels, legal, pfull,
  Q, qs` (all on the averaged matrix). `feedback.world` is the lowest-index world containing the label
  (`map_column`, `run_ourbot.py:121-128`), not evidence. So the per-world analysis needed a new logging run.
* **The eq style bot is not realisable from our side by construction.** `style_strategy`
  (`run_ourbot.py:220-230`) plays the LP equilibrium of ITS averaged matrix over ITS sampled worlds of OUR hidden
  sets; our `eq` style is the LP equilibrium of OUR Abar. The docstring says so ("Expected mismatch").

## 3. Experiments and findings (measured unless marked)

### 3.1 Per-world logging run (new data)

`research/scripts/joint/headless_logworlds.py` wraps `engine/showdown/headless.py`, monkeypatching
`run_ourbot.avg_matrix` to append `{Ms, cols, w, labels}` per decision to a side file (no project file edited;
`OURBOT_WORK` pointed at the scratchpad so nothing touched `~/sandbox/ourbot`). Runs: `hedge_bayes` vs `stock`,
seeds 1-5, one stream (posterior carried), 10 games, 372 scored turns; and `hedge_bayes` vs `style_eq`, seeds 1-4,
8 games, 219 scored turns (3.4). Raw outputs: scratchpad `jp_work/` (stock) and `jp_work/eq/` (style_eq); the row
dumps are `research/analyze_worlds_stock.json` and `research/analyze_worlds_styleeq.json`. Game JSONs carry the
true teams (`OURBOT_LOG_WORLDS=1`), so this output is already the frozen-observation benchmark format of section 6.

`research/scripts/joint/analyze_worlds.py` then recomputes, for every scored turn (feedback joined by record order;
label = the deployed `map_column` label, the same for every predictor) and every world k that contains the observed
action b: q_σ(b | M_k) for the five styles (`behavior.style_dists` on the world matrix), and five predictors of b.
P(σ) is the logged deployed posterior unless stated.

| predictor | definition | vs stock (372 turns) | vs style_eq (219) | pooled (591) |
|---|---|---|---|---|
| current (averaged matrix) | logged deployed q-hat: P(σ)-mixture of q_σ on Abar over the label union | **1.623** | 1.623 | 1.623 |
| membership × averaged style | same q-hat, restricted to each world's columns and renormalised, averaged over worlds | **1.299** | 1.296 | 1.298 |
| per-world style likelihood | Σ_k (1/K) Σ_σ P(σ) q_σ(b \| M_k) (factorised: one style posterior, per-world likelihoods) | **1.273** | 1.264 | 1.270 |
| option (iii) end to end | per-world style likelihood with a posterior updated by the per-world marginal likelihood | **1.269** | 1.274 | 1.271 |
| true joint (active set × style) | Σ_k (1/K) Σ_σ P(σ \| s_k) q_σ(b \| M_k), P(σ \| s) carried across the game's turns (below) | **1.266** | 1.274 | 1.269 |

Paired differences, mean ± game-clustered s.e. (CR1; G = 10 / 8 / 18 games):

| difference | vs stock | vs style_eq | pooled |
|---|---|---|---|
| current − membership (support fix) | **+0.324 ± 0.031** | +0.327 ± 0.053 | +0.325 ± 0.027 |
| membership − per-world style likelihood | **+0.026 ± 0.011** | +0.032 ± 0.023 | **+0.028 ± 0.011** |
| per-world (logged posterior) − option (iii) | +0.005 ± 0.017 | −0.010 ± 0.007 | −0.001 ± 0.011 |
| option (iii) − true joint (set-conditional style) | +0.002 ± 0.002 | −0.001 ± 0.001 | +0.001 ± 0.001 |

Seed-clustered s.e. (S = 5 / 4 / 9) are similar: support fix ± 0.036 / 0.032 / 0.024; per-world style
± 0.015 / 0.023 / 0.012.

**True joint predictor (option (i)/(ii) of section 4, measured offline).** Worlds are resampled every turn, but each
decision logs the active's sampled set in every world, so within a game a weight table w[mon, set key, σ] can be
carried across turns. Per scored turn, for the active mon X, each set key s sampled among the worlds containing b
gets the factor mean_{k: s_k = s} q_σ(b | M_k); keys of X already in the table but not sampled this turn get the
per-world marginal m_t(σ) = mean_{k ∋ b} q_σ(b | M_k) imputed. Prediction uses
P(σ | X = s) ∝ P_start(σ) · Π_{turns of other mons} m_t(σ) · w[X, s, σ], with P_start the option (iii) posterior at
game start; the table is updated after scoring. A key never seen before therefore gives exactly option (iii); with the
key collapsed to the species, the predictor reproduces option (iii) to 2e-15 on every row (checked on both
opponents). Switch actions are
attributed to the active's set. 57 % (stock) / 75 % (style_eq) of scored turns have ≥ 2 distinct active sets among
the containing worlds, yet the set-conditional style posterior differs from the marginal one by mean TV 0.016 / 0.022
(p90 0.035 / 0.079), and the predictive gain is +0.001 ± 0.001 nats pooled.

Reading: the support fix (predict over the opponent's actual options instead of the union) is the dominant gain,
+0.32 nats. Computing the style on each world's own matrix instead of the average adds a small, significant
+0.028 ± 0.011 nats pooled; this is a matrix-mismatch effect inside a factorised posterior, not set × style
correlation. The genuine correlation term — carrying a set-conditional style belief across turns — is
+0.001 ± 0.001, nil at this sample size. Updating the posterior with the per-world likelihood (option (iii)) instead
of the deployed averaged-matrix one changes nothing measurable (−0.001 ± 0.011).
Per style, per world vs averaged (stock): eq 1.48 vs 2.37, br_unif 1.28 vs 1.62, br_us 1.39 vs 1.79, safe 1.26 vs
1.62, uniform 1.92 vs 2.29. The eq style is the one most damaged by the union (the LP puts support on phantom columns).

### 3.2 Does the observed action re-weight the worlds?

w_k ∝ Σ_σ P(σ) q_σ(b | M_k) over the worlds containing b (98 % of scored turns have > 1 such world):
TV distance from uniform mean 0.056, median 0.031, p90 0.128; ESS/n 0.97; max/min likelihood ratio median 1.20,
p90 2.29. Re-solving the agent-form LP with the re-weighted worlds moves x by L1 0.073 on average (median 0.000)
and changes the argmax on 2.2 % of turns. On 62 % of turns the containing worlds carry ≥ 2 distinct movesets for
the active, so there is something to learn, but one action is weak evidence between them.

Accumulated over a game (option (i)): for pairs of movesets of the same active that were co-sampled on ≥ 2 turns
(n = 35 pairs, median 2 co-sampled turns), the accumulated |log-likelihood ratio| is median 0.27, p90 1.00 nats;
9 % of pairs reach ln 3. (Approximate: the moveset key is the non-tera labels among the world's first eight columns,
which after terastallization includes switches; the conclusion rests on the per-turn ratios, not on this stat.) With
per-turn ratios of 1.20 (p90 2.3), a mon that stays in for 5-8 turns accumulates roughly 0.5-1.5 nats between its
candidate movesets — an estimate, bounded above by the per-turn evidence.

### 3.3 Exact hand example (`research/scripts/joint/toy_joint.py`)

Sets S1 = {A, B}, S2 = {A, C}; styles greedy (0.9 on its best column vs the public reference mixture) and uniform;
priors 1/2. The "product" tracker is a hypothetical design that runs two separate Bayes filters (set belief with
Σ_σ P(σ) q_σ(obs|S), style belief with Σ_S P(S) q_σ(obs|S)) and multiplies. It is not what the current pipeline
does: foul-play keeps no soft set belief, only hard set filters (section 2), and only the style filter is Bayesian.
Payoff to us (rows r1, r2; columns A, B, C) = [[-2, -2, 2], [-1, -2, -2]], observations A, A (A is in both sets, so
the set is never identified):

* joint P(S, σ): (S1, greedy) 0.008, (S1, unif) 0.189, (S2, greedy) 0.614, (S2, unif) 0.189;
  product: 0.088, 0.067, 0.481, 0.364.
* next action: joint A 0.742 / B 0.102 / C 0.156; product A 0.657 / B 0.113 / C 0.230 (TV 0.085).
* best reply: joint r2, product r1; the product's reply loses 0.118 under the joint belief.

This is explaining-away between set and style within one opponent, not HBA's result on correlated beliefs across
players. Condition for correlation to matter: an action that is in several candidate sets but is the greedy choice
in only some of them (then style and set become dependent given the action). Membership-only evidence (an action in
one set only) never creates correlation — the posterior factorises after it. In randbats the second case dominates:
revealed moves are hard evidence, and among the remaining candidate movesets a 2-dp depth-2 matrix rarely makes
the observed move greedy in one world and not another (3.2); the true-joint measurement in 3.1 confirms this.

### 3.4 Realisability of the eq style (`research/scripts/joint/rq_refit.py`, `research/rq_refit.json`)

On the logged RQ runs (styles' q vectors are logged at 4 dp; inverting the 5 % mix and raising to a power β gives the
style at temperature τ/β exactly for the softmax styles, and a tempered LP strategy for eq; then a new mix):

* RQeq: 36.9 % of the eq bot's actions lie outside the eq support on our Abar (p_eq = MIX/n), out of 5,218
  multi-option turns (32.6 % of all 5,904 scored turns); RQsafe 56 % of 4,454, RQunif 76 % of 3,197 multi-option turns.
* Baseline per-style loss (nats/turn, 5904 turns): eq 2.198, br_unif 1.683, br_us 1.693, safe 1.614, uniform 2.228;
  q-hat 1.617. Best eq over the grid β ∈ {¼, ½, 1, 2} × mix ∈ {0.05, 0.2, 0.5}: 1.697 (β = ½, mix 0.5), but safe at
  that setting is 1.910 and the best q-hat over the whole grid × λ ∈ {1, 0.98} is 1.610 (λ = 0.98, no refit).
  Temperature/mix fitting does not make eq the best style and does not move q-hat.
* Per-world prediction against the live eq bot (3.1 machinery, `hedge_bayes` vs `style_eq`, seeds 1-4, 8 games,
  219 scored turns): current 1.623, membership × averaged 1.296, per-world style likelihood 1.264, true joint 1.274
  (support fix +0.327 ± 0.053; per-world style +0.032 ± 0.023, n.s. on its own; true joint vs option (iii)
  −0.001 ± 0.001). Per style, per world vs averaged: eq 1.329 vs 2.407, br_unif 1.356 vs 1.695, br_us 1.344 vs
  1.739, safe 1.266 vs 1.616, uniform 1.794 vs 2.132. The eq-safe gap shrinks from 0.79 to 0.06 nats but eq still
  trails, so the per-world posterior would still end on safe. World re-weighting is a little stronger here (TV 0.067,
  ratio p90 3.3, LP argmax change 6.1 %) but still small.

Conclusion: no change on our side of the style functional fixes realisability, because the eq bot's strategy is a
function of its sampled worlds about our hidden sets, which we do not observe. Under misspecification the posterior
concentrating on `safe` is the correct Bayesian outcome (T3(vi), `theory/tightening/bayes_dynamics.md:210-218`).
The principled fix is a style defined on public information only (an equilibrium of a matrix built from both
sides' revealed sets), which would be realisable for a bot that also used it — a different style menu, not a
different posterior.

### 3.5 CPU cost (`verify_joint/bench2.py` in the scratchpad, on the 438 logged stock decisions)

`style_dists` on the union Abar 1.0 ms; on the K = 4 world matrices 3.9 ms (ratio 3.8, ≈ 4×); agent-form LP 1.7 ms
(earlier measurement). Decisions cost 0.20-0.21 CPU-s (BH/LOG summaries) to 0.45-0.6 (this loaded machine), nearly
all in the depth-2 matrix builds. Option (iii) adds ≈ 3-4 ms (≈ 2 %). The true joint adds only a table lookup per
world on top of option (iii). Option (ii) with P particles costs one matrix build per particle (≈ 50 ms each):
32 particles ≈ 8× the current decision.

## 4. Ranked options

| rank | option | mechanism | expected gain | cost | risk / exactness | evidence |
|---|---|---|---|---|---|---|
| 1 | **Membership-aware prediction and exploit** (support fix) | restrict each style's q to the world's own columns; predict P(b) = Σ_k w_k Σ_σ P(σ) q_σ(b\|M_k); exploit Σ_k w_k x·M_k·q_k instead of x·Abar·q-hat | −0.32 nats log loss (measured, +0.325 ± 0.027 pooled, G = 18); win rate unknown (estimate ≤ +0.02) | one function in `behavior.py` + the safex call in `run_ourbot.py`; +3-4 ms | exact given the worlds; changes the safex expert's objective, so BH-type results must be re-run | 3.1 |
| 2 | **Per-world style likelihood** (q_σ on each M_k; optionally (iii), the posterior updated with Σ_k w_k q_σ(b\|M_k)) | same as 1 with q_σ computed on M_k | −0.028 ± 0.011 nats beyond rank 1, pooled (measured; per opponent n.s. vs style_eq); the (iii) posterior update adds −0.001 ± 0.011 | included in rank 1 | exact marginal likelihood of the finite mixture; no state | 3.1 |
| 3 | **Realisable style menu** (public-information equilibrium, broad fallback) | add a style computed from revealed sets only, or a QRE/tempered LP, as "eq_public"; keep uniform as the fallback | unknown; the only path to P(eq) > 0 vs an eq-type opponent | a new style function; a frozen-benchmark run | the eq bot itself would have to use the same public matrix to be exactly realisable | 3.4 |
| 4 | **(i) accumulated set weights keyed by (species, set-key)** | log-weight table over foul-play's remaining sets per revealed mon, w ∝ Π_t Σ_σ P(σ) q_σ(b_t \| S); fresh worlds scored by their sets' accumulated log-lik; feed as LP weights b_k (normalised) | ≤ 0.5-1.5 nats of evidence per mon lifetime (estimate from 3.2); planner argmax change ≈ 2 %/turn (measured for one turn); the set-conditional STYLE half of this table (true joint) gains +0.001 ± 0.001 nats (measured) | a small tracker class + K matrix lookups; the set key per world is already logged (`belief.worlds`) | weights are importance weights on a count-weighted proposal: exact only if normalised over the current filtered support and never multiplied into a world twice; bench species are sampled uniformly so their weights are prior-mismatched | 3.1, 3.2 |
| 5 | **(ii) particle table w[k, σ] with resampling** | replace the sampler by P persistent team particles × styles, resample on ESS collapse | no measured need: the carried joint (set × style) belief gains +0.001 ± 0.001 nats (3.1) | replaces `prepare_battles`; P matrix builds per turn (8× CPU at P = 32); rejuvenation must re-score history | approximate (finite particles, rejuvenation); double-counting risk only if foul-play's hard filters are re-applied as soft factors | 3.1, 3.2, 3.5 |
| 6 | scoring the 17 % not_in_worlds turns | mix an escape mass (species prior) into every predictor so skipped turns are scored | makes all comparisons honest; no win-rate gain | small | shared leak only | sec 2 |

Rejected: temperature/mix/λ refit on the averaged matrix (measured dead end, 3.4); world re-weighting of the LP by the
current action alone (2 % argmax change, 3.2); dropping the style posterior for a per-world LP equilibrium prediction
(eq is the worst-fitting style everywhere); a carried joint set × style posterior as a predictor (+0.001 ± 0.001, 3.1).

## 5. Recommended design (spec outline for Plan)

Build rank 1 + 2 as one change ("per-world style model"), optional rank 4 behind a flag, and the benchmark.

* `engine/showdown/behavior.py`
  * `style_dists_worlds(Ms, cols, p_ref) -> list[(5 x n_k)]`: `style_dists` per world (reference mixture p over our
    legal rows is shared; rows are already aligned by `ha.align`).
  * `predict_worlds(post, qs_k, w) -> dict[label, prob]`: P(b) = Σ_k w_k Σ_σ P(σ) q_σ,k(b); labels matched by
    canonical string as in `map_column`.
  * `Posterior.update_worlds(qs_k, jk, w)`: likelihood per style = Σ_{k ∋ b} w_k q_σ,k(j_k); same log-space step.
  * keep `style_dists`/`decide`/`score_and_update` for the logged `qs` so old logs stay comparable.
* `engine/showdown/run_ourbot.py` (hedge_bayes / bayes_safex branches, lines 442-470)
  * compute `qs_k` once per decision; the safex expert maximises Σ_k w_k x·M_k·q_k subject to the agent-form value
    being within ε of its optimum — a NEW LP `nash.safex_strategy_worlds(Ms, w, qs, eps)` (`nash.safex_strategy`
    takes one matrix and one q); its cost is unmeasured (the +3-4 ms covers only `style_dists` on K worlds; expect
    the same order as `agent_form_lp`, 1.7 ms). `pending` carries `qs_k` and `cols`.
  * feedback: `map_column` already returns (k, j); extend to all containing worlds (`jk`), update with `update_worlds`,
    log `p_qhat_world`, `p_qhat_bar` side by side for one run.
  * `OURBOT_LOG_WORLDS=1` already logs world teams (the active's sampled set per world, used in 3.1); add `cols` per
    world (needed for the benchmark without the side log).
* Rank 4 (flag `OURBOT_SET_WEIGHTS=1`): `engine/showdown/setweights.py`, `SetWeights.score(world) -> float`
  summing the accumulated log-lik of the world's sampled set keys; `update(cols_k, jk, lik_k)`; weights normalised over
  the containing worlds each turn; the LP receives b_k ∝ exp(score_k) (normalised, never multiplied by 1/K again).
  Do not condition the style posterior on the set key: measured worthless (3.1).
* Data flow per turn: worlds → M_k, cols_k → (LP x on b) and (qs_k) → mixture p → move; next turn: b → jk → style
  update (and set-weight update) → log.

## 6. Evaluation plan with pre-registered predictions

**Frozen-observation benchmark** (offline, minutes): the `headless_logworlds.py` output (`worlds.jsonl`, the A-side
decision log, the game JSON with `teams`) replayed through predictor variants by `analyze_worlds.py`-style code; true
teams only in the evaluator. Metrics: prequential log loss and Brier of the opponent's next action over scored turns;
log loss over all feedback turns with a shared escape mass ε = 0.05 spread over the species prior (so not_in_worlds
turns count); for set beliefs, `score_belief.py`'s marginals (log loss of the true item/ability/move/species). Sample
sizes from the measured paired per-turn s.d. 0.45 nats (current − membership): 620 scored turns detect 0.05 nats at
80 % power (two-sided 5 %), 3900 turns detect 0.02. One 10-game stream gives ≈ 370 scored turns; use ≥ 30 games
(≈ 1100 turns) against stock and ≥ 30 against style_eq. Report game-clustered standard errors (turns within a game
are correlated).

Pre-registered: (P-a) membership-aware predictor beats the current one by 0.25-0.35 nats vs stock; (P-b) per-world
style vs membership × averaged style: |Δ| < 0.02 nats, not significant at 1000 turns — **already contradicted by the
corrected pilot (+0.028 ± 0.011 pooled, 591 turns); revised prediction: +0.01 to +0.05 nats, significant at 1000
turns**; (P-b') true joint vs option (iii): |Δ| < 0.01 nats (pilot +0.001 ± 0.001); (P-c) accumulated set weights
(rank 4) change the LP argmax on < 5 % of turns and improve the set marginal log loss by < 0.05 nats; (P-d) against
style_eq, eq remains behind safe per world by 0.03-0.15 nats (measured 0.06 on 219 turns).

**Paired Showdown run** (`batch.py`, hedge_bayes-world vs stock, 10 streams, seeds shared with BH): pair-score s.d. is
0.312 (BH), so 400 pairs give a 95 % CI of ± 0.031; detecting +0.03 needs ≈ 850 pairs. Pre-registered: Δ pair score
vs BH within [−0.02, +0.03], i.e. not significant at 400 pairs; CPU per decision +≤ 10 %.

What would change the ranking: a true-joint (set-conditional style) gain ≥ 0.01 nats on the larger benchmark (then
(ii)/rank 4 rise); set-weight evidence ≥ 2 nats per mon lifetime (then rank 4 rises); any win-rate signal from the
support fix (then re-run the SX ε sweep on the per-world safex).

## 7. Literature anchors

* Bayes' Bluff (Southey et al., UAI 2005): posterior over opponent strategies (Dirichlet priors on a reduced game),
  separating uncertainty about game dynamics from uncertainty about the opponent, and best responses computed from the
  posterior. Ours is the same decomposition with the "dynamics" uncertainty being the hidden sets and the strategy
  uncertainty a finite type menu; new here is that the hidden-information half is itself sampled by a third-party
  filter whose worlds are regenerated every turn.
* Type-based reasoning / HBA (Albrecht & Ramamoorthy 2013; survey Albrecht & Stone 2018): posterior over a finite
  set of opponent types updated from observed actions, each type a policy; our styles are types defined as
  functionals of the current payoff matrix, so realisability depends on the opponent seeing the same matrix (3.4).
  The connection is structural only: nothing here uses or tests HBA's results on correlated type beliefs across
  players; the toy (3.3) is explaining-away within one opponent.
* Bayesian games (Harsanyi): the sets are the opponent's type; the agent-form LP is the Bayesian zero-sum game
  against a type-indifferent minimiser. A joint (type, style) belief is the Bayesian-game belief extended with a
  behavioural type; what is adapted is the "cheapest" version where the behavioural half is a finite mixture and the
  team half stays with the filter's proposal.
* Determinised / PIMC search with belief weighting (Cowling et al. 2012 ISMCTS; Long et al. 2010 on when PIMC fails):
  weighting determinisations by likelihood is standard; our measurement says the per-turn likelihood weights are
  nearly uniform here.
* What would be new: the measurement that, with hard-filtered randbats sets, the carried set × style correlation is
  empirically nil (+0.001 ± 0.001 nats) while the support (membership) error of an averaged-matrix predictor is
  0.32 nats and computing styles per world adds a further 0.03 — a negative result worth one paragraph.

## 8. Risks and housekeeping

* The pilot is small (18 games, 591 scored turns); the per-world style gain is significant pooled (G = 18) but not
  per opponent vs style_eq (G = 8). Use game-clustered standard errors throughout.
* The replayed deployed posterior matches the logged one to mean TV 0.0001 when replayed from the logged `qs`; a
  replay from recomputed qs (q_σ on the 2-dp Abar with `pfull` as reference) drifts to mean TV 0.0038 vs style_eq
  (0.0002 vs stock). The headline predictors use the logged posterior and q-hat, so this drift does not enter them.
* Union dilution also sits inside the exploit objective x·Abar·q-hat of the safex expert; fixing prediction without
  fixing the objective leaves the planner inconsistent with the model.
* The style_eq bot's realisability cannot be fixed by any per-world or temperature change on our side (3.4).
* Seed 6 of the stock run was killed mid-game to start the eq batch, and eq seed 5 is partial; their files were moved
  to `partial/` in the scratchpad and excluded.
* Unrevealed teammates are sampled uniformly over species and sets (not count-weighted): a prior mismatch to fix
  separately in the evaluator (bench-species log loss).
* Not this sprint's files: `engine/showdown/behavior.py` shows as modified in the worktree (another agent's opt-in
  `OURBOT_BAYES_DTP` discount-then-predict flag; default off, and at λ = 1 it is a no-op, so the analyses here are
  unaffected), as are `research/prereg_style_switch.md`, `research/scripts/switch/`, `research/scripts/voi/`.
  All 18 logged games (10 vs stock, 8 vs style_eq) finished with `error: null` and zero simulator errors.
* foul-play parses two ditto set keys with 5 fields as moves `[transform, <tera>]` (`fp/data/sets/randbats.py:45-53`);
  harmless for us, noted for the set-key design in rank 4.
