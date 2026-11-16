# Joint posterior over opponent sets and decision style: is it worth building?

Ideation sprint, 2026-11-16. Code read in `/Users/biofanat/Documents/classes/f26/cs535/final-integration`
(branch `integration/joint-belief`). Scripts in `research/scripts/joint/`. Nothing under `engine/` was edited.

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
  cannot be indexed by world — worlds do not persist across turns.
* **Unrevealed teammates are sampled uniformly over species and uniformly over that species' sets**
  (`random_battles.py:86-87`), not count-weighted. That is a prior mismatch of its own, independent of this question.
* **The style predictor normalises over the label union, not the opponent's options.** `avg_matrix` builds Abar over
  the union of the K worlds' column labels (first-seen order). Measured on 336 scored turns: the union has 13.2
  labels on average, a world has 7.8 columns. Every style distribution q_σ is a distribution over 13.2 labels of which
  the opponent actually has 7.8. This "dilution" turns out to be the dominant effect below.
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
seeds 1-5, one stream (posterior carried), 10 games, 336 scored turns; and `hedge_bayes` vs `style_eq`, seeds 1-4,
8 games, 192 scored turns (3.4). Raw outputs: scratchpad `jp_work/` (stock) and `jp_work/eq/` (style_eq); the
`analyze_worlds.json` row dumps sit next to them. Game JSONs carry the true teams (`OURBOT_LOG_WORLDS=1`), so this output is already the
frozen-observation benchmark format of section 6.

`research/scripts/joint/analyze_worlds.py` then recomputes, for every scored turn and every world k that contains the
observed action b: q_σ(b | M_k) for the five styles (`behavior.style_dists` on the world matrix), the posterior P(σ)
as the deployed code updates it (averaged-matrix likelihood), and four predictors of b:

| predictor | definition | log loss vs stock (336 turns) |
|---|---|---|
| current (averaged matrix) | P(σ)-mixture of q_σ on Abar over the label union | **1.564** (logged `p_qhat` on the same turns: 1.600; the recomputation uses `pfull` as br_us's reference mixture and 2-dp matrices, so br_us differs slightly) |
| membership × averaged style ("independent marginals") | same q-hat, restricted to each world's columns and renormalised, averaged over worlds | **1.250** |
| per-world joint, uniform worlds | Σ_k (1/K) Σ_σ P(σ) q_σ(b \| M_k) | **1.252** |
| option (iii) end to end | per-world likelihood and a posterior updated with the per-world marginal likelihood | **1.250** |

Paired differences (mean ± s.e.): current − independent **+0.314 ± 0.024** nats; independent − joint
**−0.002 ± 0.018**; joint − option (iii) **+0.002 ± 0.022**.

Reading: the whole gain is the support fix (predict over the opponent's actual options instead of the union); the
correlation term — computing the style on each world's own matrix instead of the average — is zero within noise.
Per style, per world vs averaged: eq 1.47 vs 2.40, br_unif 1.26 vs 1.56, br_us 1.37 vs 1.76, safe 1.25 vs 1.61,
uniform 1.87 vs 2.22. The eq style is the one most damaged by the union (the LP puts support on phantom columns).

### 3.2 Does the observed action re-weight the worlds?

w_k ∝ Σ_σ P(σ) q_σ(b | M_k) over the worlds containing b (98 % of scored turns have > 1 such world):
TV distance from uniform mean 0.050, median 0.025, p90 0.121; ESS/n 0.97; max/min likelihood ratio median 1.16,
p90 2.14. Re-solving the agent-form LP with the re-weighted worlds moves x by L1 0.065 on average (median 0.000)
and changes the argmax on 2.1 % of turns. On 62 % of turns the containing worlds carry ≥ 2 distinct movesets for
the active, so there is something to learn, but one action is weak evidence between them.

Accumulated over a game (option (i)): for pairs of movesets of the same active that were co-sampled on ≥ 2 turns
(n = 26 pairs, median 2 co-sampled turns), the accumulated |log-likelihood ratio| is median 0.26, p90 0.83 nats;
8 % of pairs reach ln 3. (Approximate: the moveset key is the non-tera labels among the world's first eight columns,
which after terastallization includes switches; the conclusion rests on the per-turn ratios, not on this stat.) With per-turn ratios of 1.16 (p90 2.1), a mon that stays in for 5-8 turns accumulates
roughly 0.5-1.5 nats between its candidate movesets — an estimate, bounded above by the per-turn evidence.

### 3.3 Exact hand example (`research/scripts/joint/toy_joint.py`)

Sets S1 = {A, B}, S2 = {A, C}; styles greedy (0.9 on its best column vs the public reference mixture) and uniform;
priors 1/2. The "product" tracker runs the two Bayes filters the current pipeline runs (set belief with
Σ_σ P(σ) q_σ(obs|S), style belief with Σ_S P(S) q_σ(obs|S)) and multiplies. Payoff to us (rows r1, r2; columns A, B,
C) = [[-2, -2, 2], [-1, -2, -2]], observations A, A (A is in both sets, so the set is never identified):

* joint P(S, σ): (S1, greedy) 0.008, (S1, unif) 0.189, (S2, greedy) 0.614, (S2, unif) 0.189;
  product: 0.088, 0.067, 0.481, 0.364.
* next action: joint A 0.742 / B 0.102 / C 0.156; product A 0.657 / B 0.113 / C 0.230 (TV 0.085).
* best reply: joint r2, product r1; the product's reply loses 0.118 under the joint belief.

Condition for correlation to matter: an action that is in several candidate sets but is the greedy choice in only
some of them (then style and set become dependent given the action). Membership-only evidence (an action in one
set only) never creates correlation — the posterior factorises after it. In randbats the second case dominates:
revealed moves are hard evidence, and among the remaining candidate movesets a 2-dp depth-2 matrix rarely makes
the observed move greedy in one world and not another (3.2).

### 3.4 Realisability of the eq style (`research/scripts/joint/rq_refit.py`, `research/rq_refit.json`)

On the logged RQ runs (styles' q vectors are logged at 4 dp; inverting the 5 % mix and raising to a power β gives the
style at temperature τ/β exactly for the softmax styles, and a tempered LP strategy for eq; then a new mix):

* RQeq: 36.9 % of the eq bot's actions lie outside the eq support on our Abar (p_eq = MIX/n); RQsafe 56 %, RQunif 76 %.
* Baseline per-style loss (nats/turn, 5904 turns): eq 2.198, br_unif 1.683, br_us 1.693, safe 1.614, uniform 2.228;
  q-hat 1.617. Best eq over the grid β ∈ {¼, ½, 1, 2} × mix ∈ {0.05, 0.2, 0.5}: 1.697 (β = ½, mix 0.5), but safe at
  that setting is 1.910 and the best q-hat over the whole grid × λ ∈ {1, 0.98} is 1.610 (λ = 0.98, no refit).
  Temperature/mix fitting does not make eq the best style and does not move q-hat.
* Per-world prediction against the live eq bot (3.1 machinery, `hedge_bayes` vs `style_eq`, seeds 1-4, 8 games,
  192 scored turns): current 1.596, membership × averaged 1.278, per-world joint 1.266 (support fix +0.318 ± 0.029;
  correlation term +0.012 ± 0.031, n.s.). Per style, per world vs averaged: eq 1.355 vs 2.350, br_unif 1.352 vs
  1.649, br_us 1.345 vs 1.720, safe 1.265 vs 1.587, uniform 1.753 vs 2.090. The eq-safe gap shrinks from 0.76 to
  0.09 nats but eq still trails, so the per-world posterior would still end on safe. World re-weighting is a little
  stronger here (TV 0.066, ratio p90 3.3, LP argmax change 5.9 %) but still small.

Conclusion: no change on our side of the style functional fixes realisability, because the eq bot's strategy is a
function of its sampled worlds about our hidden sets, which we do not observe. Under misspecification the posterior
concentrating on `safe` is the correct Bayesian outcome (T3(vi), `theory/tightening/bayes_dynamics.md:210-218`).
The principled fix is a style defined on public information only (an equilibrium of a matrix built from both
sides' revealed sets), which would be realisable for a bot that also used it — a different style menu, not a
different posterior.

### 3.5 CPU cost (`research/scripts/joint/bench_cost.py`, measured on the logged matrices)

`style_dists` on Abar 2.2 ms; on the K = 4 world matrices 9.8 ms; agent-form LP 1.7 ms. Decisions cost 0.20-0.21
CPU-s (BH/LOG summaries) to 0.45-0.6 (this loaded machine), nearly all in the depth-2 matrix builds. Option (iii)
adds ≈ 10 ms (≈ 5 %). Option (ii) with P particles costs one matrix build per particle (≈ 50 ms each): 32 particles
≈ 8× the current decision.

## 4. Ranked options

| rank | option | mechanism | expected gain | cost | risk / exactness | evidence |
|---|---|---|---|---|---|---|
| 1 | **Membership-aware prediction and exploit** (support fix) | restrict each style's q to the world's own columns; predict P(b) = Σ_k w_k Σ_σ P(σ) q_σ(b\|M_k); exploit Σ_k w_k x·M_k·q_k instead of x·Abar·q-hat | −0.31 nats log loss (measured, p < 1e-6); win rate unknown (estimate ≤ +0.02) | one function in `behavior.py` + the safex call in `run_ourbot.py`; +10 ms | exact given the worlds; changes the safex expert's objective, so BH-type results must be re-run | 3.1 |
| 2 | **(iii) per-world style likelihood for the current turn** | same as 1 with q_σ computed on M_k; posterior updated with Σ_k w_k q_σ(b\|M_k) | 0 ± 0.02 nats beyond rank 1 (measured) | included in rank 1 | exact marginal likelihood of the finite mixture; no state | 3.1 |
| 3 | **Realisable style menu** (public-information equilibrium, broad fallback) | add a style computed from revealed sets only, or a QRE/tempered LP, as "eq_public"; keep uniform as the fallback | unknown; the only path to P(eq) > 0 vs an eq-type opponent | a new style function; a frozen-benchmark run | the eq bot itself would have to use the same public matrix to be exactly realisable | 3.4 |
| 4 | **(i) accumulated set weights keyed by (species, set-key)** | log-weight table over foul-play's remaining sets per revealed mon, w ∝ Π_t Σ_σ P(σ) q_σ(b_t \| S); fresh worlds scored by their sets' accumulated log-lik; feed as LP weights b_k (normalised) | ≤ 0.5-1.5 nats of evidence per mon lifetime (estimate from 3.2); planner argmax change ≈ 2 %/turn (measured for one turn) | a small tracker class + K matrix lookups; needs the set key per world (not logged today) | weights are importance weights on a count-weighted proposal: exact only if normalised over the current filtered support and never multiplied into a world twice; bench species are sampled uniformly so their weights are prior-mismatched | 3.2 |
| 5 | **(ii) particle table w[k, σ] with resampling** | replace the sampler by P persistent team particles × styles, resample on ESS collapse | no measured need (3.2); correlation term ≈ 0 | replaces `prepare_battles`; P matrix builds per turn (8× CPU at P = 32); rejuvenation must re-score history | approximate (finite particles, rejuvenation); double-counting risk only if foul-play's hard filters are re-applied as soft factors | 3.2, 3.5 |
| 6 | scoring the 17 % not_in_worlds turns | mix an escape mass (species prior) into every predictor so skipped turns are scored | makes all comparisons honest; no win-rate gain | small | shared leak only | sec 2 |

Rejected: temperature/mix/λ refit on the averaged matrix (measured dead end, 3.4); world re-weighting of the LP by the
current action alone (2 % argmax change, 3.2); dropping the style posterior for a per-world LP equilibrium prediction
(eq is the worst-fitting style everywhere).

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
    takes one matrix and one q); its cost is unmeasured (the +10 ms covers only `style_dists` on K worlds; expect
    the same order as `agent_form_lp`, 1.7 ms). `pending` carries `qs_k` and `cols`.
  * feedback: `map_column` already returns (k, j); extend to all containing worlds (`jk`), update with `update_worlds`,
    log `p_qhat_world`, `p_qhat_bar` side by side for one run.
  * `OURBOT_LOG_WORLDS=1` already logs world teams; add `cols` per world and the sampled set keys (needed for rank 4
    and the benchmark).
* Rank 4 (flag `OURBOT_SET_WEIGHTS=1`): `engine/showdown/setweights.py`, `SetWeights.score(world) -> float`
  summing the accumulated log-lik of the world's sampled set keys; `update(cols_k, jk, lik_k)`; weights normalised over
  the containing worlds each turn; the LP receives b_k ∝ exp(score_k) (normalised, never multiplied by 1/K again).
* Data flow per turn: worlds → M_k, cols_k → (LP x on b) and (qs_k) → mixture p → move; next turn: b → jk → style
  update (and set-weight update) → log.

## 6. Evaluation plan with pre-registered predictions

**Frozen-observation benchmark** (offline, minutes): the `headless_logworlds.py` output (`worlds.jsonl`, the A-side
decision log, the game JSON with `teams`) replayed through predictor variants by `analyze_worlds.py`-style code; true
teams only in the evaluator. Metrics: prequential log loss and Brier of the opponent's next action over scored turns;
log loss over all feedback turns with a shared escape mass ε = 0.05 spread over the species prior (so not_in_worlds
turns count); for set beliefs, `score_belief.py`'s marginals (log loss of the true item/ability/move/species). Sample
sizes from the measured paired per-turn s.d. 0.43 nats: 580 scored turns detect 0.05 nats at 80 % power
(two-sided 5 %), 3600 turns detect 0.02. One 10-game stream gives ≈ 340 scored turns; use ≥ 30 games (≈ 1000 turns)
against stock and ≥ 30 against style_eq.

Pre-registered: (P-a) membership-aware predictor beats the current one by 0.25-0.35 nats vs stock; (P-b) per-world
style vs membership × averaged style: |Δ| < 0.02 nats, not significant at 1000 turns; (P-c) accumulated set
weights (rank 4) change the LP argmax on < 5 % of turns and improve the set marginal log loss by < 0.05 nats;
(P-d) against style_eq, eq remains behind safe per world by 0.05-0.15 nats (measured 0.09 on 192 turns).

**Paired Showdown run** (`batch.py`, hedge_bayes-world vs stock, 10 streams, seeds shared with BH): pair-score s.d. is
0.312 (BH), so 400 pairs give a 95 % CI of ± 0.031; detecting +0.03 needs ≈ 850 pairs. Pre-registered: Δ pair score
vs BH within [−0.02, +0.03], i.e. not significant at 400 pairs; CPU per decision +≤ 10 %.

What would change the ranking: a measured correlation term ≥ 0.03 nats on the eq-opponent benchmark (then (iii) is
worth its own run); set-weight evidence ≥ 2 nats per mon lifetime (then rank 4 rises); any win-rate signal from the
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
* Bayesian games (Harsanyi): the sets are the opponent's type; the agent-form LP is the Bayesian zero-sum game
  against a type-indifferent minimiser. A joint (type, style) belief is the Bayesian-game belief extended with a
  behavioural type; what is adapted is the "cheapest" version where the behavioural half is a finite mixture and the
  team half stays with the filter's proposal.
* Determinised / PIMC search with belief weighting (Cowling et al. 2012 ISMCTS; Long et al. 2010 on when PIMC fails):
  weighting determinisations by likelihood is standard; our measurement says the per-turn likelihood weights are
  nearly uniform here.
* What would be new: the measurement that, with hard-filtered randbats sets, the set × style correlation is empirically
  nil while the support (membership) error of an averaged-matrix predictor is 0.3 nats — a negative result worth one
  paragraph.

## 8. Risks and housekeeping

* The recomputed baseline (1.564) is 0.04 nats below the logged prequential loss on the same turns (1.600):
  the recomputation uses `pfull` as br_us's reference mixture and 2-dp matrices. The +0.31 effect is 8× that.
* Union dilution also sits inside the exploit objective x·Abar·q-hat of the safex expert; fixing prediction without
  fixing the objective leaves the planner inconsistent with the model.
* The style_eq bot's realisability cannot be fixed by any per-world or temperature change on our side (3.4).
* Seed 6 of the stock run was killed mid-game to start the eq batch; its partial files were moved to `partial/` in
  the scratchpad and excluded.
* Unrevealed teammates are sampled uniformly over species and sets (not count-weighted): a prior mismatch to fix
  separately in the evaluator (bench-species log loss).
* Not this sprint's files: `engine/showdown/behavior.py` shows as modified in the worktree (another agent's opt-in
  `OURBOT_BAYES_DTP` discount-then-predict flag; default off, and at λ = 1 it is a no-op, so the analyses here are
  unaffected), as are `research/prereg_style_switch.md`, `research/scripts/switch/`, `research/scripts/voi/`.
  All 19 logged games finished with `error: null` and zero simulator errors.
* foul-play parses two ditto set keys with 5 fields as moves `[transform, <tera>]` (`fp/data/sets/randbats.py:45-53`);
  harmless for us, noted for the set-key design in rank 4.
