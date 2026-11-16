# Response to "Pokemon equilibrium engine: remaining implementation plan" (2026-11-16)

This responds to `notes/pokemon_remaining_implementation_plan.txt` (prepared 2026-11-16), workstream by workstream:
what was done, what was measured, what was not done and why. The research work is on branch `integration/joint-belief`
(worktree `final-integration`), kept separate from `main` on request; the gauntlet/ladder code and the three harness
fixes are also on `main`. Every number below is from a result file in this branch
named next to it (or in the section it summarises); the status labels are: **done**, **done as a measurement** (built offline or in a controlled
run, result reported, not shipped as the default agent), **partial**, **not done**.

## 0. A note on the plan's starting point

The plan was written against branch `feat/mechanics-belief-tracker` at `1aec573` and lists as "already implemented"
a mechanics belief catalog, tracker, likelihoods, observer and world provider (`engine/belief_*.py`,
`engine/showdown/belief_observer.py`), and cites `local_play.py` and `docs/` as inspected sources. **None of these exist in this repository**: the remote
has no such branch, and a search of the course folder and the sandbox found none of the files (checked: `git ls-remote origin` lists only main and integration/joint-belief; 1aec573 is not a
valid object; no such files under the course folder or ~/sandbox). So workstreams A and B
(which validate and extend that tracker) could not be started; the rest of the plan was addressed against the code that
does exist (foul-play's hard-filter set sampler, the agent-form LP planner, the `behavior.py` style posterior). If that
branch exists elsewhere, pushing it would unblock A and B.

## Summary table

| Workstream | Status | Headline |
|---|---|---|
| A. Validate mechanics tracker | not done | code not in this repo (§0) |
| B. Public state / rounded-HP inference | not done | depends on A |
| C. Joint set-and-style posterior | done as a measurement | genuinely joint (set-conditional, carried) adds +0.001 ± 0.001 nats over factorised; not worth shipping |
| D. Style models and memory | partial | style-posterior switching test done (live test of T5, all predictions pass (one point prediction missed, below its refutation threshold)); realisability still fails for the eq bot; new styles, replay priors not done |
| E. Beliefs into planning and experts | partial (opt-in agent built and tested; E.3, E.4, E.8-9 and the correlated-example criterion not done) | `hedge_bayes_mw`: per-world prediction (−0.30 nats per scored turn, replicated) and per-world safe-exploit LP; no win-rate change (replication) |
| F. Value function | not done this round | earlier negative results stand; an exact LP-leaf child matrix is now affordable (≈ 6 extra engine calls per child; S1 costs ≈ 4× the current planner per decision, `research/s1_results.md`) (by-product of G) |
| G. Information gathering | done as a measurement | information-set planner S1 built and gated; value of the observations ≈ 0 in this engine (3/3,620 decisions change); null accepted |
| H. Concealment | done as a measurement | honest opponent model measured: concealment value median 0.24 % of range; omniscient model is the safe side (maxmin); not shipped |
| I. Benchmark / status card | partial | 9-bot gauntlet with Bradley-Terry ratings and meta-game Nash; live paired tests; adaptation; status card below |
| J. Unrun diagnostics | partial | J2 (posterior switching) done; J10 (theory-to-data ledger) maintained in `theory/theory_map.md` §11-12; others not done |
| K. Integrity and delivery | partial | three harness bugs found and fixed; pre-registration discipline; ladder delivery doc; no transactional checkpointing |

## C. Joint set-and-style posterior (plan §5)

**What was built.** Per-world logging of the sampled worlds' matrices (`research/scripts/joint/headless_logworlds.py`),
an offline benchmark replaying 18 logged games (10 vs stock foul-play, 8 vs the eq style bot), and five predictors of
the opponent's next action scored on identical turns: the deployed averaged-matrix predictor; a membership-aware
predictor (styles evaluated on each world's own columns); a per-world style likelihood (factorised posterior); option (iii): the per-world likelihood with its own posterior update; and a
**genuinely joint** predictor that keeps a weight table over (active set key, style) per game, carried across turns,
updated after scoring (`research/scripts/joint/analyze_worlds.py`, `research/joint_posterior.md`).

**Results** (corrected after independent verification; game-clustered SEs; `research/joint_posterior.md` Corrections):

| comparison | nats per scored turn |
|---|---|
| membership-aware vs averaged (support fix) | +0.324 ± 0.031 (1.623 → 1.299, 372 turns vs stock) |
| per-world style likelihood vs membership-aware | +0.028 ± 0.011 pooled (18 games) |
| genuinely joint vs factorised | +0.001 ± 0.001 pooled |

On 57-75 % of turns several distinct sets could have produced the observed action, yet the set-conditional style
posterior differs from the marginal by mean total variation 0.016-0.022. A hand-enumerated toy
(`research/scripts/joint/toy_joint.py`) shows where joint and product posteriors do diverge (explaining-away; the best
reply flips), so the null is about Pokemon, not the method.

**Against the plan's acceptance criteria.** Tiny exact games match hand enumeration: yes (toy). Mechanics-only and
behaviour-only controls: the factorised predictor is the closest behaviour-only control (it still runs on foul-play's hard-filtered worlds); mechanics evidence is foul-play's hard
filters, which carry no soft likelihood, so double counting cannot occur. Same visible history gives identical beliefs
under changed hidden truth: tested for the S1 planner's keys (924/924), not for C. Equilibrium-opponent mismatch is
reported (D below). Set-probability response to style under a switch: shown only in the toy. Duplicate / censored / forced /
out-of-support handling: not documented for C (feedback skips 16-28 % of turns, `joint_posterior.md` §2). Particle resampling / rejuvenation (§5 steps 7-8): not built, because the measured value of the
joint term is zero.

**Two bugs were caught by verification** in the first version of this analysis (tera actions scored on the stripped
label; feedback joined by repeating turn numbers). Both are fixed; the first version's "correlation term zero" was
right by accident for the joint question but wrong for the per-world term (+0.028, not 0).

## D. Style models and long-term memory (plan §6)

- **Switch test of the style posterior itself (D.8, J.2): done.** Pre-registered (`research/prereg_style_switch.md`,
  committed before any run), arms SW2 / SW4 / SW4p (forgetting λ = 0.98), 30 streams, 0 errors. All predictions pass (one point prediction missed, below its refutation threshold):
  per-stream re-concentration time predicted from each stream's own switch-time posterior (median actual/predicted
  1.06, Spearman 0.86 over 20 streams); delay scales 2.09x with doubled pre-switch evidence; forgetting cuts it to
  37.5 updates (predicted 35). One phase-1 false alarm in SW2 (predicted 0; refutation at 2). This is the first live
  test of theory result T5 (`theory/tightening/bayes_dynamics.md` §6).
- **Realisability (D.3-D.5): partial.** Temperature / mix / λ refits on the logged RQeq data (`research/rq_refit.json`)
  and per-world styles both leave the eq style behind "safe" against the eq bot (final MAP eq in 0/10 streams with
  per-world styles, pre-registered R1). 36.9 % of the eq bot's multi-option actions fall outside the eq style's support
  on our averaged matrix. Reason: the eq bot solves its own sampled worlds of our team, so no change on our side makes eq realisable against this
  bot; a public-information eq style would be realisable only against a bot that also plays it. Not built.
- **New styles (attack-heavy, switch-happy, greedy damage), replay-fitted priors, opponent-identity namespacing,
  change-point model: not done.** The *_mw posterior state files now carry their own config marker, so they cannot be
  silently shared with the old posterior's files.

## E. Beliefs into planning and expert learning (plan §7)

**Built as opt-in methods** `hedge_bayes_mw` and `bayes_safex_mw` (existing methods byte-identical, checked with a
snapshot harness): membership-aware style prediction and likelihood (`behavior.py`), and a per-world safe-exploit LP
`hidden_agents.safex_strategy_worlds`: maximise Σ_k b_k q̂_kᵀ M_kᵀ x subject to Σ_k b_k min_j (M_kᵀx)_j ≥ v_AF − ε·ptp
(plan E.5-E.6, with the factorised rather than joint style weights, justified by C). Tests: one-world reduction to the
existing predictor/update (bitwise) and LP (objective equal within the 1e-7 test tolerance), membership (labels absent from a world get zero mass), LP guarantee met
within 2.2e-12 (absolute; 5e-15 of the payoff range) on 1,200 solves (300 random instances × 4 ε), a tera-label consistency test.

**Live test, pre-registered** (`research/prereg_mw.md`): next-action log loss 0.297 nats per scored turn
[0.281, 0.314] lower than the averaged-matrix predictor's in games vs stock, 0.250 [0.224, 0.276] vs the eq bot; CPU +2 %. An unpredicted win-rate gain on seeds 1-200 (+0.0625) was not
robust under stream clustering and **did not replicate** on fresh seeds 201-400 (+0.0075, one-sided p = 0.45); the
first control was low by chance (`research/mw_w1_verification.md`). The prediction gain replicated (0.311). Reading:
better beliefs, unchanged win rate, as for the earlier Bayesian results; consistent with the leaf being the
bottleneck.

Not done: per-type opponent strategies from LP duals (E.3), a multiworld no-regret solver (E.4), counterfactual
expert feedback (E.8-E.9), and the plan's acceptance test that joint-posterior exploitation differs from an
independent-marginals control on a designed correlated example.

## F. Value function (plan §8)

Not done this round. Earlier results stand (`engine/RESULTS.md`: the PPO pilot's negative result and the leaf identified
as the bottleneck, "Next experiments"). A by-product of G: `engine/info_planner.full_d1` reconstructs the engine's exact
unpruned child matrices (5,256/5,256 cells reproduced), making an LP-value leaf affordable for future work (≈ 6 extra engine calls per child; S1 costs ≈ 4× the current planner per decision, `research/s1_results.md`).

## G. Explicit information gathering (plan §9)

**Built** `engine/info_planner.py` (S1): a depth-2 planner that groups successors across the sampled worlds by the
public observation, so one continuation strategy is shared per information set (no strategy fusion), with the plan's
matched no-information control (posterior frozen) and per-decision evaluation.

**Gates (plan acceptance criteria):** probe-worth-0.5 toy recovered exactly and chosen iff cost < 0.5; a reveal that
cannot change the continuation has information value 0 (net value −c, its cost); no clairvoyance from splitting indistinguishable histories (hidden
attribute changes leave keys unchanged, 924/924); reduction, when every child is its own information set, to an independent depth-2 agent-form reference with the same LP leaf (max rel. diff 2.3e-15).

**Result** (`research/s1_results.md`, 3,620 engine decisions, pre-registered E1-E4 from
`research/info_gathering_concealment.md`): value of observing vs the frozen control is mean 0.0022 % of the payoff
range, action changes 3/3,620 (E1b, E2a fail); grouping statistics as predicted (E4 pass). Cause: poke-engine uses
average damage and branches only on crits and KOs, so soft evidence almost never appears and hard eliminations are
credited to both arms. Clairvoyance at the children is worth a mean 0.35 % of range (p90 0.59 %). **Null accepted**; damage-roll branching
would be the next step and is a large engine change for at most that amount on average. Paired-battle evaluation (E3) not run.

## H. Concealment (plan §10)

**Measured, not shipped** (`research/info_gathering_concealment.md`, independently verified): our team is resampled
from the opponent's public view of it (no leak: 2,880/2,880 non-interference tests) and a one-stage two-sided Bayesian
LP is solved (matches brute force to 4e-16). Concealment value: median 0.24 %, mean 1.1 % of the decision's payoff
range; with our true type, mean 1.28 % (engine simulation, 6 games, 91 decisions). The current omniscient-opponent model is the safe side by maxmin (it
guarantees v_AF against any opponent); the honest model loses 3.4 % against an omniscient opponent. Tera signalling:
value 0 in 3 of 4 exact matchups; tera-specific concealment was underpowered (0 of 21 changed decisions involve tera)
and is not worth building. Not done: an opponent-view observer in live play, signalling-game planner.

## I. Benchmark and status card (plan §11)

- **Round-robin meta-game (§11.4): done.** Gauntlet G1 (`engine/showdown/results/gauntlet/G1_ratings.md`): 9
  stationary entrants, 1,800 games, 0 errors. Bradley-Terry ratings anchored at our agent: foul-play +135 to +192 Elo
  (25-1000 ms per state), top +43, pimc −41 (both CIs include 0), max-base-power −286, random −911. Meta-game Nash over
  the fixed pool with bootstrap support weights: point equilibrium all on foul-play at 1000 ms; bootstrap weight almost entirely on foul-play's four budgets, split among them. Our agent is below foul-play's smallest budget
  (25 ms ≈ 17,000 MCTS iterations per sampled battle, `research/gauntlet_fp_iterations.md`).
- **Live paired battles vs stock and simple bots (§11.2):** partial: gauntlet and MW arms done; the plan's attack-heavy / switch-happy / greedy opponents and
  equal-total-compute comparisons were not run. Failures reported, not dropped
  silently (`research/harness_bugs.md`).
- **Adaptation (§11.3):** switching test done (D). Exploiters were run earlier (`engine/RESULTS.md` Open items B3). Reset-vs-carried memory and returning-opponent tests not run.
- **Frozen-observation inference benchmark (§11.1):** partial — the per-world log format of C is that benchmark, on 18
  games; mechanics-tracker variants not applicable (§0).

**Status card (§11.5)**

| | |
|---|---|
| Strength | 0.22-0.32 vs stock foul-play; ≈ 135-190 Elo below it; level with other one-turn solvers |
| Robustness | exploiter gains small (earlier B3); omniscient-opponent model is the safe side (H) |
| Adaptation | posterior tracking after a style switch matches T5 per stream (D) |
| Beliefs (calibration, coverage and latency not measured) | membership-aware prediction −0.30 nats per scored turn (replicated); joint set×style adds nothing; eq style not realisable |
| Information use | scouting value ≈ 0 in this engine; concealment value small and measured |
| Cost / reliability | ≈ 0.4 CPU-s per decision; 0 error games after three harness fixes; all runs local (RESULTS.md, Benchmark and follow-up studies) |

## J. Unrun diagnostics (plan §12)

Done: J2 posterior switching (D). Maintained: J10 theory-to-data ledger (`theory/theory_map.md` §11-12, with T5 now
marked live-tested). Earlier: J3 (the LOG identity was computed on our side only, against stock foul-play, whose quantities are not
instrumented; `engine/analysis/open_items/REPORT.md` B1).
Not done this round: J1 (K = 16 belief scoring), J4-J9.

## K. Integrity and delivery (plan §13)

- **Bugs found and fixed** (`research/harness_bugs.md`): a simulator startup race (non-atomic per-launch rebuild),
  Revival Blessing requests (crash or stall for every method), forced recharge while trapped. 33 earlier games were
  excluded as error games non-randomly; the caveat is recorded.
- **Discipline:** every hypothesis-testing live experiment this round (style switch, hedge_bayes_mw and its replication) was
  pre-registered in a commit before its first game; the G1 gauntlet was a descriptive benchmark and was not pre-registered; results are
  appended below the unchanged predictions; an unpredicted result was re-tested on fresh seeds.
- **Delivery:** `engine/showdown/LADDER.md` + `ladder.py` (registered bot accounts only, password from the environment,
  refuses public servers without confirmation); tested locally only.
- Not done: attempt IDs and transactional state checkpoints (§13 steps 1-6). Learner state files are resumed by the
  existing rule; one audit hypothesis about aborted retries remains open (`RESULTS.md` Open items A).

## What the plan's §16 "minimum coherent research release" would still need

1. Validated mechanics inference (A/B) — blocked on the missing tracker code.
2. Genuine joint updates — measured and found to add nothing here; the factorised per-world posterior is the
   shippable version (E).
3. Integration into the planner — done as `hedge_bayes_mw` (opt-in).
4. Controlled belief-quality and paired-battle ablations — done (C, E).
5. Fixed / changing-opponent and exploiter results — done (D; earlier B3).
6. Honest documentation of mismatch, cost and negative results — this document and the files it cites.

Full-proposal items still open: a value-learning experiment beyond the earlier pilots (F), a scouting planner that sees
damage-roll evidence (G), live concealment (H), new style menu and replay priors (D).
