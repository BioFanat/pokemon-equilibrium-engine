# Plan: Bayesian inference from battle mechanics

Status: proposed implementation; no tracker behavior is changed by this document.

Baseline inspected: commit `1d37f82` (`E1 on real Showdown; E2–E5 methods and Modal runner`).

## 1. Goal and recommended scope

Upgrade the bot from remembering revealed moves and Tera types to reasoning about hidden sets using publicly observable mechanics: damage, action order, announced items and abilities, and selected activation effects.

In plain language: for every plausible opponent setup, ask whether it could have produced what we just observed. Give more weight to setups that explain the evidence, and use the updated distribution when sampling worlds for the existing agent-form solver.

Recommended scope is Gen 9 Random Battles, mechanics-based inference within a battle, with a common inference core for the internal simulator and real Showdown. Preserve the existing solver and search depth so experiments isolate the effect of better beliefs. Opponent play-style inference is a later extension: choosing a move and possessing that move are different kinds of evidence.

No user decision is required to start this plan. Assumptions are that the first deliverable is a reliable mechanics tracker, the current approximate Random Battle prior remains the starting point, and unverified mechanics are skipped with diagnostics rather than guessed.

### What success means

- Hidden-set probabilities become more accurate and calibrated on held-out battles.
- Every inference is explainable through recorded public evidence.
- No inference reads opponent-private state or unrevealed choices.
- Unsupported observations do not destroy a valid posterior.
- The agent can use updated beliefs without changing the payoff builder or solver.
- Battle strength and runtime are measured separately; improved inference need not immediately improve win rate with the existing shallow evaluator.

## 2. Current implementation and integration constraints

| Existing component | Current behavior | Required change |
|---|---|---|
| `randbats.py`: `PokeSet`, `sample_set`, `to_pokemon` | Role-based sets; approximate generator; role-specific EV/IV overrides | Add an exact candidate catalog and explicit prior masses under the same generator |
| `hidden.py`: `Tracker`, `SlotReveal` | Remembers species, moves, Tera; some chosen but unexecuted moves count as revealed | Introduce an observable-event boundary; only reveal a move when the observer can identify it |
| `hidden.py`: `role_posterior`, `Belief` | Posterior conditioned on revealed moves/Tera; rebuilt each decision | Retain as a baseline; add a persistent weighted candidate posterior |
| `hidden.py`: `make_view` | Copies opponent fields from the simulator, including fields requiring visibility review | Audit every field; represent unknown quantities explicitly |
| `hidden.py`: `_revealed_mon`, `determinize` | Builds a world; approximates hidden HP with one rounded value; reconstructs choice lock | Accept candidate-dependent HP and dynamic item/ability state |
| `hidden_experiments.py`: `play_hidden` | Constructs `Belief(view)` each turn; `Tracker.update` receives true states and both choices | Maintain a per-observer belief session; keep simulator truth outside the inference core |
| `hidden_agents.py`: `Agent._matrices`, `agent_form_lp` | Samples worlds and solves; internal posterior samples receive equal weights | Keep strategy logic fixed; introduce a world-sampling adapter |
| `showdown/run_ourbot.py`: `_recording_process`, `_finished_hook`, `make_move_fn` | Captures raw protocol history; Foul Play supplies hidden worlds through `prepare_battles` | Parse events once; add a selectable belief provider; consume final and forced-switch updates |
| `showdown/headless.py`: `public_lines` | Selects player-visible lines from simulator split messages | Reuse this visibility boundary for protocol fixtures and leakage tests |

The real-Showdown path does **not** currently use `hidden.Belief`: it uses Foul Play's sampler. Its pinned external checkout was not available for inspection here. Do not assume that sampler ignores damage or speed, and do not multiply our posterior into its weights without understanding its evidence model.

Known engine caveat: `randbats.py` documents three unsupported items becoming no-effect engine items. Treat this as a capability issue to check against the installed build, not as evidence that the true item cannot be present.

## 3. Representation: exact candidate sets, weighted by evidence

### 3.1 Enumerate the existing prior

For each species, enumerate roles, unordered four-move subsets, abilities, items and Tera types, retaining the role's level and EV/IV overrides. Under the existing generator:

`prior(role, moves, ability, item, tera)`

`= 1 / number_of_roles × 1 / choose(pool_size, min(4, pool_size))`

`  × P(ability | role) × P(item | role) × P(tera | role)`.

List frequencies matter when entries repeat. Canonicalize a complete set by its mechanics-relevant fields; if multiple role paths generate the same complete set, **sum** their masses. Do not assign uniform mass to the deduplicated candidates.

Measured from the checked-in data, role-labelled combinations before deduplication number 18 at the median species, 225 at the 95th percentile, and 2,633 at the maximum across 509 species. Exact enumeration is therefore a practical starting point. This is a count of possibilities, not a runtime benchmark.

Keep the correlated tuple intact. Independently sampling item and moves from separate posterior marginals could manufacture combinations absent from the set distribution.

### 3.2 Separate original set from current battle state

Each revealed opponent entity gets:

- Immutable original-set hypotheses with prior mass and current log weight.
- A public identity and reveal history, independent of hidden simulator team-slot numbering.
- Dynamic state: current item/ability overlays, known transformations, boosts, status, and candidate-dependent HP uncertainty.
- An event cursor and applied-event IDs, for exactly-once updates.
- Diagnostics: support count, entropy, skipped evidence, recovery events, and inference cost.

Original item and current item are different variables. An item removed, consumed or transferred during battle must not rewrite what the Pokémon originally held. Likewise, an announced replacement ability need not identify its original ability. Model supported transitions explicitly; suspend affected inference for unsupported transitions.

Use a per-entity factorization initially: the joint tuple within a Pokémon is preserved, while correlations between different unrevealed teammates remain approximate as in the current generator. If an observation couples two uncertain opponent entities, update a joint factor or skip it; do not invent an independence assumption silently.

### 3.3 Why persistence is necessary

Initialize the prior once, then apply only new evidence. Reconstructing a fresh role posterior each turn loses damage/speed evidence. Reapplying the entire history to an already updated posterior counts evidence twice. Battle reset clears all set beliefs; strategic expert weights carried across battles remain a separate mechanism.

## 4. Observable events and the no-leak boundary

The inference core consumes an allowlisted public snapshot plus normalized events, never `true_state`, a raw opponent Pokémon object, the chosen chance branch, the PRNG seed, or a submitted opponent command.

Proposed event envelope:

```python
@dataclass(frozen=True)
class EvidenceBatch:
    battle_id: str
    observer: int
    event_id: str
    pre: PublicSnapshot
    events: tuple[ObservedEvent, ...]
    post: PublicSnapshot
    completeness: str
    source_version: str
```

Events identify the actor/target, event position, observed move when identifiable, HP observations with their display denominator, and relevant public modifiers. Store the event-time context, not merely the state at the end of the turn. Preserve order across damage, healing, activation and switching.

Official Showdown protocol documentation describes move, damage/healing, critical-hit, item/ability, switch, and form-change messages, and distinguishes exact own HP from displayed opponent HP. Use that protocol as the observation contract; raw messages are not independent evidence factors. See sources below. Verify behavior against fixtures from the project's pinned Showdown revision before relying on current documentation.

### Adapter requirements

1. **Showdown adapter first:** consume observer-visible protocol and own private requests; use the existing raw-line capture and final-block hook. Retain a snapshot before the effects, because the existing move callback sees a battle already updated by Foul Play.
2. **Internal adapter:** simulator truth may enter an observation-producing harness, but output must match what the player could observe. Never identify an unexecuted move just from `m1`/`m2`. Probe whether the engine exposes enough semantic execution events. If it does not, use a trace-enabled engine adapter or restrict internal fixtures to validated observable cases; do not guess from arbitrary `repr` text and claim equivalence.
3. Audit `PUBLIC_PF`, `SF`, `opp_pending_move`, move-slot indices, sleep/rest counters, item absence and active-slot identity. Copying a field from an opponent state does not establish that it is public.
4. Resolve Illusion, transformations and ambiguous identity conservatively. Suspend affected inference until identity/state can be reconstructed; never attach evidence to an entity known to be ambiguous.

Test the boundary by changing hidden facts while keeping the visible transcript fixed: normalized observations and resulting beliefs must remain identical.

## 5. Bayesian update contract

Let `s` be an original-set candidate, `z` its latent current state, `a` our known action, `c` an observable opponent action, and `e` the supported mechanics evidence. The mechanics update is:

`w_new(s) ∝ w_old(s) × L(e | s, history, a, c)`.

For unknown current HP or other latent state:

`L(e | s) = sum_z P(z | s, history) × P(e | s, z, a, c)`.

Use normalized log weights. A reliable direct reveal can give a candidate zero likelihood. For an unsupported observation, return a typed skip result, **not** a zero likelihood.

### 5.1 Condition on actions; defer style inference

A legitimately revealed move constrains set membership. Do not otherwise reward a candidate because a guessed aggressive/passive policy would have selected that move. At this stage, the tracker is an evidence-conditioned approximation under an explicit action-conditioning assumption, not the proposal's complete posterior over set and style.

If the opponent's action is unobserved, skip mechanics inference that needs it unless a separately specified likelihood can marginalize it without importing an arbitrary policy. Do not infer a move from knowledge of the target bot's source code.

### 5.2 Avoid double-counting correlated evidence

One attack can produce a move line, damage, a critical-hit marker and an activation. They are related observations from one transition. Prefer the joint probability of the selected evidence batch.

If speed and damage are scored separately, require either disjoint randomness/known context or a valid conditional factorization such as `P(order | s) × P(damage | order, s)`. Never automatically multiply marginal likelihoods. A damage routine conditioned on a known hit/critical hit must not also charge that event's probability elsewhere unless the factorization explicitly includes it.

The existing payoff builder is not a likelihood function. Do not infer hidden sets using its heuristic leaf scores.

## 6. Evidence support, in implementation order

### Stage A: explicit reveals and state transitions

Support species, identifiable moves, Tera, explicit item/ability announcements, item removal/consumption, and supported public replacements. Use event provenance to distinguish an original attribute reveal from a temporary current-state change. Do not infer a move's hidden slot index. Repeatedly observing an already-known move adds no new membership information.

Called/copied moves require provenance: observing a called move does not necessarily mean it belongs to the original four-move set. Skip original-set filtering for cases not covered by an explicit rule.

Acceptance: with mechanics inference off, the new catalog reproduces existing reveal-only sampling probabilities on supported cases, and the real protocol never reveals choices hidden by a failure to act. If a protocol message explicitly identifies a failed move, that identification itself can be used.

### Stage B: speed/order evidence

Begin with two identifiable actions whose relevant ordering mechanics are supported. Calculate effective order using a verified engine rule/API, not a manually maintained table of base Speed alone. The local move JSON contains type/category/power, not the priority metadata needed for this feature.

- Deterministic matching order has likelihood 1, incompatible order 0.
- A true unresolved speed tie contributes the appropriate tie probability, verified against the pinned simulator.
- Different priority brackets normally provide no Speed-stat comparison; they can still involve hidden priority effects, which must be supported or skipped.
- Include known boosts, paralysis, Tailwind, Trick Room and candidate item/ability modifiers only after capability tests pass.
- Skip ambiguous order caused by unsupported priority changes, order-triggering effects, simultaneous-looking activations, mid-turn state changes or called moves.

Repeated deterministic order does not keep increasing confidence between candidates that both already predict that order. Repeated independent tie outcomes can add information. Record pre-action order context rather than end-of-turn boosts.

Acceptance: a slower candidate is eliminated only when all relevant order rules are accounted for; speed ties retain nonzero support; unknown effects produce a skip reason.

### Stage C: clean incoming damage

Start with a single, ordinary, successfully executed hit against our own Pokémon, whose HP and stats are known. For each opponent candidate, obtain a damage probability mass function from a verified mechanics interface. Compare the exact observed HP change to the distribution.

Preferred interfaces, in order: an engine damage-distribution API proven equivalent on fixtures; an engine branch projection that preserves relevant damage/order metadata; a small engine-side extension exposing those details. Determine availability at milestone 0. Do not assume `generate_instructions` exposes exact roll frequencies or critical-hit/order provenance merely because it supplies branch probabilities for gameplay.

Reject min/max-only scoring: two candidates can both admit 44 HP damage but assign different probability to it. Preserve integer rounding, duplicate damage outcomes and their probability mass.

Initial skips include critical hits unless explicitly supported, multi-hit moves, Substitute, fixed/fractional or variable-power special cases, ambiguous damage sources, unsupported abilities/items, and changes whose pre-hit context cannot be reconstructed. Apply ordinary public state transitions even when inference is skipped.

Use individual attack-associated HP changes, never total end-of-turn HP loss: residual damage, recoil, healing and hazards are separate effects. A KO is censored evidence: the observation constrains damage to at least the remaining HP; it does not reveal overkill. Either implement this tail likelihood or skip KO inference in the first release.

Acceptance: candidate likelihoods match an exhaustive reference in clean fixtures and preserve the generating set when it is supported by the prior/model.

### Stage D: outgoing damage and rounded opponent HP

Opponent display HP is not exact HP. Retain the numerator/denominator and a candidate-dependent distribution over integer HP consistent with the observed display. Derive display mapping from fixtures, including full-health and near-full-health edge cases; the current `hp_public` helper is not sufficient proof of live-Showdown rounding behavior.

For each candidate and compatible pre-hit HP, propagate possible damage to possible post-hit HP, filter against the displayed result, and retain the updated HP distribution. Do not subtract displayed percentages and treat the result as exact damage. Do not reset latent HP uniformly after every observation: that discards accumulated evidence and biases sequential updates.

For an illustrative exact-HP target at 30 HP, an observed KO supports rolls of 30, 31, 32 and larger. For a percentage-displayed target, sum over both compatible starting HP and damage outcomes. Cap damage at remaining HP when matching observed state changes.

World construction must sample HP from this filtered conditional distribution, not replace it with `round(percent × max_hp)` after inference.

Acceptance: small-state exhaustive filtering and the incremental implementation agree over multiple hits, healing and a KO; different display precision can change uncertainty without leaking true HP.

### Stage E: supported activations and negative evidence

Add residual healing/recoil, hazards, immunities and related effects one verified rule family at a time. Treat immunity as evidence allowing all supported explanations, not proof of one particular ability.

Absence of an activation is usable only when its prerequisites and transcript completeness are known. For example, no healing message is weak or irrelevant when already at full HP. Repeating one move is not mechanics evidence of a Choice item. Observing different moves only excludes a lock when reset, replacement and effect rules are all modeled.

Keep original EV/IV/level support from the current role prior. Broader arbitrary stat spreads and learned behavioral likelihoods are separate expansions, not features claimed by this release.

## 7. Failure handling and runtime

### No-support policy

Distinguish three failures:

| Failure | Response |
|---|---|
| Unsupported/ambiguous soft evidence | Leave weights unchanged for that factor; advance public state; log why |
| All candidates fail a new mechanics likelihood | Roll back that factor; retain previous valid beliefs and direct reveals; record a model-mismatch case |
| A reliable direct reveal is outside prior support | Expand support with a documented proposal consistent with all direct reveals, or enter an explicit degraded mode; never silently reset to a contradictory prior |

Recovery must also repair or invalidate any latent HP filter advanced under a rejected factor. Downstream decisions may use a documented reveal-only fallback while affected mechanics factors are suspended. Recovery never overrides an explicit reveal merely to retain a high-scoring candidate.

Do not add an unconditional likelihood floor that resurrects directly impossible sets. If later needed, specify and calibrate a contamination model for noisy mechanics factors only, and report the mixing parameter and recovery frequency.

### Cost controls

Exact set enumeration does not require a full battle simulation for every candidate every turn. Group candidates with the same mechanics signature for the observation, compute a likelihood once per group, and reuse it. Cache keys include all relevant public context and candidate modifiers, engine version and generation; clear/version the cache on transitions and rule changes.

Measure before setting a final budget. Initial target: median inference overhead at most 10% of baseline decision CPU time and p95 at most 25%, on the same fixtures/hardware; these are proposed acceptance targets, not measured results. A fixed work budget may skip complete evidence groups before scoring them. Never partially update only the candidates that happened to fit in the budget.

Track counts of eligible observations, applied updates, skips by reason, candidate/group counts, cache hits, mismatch recoveries, latency and candidate truth support. Preserve seeds for world sampling, but never feed simulator seeds to likelihood calculations.

## 8. Proposed modules and APIs

All names below are proposed, not existing commands or files.

| File | Responsibility |
|---|---|
| `belief_events.py` | Immutable observable snapshots, event types, visibility rules, grouping and evidence IDs |
| `belief_catalog.py` | Complete-set enumeration, original prior mass, deduplication, support diagnostics |
| `belief_tracker.py` | Persistent per-battle session, candidate weights, dynamic state, HP filtering, recovery |
| `belief_likelihoods.py` | Supported mechanics kernels and typed skip results; no strategy evaluator calls |
| `belief_worlds.py` | Posterior-to-world adapter, latent HP sampling, dynamic overlays, existing belief compatibility |
| `showdown/belief_observer.py` | Observer-specific protocol parsing and event-time snapshots |
| `test_belief_tracker.py` | Catalog, Bayesian update, recovery, persistence and determinization tests |
| `test_belief_observer.py` | Protocol fixtures, identity, final-block and leakage tests |
| `belief_eval.py` | Replay likelihood/calibration evaluation and experiment summaries |

Core interface sketch:

```python
session = BeliefSession(prior=catalog, config=config)
session.observe(batch)                    # idempotent; public data only
posterior = session.snapshot(view)
worlds = posterior.sample_worlds(K=4, rng=rng)
strategy = agent_form_lp(build_matrices(worlds), worlds.weights)
```

Preserve a wrapper implementing `sample(slot, rng)`, `map_set(slot)`, and `sample_unrevealed(rng)` where practical. Extend world construction separately for latent HP and dynamic attributes. Keep the old tracker selectable for reproducing historical results, but evaluate a corrected reveal-only observer as the main ablation control so fixing visibility leakage is not mistaken for a mechanics-inference effect.

Sampling convention: independent samples **from the posterior** get equal Monte Carlo weights. Do not multiply by posterior probability again. A deliberately selected weighted support uses its normalized masses/importance weights instead; every world provider declares which convention it uses.

## 9. Real-Showdown integration without double-counting

Add a belief-source selection separate from strategy-method selection. Suggested values are `foulplay_baseline`, `reveals_v2`, and `mechanics_v2`.

Recommended supported route: use Foul Play for transport and battle bookkeeping, while a fresh catalog prior plus our normalized observable events owns the new posterior and world sampler. Audit all imported snapshot fields so Foul Play's inferred/private values do not become public facts in our tracker. Validate conversions and own legal options before replacing `prepare_battles` in the new path.

Do not simply take Foul Play's already-conditioned worlds and multiply by likelihoods for the entire damage/speed history. Reweighting such proposals would require their known proposal probabilities and evidence provenance, with a valid importance-weight calculation. Until that contract is available, keep the providers independent.

Run `mechanics_v2` in shadow mode first: parse, update and log without affecting actions. Process observations at every protocol boundary, including forced switches and the last game block, not only when the bot gets an ordinary move choice.

For controlled comparisons, `reveals_v2` and `mechanics_v2` use the same prior, observer, K=4, solver, leaf evaluator and search depth. Compare stock Foul Play separately as a whole-system baseline; an improvement over the old internal tracker does not establish an improvement over Foul Play's inference.

## 10. Validation and experiments

### Required correctness checks

1. **Catalog:** masses sum to one; duplicate generation paths retain their combined probability; direct conditioning matches hand calculations and existing role posterior results under the same assumptions.
2. **Visibility:** identical public history produces identical observations/posteriors despite changed hidden state; unexecuted unknown choices stay unknown; own private HP never leaks to the other observer.
3. **Mechanics contract:** supported damage PMFs, priority, effective Speed, ties, critical hits and rounding are tested against the pinned simulator. Unsupported families reliably skip.
4. **Sequential filtering:** no double application on replay/reconnect; multiple hits and healing preserve HP distributions; original/current item and ability transitions remain distinct.
5. **Joint evidence:** known correlated event fixtures agree with an exhaustive transition reference; order/damage are not multiplied as independent marginals.
6. **Recovery:** impossible soft evidence cannot wipe out the posterior; out-of-prior direct reveals trigger documented degraded behavior; NaNs and zero normalization are never silently returned.
7. **Agent compatibility:** posterior samples contain all direct reveals, legal actions remain valid, no sampled world leaks true unseen teammates, and weighted/equal sampling conventions match the solver input.

### Belief quality before battle strength

Use synthetic battles with held-out generation seeds for exact ground truth. Separately use pinned Showdown-generated logs for external mechanics validation and realistic prior-support mismatch. Evaluators may inspect true teams; inference code may not.

Measure original item/ability/Tera and unrevealed-move marginal log loss, Brier score, calibration, credible-set coverage, posterior mass of the true complete set when supported, and prior-support rate. Report out-of-support cases separately; do not exclude them from system-level coverage/failure statistics. Track current item/ability state separately from original-set accuracy.

Do not use minimum entropy as the success criterion: a confidently wrong tracker is worse. Repeated positions within a battle are correlated; aggregate uncertainty by battle/paired seed, rather than pretending each turn is independent.

### Battle ablations

Compare corrected reveals-only, +explicit attributes, +speed, +clean damage, and the supported combined tracker. Keep K fixed at 4 initially; study K separately. Use paired team/battle seeds with seat swaps, held-out data, fixed solver/search settings, and log inference CPU separately from decision CPU.

Begin with 50 pairs per comparison as an engineering pilot. Use its paired-score variance to plan a confirmatory run for a prespecified practically relevant gain; 300 pairs is a reasonable starting proposal, not a guarantee of adequate power. Predeclare the primary comparison, report all variants, retain errors and turn-cap draws, and account for correlated runs/shared seeds. A null win-rate result can coexist with improved beliefs because the heuristic leaf remains a bottleneck.

Store new experiment outputs under a new versioned tag, with engine/data versions, configuration, public evidence, recovery counts, inference timings and evaluation seeds. Do not overwrite E1/E4 or their historical artifacts.

## 11. Implementation milestones and exit gates

| Milestone | Deliverable | Exit gate |
|---|---|---|
| M0: mechanics/visibility spike | Confirm installed engine APIs, inspect pinned Foul Play sampler, create pinned Showdown fixtures, audit copied fields | Written capability matrix; ambiguous engine features marked unsupported; reproduce baseline before integration |
| M1: exact reveal-only posterior | Catalog, persistence, normalized direct-reveal events, dynamic attribute skeleton | Same prior as baseline; correct reveal probabilities; no chosen-move leakage; exactly-once replay |
| M2: speed inference | Supported action-order likelihoods and context capture | Priority/tie/modifier fixtures pass; conservative skips; truth retained in supported cases |
| M3: incoming damage | Exact own-HP likelihoods for clean hits and optional censored KOs | Verified PMFs; damage/source association; no end-of-turn subtraction; recovery tests pass |
| M4: rounded HP and transitions | Candidate-dependent HP filters and selected additional mechanics | Exhaustive sequential-filter agreement; supported transformations/items reconstruct correctly |
| M5: world sampling and shadow rollout | New belief provider for real Showdown; safe internal adapter for applicable ablations | No inference truth access; valid worlds/actions; shadow logs show calibrated inference and acceptable latency |
| M6: controlled evaluation | Belief-quality report, fixed-K ablations, confirmatory battle comparison | Reproducible metrics and confidence intervals; prior mismatch and skips reported; no unsupported performance claims |

Each milestone should be a reviewable change. M0–M3 define the first useful release; M4 onward broadens coverage and validates end-to-end strength. The protocol observer and core tests can be implemented before an internal trace extension if the engine cannot faithfully expose execution events.

Do not launch Modal jobs or install/rebuild dependencies merely to complete this planning document. Those are implementation/evaluation tasks after the plan is adopted.

## 12. Later extension: joint set and opponent-style inference

Once mechanics likelihoods are validated, add an explicit opponent action model and consider:

`P(S, style | history) ∝ P(effects | S, actions, history)`

`                              × P(opponent action | S, style, history)`

`                              × P(S, style | previous history)`.

Unlike mechanics inference, this requires assumptions about how the opponent chooses actions. Repeated attacks, switching tendencies and apparent choice locking become behavior evidence only through that model. Keep the action model versioned and ablated, and do not claim the current mechanics-conditioned update already implements the proposal's full joint posterior.

## Sources and evidence limits

- Local code and set data inspected at commit `1d37f82`: `hidden.py`, `hidden_agents.py`, `hidden_experiments.py`, `randbats.py`, `pkbattle.py`, `showdown/run_ourbot.py`, `showdown/headless.py`, `showdown/SETUP.md`, and `data/SOURCES.md`.
- [Official Showdown simulator protocol](https://github.com/smogon/pokemon-showdown/blob/master/sim/SIM-PROTOCOL.md), consulted for observable event families and HP visibility. The current documentation was accessible; the commit-specific page was not retrievable during planning. Pin behavior with fixtures from Showdown commit `7e743761cd13cdb361fc0f58eb5a866284894f5b` before implementation.
- Existing setup pins Foul Play to `8f541f80f91f960603673796559f796b10ccf0fb` and poke-engine to `0.0.48` with Terastallization. Their external source/runtime was not verified in this planning task. Engine inference support is an M0 question, not an assumed capability.
- Candidate counts were computed from checked-in JSON using the existing role-generation structure. No simulator tests, training jobs or live battles were run for this document.
