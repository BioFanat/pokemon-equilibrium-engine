# Information gathering (scouting) and concealment: where the value is, what is buildable, what is not worth it

## Corrections (independent verification, 2026-11-16)
Verifier scripts: `recompute.py`, `lp_check.py`, `leak_test.py`, `claim5.py`, `toy_out.txt` (session scratchpad `verify_voi/`). No information leak was found (opponent-view sampler non-interference, 2880/2880 tests); the two-sided LP matches the brute-force normal form to 4e-16 on 300 instances; the dual extraction is correct. Corrections applied in place below:
- a. "Species uncertainty adds only 5 %" / "P5 refuted" is replaced by the two conditions reported separately (species known mean 1.113 % / median 0.241 %; back-row species unknown 1.515 % / 0.73 %; ratios 1.36× of means, 3.0× of medians); the per-decision comparison is Monte Carlo noise-dominated, so P5 is "not decided (noise-dominated)".
- b. "0/91 argmax changes involve `-tera`" becomes 0 of 21 (species known) and 0 of 26 (species unknown), with almost no power (2/91 decisions had a tera argmax); "never carried by a tera decision" is deleted; the drop after tera use (1.47 % → 0.22 %) is confounded with move reveals; headline "≈ 0.1 %" fixed to 0.22 %.
- c. Added: argmax changes in 21/91 decisions, one-stage regret > 1 % of range in 4/91, 16/91 involve an unrevealed move.
- d. Added the interim value for our true type, br_true − v_AF.
- e. "Safe side" follows from theory (maxmin), not from comparing 1.0 % with 3.4 %.
- f. §6 "upper bound on what a foul-play opponent leaves on the table" removed; the direction of the bias is not established.
- g. Scouting "ceiling" is an along-play proxy (a lower bound in expectation) for the children's η, which bounds the capturable value.
- h. Toy count 44/44 becomes 47/47, with caveats (single-agent decision problem; strict tie-breaks; no horizon-1 case).
- i. Turn-30+ bucket: 16 of 24 decisions from one game (seed 120066).
- j. Leaf statistics come from one worker only; other worker and combined lower bounds added.
- k. "u_true = BR to 1e-14" is relative to the payoff range; absolute maximum 2.3e-12 engine units.
- l. Stale anchors: world loop at `run_ourbot.py` ≈ :441–467 (not :416–437); `nash.py:154` is the definition of `full_matrix`.
- m. Minor formal leak added to Risks (`switch_out_move_second_saved_move` copied unmasked; chosen-but-unexecuted moves counted as revealed).

Status: ideation sprint with measurements, 2026-11-16. Interpreter `final/engine/.venv/bin/python` (the worktree has no
`.venv`; imports resolve to `engine/` of this worktree via `sys.path`), `OURBOT_DET_LEAF=1`, 2 workers, machine load
average 17–23 (a 10-worker gauntlet was running). Everything marked **measured** was computed in this session by the
scripts in `research/scripts/voi/` (raw records in `research/scripts/voi/out/`); *estimate* / *speculative* are
labelled. No project file outside `research/` was changed. Existing results are cited from `theory/tightening/voi.md`
(46 exact tera subgames, two-sided subgames, $\hat\eta$ along play) and were spot-checked here where used
(`eta_hat_det.jsonl` re-summarised: mean 0.0134 / median 0.0043 / 87 % tera-zero, identical to voi.md §2.3).

**Headline.**
1. *Measured.* Along real 6v6 play the hidden variables of the opponent's **active set** change our optimal action
   rarely and by little: per-attribute strategy-fusion value $\hat\eta$ is median 0 for moves, item and tera; means
   0.53 % / 0.45 % / 0.22 % of the payoff range; the per-world best action disagrees across attribute worlds in 16 % /
   6 % / 9 % of decisions. Item uncertainty is a *mis-valuation* (world values spread by a median 3.7 % of range) but
   not a *mis-decision* ($\hat\eta_{\rm item}$ median 0). All of this is "VOI as a property of positions".
2. *Measured, new.* Making the opponent's model of us honest (our team sampled from the opponent's public view, a
   one-stage two-sided Bayesian LP) is cheap in the engine simulation and changes the LP argmax in 21/91 decisions,
   but the change carries a one-stage regret > 1 % of range in only 4/91 decisions (16/91 involve an unrevealed move); the **ex-ante concealment value**
   (what the opponent's ignorance of our set is worth) is median 0.24 % and mean 1.1 % of range, decays to ≈ 0.1 %
   once four of our moves are revealed and to 0.22 % once our tera is used (confounded with move reveals), and no argmax change involves `-tera`
   (0 of 21 species known, 0 of 26 species unknown; only 2/91 decisions had a tera argmax, so this test has almost no power).
   The interim value for our true type, br_true − v_AF, is mean 1.28 %, median 0, > 1 % in 22 % (species known); mean 1.71 %, median 0.16 % (species unknown).
   The omniscient-opponent assumption is the safe side by theory: x_AF is maxmin against opponents who can condition on our
   type, a set that includes all type-blind opponents, so it guarantees v_AF against any opponent. Against one solver-picked
   (non-unique) equilibrium y it loses mean 1.0 % (0.948 %, median 0) of range, while the honest-model strategy would lose
   mean 3.4 % (median 0.9 %) against a best-responding omniscient opponent (species unknown: 1.05 % / 0 and 3.09 % / 1.07 %).
3. *Measured.* The scouting toy of workstream G is exact and cheap (`toy_scout.py`: probe worth 0.5 recovered,
   chosen iff cost < 0.5; zero-VOI reveal worth exactly $-c$; matched no-information control isolates the 0.5;
   47/47 checks; single-agent toy, see §8). The along-play proxy (a lower bound in expectation) for the children's $\eta$,
   which bounds the capturable value of the same mechanism in the live depth-2 agent, is the strategy-fusion value at the
   next decision, $\hat\eta_{\rm set}$ at the realised child: mean 1.3 %, median 0.4 % of range (977 consecutive
   own-decision pairs in `eta_hat_det.jsonl`; essentially the unconditional distribution, 1.34 % / 0.43 % over all 1003 decisions); an effect of this size per decision is unlikely to be visible at the
   0.04-score resolution of a 300–400-pair run (*estimate*; no unit map from range to score exists).
4. *Verdict.* A principled, measurable scouting/concealment module is buildable (options S1 and C1 below, 2–4 days
   each) and would make good report material because it is **exact at one stage and has matched controls**, but the
   measured value it can capture is at or below run resolution. Recommend: build S1 (public-observation grouping at
   the children, in the engine simulation) as the "VOI the agent captures" experiment with its no-information control,
   report C1 as a measurement (done here) rather than a planner change, and do not build the tera signalling game.

## 1. Findings

### 1.1 Setting (verified against code)
- The live agent (`engine/showdown/run_ourbot.py:441–467`) takes K = 4 worlds from foul-play's sampler with uniform
  weights (`fp/search/random_battles.py:61`); our side is our true team from the request JSON
  (`fp/battle/state.py:516–544`), the opponent's revealed mons get count-weighted sets filtered by hard evidence
  (`fp/data/sets/base.py:395–460`), unrevealed mons uniform species and sets. Each world's depth-2 matrix uses our
  TRUE team on both sides (`engine/nash.py:154`, the definition of `full_matrix`), so inside every world the opponent is omniscient about us.
- foul-play keeps **no record of what the opponent has seen of us** (only `revealed` on switch-in, tera, impossible
  items/abilities; `fp/battle/protocol.py:369, 1098–1105, 465–481`); our lead's `revealed` flag is not even set
  (`fp/modes/random_battle.py:46–50`). A live mirrored sampler needs a Tracker rebuilt from `battle._ourlines`, masking of
  our mons' moves/item/ability/tera, a fresh dataset object (`BattleMode.__deepcopy__` returns `self`), and it would still
  miss the one-sided inference (speed range, damage pruning) foul-play runs for the opponent only.
- The engine simulation already has the mirror: `hidden.Tracker` per side, `make_view(state, tr[3-s])` +
  `Belief` + `determinize` samples OUR side from the opponent's public view in 3 ms (`engine/hidden.py:101–342`,
  `engine/hidden_experiments.py:46–58`). All new measurements use it.
- Theory anchors: Theorem C (`theory/full_game.md:166`) with $\eta(t,s',b')=\sum_\theta b'(\theta)V(s',\delta_\theta)-V(s',b')$;
  assumption A3 (`theory/theory_map.md:148`, "world-only values; then $\eta_{\rm NL}\equiv0$ and concealment has no
  value"); the two-sided Remark (`full_game.md:188`, concealment value added once, $\ge0$). RESULTS.md:176 "concealment
  has zero value in our model" is a statement about the planner, not the game.

### 1.2 Where the information value is (VOI as a property of positions)

| hidden variable | measurement | decision changes? | size | source |
|---|---|---|---|---|
| opponent tera type (exact, 1v1, H=2) | VOI$_{\rm total}$ > 0.1 %Δ in 14/46 matchups, max 5.2 %Δ; expl(A3) > 0.1 %Δ in 3/46, max 3.1 % | rarely (A3's loss ≈ 0 in 43/46) | ≤ 5 %Δ | voi.md §2.1 (**measured**, prior) |
| opponent tera type, 6v6 along play | $\hat\eta_{\rm tera}$ median 0, 0 in 79 % (87 % in the 1003-decision run), mean 0.22 % | per-world argmax disagrees in 9 % (n=34) | max regret of the AF action in a tera world: median 0, p90 1.3 %, max 8.1 % | **measured** here (n=34) + voi.md §2.3 |
| opponent unrevealed moves (active mon) | $\hat\eta_{\rm moves}$ median 0, 0 in 65 %, mean 0.53 %, max 7.0 % | disagree in 16 % (n=49) | max regret median 0, p90 2.4 %, max 18 % | **measured** here |
| opponent item (active mon) | $\hat\eta_{\rm item}$ median 0, 0 in 53 %, mean 0.45 %, max 3.0 % | disagree in 6 % (n=17) | world *values* spread median 3.7 %, p90 9.9 % (mis-valuation, not mis-decision) | **measured** here |
| opponent ability | 0 in 3/3 | 0/3 | 0 | **measured** here (n=3; no conclusion) |
| Choice Scarf / Booster speed specifically | not separable at n=17 items; the item worlds include them when the role lists them | — | — | not measured |
| all opponent sets jointly (K=4 full-set worlds) | $\hat\eta_{\rm set}$ mean 1.2 %, median 0.3 %, > 5 % in 5 %; falls from 2.4 % (turns 0–9) to 0.03 % (turns 30+; 16 of the 24 decisions in that bucket come from one game, seed 120066) | per-world argmaxes disagree in 37 %; AF argmax outside the per-world set in 5 % | max regret of the AF action across worlds median 0.8 %, p90 8.1 %, max 24 % | **measured** here (n=91) + voi.md §2.3 (1003 decisions) |
| our tera type (two-sided exact 1v1) | concealment value 0 in 3/4, 0.39 %Δ in 1/4 | eq. tera timing differs by > 0.1 in 14 % of (matchup,type) pairs | ≤ 0.4 %Δ | voi.md §2.2 (prior) |
| our whole set, 6v6 along play | ex-ante concealment value mean 1.113 %, median 0.241 %, > 1 % in 34 %, > 5 % in 7 % (species known); mean 1.515 % / median 0.73 % with back-row species unknown (ratios 1.36× / 3.0×); interim value for our true type (br_true − v_AF): species known mean 1.28 %, median 0, > 1 % in 22 %; unknown mean 1.71 %, median 0.16 % | argmax(x) changes in 21/91, but with one-stage regret > 1 % in only 4/91; **0 of 21 (known) and 0 of 26 (unknown) involve `-tera`**, but only 2/91 decisions had a tera argmax (almost no power; hiding our tera can still change the opponent's play) (our tera was still available in 71 % of these decisions); 16/91 involve an unrevealed move | by reveals: 0 moves revealed 1.4 % → 4 revealed 0.1 %; tera unused 1.47 % → used 0.22 % (confounded: 18 of 26 tera-used decisions also have all 4 moves revealed) | **measured** here |

Units: fraction of that decision's depth-2 payoff range (engine leaf units; 1v1 rows use the subgame's Δ). No map to
score exists (voi.md §2.3); the oracle gap of ≈ 0.1 score (RESULTS.md:97–103) is the total of *all* opponent hidden
information including belief error, and is not decomposed into these terms.

How often is a decision in a regime where information changes the optimal action? *Measured:* for the whole hidden
set, the per-world best action disagrees in 37 % of decisions (early game more), but the agent-form action is already
one of the per-world best actions in 95 %; for a single attribute, 6–16 %. The size when it does matter: the
type-informed action beats the AF action by a median 0 and a p90 of 1.3 % (tera) / 2.1 % (item) / 2.4 % (moves) of
range; the whole-set p90 is 8 %.

### 1.3 VOI the agent could capture (needs a planner change)
- *Scouting.* The present agent is agent-form at the root (no fusion at stage 0) and world-only at the children
  (A3). Any depth-2 scouting mechanism can only recover the children's $\eta$, whose along-play proxy is
  $\hat\eta_{\rm set}$ at the next own decision: **measured** mean 1.32 %, median 0.38 %, p90 3.69 % of range (977 pairs;
  a lower bound in expectation of the children's $\eta$ (voi.md:57), essentially the unconditional distribution, 1.34 % / 0.43 % over all 1003 decisions).
  In the exact 1v1 tera subgames the realised along-play $\eta$ is median 0, max 2.8 %Δ, and the A3 agent's actual loss
  is ≤ 0.1 %Δ in 43/46. So the capturable scouting value is at most a few per cent of range at the decisions where
  it exists, concentrated in the first ten turns.
- *Concealment.* The capturable one-stage quantity is the regret of today's $x_{AF}$ against an opponent who does not
  know our set: **measured** mean 0.95 % (0.948 %), median 0, > 1 % in 14 %, > 5 % in 5 % (species known; species unknown 1.05 % / 0). The cost of switching
  (regret of the honest-model strategy against an omniscient opponent) is mean 3.4 %, median 0.9 % (species unknown 3.09 % / 1.07 %). The current
  assumption is the safe side by theory, not by this comparison: x_AF is maxmin against opponents who can condition on our
  type (a set including all type-blind opponents), so it guarantees v_AF against any opponent; the 1.0 % is against one
  solver-picked, non-unique equilibrium y, and the 3.4 % against a best-responding omniscient opponent. The honest model is the exploitative side.

### 1.4 Pre-registered predictions and outcomes (written in `two_sided_live.py` before the run)

| | prediction | outcome | verdict |
|---|---|---|---|
| P1 | per-world argmax disagreement 40–60 %; AF argmax outside the set < 15 % | 37 %; 5 % | borderline / holds |
| P2 | $\hat\eta$ order moves > item > tera; tera median 0; item median < 1 %; moves median < 2 % | means 0.53 > 0.45 > 0.22 %; all medians 0 | holds on means (medians uninformative) |
| P3 | concealment value (species known) median 0, > 1 % in < 10 % | median 0.24 %, > 1 % in 34 % | **refuted** |
| P4 | argmax change < 15 %; tera-involving < 3 % | 21/91 (23 %); 0 of 21 involve tera (underpowered) | refuted / not decided (2/91 tera argmax). Post hoc: argmax of a non-unique LP solution was a poor metric; with regret > 1 % the rate is 4/91 |
| P5 | back-row species unknown ⇒ concealment ≥ 2× species-known | means 1.113 % vs 1.515 % (1.36×), medians 0.241 % vs 0.73 % (3.0×); unknown > known in 42 % of decisions | **not decided (noise-dominated)**: 3 Monte Carlo samples per condition dominate the per-decision comparison |

Leaf determinism under load (from one worker, 4 of 6 games): 2.44 M leaf calls, 0.7 % first-try non-depth-1 (17,419 returns; wrapped i8 values appear in the depth
histogram), 21,720 retries, **0 fallbacks** — the `OURBOT_DET_LEAF` guard held at load average ≈ 20. The other worker had
810,264 calls, 5,343 retries and 4,114 non-depth-1 returns. Combined lower bounds: ≥ 3.25 M calls, ≥ 27,063 retries, 0.66 % non-depth-1.

## 2. Ranked options (impact × confidence ÷ cost)

| # | option | mechanism | expected gain (basis) | cost | risk | evidence |
|---|---|---|---|---|---|---|
| S1 | **Public-observation grouping at the children** (one-step lookahead over observation branches with posterior re-solve, on the K worlds) | In `full_matrix`, keep each child with its branch; key children of all K worlds by the belief-relevant observation (opponent move id used, tera flag, switch-in species, faint flags; HP rounded as `hp_public`); at each key compute the AF-LP over the group's **unpruned** depth-1 child matrices at the posterior $b'\propto b_k P(o\mid k)$; cell value = $\sum_o P(o)\,V(o)$ | ≤ children's $\eta$: mean 1.3 %, median 0.4 % of range at the next decision (**measured** proxy); below score resolution (*estimate*) | 2–3 days in the engine simulation (no foul-play changes): rewrite of `full_matrix` to return children; one LP per (cell, key): ≈ 169 cells × 2–4 keys × 1–3 ms ≈ 0.5–2 s per decision for the LPs (*estimate*). **Open cost item:** `pe.id(child,0).matrix` is alpha-beta pruned (55–80 % NaN cells, `nash.py:128–135, 155–156`), so the group LP needs an unpruned child matrix — either poke-engine exposes one (unknown; check first) or each child is expanded with `generate_instructions` + scalar leaf, which is `tera_subgame`'s cost class (15–110 s per 1v1 matchup) and makes S1 offline-only like S2 | correctness: posterior must use the same `Belief` the next decision uses, else the planner gains clairvoyance; latency 2–5× | toy exact (47/47; single-agent toy, see §8); voi.md option 4 deferred on the same numbers |
| S2 | **Information-set-correct depth-2 agent-form tree** (sequence-form LP, one continuation per information set) | `tera_subgame.seqform` generalised to K sampled worlds with the S1 key; root and children consistent | S1's gain plus root consistency; the extra is the root's $\hat\eta_{\rm set}$ already removed by agent-form (0) — so ≈ S1 (*estimate*) | 4–6 days; LP with ≈ $13\times$(keys)$\times13$ sequences, 4–40 s per matchup in 1v1 today | latency makes it offline-only; same clairvoyance risk | voi.md §2.1 (1v1 version runs) |
| S3 | **Scouting probes as explicit root actions** (Protect / status move / switch valued by S1 with a matched no-information control) | S1 plus logging of `V(a) − V_noinfo(a)` per root action | the information part of a Protect is the $\eta$ at its child; measured $\eta$ along play median 0 ⇒ rarely positive (*estimate*) | included in S1 | none beyond S1 | workstream G acceptance criteria met by the toy |
| C1 | **Honest opponent model** (our team sampled from the opponent's public view; two-sided one-stage Bayesian LP, play $x_{\rm true}$) | `two_sided_lp` in `two_sided_live.py`; L = {true} ∪ 3 samples, 16 matrices per decision | one-stage gain vs an honest-equilibrium opponent mean 1.0 %, median 0 of range; cost vs an omniscient opponent mean 3.4 % (**measured**) | engine side: done (4× matrices); live side: days (Tracker from `_ourlines`, masking, dataset copy, one-sided inference gap) | makes the agent *more* exploitable by an opponent who does know us (foul-play does not); value is median 0 | 91 decisions, 6 games |
| C2 | **Honest model restricted to unrevealed moves/tera of the active mon** (species, item, ability as the opponent sees them; only unrevealed moves and tera sampled) | same LP, cheaper sampler | the part of C1 that is about concealment proper; 0 of 21 / 0 of 26 tera-related changes (underpowered: 2/91 tera argmax), 16/91 unrevealed-move-related changes (**measured**) ⇒ ≤ C1 | same as C1 | same | same run |
| C3 | **Tera-timing signalling game** (two-sided exact subgame with our tera hidden; equilibrium tera timing) | `tera_two_sided.py` already computes it in 1v1 | concealment value 0 in 3/4, 0.39 %Δ in 1/4; timing differs by > 0.1 in 14 % of pairs (**measured**, voi.md §2.2) | 2 days to extend to live positions (H=2 exact) | result known to be ≈ 0 | voi.md §2.2 |
| C4 | **Reveal-penalty / entropy bonus** | leaf $-\lambda\,\mathbf 1[\text{reveal}]$ | effect size = $\lambda$, no theorem | hours | arbitrary; not a VOI | keep only as a labelled baseline (plan G.6) |
| M1 | **Oracle-gap decomposition by attribute** (oracle told only tera / only items / only moves) | 100 paired engine games per arm, 2 workers | descriptive: which attribute carries the 0.1 score gap | 30–60 min per arm | load-dependent leaf (guarded now) | voi.md option 6 (not run) |

Ranking: S1 > C1 (as a measurement) > M1 > S3 > C2 > S2 > C3 > C4. S1 is the only one that turns a Theorem C term
on inside the live planner with an exact matched control and a measured along-play proxy for the capturable value; C1 is already a result (this note),
and shipping it would trade safety for a median-zero gain. The ranking is unchanged by the corrections; tera
concealment (C3) stays "not worth building", but its measurement was underpowered rather than shown to be zero.

## 3. Recommended design outline (S1, engine simulation first)

1. `research/` prototype of `full_matrix_children(state)` returning, per cell $(a,\beta)$, the list of
   `(child, prob, key)` with `key = (opp move id or switch target species, opp tera flag, our/their faint flags,
   hp_public of both actives)`. First check whether poke-engine can return an **unpruned** depth-1 matrix at a child
   (`pe.id(child, 0).matrix` is alpha-beta pruned; `_child_value` survives that only because it takes a nanmax-nanmin);
   if not, each child needs its own `generate_instructions` expansion with the scalar leaf, and S1 is offline-only.
2. For the K worlds, build cells in parallel; for each $(a,\beta)$ pool the children of all worlds by key; posterior
   over worlds in a key group $b'_k\propto w_k\,P(\text{key}\mid k,a,\beta)$; group value = `agent_form_lp` over the
   group's child matrices at $b'$ (rows = our child actions, aligned by label; columns per world, as today).
   Cell value for world $k$ = $\sum_{\text{keys}}P(\text{key}\mid k)\,V(\text{key})$. The root stays today's
   agent-form LP over the K cell-value matrices.
3. **No-information control** (same code, `b'_k = w_k$ frozen for every key): isolates the physical effect of an
   action from the learning value, as in the toy's `hit` vs `hitprobe`.
4. **Clairvoyance guard**: assert that two worlds with identical keys get identical posteriors, and log, per decision,
   the number of keys per cell and the fraction of children merged (the toy's "coin" case must give exactly the
   frozen-posterior value).
5. Logging per decision: $V_{S1}(a)-V_{\rm noinfo}(a)$ for every root action (the estimated information value),
   the chosen action under each, and later the realised next-decision $\hat\eta$.
6. Live port (only if 4 shows an effect): foul-play's `prepare_battles` is unchanged; the change is in
   `nash.full_matrix` and `run_ourbot.find_best_move`'s matrix loop (`run_ourbot.py`, world loop ≈ :441–467).

## 4. Evaluation plan (pre-registered style, bayes_dynamics.md §4)

Arms (engine simulation, `hidden_experiments.play_pair`, paired seeds, both sides swapped, K = 4, `OURBOT_DET_LEAF=1`,
≤ 2 workers):
- A0 `agentform:4` (today) vs A1 S1 (grouped children, live posterior) vs A2 S1-noinfo (frozen posterior) vs A3
  S1 with the opponent told our team (public-team control for the concealment side; here the opponent *already*
  knows our team, so A3 = A0 for the opponent and is the baseline for C1-type arms).
- Primary metric (per decision, no score conversion): mean and p90 of $V_{A1}(a^*)-V_{A2}(a^*)$ over decisions, and the
  fraction of decisions where A1 and A2 pick different actions; secondary: paired score A1 vs A0 and A2 vs A0.
- Sample sizes: per-decision metrics need ≈ 2,000 decisions (25 games × 2 sides × ≈ 40) for a 0.1 % standard error on a
  1 % mean (per-decision s.d. ≈ 3 %, *estimate* from §1.2's distributions); score needs 400 pairs for ± 0.045 at 95 % — not reachable on 2 cores
  in a session (≈ 6 h), so score is reported as a bound, not a result.

Predictions (written before any S1 run exists):
- E1. $V_{A1}(a^*)-V_{A2}(a^*)\ge0$ at every decision (Jensen; a violation is a bug). Mean 0.5–1.5 % of range, median
  ≤ 0.4 %, p90 ≤ 4 %; larger in turns 0–9 than after turn 30 by ≥ 5×.
- E2. A1 and A2 choose different actions in 5–12 % of decisions; of those, < 1 % involve `-tera` and most involve a
  switch or a status/pivot move (Protect, U-turn), i.e. actions whose children are shared across worlds.
- E3. Paired score A1 vs A0 within ± 0.03 of 0.5 at n = 100 pairs (no detectable effect); A2 vs A0 likewise.
- E4. Keys per cell median 2–3 for the opponent's move choices and 1 for switches; fraction of children merged
  across worlds > 50 % (because the key hides damage rolls and set-dependent stats). If < 20 %, S1 degenerates to A0
  and E1's mean will be < 0.2 %.
- Concealment arm (if C1 is ever shipped): C1 vs A0 paired score within ± 0.03; C1 vs an exploiter that knows our
  team loses by ≥ 0.03 more than A0 does (the measured 3.4 % vs 1.0 % asymmetry); C1's behaviour change (argmax
  differences with regret > 1 %) disappears in the public-team control by construction.

What would change the ranking: E1 mean > 3 % of range or E2 > 20 % (then S2 and a live port are worth it); or an
attribute-decomposed oracle gap (M1) putting > 0.05 score on moves/items (then belief upgrades beat S1).

## 5. Rejected or not-worth-it ideas
- **Tera signalling game as a planner (C3)**: the exact two-sided value is 0 in 3/4 matchups and 0.39 %Δ in the
  fourth; 0 of 21 (0 of 26) honest-model argmax changes involve tera, but only 2/91 decisions had a tera argmax, so the
  test was underpowered rather than showing zero. Still not worth building; report the measurement, do not build the planner.
- **Honest opponent model in the live bot (C1 live)**: median-zero one-stage gain, 3.4 % mean cost against an
  informed opponent, days of foul-play surgery with a known inference gap (speed/damage pruning is one-sided).
- **Entropy / reveal penalty (C4)**: not a VOI; effect size is the knob. Baseline only.
- **Full information-set depth-2 LP in the live agent (S2 live)**: 4–40 s per position in 1v1; gain ≈ S1.
- **Species-level concealment (back row)**: P5 not decided (noise-dominated); species unknown gives mean 1.515 % /
  median 0.73 % vs 1.113 % / 0.241 % species known (ratios 1.36× / 3.0×).
- **Item/speed scouting (Scarf/Booster) as a target**: items are a mis-valuation (value spread median 3.7 %) with
  median-zero decision impact ($\hat\eta_{\rm item}$ 0 in 53 %, argmax disagreement 6 %); n = 17, so this is a lead,
  not a verdict.
- **Reading scouting off Showdown logs** (tera timing, Protect rate): both bots are A3-type; no equilibrium reference
  (voi.md §2.4).

## 6. Risks
- Units: all new numbers are fractions of the depth-2 payoff range; no score mapping; comparisons to the 0.1 oracle
  gap are of order only.
- The honest-model prior over our sets is uniform over {true, 3 samples} from `hidden.Belief` (role posterior on
  revealed moves + tera; no item/ability/damage inference). foul-play's real inference about us is sharper, but the
  direction of the bias is not established. The model's opponent prior always contains our true set at weight ≥ 1/4,
  which biases the concealment value down. The 4-type truncation and foul-play's inference are not ordered garblings,
  so no Blackwell ordering applies.
- Minor formal leak: the side field `switch_out_move_second_saved_move` is copied unmasked (`hidden.py:175`). It is
  stale-unrevealed in 469/6843 random-play decisions, with zero effect on matrices (max difference 0.0 in 4/4 states)
  or values. Also, `hidden.py:10–12` counts chosen-but-unexecuted moves as revealed, which over-informs the opponent
  and pushes the concealment value down.
- Argmax-of-LP is unstable (non-unique solutions); use regret-weighted change rates (post hoc here, pre-registered in §4).
- Sample: 6 games, 91 decisions, one seed series; items n = 17, ability n = 3. The attribute split is on the MAP set
  of the active opponent only (unrevealed back-row sets are held fixed).
- The leaf is time-budgeted; the det-leaf guard produced 0 fallbacks here, but any rerun must check `leaf_stats`.
- S1's posterior must be the same object the next decision uses (`hidden.Belief`), or the lookahead is clairvoyant
  (plan G acceptance criterion 3); the toy's `coin` check is the regression test.

## 7. Literature anchors (what is known / adapted / new here)
- **Strategy fusion and non-locality** — Frank & Basin 1998 (*AI* 100:87–123; wording of the definitions not
  verified from the full text); Long, Sturtevant, Buro & Furtak 2010 (AAAI, pp. 134–140): PIMC's failure is governed by
  leaf correlation, bias and the **disambiguation factor**; low correlation hurts PIMC unless disambiguation is high
  (0.7–0.9). *Adapted:* our measured disambiguation analogue is the opponent's revealed moves of the active mon:
  0.8 / 1.1 / 1.9 / 3.5 of 4 in turn buckets 0–9 / 10–19 / 20–29 / 30+, which matches $\hat\eta_{\rm set}$ falling from
  2.4 % to 0.03 % — Pokémon is a high-disambiguation game by the midgame, exactly where Lon10 predicts PIMC-style
  (world-only) play to be nearly safe. *New:* the per-attribute $\hat\eta$ in a real game.
- **Public belief states** — ReBeL (Brown, Bakhtin, Lerer, Gong, NeurIPS 2020) and Student of Games (Schmid et al.,
  *Science Advances* 9, 2023; not *Science*): value functions of public belief states; "safe" re-solving is Brown &
  Sandholm 2017, not ReBeL. *Adapted:* S1's group value `agent_form_lp` at the posterior is the exact PBS value at H=2
  (voi.md §0); a PBS value net is its learned approximation and was rejected for lack of training targets.
- **Bayes' Bluff** (Southey et al., UAI 2005): posterior over opponent strategies and a response to it. *Adapted:*
  our `hedge_bayes` style learner is this on styles; it is orthogonal to set concealment.
- **Value of information** — Howard 1966 (single-agent VOI ≥ 0); in games VOI can be negative except in zero-sum
  classes (Gossner & Mertens, abstract; Bassan, Gossner, Scarsini & Zamir 2003 give positive-value conditions;
  Kamien, Tauman & Zamir 1990 not verified). *Known here:* Theorem C's $\eta\ge0$ is the zero-sum case; the measured
  concealment value $\ge0$ (asserted, 91/91) is the two-sided Remark.
- **Signalling games** — Spence 1973, Crawford & Sobel 1982, Cho & Kreps 1987 (off-path beliefs). *Adapted:* the tera
  timing game is a signalling game with a verifiable message; measured value ≈ 0 (C3). Cowling, Whitehouse & Powley
  2015 obtain bluffing in MCTS by self-determinisation (adding worlds for what the opponent believes about us) — the
  closest sampling-based analogue of C1.
- **Pokémon** — PokaiTrainer (Yu26d §4.1) concedes our configuration to the opponent, as we do; foul-play filters set
  lists by hard evidence and samples worlds; PokéChamp, PokéLLMon, Metamon have no scouting or concealment model;
  no paper measures VOI or tera timing (`lit/joint_belief_voi_concealment.md:110`). *New:* measured per-attribute VOI
  and the honest-vs-omniscient asymmetry along real play.

## 8. Experiments run and reproduction
- `research/scripts/voi/two_sided_live.py` — 6 games (seeds 20011·i, i=1..6), 91 analysed decisions (every 4th, one
  dropped for payoff range 0), 18 min wall on 2 workers; output `out/two_sided_live.jsonl`, log
  `out/two_sided_live.log`. Launched as an 8-game run; the driver's progress print divided by a zero payoff range after
  the 6th game (fixed in the script; the 6 written games are complete and unaffected). A 2-game top-up was started and
  killed unfinished to keep the output file fixed; no second seed series is included.
  `OURBOT_DET_LEAF=1 final/engine/.venv/bin/python two_sided_live.py --games 8 --workers 2 --every 4`
- `research/scripts/voi/summarize_live.py` — all §1 numbers; checks: $u_{\rm true}=\mathrm{BR}_{\rm true}$ to $10^{-14}$
  relative to the payoff range (absolute maximum 2.3e-12 engine units; dual extraction consistent, 182/182), concealment value $\ge0$ asserted in every record.
- `research/scripts/voi/toy_scout.py` — exact scouting toy, 47/47 checks (3 + 6×6 + 4×2); `final/engine/.venv/bin/python toy_scout.py`.
  Caveats: the stage-1 matrices have identical columns and the opponent is passive, so the toy is a single-agent
  decision problem; "chosen iff c < ½" and "fused never probes" rely on strict tie-breaks; it reproduces Frank & Basin
  strategy fusion qualitatively but not the I-POMDP horizon-1 statement (the toy has no horizon-1 case; Gmytrasiewicz &
  Doshi 2005, arXiv 1109.2135, quote verified).
- Spot-checks of prior results: `theory/tightening/verify/voi/out/eta_hat_det.jsonl` re-summarised (matches voi.md
  §2.3); next-decision $\hat\eta$ from the same file.
- Not run (budget): M1, any S1 prototype, an item-only sweep large enough to separate Scarf/Booster.
