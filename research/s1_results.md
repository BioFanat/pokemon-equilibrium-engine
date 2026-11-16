# S1: information-set-correct depth-2 planner — implementation, gates, and the pre-registered E1/E2/E4 scores

Status: 2026-11-16, engine simulation only, `OURBOT_DET_LEAF=1`, ≤ 2 workers (a 10-worker gauntlet was running),
interpreter `final/engine/.venv/bin/python`. Design and predictions: `research/info_gathering_concealment.md` §3–§4
(option S1). Code: `engine/info_planner.py` (new; nash.py, hidden_agents.py, behavior.py, run_ourbot.py untouched),
scripts and tests in `research/scripts/voi/s1/`, raw records in `research/scripts/voi/s1/out/`.

**Headline.**
1. *Built and verified.* `engine/info_planner.py` implements S1 exactly as §3 designs it, with arms A1 (live
   posterior) and A2 (frozen posterior), plus a clairvoyant reference A0L. Five gates pass: the matrix extraction,
   (a) reduction, (b) toy, (c) Jensen and (d) no leak.
   **The report's open cost item is resolved.** Unpruned child matrices *can* be had from poke-engine as it stands, at
   ≈ 6 extra `pe.id(·, 0)` calls per child. They reproduce every stock cell, 5,256/5,256. S1 costs a mean of 1.56 s
   per decision (CPU) against 0.31 s for A0, a median ratio of 3.9×. That is not offline-only.
2. *Measured, n = 3,620 decisions (32 games, both sides).* V_A1(a*) − V_A2(a*) is mean 0.0022 % of range, median 0,
   p90 0; it is nonzero in 0.6 % of decisions. A1 and A2 choose different actions in 3/3,620 decisions (0.08 %).
   **E1 (mean) and E2 (rate) FAIL**; the Jensen and shape sub-predictions pass. **E4 passes** on all three counts:
   keys per cell median 2 (opponent moves) and 1 (switches), with 67.9 % of children merged across worlds.
3. *Why E1 is ≈ 0: it is the control, not a bug.* In the engine's chance model a public observation is (almost
   always) either impossible in a world or equally likely in every world that can produce it. Damage is the average
   roll, and branching happens only on crit or kill (§1). So the posterior inside a group equals the frozen one: the
   A1/A2 node weights coincide in 99 % of multi-world groups (TV ≤ 0.01; `diag_weights.py`, 1,003 groups, `out/diag_weights.txt`), and V1 > V2 in 0.28 % of them. The information
   arrives as **hard eliminations**: 38 % of the branch mass lies in groups whose world support is a strict subset of
   the worlds that have the cell. §3.3's "same code, b'_k = w_k" control credits these to both arms. As specified,
   E1 measures soft (likelihood-ratio) updating only, which this engine model nearly cannot produce.
4. *Supplementary, measured.* The full price of information-set correctness at the children (A0L − A1, per-world
   form, at A1's action) is mean 0.35 %, median 0, p90 0.59 % of range (> 0 in 28 %). It is of the same order as
   the report's η̂_set proxy (mean 1.3 %; a differently defined quantity). It moves the root action in 2.6 % of decisions, with regret > 1 % in 1.4 %.
5. *Ranking.* Neither §4 trigger fires: E1 mean > 3 % no (0.002 %); E2 > 20 % no (0.08 %). The ranking stands. S1's
   information value is not visible per decision, so a live port and S2 are not justified by this run.


## 1. Method

**Planner.** For each of the K = 4 sampled worlds (the same worlds A0 draws), every root cell (a, β) is expanded with
`pe.generate_instructions` into chance children. Each child gets a **public observation key** from the decider's view:
our action a; the opponent's option label β (move id, `-tera` included, or the switch-in's species); our whole side
after the turn (exact — it is ours); the opponent's public fields only (active slot/species, `hp_public`, status, tera
flag and type once used, types, item-gone flag, boosts, volatile statuses, side conditions with durations reduced to
presence, wish/future-sight presence, and species/`hp_public`/status of the slots revealed in our view); the field
without turn counters. Children of all worlds with the same key form one information set; one continuation strategy
x_o is shared within it. The group LP is `hidden_agents.agent_form_lp` with **one block per child node** (the opponent
sees the true node), over the children's unpruned depth-1 matrices, rows aligned by label (colmin fill).

- **A1** (live posterior): node weights π_n ∝ w_k · P(branch | k, a, β), i.e. b'_k ∝ w_k P(o | k, a, β).
- **A2** (no-information control, "b'_k = w_k frozen for every key"): each world present in the group keeps its prior
  weight (π̃_n ∝ w_k P(branch | k)/P(o | k)). A world with no node in the group cannot enter the LP, so it drops
  out in both arms: **hard eliminations are credited to both arms and V_A1 − V_A2 measures likelihood-ratio ("soft")
  updating only** (a lower bound on the learning value; on the toy, a perfect probe has A1 = A2). Both arms' x_o
  are *evaluated* under the live posterior, so V_A1(o) ≥ V_A2(o) by LP optimality.
- **Cell value.** Primary = the report's formula (§3.2): R_k[a, β] = Σ_o P(o | k, a, β) V(o). Variant `pw`
  (per-world): R_k[a, β] = world k's own payoff under the shared x_o. Each satisfies one bound and not the other
  (gate c): `rep` gives Jensen V_A1(a) ≥ V_A2(a); `pw` gives the clairvoyance bound V_A0L(a) ≥ V_A1(a).
- **Root:** today's agent-form LP over the K cell matrices; V(a) = Σ_k w_k min_β R_k[a, β]; a* = argmax x_A1.
- **A0L** (reference): every child its own group (clairvoyant children) with the same LP leaf.
- **A0** = today's `agentform:4` (stock `nash.full_matrix`, midpoint leaf) on the same worlds; it is the action played.

**Leaf.** The continuation is the LP value of the child's depth-1 matrix, not A0's midpoint leaf (`nash._leaf`), because
an information-set LP needs the mixed value. They agree at pure saddles (26 % of children in the check below); on
the others |val − mid| / ptp is mean 0.70 %, p90 2.5 %, max 9.6 %.

**Unpruned child matrices (the report's open cost item).** poke-engine 0.0.48 exposes no static `evaluate` and
`pe.id(child, 0).matrix` is alpha-beta pruned (each row is cut after its first cell ≤ alpha; row 0 and column 0 are
always complete; ≈ 50 % of cells NaN). A depth-1 return also carries the engine's row-label bug: the matrix is in
`get_all_options` order while `s1` is that list stably sorted by row minimum (`search.rs`,
`re_order_moves_for_iterative_deepening`); inverting the stable sort relabels rows exactly. A pruned row r is
recomputed with one more `pe.id(·, 0)` call on a copy of the child where r is side one's **first** option (never
pruned): move rows keep only that move enabled and set `force_trapped`; `-tera` rows also pre-terastallize the active
mon; switch rows disable all moves and swap the target into the first alive bench slot. `evaluate()` reads neither
`disabled` nor `force_trapped`. So S1 is **not** offline-only for cost reasons: it costs ≈ 6–7 `pe.id` calls per child (6.85 in the extraction test, 6.0 in the run)
(measured ≈ 1.6 ms/child vs ≈ 0.24 ms for A0's two leaf calls), not a `generate_instructions` re-expansion.
No mirror midpoint is used (the mirror's stock cells equal −Fᵀ in 5,047/5,071 cells).

**Engine chance model (matters for E1).** With damage branching on (the Python `generate_instructions`), poke-engine
uses *average* damage (0.925 × max roll) and branches only on crit vs non-crit, or kill vs non-kill at a KO threshold
(`generate_instructions.rs:2380–2440`). Public HP after a hit is therefore a deterministic function of (world,
crit/kill event); the event probabilities depend on the world only at KO thresholds (num_kill_rolls/16) and through a
few abilities/items. Two worlds that produce the same public key almost always produce it with the same probability,
so P(o | k) is 0 or the same constant across worlds: the information arrives as hard eliminations, and the soft-update
channel that A1 − A2 measures is nearly empty *by construction of the engine's chance model*. Showdown's 16 uniform
damage rolls would populate it (*speculative*, not measured).

## 2. Correctness gates (`research/scripts/voi/s1/tests/`, outputs in `out/gates/`)

All gates were rerun on the final code after the evaluation (`out/gates/*.txt`; every file ends `ALL OK`).

| gate | test | result |
|---|---|---|
| 0 matrix extraction | `test_full_d1.py`: 182 children of real positions | stock cells reproduced exactly: move rows 621/621, `-tera` rows 182/182, switch rows 988/988, complete rows 3,465/3,465. Pure maximin of the full matrix == `nash._child_value` 182/182. Mirror: 24/5,071 cells differ from −Fᵀ |
| (a) reduction | `test_a_reduction.py`: 6 decisions, K = 1, 2, 4 | all nodes distinguishable: A1 = A2 = A0L = the independent depth-2 AF reference (LP leaf), max relative diff 2.3e-15, for the root value and every V(a), both cell forms. K = 1 with real keys: A1 = A2 (6/6); A1 ≤ reference; A1 = reference in the 3 decisions with no within-world merge |
| (b) toy | `test_b_toy.py` on `toy_scout.py` | 129/129. A1: wait 0.5, probe 1 − c, probe chosen iff c < 0.5 (probe worth 0.5), coin 0.5 − c (zero-VOI reveal), noisy 0.8 − c, hitprobe − hit = 0.5 for every d. A0L = the toy's fused values. A2: coin = A1 (the §3.4 regression), noisy = 0.5 − c (soft update worth 0.3), probe = A1 (the hard elimination is credited to both arms) |
| (c) Jensen | `test_c_jensen.py`: 2,000 random instances + 3 real decisions | `rep`: per-group V1 ≥ V2 (0 violations), V_A1(a) ≥ V_A2(a) for every a (0/2,000), also at a*, and on real decisions. `pw`: clairvoyance bound V_A0L ≥ V_A1 (0 violations). Not theorems, reported: `pw` Jensen fails in 724/2,000; the `rep` clairvoyance bound fails in 439/2,000 |
| (d) no leak | `test_d_leak.py`: 924 children, 8 decisions | mutating the opponent's item, ability, unrevealed moves and their order, unused tera type, stats, EVs, exact HP inside a public bucket, every unrevealed slot, and `switch_out_move_second_saved_move` leaves the key unchanged in 924/924. Public opponent HP or our own HP changes it in 924/924. No group mixes root actions or opponent labels |


## 3. Evaluation run

`run_s1.py`: `hidden_experiments.play_pair` (paired seeds, both sides swapped), both seats today's A0 (`agentform:4`,
sampled); the A-seat is wrapped at runtime (`hidden_agents.Agent` replaced by a factory inside the worker; no project
file edited) to also run `info_planner.plan` on the same K worlds at every decision with > 1 legal option. A0's action
is played, so the rng stream and games are exactly A0 self-play; the a_side = 2 game of a pair replays the a_side = 1
game and logs the other side. Analysis time is excluded from the play clocks. Seeds 5,000,011 + 7,919 k. 85-minute
analysis deadline. Values are fractions of the decision's depth-2 payoff range (A0's K matrices, max − min).

**Sample.** 32 pairs = 32 distinct games (the a_side = 2 game replays the first; verified: 32/32 pairs have identical turns and winner, and both sides' decisions are logged), 64 game records, 0 errors.
3,627 analysed decisions; 3,620 have range > 0. 0 analysis errors. Det-leaf: 14.1 M leaf calls, 30,168 first-try
non-depth-1 returns retried, **0 fallbacks**. Wall time 88 min. Two games are long (587 and 590 decisions), so
cluster views are given too. Raw data: `out/s1_eval.jsonl`, log `out/s1_eval.log`, summary `out/s1_summary.{txt,json}`.

### E1 — V_A1(a*) − V_A2(a*), fraction of range (a* = argmax x_A1)

| | n | mean | median | p90 | max | > 0 |
|---|---|---|---|---|---|---|
| `rep` (primary) | 3,620 | 0.0022 % | 0 | 0 | 2.85 % | 0.6 % |
| `rep`, turns 1–9 | 524 | 0.0086 % | 0 | 0 | | |
| `rep`, turns 10–19 | 549 | 0.0018 % | 0 | 0 | | |
| `rep`, turns 20–29 | 484 | 0.0008 % | 0 | 0 | | |
| `rep`, turns 30+ | 2,063 | 0.0009 % | 0 | 0 | | |
| `rep`, largest game left out | 3,030 | 0.0022 % | | 0 | | |
| `rep`, mean of 32 per-game means | 32 | 0.0044 % | | | | |
| `rep`, max over all root actions a | 3,620 | 0.0166 % | | 0 | 4.39 % | |
| `pw` (per-world cells) | 3,620 | 0.0026 % | 0 | 0 | 3.99 % | 0.5 % |

Under `rep`, V_A1(a) ≥ V_A2(a) held for every root action in every decision. Under `pw` (no theorem), 3 decisions
violate it at a* and 19 at some a; the minimum is −0.33 % of range.

### E2 — A1 vs A2 action choice

`rep`: 3/3,620 decisions (0.08 %) differ: attack↔status twice and attack↔switch once. None involves `-tera`; all 3
involve a switch or a status move. Regret-weighted (A2's action loses > 1 % of range under A1's values): 0 %. `pw`:
2/3,620 (switch↔tera, attack↔status); regret > 1 % in 1 decision.

### E4 — keys and merging (public key; the report's literal §3.1 key in brackets)

- Keys per world-cell, opponent move choices: median 2, mean 2.48, p90 4 (703,093 cells) [median 2, mean 2.13].
- Keys per world-cell, opponent switches: median 1, mean 1.51, p90 3 (358,836 cells) [median 1, mean 1.40].
- Children merged across worlds: 67.9 % pooled; per-decision mean 76.4 % by count and 75.4 % by branch mass
  [78.5 % / 84.6 % / 83.1 %].
- Strict-support mass (hard eliminations): mean 38.3 % [13.1 %]. Within-world merges: 5.5 % of children [18.1 %:
  the literal key fuses distinct own-HP and status outcomes]. Root cells whose opponent label is missing in some
  world: 40.8 %.
- Multi-world groups per decision: mean 147. Groups with V1 > V2: 0.28 %. Groups needing colmin row alignment:
  0.3 % of decisions.

### CPU per decision (process CPU incl. engine threads, under load average 4–17; wall ≈ CPU)

| | mean | median | p90 |
|---|---|---|---|
| A0 (`agentform:4`: 4 × `full_matrix` + LP) | 0.31 s | 0.20 s | 0.76 s |
| A1 standalone (expand + unpruned child matrices + group LPs + root) | 1.56 s | 0.71 s | 4.03 s |
| A2 standalone | 1.56 s | 0.71 s | 4.02 s |
| of which: expansion and child matrices (shared) | 1.32 s | 0.58 s | 3.37 s |
| of which: group LPs, one-world groups (shared) / A1 multi-world / A2 multi-world | 0.13 / 0.11 / 0.11 s | | |
| full analysis (A1, A2, both cell forms, A0L) | 1.92 s | 1.01 s | 4.88 s |
| A1 / A0 ratio | 4.1× | 3.9× | 6.2× |

Children per decision: mean 649, median 405, p90 1,444. Unpruned-matrix engine calls per decision: mean 3,897
(≈ 6.0 per child). **Cost item verdict:** unpruned child matrices are needed (a singleton group already needs the
mixed value) and cost ≈ 4× A0 at the mean and 6× at p90 (p90 ≈ 4 s at K = 4, load average 4–17). That is inside the report's 2–5×
latency-risk band at the mean and above it at p90, and far below `tera_subgame`'s 15–110 s.

### Supplementary (not pre-registered)

- Clairvoyance gap at the children, V_A0L(a*) − V_A1,pw(a*): mean 0.35 %, median 0, p90 0.59 %, max 22.9 %;
  > 0 in 27.9 %. Root value gap: mean 0.38 %. Root argmax A0L ≠ A1 in 2.6 % (`pw`) / 4.1 % (`rep`); regret > 1 % in
  1.4 %. This is the size of the strategy-fusion term S1 removes, i.e. the most any scouting/no-information
  comparison on these worlds could show. It is of the same order as the report's along-play proxy η̂_set (mean 1.3 %), which is a differently defined quantity (next-decision AF proxy, midpoint leaf).
- Leaf change (A0 midpoint leaf vs A0L LP leaf, both clairvoyant): root argmax differs in 2.7 %; root value
  difference median 0.
- Today's A0 vs A1 (`rep`) root argmax differs in 5.3 %. This mixes the leaf change and information-set correctness.
- `rep` vs `pw` cell formulas: root argmax differs in 3.7 %; root value difference mean 0.008 %, range −13.9 % to
  +10.2 %.

## 4. Pre-registered scores (§4 text, scored literally)

| | prediction | outcome | verdict |
|---|---|---|---|
| E1a | V_A1(a*) − V_A2(a*) ≥ 0 at every decision | 0 violations in 3,620 (`rep`); also for every root action | **PASS** (by construction: a theorem under `rep`, gate c) |
| E1b | mean 0.5–1.5 % of range | 0.0022 % | **FAIL** (> 200× below the band) |
| E1c | median ≤ 0.4 % | 0 | PASS |
| E1d | p90 ≤ 4 % | 0 | PASS |
| E1e | turns 0–9 ≥ 5 × turns 30+ | 0.0086 % vs 0.0009 % (9.6×) | PASS (both tiny; driven by a handful of decisions) |
| E2a | A1, A2 differ in 5–12 % of decisions | 0.08 % (3/3,620) | **FAIL** |
| E2b | of those, < 1 % involve `-tera` | 0/3 | PASS (n = 3) |
| E2c | most involve a switch or status/pivot move | 3/3 | PASS (n = 3) |
| E3 | paired score A1 vs A0, A2 vs A0 within ± 0.03 at n = 100 pairs | not run (bound only, per task) | — |
| E4a | keys/cell median 2–3 for opponent moves | 2 | **PASS** |
| E4b | keys/cell median 1 for switches | 1 | **PASS** |
| E4c | children merged across worlds > 50 % | 67.9 % | **PASS** |
| E4d | if merged < 20 %, E1 mean < 0.2 % | antecedent false | not triggered |

E4's own prediction was that merging > 50 % implies E1 is non-trivial. Merging happened, but the merged worlds
carry identical likelihoods for their shared key, so it bought no soft information. E4c passing and E1b failing are
consistent once §1's chance-model point is taken in.

## 5. What would change the ranking (§4) and what to do next

- §4's triggers (E1 mean > 3 % or E2 > 20 %) are far from met: **the ranking stands**. Do not port S1 live, and do
  not build S2, on the strength of this run.
- The *control* is the weak link. As §3.3 specifies it, A2 isolates soft updating only, and the engine's chance model
  (average damage rolls) almost never produces soft evidence. Two things could revive a measurable E1, both outside
  this task's scope:
  1. Damage-roll branching (16 rolls or roll buckets) in the expansion, so that `hp_public` outcomes carry
     likelihood ratios. Cost grows with the number of branches; the matrices per child are unchanged.
  2. A control that also removes hard eliminations. That needs a counterfactual continuation for worlds absent from
     a group, and no principled choice was found. The clairvoyance gap above (mean 0.35 %) bounds what any such
     control could add on these worlds.
- The useful by-product is the unpruned depth-1 matrix extraction (`info_planner.full_d1`): exact, ≈ 6 calls per
  child. It makes any mixed-value leaf (or an LP-leaf A0) affordable.

## 6. Deviations from the report's design, and caveats

- **Key:** the primary key is the full *public* observation (task wording: "publicly distinguishable observations").
  The report's literal §3.1 key is computed alongside, for E4 only. The literal key uses `hp_public` for our own
  active and ignores status and boosts, so it fuses outcomes we can see (18 % within-world merges against 5.5 %).
- **Leaf:** the LP value of the unpruned child matrix, not A0's midpoint leaf; gate (a) is therefore against the LP-leaf
  depth-2 AF reference. Leaf change: |val − mid|/ptp has mean 0.7 % per child; root argmax changes in 2.7 %.
- **A2 on worlds absent from a group:** they are renormalised away (no counterfactual nodes), so hard eliminations
  count for both arms (§1, §2(b)).
- **Cell formula:** the primary is the report's Σ_o P(o|k) V(o). The per-world variant `pw` (world k's own payoff
  under the shared continuation) is reported alongside. It is arguably the information-set-correct one for the root's
  per-world minimum, but Jensen is not a theorem for it.
- **Posterior vs the next decision's Belief:** inside the tree A1 uses the exact Bayes posterior over the K sampled
  worlds given the public key. `hidden.Belief` at the next real decision uses hard evidence only. Since A1 ≈ A2 here,
  this gap does not affect the numbers.
- **Positions** are A0 self-play (A0's action is played), which matches the report's earlier per-decision measurements.
  Units are fractions of the depth-2 payoff range, with no mapping to score.


## Decision (2026-11-16)

Accept the null (option i). The pre-registered E1b/E2a failures are reported as failures. They are explained by the
engine: poke-engine branches only on crit/kill, so soft evidence is almost absent and the matched frozen-posterior
control removes almost nothing. Damage-roll branching (option ii) is a large engine change for at most the 0.35 %
clairvoyance bound at the children; no principled alternative control (option iii) was found. The per-decision log
`out/s1_eval.jsonl` (12 MB) is not in git; it is archived at `~/sandbox/ourbot/research_logs/s1_eval.jsonl.gz`.
