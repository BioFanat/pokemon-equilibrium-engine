# Final ladder bot handoff: stratified native MCTS

Recorded November 17, 2026. Branch: `feat/mechanics-belief-tracker`.

The final bot is **`mcts_native` with `item_stratified` sampling, four hidden worlds, and a total 1000 ms native search budget per decision**. It plays Gen 9 Random Battle as `bot_bonp`, with up to five simultaneous games through one connection. The user reported reaching the **top 500**. The saved game ledgers establish an ending Elo of **2090**, a peak of **2141**, and **66 wins / 34 losses across 100 completed games with this configuration**. The rank itself was not captured in the ledgers.

This file is the entry point for another human or agent. The machine-readable configuration, dependency versions, dataset hash, and runtime source hashes are in [final_ladder_bot_manifest.json](../research/final_ladder_bot_manifest.json).

## Exact configuration

| Setting | Final value |
| --- | --- |
| Method | `mcts_native` |
| Sampler | `item_stratified` |
| Base belief source | `foulplay_baseline` |
| Worlds per decision | 4 |
| Native budget | 1000 ms total, normally 250 ms per world |
| Native search threads | 1 per world search |
| Root selection | Prior-weighted normalized legal root visits, then deterministic argmax |
| Tie break | First maximum in the legal action order |
| Concurrency | At most 5 games; solver calls are serialized |
| Format | `gen9randombattle` |
| Evaluated server | `wss://sim3.psim.us/showdown/websocket` |
| Cross-game learned state | None for this method |
| Neural network / newly trained model | None |

The ordinary CLI defaults still select older methods. **Pass the method and sampler explicitly** to reproduce the ranked bot. Increasing the search budget, replacing the catalogue, enabling the mechanics provider, or selecting S1 creates a different candidate.

## Decision pipeline and changes used by this bot

### 1. Foul Play tracks the live game and supplies compatible hidden sets

The pinned Foul Play client handles the WebSocket protocol, current teams, observed moves, item/ability inference, HP, PP, and battle conditions. Its Random Battle set catalogue and compatibility logic remain the source of hidden-set proposals. We do not inspect the opponent's actual hidden team.

The earlier `integration/joint-belief` work was merged at `014ad4b` through merge `bacfe12`, after the local main merge `7db176f`. The final ladder runs used base commit `5e7cb77` plus the runtime changes now committed with this handoff. Later upstream integration changes are not part of the measured configuration.

### 2. Item/ability stratification improves four-world threat coverage

Implementation: [stratified_worlds.py](../engine/showdown/stratified_worlds.py).

The old baseline independently sampled four compatible sets and could omit an important active-opponent hypothesis. The new sampler groups compatible active sets by their **effective current item and ability**. It respects already known items and abilities, including an item that is currently absent.

For class masses `p_i` and four available slots:

- With at most four positive-mass classes, give each class one slot, then assign remaining slots to the largest deficit `4*p_i - quota_i`. Draw full sets within each class by their original catalogue counts. Each sample receives weight `p_i / quota_i`, preserving the class's exact probability mass.
- With more classes than slots, use randomized systematic resampling. Each selected world has weight `1/4`; class probabilities are preserved in expectation, but not every class can appear in one decision.
- Shuffle the allocated slots. Fill revealed reserves and unrevealed teammates using the existing upstream sampling and constraints, then apply the existing move locks.
- If no positive active-set candidates remain, use the baseline sampler and log that fallback explicitly.

This changes finite-sample coverage rather than the underlying frequency prior. It does not give rare threats the same mass as common threats, stratify the entire unrevealed team, or create a carried joint set/style posterior.

In a 5000-draw Dragalge item diagnostic, the original four draws omitted Choice Specs 1691 times (33.82%); stratification omitted it zero times while retaining its catalogue mass. This diagnostic establishes coverage behavior, not a causal ladder strength estimate.

### 3. Native MCTS replaces the matrix/Hedge action-selection layer

Implementation: [native_mcts.py](../engine/showdown/native_mcts.py), registered by [run_ourbot.py](../engine/showdown/run_ourbot.py).

For each sampled world, call the installed native `poke_engine.monte_carlo_tree_search` with one thread. Split the **total** 1000 ms budget exactly among the sampled worlds; it is not 1000 ms per world. World searches run sequentially.

Let `n_k(a)` be world `k`'s root visits to a server-legal action, and `w_k` its normalized prior weight. The aggregate score is:

```text
score(a) = sum_k w_k * n_k(a) / sum_{b in legal} n_k(b)
action   = first argmax_a score(a)
```

Worlds with no legal visits contribute no score; the final score vector is normalized again. If every world has no legal visits, choose the first legal action and log an explicit fallback. No such fallback occurred in either stratified ladder batch. With only one legal action, bypass native search.

Normalizing inside each world is essential: pooling raw visits would let a faster world change its prior probability merely by completing more iterations. The selected action is the **argmax**, not a random sample from the aggregate visit distribution. Logs distinguish `mcts.visit_policy` from the actual one-hot `pfull` played policy.

These are separate searches over determinizations with aggregation at the root. They do not share a single information-set search tree. The native engine supplies its existing mechanics, tree search, and evaluation; no new learned value function was introduced.

### 4. Preserve the mechanics and legal-action boundary

Shared support, already included in the branch history:

- [battle_adapter.py](../engine/showdown/battle_adapter.py) preserves accuracy and evasion stages for both sides, which the pinned upstream converter otherwise resets. Other state conversion remains delegated to the pinned client.
- Intersect native root options with the live server request. Honor explicit disabled flags, exhausted PP, trapping/forced switches, and tera availability. `maybeDisabled` remains uncertainty rather than a prohibition.
- Map moves, tera variants, and switches by identity rather than indices that the engine may reorder. Do not globally ban immune attacks or setup moves; switches, tera, forced choices, and other mechanics can make those actions reasonable.
- The native API has no root-action mask. Filter unavailable root visits after search and log the discarded fraction; future-state mechanics are not modified to imitate a root mask.
- [revival_guard.py](../engine/showdown/revival_guard.py) bypasses search for Revival Blessing's mandatory fainted replacement and forced recharge. It preserves the actual last selected move through the mid-turn revival selection. Revival targets use a fixed first-fainted-reserve rule, not strategic target evaluation.

### 5. Run five games without mixing their battle state

Implementations: [ladder.py](../engine/showdown/ladder.py), [ladder_concurrency.py](../engine/showdown/ladder_concurrency.py), and [headless.py](../engine/showdown/headless.py).

One signed-in connection has a central reader and isolated room queues. Each battle gets its own decision function, context, and log through `BattleMovePool`; completed or failed battles release their slot. A solver lock serializes computations because upstream sampling uses global RNG state. Network waits overlap, so five active games do not imply five simultaneous CPU searches. Public matchmaking starts at most one new search every 20 seconds and never exceeds the reserved capacity.

For older Hedge methods, each stream also has its own persistent learner/checkpoint and final feedback. The final MCTS method uses the isolation machinery but does not read or update Hedge/style checkpoints. An error stops new matchmaking and attempts to close the failed battle while draining the others.

The headless harness supports `mcts_native`, paired identical teams with seats swapped, and the existing forced-action guards. It uses the built simulator and handles early simulator exits.

### 6. Make evaluation resumable and auditable

Implementations: [run_public_evaluation.py](../engine/showdown/run_public_evaluation.py), [ladder_metrics.py](../engine/showdown/ladder_metrics.py), [audit_decisions.py](../engine/showdown/audit_decisions.py), and [summarize_tactical_run.py](../engine/showdown/summarize_tactical_run.py).

The wrapper pins method, sampler, world count, budget, and target in `evaluation.json`. Resuming counts completed ledger games toward the original target and rejects incompatible settings. Concurrency changes are recorded. An account lock prevents overlapping managed runs. The password is read from the Terminal environment or a secure prompt and is not saved.

Per-battle records contain sampled native states, world weights, sampler strata/quotas, root visits/scores, selected legal action, timings, filtering, and fallback diagnostics. Local replays omit private request payloads. Elo summaries deduplicate by battle ID and report the first recorded starting rating, last ending rating, net change, peak/range, and results. Source copies and hashes were frozen before each measured batch.

Those source archives were prepared before launch by the evaluation operator; the current wrapper does not automatically archive source files. For a new measured candidate, record its commit and source hashes or copy its runtime sources before connecting. The checked-in final manifest provides the reference hashes for this bot.

## Results and what they establish

| Candidate / cohort | Completed games | Wins–losses | Starting Elo | Ending Elo | Change | Peak |
| --- | ---: | --- | ---: | ---: | ---: | ---: |
| Corrected per-world Hedge reference | 50 | 25–25 | 1521 | 1470 | -51 | See its report |
| Native MCTS, baseline sampler | 50 | 33–17 | 1470 | 1677 | +207 | 1741 |
| Final bot, first stratified batch | 50 | 33–17 | 1677 | 1885 | +208 | 1979 |
| Final bot, extension | 50 | 33–17 | 1885 | **2090** | **+205** | **2141** |
| Final bot, combined stratified cohorts | **100** | **66–34** | **1677** | **2090** | **+413** | **2141** |

The extension also had **five interrupted attempts with unknown local outcomes** during a connection loss. They are excluded from completed-game win rate. The server rating fell from 1984 to 1920 during that gap; the actual endpoint change above includes that 64-point decrease. After reconnecting, 36 games finished 23–13 and raised Elo from 1920 to 2090. There are 105 attempts and 100 completed local results across the two final-bot cohorts.

The final extension has all 50 completed-game logs/replays and 1548 decisions, with zero selected illegal actions and zero native no-visits fallbacks. Mean policy time was 1.060 seconds, maximum 1.436 seconds; including queue/dispatch time, mean was 1.194 seconds and maximum 3.325 seconds. Two decisions discarded unavailable `struggle-tera` root visits, at most 0.969% of a world's visits, and selected legal switches. All frozen source hashes matched.

These are sequential public ladder observations at different ratings and against different opponents. They support retaining this configuration as the leading measured candidate, but do not isolate a causal Elo gain for each change or establish a settled rating.

## Changes present in the branch but inactive in the ranked bot

| Work | Status / why it matters for handoff |
| --- | --- |
| Matrix terminal rewards | Earlier fixes in `nash.py` and `info_planner.py` give completed wins/losses/draws exact +/-10000/0 values. They affect matrix/S1 methods; native MCTS delegates evaluation to the installed engine. |
| Corrected safest expert | `world_learning.py` uses `argmax_a sum_k w_k min_b M_k[a,b]` in per-world Hedge, instead of minimizing the averaged union matrix. Native MCTS does not call it. |
| Consistent learning feedback | Per-world Hedge conditions credit on the exact observed action with `alpha_k ∝ w_k*qhat_k(b)`, using saved pre-action predictions, one update per observation, and exact tera identity. Legacy checkpoints are rejected by semantic version. Not used by MCTS. |
| Carried joint/mechanics belief provider | Available from the integration work, but not enabled by the final `foulplay_baseline` MCTS configuration. Setting the harness's shadow flag does not activate this provider in the native early-return path. |
| S1 live adapter | `s1_agentform` / `s1_live.py` selects existing `A1_rep` on the matrix-planner path with baseline sampling. Its local paired smoke test was 0–2 without errors; max decision time 18.481 s. No public S1 batch was run; the user paused it. |
| Reward rescaling | `rescaled_credit.py` and helper tests are prepared, but no rescaled live method is wired/registered or ladder-tested. It is not part of the final reward/evaluation path. |
| Continuous legacy runner | `continue_public_evaluation.py` is for the earlier Hedge continuation. Do not use it to reproduce this capped MCTS evaluation. |

Do not attribute the final ladder result to neural training, S1, reward rescaling, or a fully joint particle posterior.

## Reproduce or continue the final configuration

Run from the repository root. Setup is documented in [SETUP.md](../engine/showdown/SETUP.md). The managed setup creates `.local-play/environment.json`, which the public wrapper requires:

```bash
python3.12 engine/showdown/local_play.py setup

# Restore the exact public set catalogue used in the measured runs.
cp research/data/gen9randombattle_20261117.json \
  .local-play/foul-play/fp/data/pkmn_sets_cache/gen9randombattle.json

# No network/game is started by this preflight.
bash run_decision_audit.command --method mcts_native --sampler item_stratified \
  --mcts-ms 1000 --games 50 --concurrency 5 --dry-run

# Start a fresh capped batch in an interactive Terminal.
bash run_decision_audit.command --method mcts_native --sampler item_stratified \
  --mcts-ms 1000 --games 50 --concurrency 5
```

The wrapper is account-specific (`bot_bonp`). Use its existing secure login flow; credentials are not in this repository. No games are started by this handoff or by its tests. Changing the account for a separate experiment requires adjusting the wrapper's account/lock configuration or using `ladder.py` directly with the same method/environment settings.

To resume an incomplete batch, supply its existing directory and the **same original total target**:

```bash
bash run_decision_audit.command --method mcts_native --sampler item_stratified \
  --mcts-ms 1000 --games 50 --concurrency 5 --out <existing-batch-directory>
```

An already completed 50-game directory will start zero additional games. Use a fresh directory for a further 50-game cohort, then link its results to the previous cohort. Do not change policy files while a measured batch is active.

Ctrl-C once stops new searches and lets active games finish; it cannot freeze rated server games. Maintain connectivity until they drain. A network loss can leave unknown local outcomes and change the server rating. Ctrl-C twice aborts immediately. At this handoff, all requested completed games have finished and there are no active managed matches.

Useful batch artifacts: `evaluation.json`, `evaluation_results.json`, `run_status.json`, `progress.json`, `ladder_games.jsonl`, `decisions/`, `replays/`, `source_manifest.json`, and the `tactical_audit*` / `evaluation_summary*` reports. These local raw logs are ignored by Git; checked-in summaries retain the measured results, while private room suffixes and machine-specific checkout paths are omitted from published summaries.

## Dependencies and validation

The measured environment used Python **3.12.16**, `poke-engine==0.0.48` built from source with the `poke-engine/terastallization` feature and no default features, NumPy **2.5.4**, SciPy **1.18.1**, and websockets **14.1**. Pinned source revisions are:

- Foul Play: `8f541f80f91f960603673796559f796b10ccf0fb`.
- Showdown simulator/server: `7e743761cd13cdb361fc0f58eb5a866284894f5b`.
- Showdown browser client, needed for local browser play: `1cf9b385a97ad5c77a25b91a1c799078827ec306`.

The frozen catalogue is 461321 bytes with SHA-256 `4264b4d70fd86c7ea8ae362bc6aa2c027d2524a73b6df3f286dc0dffbabe5d3f`. Its provenance is in [research/data/README.md](../research/data/README.md). Timed native searches remain stochastic; a fixed catalogue and code revision do not reproduce an identical game trajectory.

Run the offline unit/native regressions and the imported exact-tera feedback check:

```bash
export FOULPLAY_DIR="$PWD/.local-play/foul-play"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
export OURBOT_DET_LEAF=1
PYTHONWARNINGS=ignore::ResourceWarning .local-play/venv/bin/python \
  -m unittest discover -s engine/tests -q
.local-play/venv/bin/python research/scripts/joint/tests/test_tera_label.py
```

The final suite contains 128 tests. Coverage includes actual native search, weighted visit aggregation, world-order and iteration-count independence, root legality/fallbacks, forced actions, exact budget splitting, real-catalogue sampling and current-state preservation, concurrent isolation, rating metrics, and the inactive matrix/feedback regressions. Local paired headless checks were 2–0 for baseline MCTS against corrected Hedge (seed 5), and 2–0 for stratified MCTS against baseline MCTS (seed 6). Those tiny paired runs are integration checks, not strength estimates.

## Remaining limitations and concrete follow-ups

1. **Native item support:** unsupported items such as Light Clay/Leppa Berry round-trip to `UNKNOWNITEM`. The final extension has 23 aliased records across all logs, including interrupted games. This can erase a sampled distinction even when sampler allocation is correct.
2. **Re-acquired item conversion:** the upstream converter lets a stale `knocked_off` flag override a newly acquired item. Four Weavile decisions in the first stratified batch lost a re-acquired Choice Band after Pickpocket. This inherited bug was documented, not fixed inside the frozen trials.
3. **Finite hidden-world coverage:** overflow cannot cover all item/ability classes in four worlds, and active-item stratification does not cover every unknown teammate/move combination. Baseline fallback occurred 16 times in the first batch and zero times in the extension.
4. **Root-only legality filtering:** unavailable actions can consume native visits before filtering. Both filtered `struggle-tera` cases in the extension selected valid switches; a native root-mask API would be a separate implementation change.
5. **Tactical choices still need review:** the extension's heuristic audit flags three setup-then-KO candidates (96 setup uses), two Ground/Flying candidates (44 Ground uses), and one Punch/Ghost candidate. These are positions for review, not proven mistakes or a complete error rate. There were no repeated/reflected rocks, Toxic/Steel, capped-setup, or risky Hurricane flags in that batch.

Any subsequent policy change should receive a fresh frozen 50-game ladder batch with starting/ending Elo as the primary metric. Preserve the current configuration as the measured reference.

## Detailed supporting artifacts

- [Combined final-bot results](../research/stratified_mcts_climb_100completed_results.json).
- [First stratified batch](../research/stratified_mcts_v1_ladder_results.json).
- [Native MCTS baseline](../research/native_mcts_v1_ladder_results.json) and [implementation report](native-mcts-evaluation.md).
- [Sampler and climb report](expected-benefit-ladder-tests.md).
- [Mechanics review](bot-decision-review.md) and [integration reassessment](joint-belief-reassessment.md).
- [Earlier Hedge feedback correction](per-world-expert-feedback.md) and [its 50-game evaluation](50-game-v2-evaluation.md).

Local measured batches remain under `.local-play/work/ladder/`: `mcts_native_v1_50games_20261117T135639Z`, `mcts_stratified_v1_50games_20261117T151646Z`, and `mcts_stratified_v1_next50_20261117T160925Z`. The directory timestamps use UTC; the handoff date is November 17 in America/New_York.
