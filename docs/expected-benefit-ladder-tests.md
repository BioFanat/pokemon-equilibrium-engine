# Independent ladder tests by expected benefit

The requested order was changed from implementation cost to expected ladder benefit, following the user's confirmation. Each candidate receives a separate frozen **50-game** run with up to five concurrent games. Primary reporting is starting/ending Showdown Elo and net change; previous interim peaks are not substituted for endpoints.

1. **Coverage-aware sampler + native MCTS.** Four worlds and the 1,000 ms total native search budget remain unchanged. Only sampling of the active opponent's item/ability hypotheses changes.
2. **S1 activation on the matrix-planner path.** Independent architecture challenger using baseline sampling and the existing A1_rep information-set planner. This is not MCTS plus S1.
3. **Reward rescaling on the Hedge path.** Independent change to the corrected per-world Hedge learner; no sampler or S1 changes are included. Native MCTS has no such reward updates.

The user explicitly chose tests on the applicable architectures. Their sequential ladder changes will start at different ratings and face different opponents, so raw deltas are descriptive, not a controlled causal ranking.

## Sampler design and local checks

The sampler groups the existing compatible set proposals by the active Pokémon's **effective item and ability**. Known current items/abilities are respected, including the item having been removed. Where at most four classes remain, each class gets at least one sample. Remaining slots go to the largest probability deficits. Each world receives its class probability divided by the class's sample quota. This preserves exact class mass rather than assigning equal weight to rare and common threats.

For more classes than available slots, randomized systematic resampling keeps equal world weights and preserves class probabilities in expectation. It cannot represent every class in four worlds; the mode and represented prior mass are logged. Within each class, full sets retain their catalogue frequency weights. Known reserves and unknown-team generation use the existing upstream sampling/constraints.

This does not change the set catalogue or solve all possible unrevealed-team coverage gaps. No cross-game learning is added.

Local checks: **123 tests pass**. In a 5,000-draw item-only Dragalge diagnostic, baseline four-world draws miss Choice Specs 1,691 times (33.82%); stratification misses it zero times while retaining its exact catalogue probability. Two seat-swapped headless games, seed 6, against baseline MCTS complete without errors and are both won by the stratified candidate. This is an integration smoke test, not a strength estimate.

## Ladder results

| Candidate | Games | Wins–losses | Starting Elo | Ending Elo | Change |
| --- | ---: | --- | ---: | ---: | ---: |
| Native MCTS baseline, completed reference | 50 | 33–17 | 1470 | 1677 | +207 |
| MCTS + item/ability stratification | 50 | **33–17** | **1677** | **1885** | **+208** |
| S1 matrix planner | Not started | — | — | — | — |
| Rescaled per-world Hedge | Not started | — | — | — | — |

Sampler batch: `.local-play/work/ladder/mcts_stratified_v1_50games_20261117T151646Z/`. All 50 games completed with no runner errors, invalid selected actions, native-search fallbacks, or discarded root visit mass. All source/catalogue hashes match the pre-launch snapshot. Peak Elo was 1979; the reported endpoint is **1885**, not the peak. Mean policy time was 1.060 seconds, maximum 1.552 seconds. Queue/dispatch-inclusive mean was 1.172 seconds, maximum 3.297 seconds.

The 1,481 logged decisions include 314 with multiple item/ability classes and 307 with unequal world weights. Exact class-mass allocation applies to 1,458 decisions; seven use systematic overflow, and 16 explicitly fall back to the baseline sampler because no positive active-set candidates remain. Snapshots verify the weights used by MCTS. The tactical audit reports zero setup-then-KO candidates in 58 matched setup uses, four Ground/Flying immunity candidates, and no Punch/Ghost, repeated rocks, Toxic/Steel, or capped-setup failures. These are descriptive candidates, not causal evidence or an exhaustive error rate.

The audit also exposed inherited conversion limitations: native round-tripping maps unsupported Light Clay/Leppa Berry to UNKNOWNITEM (10 decisions). Four decisions in one Weavile game incorrectly lose a re-acquired Choice Band in the converter, because its old `knocked_off` flag overrides the item obtained through Pickpocket. These were **not changed inside the test**; they remain separate adapter issues. Sampling allocation is verified subject to these documented conversion losses.

Compact results: [stratified_mcts_v1_ladder_results.json](../research/stratified_mcts_v1_ladder_results.json). The frozen configuration remains available for reuse. The user paused the S1 and reward-rescaling ladder tests to extend the promising sampler/MCTS run by another 50 games. S1 local validation completed 0–2 against baseline agent-form, without errors; its maximum decision time was 18.481 seconds. No S1 public games were started.

## Additional 50-game climb test

The unchanged stratified MCTS candidate completed 50 further public games with concurrency five, beginning after the 1885 Elo endpoint. Output: `.local-play/work/ladder/mcts_stratified_v1_next50_20261117T160925Z/`. This is a separate linked batch so the original 50-game result remains intact. Source snapshots and hashes are saved again. The only common-file differences since the previous run register the inactive S1 method; the selected MCTS policy and sampling code are unchanged. S1 and reward-rescaling tests are paused at the user's request.

The extension was paused at the user's request after 14 completed games (10–4, last recorded Elo 1984, peak 2004). Five active games were interrupted by the connection loss and have unknown local outcomes. The user subsequently resumed the batch. The first resumed game starts at 1920 Elo, exposing a **64-point rating decrease during the interruption**. The original 50-completed-game target is retained, so those five interrupted games are additional attempts excluded from the completed-game win rate. Final account Elo includes the rating decrease. Archived interruption status and resume metadata remain inside the run directory.


## Completed climb results

| Stratified MCTS batch | Completed games | Wins–losses | Win rate | Starting Elo | Ending Elo | Change | Peak Elo |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| Original sampler test | 50 | 33–17 | 66% | 1677 | 1885 | +208 | 1979 |
| Additional climb test | 50 | 33–17 | 66% | 1885 | **2090** | **+205** | **2141** |
| Combined recorded results | 100 | **66–34** | **66%** | 1677 | **2090** | **+413** | **2141** |

The extension has 50 completed games plus five interrupted attempts. Its win rate excludes the five unknown local outcomes. The actual account Elo change **includes** the 64-point decrease during the interruption. The resumed portion alone was 23–13 over 36 games, from 1920 to 2090 (+170). Completed recorded games contribute +269 in their individual rating updates; the unrecorded interruption gap reconciles that total to the actual +205 endpoint change. This is an accounting identity, not an estimate of the counterfactual result without the outage.

The same four-world, 1000 ms stratified MCTS policy was used throughout. Policy source hashes match the frozen snapshot. The resumed runner finished with exit status 0, no active rooms, and no pending search; the account run lock is free. S1 and reward-rescaling ladder tests remain paused as requested.

All 50 completed-game logs and replays are present, containing 1548 decisions. None selected an illegal action or used the native no-visits fallback. Mean policy time was 1.060 seconds, maximum 1.436 seconds; queue/dispatch-inclusive mean was 1.194 seconds, maximum 3.325 seconds. In two decisions, the legality filter discarded native visits to server-unavailable `struggle-tera` (at most 0.969% of a world's root visits); both selected legal switches.

Across all 1631 decision records, including interrupted games, the sampler audit finds no class/weight inconsistencies after native item canonicalization. There are 23 records with native item/ability aliases, which remain a modeling limitation. All source hashes still match. The audit's empty-item representation was corrected to reflect the existing converter's deliberate empty-item-to-NONE behavior; no policy or converter changes were made during this run.

The completed-game tactical audit flags three setup-then-KO candidates in 96 setup uses, two Ground/Flying immunity candidates in 44 Ground uses, and one Punch/Ghost candidate. It flags no repeated/reflected rocks, Toxic/Steel, capped setup, or risky Hurricane switches. These are heuristic review candidates, not proven decision errors or a complete measure of playing strength.

Compact machine-readable results, including the interruption, post-resume breakdown, final health audit, and combined 100 completed games: [stratified_mcts_climb_100completed_results.json](../research/stratified_mcts_climb_100completed_results.json). Detailed run artifacts remain in `.local-play/work/ladder/mcts_stratified_v1_next50_20261117T160925Z/`.
