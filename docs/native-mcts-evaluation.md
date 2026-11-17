# Native MCTS v1: implementation and ladder evaluation

November 17, 2026. **Native MCTS v1 finished 33–17 and increased Showdown Elo from 1470 to 1677: +207 Elo.** All 50 games completed, peak concurrency was five, and there were no runner failures or invalid choices. The ladder test took approximately 40 minutes.

Primary endpoint: **Showdown ladder Elo change over 50 completed games**, with up to five concurrent games. Secondary diagnostics are wins/losses, decision time, invalid choices, and tactical replay flags. Each subsequent policy revision should receive its own fresh, frozen 50-game batch.

## Candidate

`mcts_native` replaces the matrix/Hedge decision layer with the installed native engine's Monte Carlo tree search. It retains the same Foul Play baseline sampler, four hidden worlds, corrected state conversion, server-action mapping, battle isolation, and forced revival/recharge guards.

- Total native search budget: **1,000 ms per decision**, split across four worlds; one native thread per search.
- Each world contributes its prior weight times its normalized legal root visit counts. Worlds are not reweighted by how many iterations they happened to complete.
- The selected action is the largest aggregate visit share, with a stable first-label tie break. The logged visit distribution is distinguished from the one-hot played policy.
- The native API cannot mask individual root actions. Server-illegal actions are excluded when aggregating returned root visits, with discarded mass explicitly logged. Future-state mechanics are not altered to imitate a root mask. A no-legal-visits fallback is legal and explicitly logged.
- Single-legal-action decisions bypass native search. No matrix is constructed and no Hedge/style checkpoint is updated.
- Exact sampled states, per-world root visits/scores, budgets, durations, selected action, and fallback/filter diagnostics are saved per battle.

There is no neural network, new sampler, reward rescaling, or S1 planner activation in this candidate. Combining those changes with the first MCTS test would obscure which change affected Elo.

## Local validation

**114 tests pass.** New coverage includes prior-weighted aggregation, iteration-count independence, world ordering, switch labels, missing/illegal root actions, explicit fallback, exact budget splitting, forced-action bypass, real native search, matrix bypass, and pinned evaluation settings.

Two headless Showdown games against corrected `hedge_bayes_mw`, seed 5, use identical teams with seats swapped. MCTS wins both, in 29 and 34 turns, with zero simulator errors. It logs 69 decisions, no no-legal-visits fallback, and no discarded root mass. Mean MCTS decision time is approximately 1.086 seconds. This is an integration smoke test, not a strength estimate.

On saved worlds from the latest ladder failures, one diagnostic search selects Ice Hammer instead of Crabominable Drain Punch, Earthquake instead of Gyarados Dragon Dance, and Volt Switch instead of Sandy Shocks Earth Power. These are timed stochastic searches with frozen sampled worlds, not counterfactual game outcomes. In particular, switching from Dragon Dance to Earthquake does not resolve an omitted Choice Specs hypothesis.

## Ladder test

Run directory: `.local-play/work/ladder/mcts_native_v1_50games_20261117T135639Z/`.

The policy configuration and source copies/hashes were frozen before launch. `evaluation_results.json` is automatically updated with starting/ending Elo, net change, observed Elo range, game counts, and wins/losses. The batch stops at 50 games; no policy changes will be made inside it.

| Policy | Games | Wins–losses | Starting Elo | Ending Elo | Elo change |
| --- | ---: | --- | ---: | ---: | ---: |
| Corrected per-world Hedge, previous batch | 50 | 25–25 | 1521 | 1470 | −51 |
| Native MCTS v1 | 50 | **33–17** | **1470** | **1677** | **+207** |

The policy and total budget were unchanged throughout the batch. All 50 games have rating records, and the source hashes still match the pre-launch manifest. The final rating is the endpoint; the interim peak of 1741 is not substituted for it. The run stopped at the requested 50 with no active games or pending searches.

Different starting ratings, opponents, and finite samples make sequential ladder deltas descriptive rather than controlled estimates of causal strength gain. The practical result supports keeping MCTS as the leading ladder candidate. It does not establish an exact +207 causal Elo advantage over Hedge, or that 1677 is its settled rating.

### Runtime and audit

- **1,359 logged decisions**, all with sampled-state snapshots; 32 single-action decisions bypassed search.
- **5,308 native world searches**, with median 113,000 visits per world. All records used four worlds and the pinned 1,000 ms total budget.
- No fallback, no selected illegal action, and no discarded root visit mass.
- Mean policy processing time **1.060 s**, maximum **1.491 s**. Including queue/dispatch time, the ladder ledger reports mean **1.169 s**, maximum **3.524 s**.
- Tactical replay audit matches 1,221 executed actions, with no missing game logs. Interrupted/automatic/aliased decisions account for the remaining coverage gap; these are not all failed commands.
- No reflected/redundant rocks, Toxic-into-Steel, capped setup, or Punch-into-Ghost flags. There are seven Ground/Flying immunity candidates and three setup-then-KO candidates. Of the seven Ground cases, four follow opponent switches and three occur against active Tropius in one Poison Heal Gliscor stall line; that game was won. These flags require tactical context and are not an error-rate or win-rate estimator. The revealed-Hurricane switch is not flagged risky by the existing criterion.

This is a material improvement in the observed ladder endpoint despite some tactical warning signs remaining. I would evaluate the next search/sampling change in another separate 50-game batch, retaining Elo change and ending Elo as the primary reported results.

Compact results: [native_mcts_v1_ladder_results.json](../research/native_mcts_v1_ladder_results.json). The run directory contains `evaluation_results.json`, `mcts_health.json`, `evaluation_summary_first50.json`, `tactical_audit_first50.md`, the source snapshot, and per-game decisions/replays.

Commands:

```sh
PYTHONWARNINGS=ignore::ResourceWarning .local-play/venv/bin/python -m unittest discover -s engine/tests -q
bash run_decision_audit.command --method mcts_native --games 50 --concurrency 5 --mcts-ms 1000
```

The launcher handles the secure account login and records Elo automatically. Use a new output directory for each policy revision; resuming a batch requires its original method and MCTS budget. The existing account lock prevents simultaneous managed runs on the account.
