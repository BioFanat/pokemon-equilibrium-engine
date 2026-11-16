# Harness bugs found during the G1 gauntlet (2026-11-16) and their effect on earlier results

Three bugs in the real-Showdown harness, all fixed (commits on `main` 3a537f5, 959a91b, df9ba59 and on this branch).
Diagnoses were made by replaying the failing games; scripts and replays were in the session scratch
(`zdiv/`, `hang/`), the fixes are in `engine/showdown/headless.py` and `engine/showdown/revival_guard.py`.

## 1. Simulator startup race (local runs only)

Every `pokemon-showdown simulate-battle` launch ran `node build`, which rewrites `dist/` in place. A sim starting at
the same moment could `require` a half-written file and exit with no output; headless then waited for the game
timeout and crashed in `proc.kill()` (ProcessLookupError), losing both games of the seed. Reproduced (SyntaxError at a
4 KiB boundary of a copied JSON; 1/80 launches with the default launcher, 0/80 with `--skip-build`).
Effect: 5 lost G1 seeds (rerun). Modal runs were not exposed (one game at a time per container).
Fix: launch with `--skip-build` (dist/ is built once at setup), fail fast when the sim exits before `|end|`, guard
`proc.kill()`.

## 2. Revival Blessing (all methods, all runs)

After Revival Blessing, Showdown requests a fainted reserve (`forceSwitch` with `"reviving": true`). foul-play
(8f541f8) stores the flag but never uses it and poke-engine has no notion of reviving, so:
- stock foul-play with every reserve fainted returned 'No Move' with 0 visits -> `ZeroDivisionError` in
  `select_move_from_mcts_results`;
- our solvers in the same state -> `AttributeError ... can_z` in `format_decision`;
- any method with a living reserve picked it -> Showdown rejected it -> `invalid_choice_stall`.

Past occurrences found in game JSONs (error field), all excluded from earlier scores as error games:

| error | games | where |
|---|---|---|
| ZeroDivisionError (stock) | 10 | G1 x3; BH s291-g1, s294-g2; E2 s352-g1; E2fs s4-g2; E3f s4-g2; LOG s4-g2; SX30 s93-g1 |
| invalid_choice_stall ("pass to a fainted Pokemon") | 15 | stock (AF s52-g2, E3a s52-g2), bandit_exp3 (E4e x3), bandit_exp3_mv (E4emv x3), bandit_ucb1 (E4u, XU), style_uniform (RQunif x2), hedge (E3a), agentform (E5s), pimc (E1 s283-g1) |
| AttributeError can_z | 8 | G1_stockat1000_pimc s4-g2 (confirmed); 6 more with Revival Blessing in their decision logs; RQunif s3-g1 not checked |

The ZeroDivision crasher was the stock player in all 10, each time on a team with Pawmot or Rabsca.
**Caveat for earlier results:** these 33 games (a small fraction of the thousands played) were dropped as error games,
but not at random: they are games in which a Revival Blessing user was on the team and actually used the move, and
seed 4 / seat p1 (Pawmot) recurs (E2fs, E3f, LOG and two G1 games). The direction of the resulting bias on any one
pair score is not established.
Fix: `revival_guard.guard` answers such requests with the first fainted reserve (a fixed legal rule; no method
evaluates revival targets) and keeps `last_selected_move` at the move used (otherwise foul-play's speed inference
looks up `switch <name>` as a move and raises KeyError). Verified by deterministic replay of
G1_stockat25_stockat400-s4-g1 (unguarded: ZeroDivisionError; guarded: the game finishes, 22 turns).

## 3. Forced recharge while trapped

After Hyper Beam / Giga Impact the request offers only `recharge`; if the user is also trapped (Arena Trap),
poke-engine sees no option and returns 'No Move' -> `AttributeError ... can_z`. Seen in G1_stockat1000_random-s3-g1.
Fix: `revival_guard.recharge_choice` answers `recharge`, the only legal choice (unit-checked on the real request
with foul-play's Pokemon object).

All three fixes leave decisions in every other state unchanged. After them, the 11 lost G1 seeds were rerun and G1
has 0 error games in 1,800.
