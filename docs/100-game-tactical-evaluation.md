# Public ladder tactical evaluation: hedge_bayes_mw

The first 100 games finished **52–48**, with no runner failures or invalid-choice errors. Ground immunity and risky setup choices recurred. Several other earlier mistakes were absent, but absence in this sample does not establish a fix.

The batch used the merged joint-belief integration at commit `5e7cb77`, `hedge_bayes_mw`, four sampled worlds, and the existing native evaluator. The experimental S1 planner remained disabled. Game 1 ran sequentially; the remaining 99 ran with up to five concurrent games. Each of five streams retained its own learning checkpoints and fresh feedback context per battle, warm-started from game 1. The policy implementation stayed fixed during the batch.

Recorded Elo moved **1577 → 1521**. Public opponents, rating, and adaptive learning varied; this is not a paired test or evidence of improved playing strength. Average logged decision time was 0.348 seconds; maximum was 2.418 seconds.

## Recurrence

| Earlier concern | First 100 games | Interpretation |
| --- | ---: | --- |
| Stealth Rock reflected by Magic Bounce | 0 | No sampled Magic Bounce matchup with rocks selectable; untested. |
| Stealth Rock while rocks already present | 0 | Rocks were selectable in 71 decisions with rocks already present. |
| Toxic into Steel | 0 | Toxic was selectable against Steel without Corrosion in 13 decisions. |
| Ground into Flying immunity | 1 | A clear recurrence against an already active Flying opponent. |
| Setup at relevant boost caps | 0 | Only one decision exposed this opportunity. |
| Setup followed by own KO that turn | 7 | One forced choice, one critical-hit case, five other freely selected cases. |
| Risky switch into revealed Hurricane | 0 | The one revealed-Hurricane switch was a quarter-resisting Iron Thorns, losing 15.6% HP. |
| Mach/Drain Punch into Ghost immunity | 0 | Six Punch choices with Knock Off available had no Ghost immunity; reviewed examples had healing, priority, or type advantages. |

There were 194 executed setup uses and 82 Ground attacks. Counts are replay-backed review candidates, not counterfactual estimates of lost games.

## Decisions that still need work

**Rhyperior used Earthquake into Drifblim**, battle `2715109969`, turn 4. Drifblim was already active. Earthquake had 41.4% mixture probability. The legacy pure “safest” expert supplied 22.7 percentage points of that probability, while its per-world worst-case value was worse than Stone Edge: −69.02 versus −41.64.

The new method updates the fourth, safe-exploitation expert using per-world action membership, but the third, pure “safest” expert still minimizes columns of the averaged union matrix. Across 2,951 comparable decisions, that expert selected a strictly worse per-world pure security action in **495 cases (16.8%)**, using a 0.001 evaluation-unit tolerance. A shadow substitution of only this expert reduced the logged Earthquake probability from 41.4% to 18.7%; it did not eliminate it. This is a diagnostic recomputation with recorded weights, not a played intervention or win-rate result. Aligning these expert objectives is the clearest next implementation task.

**Lucario used Nasty Plot before Weavile's Triple Axel KO**, the same battle, turn 14. The selected probability was 20.1%. Conditional on Triple Axel, supported sampled worlds predicted roughly 90% KO risk for setup and zero for the available attacks. This shows that risky setup remains possible despite the posterior changes.

Two other setup deaths were low-probability mixture selections: Blaziken Swords Dance against Gumshoos (`2715127582…`, turn 7, 1.5%) and Delphox Nasty Plot against Dunsparce (`2715129014`, turn 10, 1.5%). Their sampled models contained materially safer attacks. Separately, Iron Leaves chose Swords Dance against Sinistcha (`2715113441`, turn 14) when a tera variant survived in the conditional model; the leaf still preferred preserving tera. A probe did not establish a bug that rewards boosts on fainted Pokémon.

Not every setup death establishes a mistake: Bellossom had one legal action, and Drifblim's Calm Mind death involved a critical Water Pulse. Conditional survival figures use the native engine's sampled worlds and chance approximation, not the opponent's true hidden state or exhaustive Showdown damage rolls.

## Evidence and coverage

All 100 games have replay logs and all **3,019 decisions have exact sampled-state snapshots**. The strict turn/action join matched 2,747 executed actions. The remaining 272 include KOs before acting, status/flinch, games ending before execution, Encore replacing the submitted move, automatic locked moves, and form/Illusion aliases; they are not all failed commands. The strict matcher excludes these records from executed-action findings. This limits exhaustive coverage of automatic or aliased actions.

Artifacts are under `.local-play/work/ladder/decision_audit_hedge_bayes_mw_20261116T233404Z/`:

- `evaluation_summary_first100.json`: fixed-cohort results, timing, opportunities, expert comparison.
- `tactical_audit_first100.md` and `.json`: findings, battle IDs, turns, replay line references.
- `tactical_details_first100.json`: expert contributions and conditional survival probes.
- `unmatched_action_review_first100.json`: strict-join exclusions and examples.
- `replays/`, `decisions/`, and `learning/`: public battle logs, detailed decisions, persistent stream checkpoints.

Reproduce without account access:

```sh
OUT=.local-play/work/ladder/decision_audit_hedge_bayes_mw_20261116T233404Z
.local-play/venv/bin/python engine/showdown/audit_decisions.py "$OUT" --limit 100 --write
.local-play/venv/bin/python engine/showdown/inspect_audit_positions.py "$OUT" --limit 100 --write
.local-play/venv/bin/python engine/showdown/summarize_tactical_run.py "$OUT" --limit 100 --write
```

The user's requested continuation has a separate launcher and status file, preserves these first-100 artifacts, and runs up to five games until a stop request. At report creation it was **awaiting the secure Terminal password entry**; no game beyond 100 had started. Touching `STOP_CONTINUOUS` in the run directory prevents new searches and lets active games drain.
