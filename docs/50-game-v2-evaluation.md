# Corrected per-world policy: 50-game evaluation

November 17, 2026. **The correction worked as implemented, but this run does not show a playing-strength improvement.** The bot finished **25 wins, 25 losses**, with recorded Elo **1521 → 1470**. The earlier uncorrected batch was 52–48. These are different public opponents and rating ranges, and this run started fresh learning streams; they do not isolate the correction's effect on win rate.

All 50 games finished, peak concurrency reached five, and there were no runner failures, tracebacks, or invalid-choice errors. Average decision time was 0.437 seconds; maximum was 2.573 seconds. The implementation files match the source hashes saved before launch. No policy changes were made during this analysis.

## Did the intended correction actually run?

**Yes.** Across all **1,598 decisions with expert strategies**, the safest expert maximizes the intended pure per-world security objective. There are **zero mismatches** at tolerance 0.001, versus 495/2,951 in the earlier batch. This verifies the objective correction; it does not establish that every selected move is tactically correct.

All **1,330 credited updates** use the exact observed action, zero mass on worlds lacking that action, and the conditional world weights implied by the saved pre-action predictions. Of these, **1,293** combine multiple worlds and **27** concern tera actions. Credit and style-feedback predictive probabilities agree within the log's six-decimal precision. Reconstructed rewards and utilities agree within bounds implied by four-decimal expert logs. Every stream's final update count and cumulative rewards match its logs exactly.

There are 268 skipped observations: 114 actions outside sampled support, 86 opponent KOs before acting, 36 `cant` events, and 32 with no action observed. All 1,598 pending learner decisions have paired feedback, including final feedback.

## Earlier tactical concerns

| Concern | This run | Interpretation |
| --- | ---: | --- |
| Rocks reflected by Magic Bounce | 0 | No rocks-selectable sampled Magic Bounce matchup; untested. |
| Rocks attempted while already present | 0 | Rocks were selectable with rocks present in 21 decisions. |
| Toxic into Steel | 0 | No Toxic-selectable Steel matchup without Corrosion; untested. |
| Ground into Flying immunity | 1 | The opponent switched Quagsire out for Squawkabilly. |
| Setup at relevant boost caps | 0 | One eligible decision; little exposure. |
| Setup followed by own KO | 1 | Gyarados Dragon Dance into Dragalge Draco Meteor; not a critical hit. |
| Switch into previously revealed Hurricane | 0 | No matched occurrence; does not establish a targeted fix. |
| Punch into Ghost immunity | 1 | The opponent switched Tatsugiri out for Misdreavus. |
| Punch chosen with Knock Off available | 12 | Includes the Ghost switch; this count alone is not an error count. |

There were 71 matched setup uses and 39 Ground attacks. The earlier batch had seven setup-then-KO flags in 194 setup uses; the lower raw rate here is descriptive, with different situations and opponents. A scan of the remaining parsed replay actions found no additional setup deaths omitted by the strict decision join.

## Three decisions worth understanding

### 1. Crabominable Drain Punch met an unsampled Ghost switch

Battle `2715214922`, turn 2, replay line 41. The decision was made against **Tatsugiri**. The opponent switched to **Misdreavus**, making Drain Punch immune. Its played probability was **22.65%**: approximately 10.76 percentage points came from the sampled equilibrium expert and 11.89 from safe exploitation. The corrected pure safest expert preferred **Knock Off** and supplied no Drain Punch probability.

**None of the four sampled worlds contained Misdreavus**, so the predictor assigned that exact switch zero probability. Other Ghost species did occur among sampled reserves, so the policy was not entirely unaware of Ghost-switch risk. This is a failed prediction involving finite hidden-team coverage, rather than Punch knowingly selected against an already active Ghost. The other eleven Punch-over-Knock candidates do not reproduce this Ghost-immunity event; healing, type, and survival values can justify Punch.

### 2. Sandy Shocks Earth Power met a represented but underestimated switch

Battle `2715222661…`, turn 23, replay line 298. **Quagsire was active** when Earth Power was chosen; the opponent switched to **Squawkabilly-White**. Earth Power's probability was **75.60%**, and all four experts had it as their highest-probability action.

Squawkabilly was present in all four worlds, but its switch received only **about 4.05% predicted probability**. Given that switch, the recorded model values Volt Switch at −186.91 versus −337.94 for Earth Power. Against the full response set, however, Earth Power has the better pure security value: −337.94 versus −376.87. Quagsire itself is immune to Electric attacks. This is a prediction/coverage tradeoff, not a recurrence of the earlier Earthquake into an already active Drifblim, and not a failure of the corrected safest calculation.

### 3. Gyarados's setup death exposes hidden-set risk

Battle `2715218126`, turn 1, replay line 24. Full-health Gyarados used **Dragon Dance** with **13.06% probability**, then died to a non-critical Draco Meteor. The pure safest and top experts preferred switching to Scizor; the Dragon Dance probability came from the two mixed experts.

All four sampled Dragalge sets had **Leftovers**, with identical relevant attacking stats. Conditional on Draco Meteor being chosen, the native model estimated **3.75% KO probability** for Dragon Dance. Earthquake had essentially the same KO risk in those worlds; this was not a choice that those samples predicted would usually die.

An item-only diagnostic changes that estimate substantially:

| Dragalge item, other saved state held fixed | Native modeled KO probability |
| --- | ---: |
| Leftovers | 3.75% |
| Assault Vest | 3.75% |
| Choice Specs | **68.44%** |

Choice Specs exists in the cached set catalogue but was absent from all four samples. The replay never reveals Dragalge's item, so this probe **does not identify its actual set**. Native chance branches also approximate Showdown's full damage distribution. It nevertheless demonstrates that this decision's survival estimate is sensitive to a plausible hidden alternative excluded from the sampled ensemble.

## Reward saturation remains consequential

**43/1,330 updates** give every expert reward one. In **12** of those, the expected raw utilities differ. Across all reward ties, **13 updates** erase raw-utility distinctions.

For example, battle `2715221095…`, turn 18: raw utilities are approximately `[-8104, -8104, -8108, -5180]` for top, sampled, safest, and safe exploitation, yet all four receive reward **1.0**. The fixed scale of 400 and clipping erase a roughly **2,929-unit** difference. These are relative surrogate rewards, not win probabilities: a reward of one does not mean a winning position. More training on these identical rewards cannot teach this distinction.

This demonstrates information loss in the learning signal. Its contribution to lost games has not been estimated; these 13 updates are not established as the main cause of the batch's losses.

On the **1,330 scored observations only**, the per-world predictor's average log loss is 1.479 nats versus 1.782 for the averaged-matrix predictor computed alongside it. This is a useful diagnostic, not a new v1-versus-v2 comparison: both revisions already used per-world prediction. It excludes unsupported and censored actions, so it is not a full unconditional prediction score.

## What I would do next

1. **Audit and revise the reward transformation**, with regression cases where clipping removes large utility differences. Measure whether better credit improves expert selection before scaling training.
2. **Test hidden-set coverage and survival calibration**, including risk-relevant item diversity rather than only increasing random sample count. Gyarados/Dragalge is a concrete benchmark.
3. **Evaluate switch prediction and the final mixture's risk constraint separately.** The correction makes the pure expert consistent; it does not constrain the entire Hedge mixture or guarantee coverage of unseen switches.

The evidence supports these targeted follow-ups. It does not support declaring the architecture solved, abandoning the correction, or claiming that additional ladder training alone will resolve the remaining failures.

## Evidence and limits

All **1,638 decisions** have exact sampled-state/payoff snapshots; all 50 games have replay logs. The strict action join matches **1,484** actions. The remaining 154 consist of 96 own KOs before the chosen move, 44 `cant` interruptions, eight games ending before execution, five automatic/replaced actions, and one switch alias/timing case. These are distinct from opponent-feedback skips. Findings are candidate checks for the earlier issues, not an exhaustive review of every strategic mistake.

The compact [evaluation JSON](../research/credit_v2_50game_evaluation.json) records results, verification, and case studies. Full artifacts are under `.local-play/work/ladder/credit_v2_50games_20261117T051258Z/`:

- `tactical_audit_first50.md` / `.json`: cases and exact replay references.
- `tactical_details_first50.json`, `case_studies.json`: expert contributions and native probes.
- `runtime_learning_audit.json`: all credit/objective checks and stream accounting, with no material discrepancies.
- `evaluation_summary_first50.json`, `additional_diagnostics.json`: results, coverage, timing, reward ties, and scored-subset prediction diagnostics.
- `analyze_runtime.py`: repeatable verification, with tolerances derived from log precision.

No playing policy was modified and no additional games were started during this review.
