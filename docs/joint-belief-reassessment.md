# Joint-belief integration and decision reassessment

November 16, 2026. Fetched `origin` and merged its actual branch name,
`integration/joint-belief` at `014ad4b`, into the existing local branch
`feat/mechanics-belief-tracker`. Merge commit: `bacfe12`. The earlier unfinished
`origin/main` merge was completed first (`7db176f`). Both ladder conflicts were
resolved by retaining concurrent battle isolation and the incoming revival and
recharge guards. No remote branch was changed.

The earlier local review's terminal rewards, accuracy/evasion conversion,
server-request legality checks, and optional exact decision snapshots are all
preserved. The remote branch did not independently contain those changes.

## What the branch fixes

| Area | Reassessed status |
| --- | --- |
| Probability assigned through worlds that lack an opponent action | **Fixed in opt-in methods** `hedge_bayes_mw` and `bayes_safex_mw`. Prediction and likelihood use each world's own columns. A missing label contributes exactly zero from that world. |
| Exploitation using the same world model | **Implemented in those methods.** `safex_strategy_worlds` optimizes the per-world predicted return while retaining the agent-form guarantee within its epsilon allowance. |
| Full joint set × style posterior | **Measured offline, not deployed as a carried joint particle posterior.** The shipped per-world style posterior remains factorized; the branch's measurements found negligible extra benefit from the genuinely joint term. |
| Strategy fusion at child information sets | **Experimental S1 implementation exists.** It shares a continuation strategy across indistinguishable children and uses an exact unpruned LP leaf. It is depth 2 and is not called by `run_ourbot.py`; merging does not activate it. |
| Revival Blessing and trapped recharge failures | **Guarded in headless and ladder entry points.** These bypass search for the mandatory fainted replacement or recharge action. They do not evaluate revival targets. |
| Headless simulator build/startup race | **Fixed.** The harness uses the existing build with `--skip-build` and handles an early simulator exit. |

The live default remains `agentform` using the Foul Play belief source. The
local mechanics provider remains separately selectable and shadowed by the
local launcher. The merge makes both workstreams available in one checkout;
it does not automatically combine their inference or change the selected agent.

## Do the reported tactical issues disappear?

**They are not established as fixed by this branch.** The default agent's
matrix construction and root decision rule are unchanged by the incoming
branch. The existing Bayes methods also preserve their decisions in the
synthetic pre-merge snapshot comparison. The opt-in methods change prediction
and safe exploitation; they add no targeted hazard, immunity, setup, or
incoming-threat handling.

| Earlier issue | Assessment after merge |
| --- | --- |
| Repeated rocks into Magic Bounce / existing rocks | Mechanics are already represented. No new targeted prevention is present; the historical failures remain relevant. |
| Toxic into Steel; Fighting attacks into Ghost | Native mechanics tests still confirm immunity. Better opponent predictions may change choices under `*_mw`, but the old decisions cannot be re-solved exactly from their mixture-only logs. No verified tactical fix is claimed. |
| Ground into Flying | The identified Pelipper example was a switch-in on the chosen move. The branch adds no blanket type restriction, which would also mishandle switches, tera, Roost, and ability interactions. |
| Setup before a KO | S1 changes continuation information and the leaf, not search depth. There is no matched replay evidence that it repairs the Espathra/Noctowl position. |
| Setup at maximum boosts | Latias's repeated Calm Mind was choice-locked and trapped; Gurdurr's Encore explains one instance, while its later freely selected capped Bulk Up remains a weakness. No new capped-setup rule was added. |
| Switches into revealed Hurricane | The branch does not add a revealed-threat guard or deeper search. The known Urshifu KO is not shown to be repaired. |
| Mach/Drain Punch over Knock Off | Move identity mapping and Ghost immunity regressions pass. The new posterior is not evidence of improved tactical ranking in those saved positions. |

Detailed battle IDs, turns, and the distinction between forced actions and
poor predictions remain in [the original review](bot-decision-review.md).

## Integration corrections made locally

The experimental S1 leaf still used a material heuristic for a battle already
won or lost. A completed 1v1 win scored about **110.74** through S1, versus the
live planner's exact **10000** reward. S1's `own_child_matrix` and LP-leaf
reference now share `nash.terminal_value`: +10000 win, -10000 loss, 0 draw,
including the perspective change for side two. The raw `full_d1` extractor
remains unchanged so its native-cell identity contract is preserved. This
correction does not activate S1 in live play.

Imported research tests also needed two integration repairs: the analyzer
test had an absolute path to the author's machine, and the stubbed live-bot
tests lacked the request/user fields and matrix keyword accepted by the local
adapter. Their assertions and posterior logic are retained.

## What the existing experiments support

These are **imported experiment results**, not new ladder measurements:

- Per-world prediction improvement replicated: about **0.30–0.31 nats per
  scored turn** against stock. The win-rate improvement did **not** replicate
  on seeds 201–400: +0.0075, one-sided p = 0.45. See
  [results](../engine/RESULTS.md) and
  [verification](../research/mw_w1_verification.md).
- The S1 A1-versus-frozen-A2 action difference was **3/3,620 decisions** in the
  stored evaluation. That compares two S1 variants, not S1 against the live
  baseline. No live port or tactical win-rate gain was established. See
  [S1 results](../research/s1_results.md).

Historical report figures and raw result files were preserved. Their measured
scores predate the local terminal reward correction and have not been relabeled
as results of this merged evaluator.

## Local validation

- Local unit suite: **71 tests pass**, including earlier mechanics and action
  regressions, win/loss/draw continuation values in S1, terminal wins through
  its root analysis, and forced revival/recharge bypasses.
- Per-world tests: **2,616 support checks**, **2,000 one-world predictor/update
  checks**, **600 one-world LP reductions**, and **1,200 safety LP solves** pass.
- Tera-feedback checks and the synthetic results-analyzer test pass. Existing
  `hedge_bayes` / `bayes_safex` move sequences, records, and saved state files
  match the pre-merge local snapshot (`d2b0003`) on 12 controlled sequences of
  40 decisions. Both new `*_mw` methods execute through the same harness.
- S1's extraction, reduction, scouting toy, Jensen, and public-key leakage
  gates **all pass** on the merged code. The extraction gate reproduces all
  **5,256** native cells, including tera and switch rows.
- The full matrix suite passes, including 20 random positions, mirrored
  positions, solver/policy checks, and 155 turns of whole games. The gauntlet
  meta-game solver and discount-then-predict acceptance tests also pass.

- A local paired headless run of `hedge_bayes_mw` against `agentform`, seed 4,
  K=4, completed **both games with zero simulator or harness errors**, in 30
  and 27 turns. The pair used identical teams with seats swapped. All **65**
  saved `agentform` decisions re-solve to their logged mixtures within logging
  precision. Logs and exact snapshots are under
  `.local-play/work/joint-merge-review-6t4l8o2a/`.

No login or public game was started. The two headless games are an integration
smoke check, not a new strength estimate.
