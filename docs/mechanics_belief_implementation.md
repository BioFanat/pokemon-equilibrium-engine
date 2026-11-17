# Mechanics belief tracker: first implementation

This patch implements the first useful mechanics-inference release described in
`belief_tracker_plan.md`. It is experimental and opt-in. Historical experiments
and the default Foul Play belief source retain their existing behavior.

## Implemented

| Area | Behavior |
|---|---|
| Prior | Exact enumeration of complete sets under the existing approximate role-based generator; duplicate generation paths retain summed mass |
| Updates | Persistent log-weight posterior within a battle; original-set tuples remain correlated; event IDs prevent duplicate updates |
| Reveals | Species, legitimately observed moves, Tera and explicit item/ability messages; original attributes are separate from their current values |
| Speed | Ordinary same-priority attacks, supported neutral abilities/items, Choice Scarf and a single supported Speed modifier; ties retain likelihood 1/2 |
| Incoming damage | Exact HP observation likelihoods, duplicate-roll probability masses and censored KO tails; requires an explicit-roll damage oracle |
| Rounded HP | General candidate-dependent sequential HP-filter kernel and tests; outgoing-damage protocol wiring is not enabled in this release |
| Observation | Player-visible protocol parser; no hidden commands, no inferred opponent fields; event-time snapshots guard against using an item after it was removed |
| Sampling | Independent new world provider using this posterior, equal weights for posterior samples; no multiplication into Foul Play's already-conditioned weights |
| Recovery | Unsupported factors skip transactionally; all-zero mechanics likelihood rolls back; direct evidence outside the prior degrades explicitly |
| Rollout | Shadow mode by default; explicit fallback to Foul Play when public state cannot yet be faithfully reconstructed |

The old internal `hidden.py` harness remains a reproduction baseline. It does
not gain mechanics inference from this patch: its engine instruction stream
does not expose a verified public execution trace. New inference is attached
to the real-Showdown protocol path and is independently testable without Rust.

## Run the dependency-free tests

From the repository root:

```bash
python3 -m unittest discover -s engine/tests -v
```

These tests cover all 509 species in the checked-in prior, Bayesian update
arithmetic, likelihood rejection, repeated evidence, hidden/unexecuted moves,
item transfers, future-state leakage, KO censoring, sequential rounded-HP
filtering, and posterior-driven world sampling. World-conversion contracts use
test doubles; they are not live simulator validation.

## Enable shadow observation

Use the existing pinned Showdown/Foul Play setup and Python environment:

```bash
OURBOT_BELIEF_SOURCE=mechanics_v2 \
OURBOT_BELIEF_SHADOW=1 \
  "$BOT_PYTHON" engine/showdown/headless.py --seed 70001 --a agentform --b stock --tag MBshadow
```

`OURBOT_BELIEF_SOURCE` accepts `foulplay_baseline` (default), `reveals_v2`, or
`mechanics_v2`. The last two use the same catalog and observer, while only
`mechanics_v2` applies speed/damage likelihoods. This is separate from opponent
style learning (`hedge_bayes` / `bayes_safex`).

Decision and final logs include `mechanics`, with candidate support, item and
ability marginals, entropy, skip reasons and fallback reasons. Inspect coverage
and model-mismatch counts before trying move-selection integration.

## Damage backend contract

Set `POKE_ENGINE_BIN` to a trusted, locally built Gen 9/Tera `poke-engine`
0.0.48 executable. The backend serializes candidate states using the matching
Python bindings and invokes the engine's `calculate-damage` CLI. It requires
two explicit roll lists and validates that the selected list has 16 ordered
integer outcomes. It will not turn two extrema or normal/critical maxima into
a purported exact distribution.

```bash
POKE_ENGINE_BIN=/absolute/path/to/poke-engine \
OURBOT_BELIEF_SOURCE=mechanics_v2 \
OURBOT_BELIEF_SHADOW=1 \
  "$BOT_PYTHON" engine/showdown/headless.py --seed 70001 --a agentform --b stock --tag MBdamage
```

If the executable, Python bindings or roll contract is unavailable, damage
inference skips with a diagnostic; reveals and supported speed updates still
work. A bounded number of mechanics-equivalent groups is evaluated, with an
LRU cache. Actual latency must be measured in the pinned environment before
enabling this in normal play.

The CLI/API contract and real damage values have **not** been tested against
the pinned runtime in the implementation environment: the engine/Foul Play
checkouts were absent, and dependency installation was blocked by network
restrictions. Do not interpret passing pure-Python tests as verified battle
mechanics parity. Consult the engine's [official CLI documentation](https://github.com/pmariglia/poke-engine#usage)
for its damage-roll interface, then validate the installed build with fixtures.

## Enable posterior-driven moves after simulator validation

```bash
POKE_ENGINE_BIN=/absolute/path/to/poke-engine \
OURBOT_BELIEF_SOURCE=mechanics_v2 \
OURBOT_BELIEF_SHADOW=0 \
  "$BOT_PYTHON" engine/showdown/headless.py --seed 70001 --a agentform --b stock --tag MBplay
```

The payoff builder and agent-form LP are unchanged. Unsupported world states
fall back to Foul Play's normal sampler, explicitly logged, without reweighting
its posterior. The new sampler currently supports ordinary public states;
weather/terrain/side conditions, transformations, volatile effects, stat
changes, ambiguous identity, and pending switches trigger conservative
fallback. Field/volatile support is intentionally coarse and can remain in
fallback for the rest of a battle. Keep this in shadow mode until live coverage
and conversion tests are satisfactory.

## Remaining milestones

1. Validate the pinned CLI distributions, priority/Speed cases and HP displays
   against real Showdown fixtures; add those fixtures to regression tests.
2. Expand supported world reconstruction, public activations and incoming
   damage cases, measuring inference cost and model mismatches.
3. Wire the rounded-HP filter to outgoing-damage protocol events, including
   healing transitions rather than resetting HP uncertainty.
4. Add an internal simulator adapter only when observable execution events can
   be extracted without the old chosen-move leakage.
5. Run the plan's held-out calibration and paired battle ablations. No new
   battle-strength claim is made by this patch.

## Delivery constraint

The implementation was prepared in an isolated directory because the workspace
rejected Git branch creation and the managed-worktree tool failed. It is
delivered as an applyable patch; no tracked files on `main` were modified.
Create the requested branch in a Git-writable terminal, then apply the patch
from the repository root:

```bash
git switch -c feat/mechanics-belief-tracker
git apply docs/mechanics_belief_implementation.patch
python3 -m unittest discover -s engine/tests -v
```
