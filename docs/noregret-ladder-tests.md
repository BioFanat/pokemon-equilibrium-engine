# No-regret ladder tests

Three variants of the final ladder bot (`mcts_native` + `item_stratified`). Each uses the same four sampled
worlds, the same world sampler (`OURBOT_WORLD_SAMPLER`), the same total budget (`OURBOT_MCTS_MS`, split exactly
across the worlds), the same forced-action bypass and server-legality filter, and the same decision-record schema.
Code: `engine/showdown/noregret_methods.py`, `engine/showdown/smmcts_rm.py`; tests:
`engine/tests/test_noregret_methods.py`.

## Methods

- **`mcts_native_sampled`**: identical search to `mcts_native` (native MCTS per world, root visits filtered to legal
  actions, normalised per world, weighted by world priors), but the move is *sampled* from the aggregate share vector
  instead of taking its argmax. Policy string `weighted_root_visit_sample_v1`.
- **`hedge_mcts`**: per decision runs `mcts_native`'s search (full budget) and the per-world full depth-2 matrices
  with the agent-form LP. Hedge (anytime step, scale 400) over three experts, in this order: `mcts_top` (argmax of
  the shares), `mcts_sampled` (the shares), `lp_sampled` (the LP mixture). Plays a sample of the weight mixture.
  After each decision the opponent's realised action is read from the protocol and credited in its highest-weight
  world (run_ourbot's hedge credit). Weights carry across games in a state file (`semantics = hedge_mcts_v1`; files
  of other methods/experts are rejected). The matrices add time on top of the MCTS budget.
- **`smmcts_rm`**: per world, a Python simultaneous-move MCTS with regret matching (RM+) at every node
  (outcome sampling, exploration γ = 0.1, importance-weighted regret updates, chance branches sampled by their
  probability). Depth limit `OURBOT_SMRM_DEPTH` joint turns (default 3), then the squashed midpoint leaf
  `tanh(leaf/160)`; terminal positions score ±1/0. The world-weighted root average strategy, restricted to legal
  actions, is sampled. Policy string `smmcts_rm_avg_sample_v1`.

## Public ladder (same settings as the final bot)

Dry run (prints the pinned configuration; no login, no connection):

```bash
bash run_decision_audit.command --method mcts_native_sampled --sampler item_stratified --mcts-ms 1000 --games 50 --concurrency 5 --dry-run
bash run_decision_audit.command --method hedge_mcts --sampler item_stratified --mcts-ms 1000 --games 50 --concurrency 5 --dry-run
bash run_decision_audit.command --method smmcts_rm --sampler item_stratified --mcts-ms 1000 --games 50 --concurrency 5 --dry-run
```

Real runs (Terminal; prompts for the password):

```bash
bash run_decision_audit.command --method <method> --sampler item_stratified --mcts-ms 1000 --games 50 --concurrency 5
```

Resume an incomplete batch with the same settings and its directory (pinned values in `evaluation.json` must match):

```bash
bash run_decision_audit.command --method <method> --sampler item_stratified --mcts-ms 1000 --games 50 --concurrency 5 \
  --out <existing-batch-directory>
```

`hedge_mcts` with `--concurrency 5` keeps one independent learner per stream in
`<out>/learning/hedge_mcts-stream<k>.json` (sequential runs use `<out>/hedge_mcts_state.json`, which warm-starts
the streams if present).

## Local headless play against `mcts_native`

One seed (two games, seats swapped, same teams):

```bash
FOULPLAY_DIR=~/sandbox/foul-play SHOWDOWN_DIR=~/sandbox/pokemon-showdown OURBOT_WORK=<work dir> \
OURBOT_MCTS_MS=1000 OURBOT_WORLD_SAMPLER=item_stratified OURBOT_DET_LEAF=1 \
  ~/sandbox/bot-venv/bin/python engine/showdown/headless.py --seed 5 --tag smoke --a <method> --b mcts_native --k 4
```

Several seeds with `engine/showdown/batch.py` (resumable; summary in `<OURBOT_WORK>/replays_headless/<tag>_summary.json`).
For `hedge_mcts` use `--streams` so each carried state file (`<OURBOT_WORK>/hedge_state/<tag>-stream<k>-<A|B>.json`)
is used by one worker at a time:

```bash
FOULPLAY_DIR=~/sandbox/foul-play SHOWDOWN_DIR=~/sandbox/pokemon-showdown OURBOT_WORK=<work dir> \
OURBOT_MCTS_MS=1000 OURBOT_WORLD_SAMPLER=item_stratified OURBOT_DET_LEAF=1 \
  ~/sandbox/bot-venv/bin/python engine/showdown/batch.py --tag nr_<method> --a <method> --b mcts_native \
  --seeds 1-50 --k 4 --workers 5 --streams 5
```
