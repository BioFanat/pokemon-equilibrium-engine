# Running our bot on the public Showdown ladder

The measured final ladder bot is **native MCTS + item/ability stratification**:
`mcts_native`, sampler `item_stratified`, four worlds, 1000 ms total search,
and up to five concurrent games. It ended at **2090 Elo**, peaked at **2141**,
and went **66–34 over 100 completed games** with this configuration. The user
reported top 500. See the [final bot handoff](../../docs/final-ladder-bot-handoff.md)
for all active changes, exact reproduction steps, dependencies, and limitations.
The ordinary CLI defaults below still select older matrix methods.

From the repository root, with the managed environment installed:

```bash
bash run_decision_audit.command --method mcts_native --sampler item_stratified \
  --mcts-ms 1000 --games 50 --concurrency 5
```

No-regret variants of this bot (`mcts_native_sampled`, `hedge_mcts`, `smmcts_rm`) run with the same command; see
[no-regret ladder tests](../../docs/noregret-ladder-tests.md).

This guide runs our equilibrium bot, `agentform` (the per-turn agent-form matrix-game solver), in rated **Gen 9 Random
Battle** games on play.pokemonshowdown.com. The goal is a rating against the live ladder.

The ladder entry point is `ladder.py`. It logs every game and stops after N games.
The default is one battle at a time. `--concurrency 5` manages up to five battles
through one signed-in connection, with isolated queues, beliefs and decision logs.
A three-game public trial completed on 2026-11-16 (2 wins, including one opponent
disconnect, and 1 loss). Larger batches remain evaluations rather than established ratings.

## 0. Before anything: register the account as a bot

Showdown's bot policy (Smogon announcement "Initial policy on bots", 9 Oct 2026,
https://www.smogon.com/forums/threads/3789528/) requires bots to be registered. **An unregistered bot account is
bannable.**

1. Create a Showdown account whose name contains **`bot`** (e.g. `cs535eqbot`), with a password.
2. Log in as that account in the browser and type `/ht` in chat. Choose **"Something else" → "I want to register my
   bot"** and follow the staff's instructions. Mention that it is a course research evaluation (CS535), one game at a
   time, about 100 games.
3. Wait for confirmation before running rated games.

Under the current policy, bots are matched only with other bots and with humans who opted in, and games against bots
do not change human ratings. So the opponent pool is not the full ladder; say this when reporting the number.

## 1. Setup (once)

Follow `SETUP.md` §"Setup" for the Python venv and foul-play: steps 2 and 3. You do **not** need the local Showdown
server (step 1) for ladder play. Then:

```bash
cd <repo>/engine/showdown
PY=~/sandbox/bot-venv/bin/python        # or wherever the bot venv is
export FOULPLAY_DIR=~/sandbox/foul-play # if not at the default path
```

## 2. Dry run (no network)

```bash
$PY ladder.py --server wss://sim3.psim.us/showdown/websocket --username cs535eqbot --games 1 --dry-run
```

This prints the resolved config and the reasons it would refuse to start. It does not open a connection.

## 3. Guards (why it may refuse)

For any non-localhost server, `ladder.py` exits with code 2 and does not connect unless all of these hold:

- `--i-am-registered` is passed (you confirm step 0 is done);
- the password is in the environment variable `PS_PASSWORD`. It is never accepted on the command line, never printed,
  and scrubbed from logs and tracebacks;
- the username contains `bot`.

## 4. One-game trial

```bash
read -s PS_PASSWORD && export PS_PASSWORD     # type the password; it is not echoed
$PY ladder.py --server wss://sim3.psim.us/showdown/websocket --username cs535eqbot --games 1 --i-am-registered
```

Check that the game finished, that `~/sandbox/ourbot/ladder/cs535eqbot/ladder_games.jsonl` has one line, and that the
replay opens.

## 5. Full run

```bash
$PY ladder.py --server wss://sim3.psim.us/showdown/websocket --username cs535eqbot --games 100 --i-am-registered
# Up to five overlapping battles, one account and one connection:
$PY ladder.py --server wss://sim3.psim.us/showdown/websocket --username cs535eqbot --games 100 --concurrency 5 --i-am-registered
```

- **Stopping:** press Ctrl-C once to stop new searches and finish the active games; press it twice to abort now. Rerunning continues the
  same account's rating and appends to the same log.
- **Length:** a game takes about 5–10 minutes, so plan about 10–15 hours for 100 games. It can be split over several
  sessions.
- **Time per move:** decisions take about 0.3–1 s (local test: mean 0.3 s, max 1.1 s), far inside the 150 s
  ladder timer.
- **Hardware:** use a machine that is otherwise idle. Our bot needs one core.
- **Do not** run two separate instances on the same account, run the account against itself, or play more games than planned to improve
  the number.
- **Stopping rule:** 100 games, or earlier once the Glicko deviation (RD, shown on the ladder page) is below 50.

## 6. Outputs (send these back)

Everything lands in `~/sandbox/ourbot/ladder/<username>/` (change it with `--out`):

| File | Contents |
|---|---|
| `ladder_games.jsonl` | One line per game: UTC time, battle id, opponent, result, turns, our Elo before → after, replay path, decisions, mean/max decision time, timer messages |
| `decisions.jsonl` | Per-decision log of our solver (matrices, strategy played) |
| `replays/` | HTML + log replay of every game |

Also send a screenshot or copy of the account's ladder row at the end: **Elo, GXE and Glicko rating ± RD**. Elo
has a floor of 1000 on Showdown. For the current evaluation protocol, report
starting/ending **Elo and net change** as the primary metric; GXE and RD are
additional ladder diagnostics.

## Options

| Flag | Default | Meaning |
|---|---|---|
| `--method` | `agentform` | which of our agents plays (any `run_ourbot.METHODS`; `random`, `maxpower` for testing) |
| `--k` | 4 | sampled hidden-information worlds per decision |
| `--format` | `gen9randombattle` | ladder format |
| `--games` | 1 | games to play, then exit |
| `--concurrency` | 1 | maximum active battles, from 1 to 5 |
| `--local` | off | local `--no-security` server, no password (testing only) |

Concurrent play supports Gen 9 Random Battle with `agentform`, `pimc`,
`top`, `random`, `maxpower`, `hedge_bayes_mw`, `bayes_safex_mw`, or `mcts_native`.
The per-world Bayesian methods use one independent persistent learner per stream,
with a fresh pending-action context and decision log for every battle. A stream
is reused only after its prior battle and final feedback complete. Checkpoints
live in `learning/`; `learning_streams.jsonl` maps games to streams. When switching
a sequential evaluation to concurrent mode, each stream starts from the last
sequential checkpoint, then learns independently. The learners are not merged.
The concurrent runner serializes solver calls because upstream sampling uses
global RNG state. Network waits and battle state machines overlap. New searches
are paced at least 20 seconds apart on public servers. Reaching five active games
depends on matchmaking and game length. A battle error stops new searches and
forfeits the failed battle while allowing other active battles to finish.

The corrected `hedge_bayes_mw` uses the per-world pure security objective and
action-conditional credit over all supporting worlds. Its v2 checkpoints are
incompatible with the earlier highest-world credit rule. Use a **fresh output
directory** when upgrading; sequential and concurrent startup reject legacy
Hedge checkpoints before connecting. Historical non-mw methods keep their
existing semantics. See [design and measured effects](../../docs/per-world-expert-feedback.md).

`progress.json` records started/completed/failed counts, active room IDs, peak
concurrency and a heartbeat. Concurrent decision logs live in `decisions/<battle>.jsonl`.
`errors.jsonl` records failures. The game ledger and local replay locations are unchanged.

## Detailed 100-game tactical audit

For a frozen 50-game native MCTS test, with ladder Elo change as the primary
endpoint:

```bash
bash run_decision_audit.command --method mcts_native --games 50 --concurrency 5 --mcts-ms 1000
```

This uses four sampled worlds, one native search thread, and a total 1,000 ms
native budget split across the worlds. Root visits are filtered to server-legal
actions, normalized per world, weighted by world priors, and the largest aggregate
share is played. Exact states/root statistics are logged without building payoff
matrices. `evaluation_results.json` reports starting/ending Elo and net change;
`run_status.json` includes the current Elo delta while running. Keep the method
and budget fixed within each 50-game test. See [implementation and evaluation](../../docs/native-mcts-evaluation.md).

The secure launcher creates a fresh batch with `hedge_bayes_mw`, four worlds,
up to five concurrent games, and exact state/payoff snapshots:

```bash
bash run_decision_audit.command
```

It prompts for the account password in Terminal and never saves it. To resume
the same batch, pass its existing directory; the completed ledger games count
toward the original 100 rather than starting another 100:

```bash
bash run_decision_audit.command --out <existing-batch-directory> --concurrency 5
```

Concurrency changes are recorded in `evaluation.json`. Each Bayesian stream
carries its own Hedge weights and style posterior between games; active games
never share pending feedback or checkpoint files. A sequential-to-concurrent
transition seeds each stream with the last sequential checkpoint.

`run_status.json`, `progress.json`, per-game decisions and replays track the run.
When it stops, the launcher writes `tactical_audit.json` and `tactical_audit.md`.
The auditor also runs independently on an incomplete batch:

```bash
.local-play/venv/bin/python engine/showdown/audit_decisions.py <batch-directory> --write
```

It joins executed moves to saved decisions and flags reflected/redundant rocks,
Toxic into Steel, Ground into Flying, capped setup, setup before a KO, switches
into revealed Hurricane, and Punch selections when Knock Off is available.
Forced actions and opponent switch-ins are labeled separately. Flags identify
positions to inspect; move priority, healing, switch prediction and modeled
payoffs still matter when judging whether a choice was poor.

For an explicitly authorized continuation after that cohort, use
`continue_public_evaluation.py --out <batch-directory>` in Terminal. It prompts
once, waits for the original run's account lock, then uses five existing learning
streams with no game-count cap. It writes `continuous_status.json` and
`continuous.log`. Creating `<batch-directory>/STOP_CONTINUOUS` stops new searches
and drains active games. It never starts if that stop request already exists.

Once play continues beyond 100, `audit_decisions.py --limit 100 --write` and
`summarize_tactical_run.py --limit 100 --write` keep the original 100-game cohort
fixed in `*_first100` artifacts. Extra games remain separately identifiable in
the same ledger and replay logs.

Local test, two terminals against a local server started as in `SETUP.md`:

```bash
$PY ladder.py --local --server ws://127.0.0.1:8000/showdown/websocket --username localeqbot --games 3
$PY ladder.py --local --server ws://127.0.0.1:8000/showdown/websocket --username localfoebot --method maxpower --games 3
```

## Reference

foul-play (the MCTS bot whose client we use) reports peaks of GXE 88% and Elo 2341 in gen9randombattle
(author's post, 11 May 2025, https://pmariglia.github.io/posts/foul-play). Locally, our agent scores about 0.31 against
stock foul-play at 100 ms per state, so expect a clearly lower rating.
