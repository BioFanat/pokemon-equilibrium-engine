# Running our bot on the public Showdown ladder

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
has a floor of 1000 on Showdown, so a weak bot sits at 1000; GXE (estimated win chance against an average ladder
player) is the number to report.

## Options

| Flag | Default | Meaning |
|---|---|---|
| `--method` | `agentform` | which of our agents plays (any `run_ourbot.METHODS`; `random`, `maxpower` for testing) |
| `--k` | 4 | sampled hidden-information worlds per decision |
| `--format` | `gen9randombattle` | ladder format |
| `--games` | 1 | games to play, then exit |
| `--concurrency` | 1 | maximum active battles, from 1 to 5 |
| `--local` | off | local `--no-security` server, no password (testing only) |

Concurrent play currently supports Gen 9 Random Battle with `agentform`, `pimc`,
`top`, `random`, or `maxpower`. Stateful Hedge/style learning keeps the sequential
runner until its cross-game update semantics support overlapping battles.
The concurrent runner serializes solver calls because upstream sampling uses
global RNG state. Network waits and battle state machines overlap. New searches
are paced at least 20 seconds apart on public servers. Reaching five active games
depends on matchmaking and game length. A battle error stops new searches and
forfeits the failed battle while allowing other active battles to finish.

`progress.json` records started/completed/failed counts, active room IDs, peak
concurrency and a heartbeat. Concurrent decision logs live in `decisions/<battle>.jsonl`.
`errors.jsonl` records failures. The game ledger and local replay locations are unchanged.

Local test, two terminals against a local server started as in `SETUP.md`:

```bash
$PY ladder.py --local --server ws://127.0.0.1:8000/showdown/websocket --username localeqbot --games 3
$PY ladder.py --local --server ws://127.0.0.1:8000/showdown/websocket --username localfoebot --method maxpower --games 3
```

## Reference

foul-play (the MCTS bot whose client we use) reports peaks of GXE 88% and Elo 2341 in gen9randombattle
(author's post, 11 May 2025, https://pmariglia.github.io/posts/foul-play). Locally, our agent scores about 0.31 against
stock foul-play at 100 ms per state, so expect a clearly lower rating.
