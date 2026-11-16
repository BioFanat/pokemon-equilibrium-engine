# Real-Showdown test environment

Runs our agents in real Pokémon Showdown games, two ways:

- **Headless** (experiments): the real Showdown simulator (`pokemon-showdown simulate-battle`) over stdin/stdout,
  no server. Paired, seeded games: same teams and battle PRNG, seats swapped. `headless.py`, `batch.py`.
- **Live server** (play in the browser): a local Showdown server, our bot logged in as `eqbot`. `run_ourbot.py`,
  `run_fp_challenger.py`.

Our move choice plugs into **foul-play**'s client (battle tracking, sampling of hidden opponent sets). foul-play is
GPL-3.0 and is NOT vendored here; it is cloned separately and imported at runtime.

## Play against the bot: one-command local setup

From this repository's root, run:

```bash
bash play_bot.command
```

The first run installs a project-local Node 22 runtime if needed, an isolated
Rust toolchain if needed, the pinned Showdown/Foul Play sources, and a Python
3.12 environment under `.local-play/`. It builds the simulator and starts both
the local server and our bot. It does not change global packages or shell
profiles. Initial installation requires internet access and may take several
minutes to compile the Rust engine. Later runs reuse the environment.

Once `eqbot` is confirmed online, the launcher opens
[the local server](http://127.0.0.1:8000). The server redirects to Showdown's
hosted client configured for your local server, so the browser also needs
internet access for the client/assets. Pick an unregistered nickname, find
`eqbot`, click **Challenge**, and select **[Gen 9] Random Battle**.
Both battle participants connect to your local server; no port forwarding or
Pokémon Showdown account is needed. Keep the Terminal open; **Ctrl+C** stops
the server and bot started by this launcher.

Options:

```bash
bash play_bot.command --method hedge_bayes    # opponent-style adaptation
bash play_bot.command --mechanics-shadow     # log new mechanics beliefs without changing moves
bash play_bot.command --port 8001            # if port 8000 is occupied
bash play_bot.command --no-browser           # print the URL without opening a browser
python3.12 engine/showdown/local_play.py doctor
python3.12 engine/showdown/local_play.py setup # install/build without starting games
```

The default method is `agentform`, with four sampled worlds. The experimental
mechanics tracker is optional and runs only in shadow mode through this
launcher; its unverified damage backend is not enabled automatically.

Logs are `.local-play/logs/server.log` and `.local-play/logs/bot.log`.
Battle replays and decisions are under `.local-play/work/`.
An occupied port produces an error rather than stopping another server.
The older manual instructions below remain available for an existing setup.

The launcher was checked with dependency-free tests. Full installation and
play could not be exercised in the restricted tool environment because GitHub,
PyPI and npm downloads were unreachable. If that restriction affects the first
run, execute the command in your normal macOS Terminal.

## Pinned components

| Component | Source | Version |
|---|---|---|
| Showdown server/simulator | https://github.com/smogon/pokemon-showdown | commit `7e743761cd13cdb361fc0f58eb5a866284894f5b` |
| foul-play | https://github.com/pmariglia/foul-play | commit `8f541f80f91f960603673796559f796b10ccf0fb` |
| Node.js | | ≥ 22 (tested 25.2.1) |
| Python | | 3.12, packages in `requirements-bot.txt` |
| poke-engine | PyPI sdist, built with `terastallization` | 0.0.48 |

## Setup (safe defaults)

```bash
mkdir -p ~/sandbox && cd ~/sandbox

# 1. Showdown: exact lockfile versions, no install scripts, localhost only
git clone https://github.com/smogon/pokemon-showdown.git && cd pokemon-showdown
git checkout 7e743761cd13cdb361fc0f58eb5a866284894f5b
npm ci --ignore-scripts
cp config/config-example.js config/config.js
sed -i '' "s/exports.bindaddress = '0.0.0.0'/exports.bindaddress = '127.0.0.1'/" config/config.js   # GNU sed: -i without ''
node build            # compiles TypeScript to dist/ (no network)
cd ..

# 2. foul-play (GPL-3.0), kept outside this repo
git clone https://github.com/pmariglia/foul-play.git && git -C foul-play checkout 8f541f80f91f960603673796559f796b10ccf0fb

# 3. Bot venv
uv venv --python 3.12 bot-venv
uv pip install --python bot-venv/bin/python -r <repo>/showdown/requirements-bot.txt
uv pip install --python bot-venv/bin/python --no-binary poke-engine "poke-engine==0.0.48" \
  --config-settings 'build-args=--features poke-engine/terastallization --no-default-features'
```

Security notes: `npm ci --ignore-scripts` installs exactly the lockfile (396 packages, all from registry.npmjs.org
with integrity hashes) and runs no install scripts. The server binds to 127.0.0.1. The bot's login is patched to
never contact play.pokemonshowdown.com (the local server runs with `--no-security`). Replay pages load Showdown's
viewer script from play.pokemonshowdown.com (read-only).

## Paths (environment variables, with defaults)

| Variable | Default | Meaning |
|---|---|---|
| `SHOWDOWN_DIR` | `~/sandbox/pokemon-showdown` | Showdown checkout |
| `FOULPLAY_DIR` | `~/sandbox/foul-play` | foul-play checkout |
| `ENGINE_DIR` | parent of this folder | this repo (nash.py, hidden_agents.py, pkbattle.py) |
| `OURBOT_WORK` | `~/sandbox/ourbot` | logs, replays, `replays_headless/` outputs |
| `BOT_PYTHON` | `~/sandbox/bot-venv/bin/python` | interpreter used by `batch.py` |
| `OURBOT_K` | `4` | sampled worlds per decision |

## Run

```bash
PY=~/sandbox/bot-venv/bin/python
# one paired game: our agent-form bot vs stock foul-play (MCTS 100 ms), real simulator
$PY showdown/headless.py --seed 1
# head to head, e.g. experiment E1 (agent form vs PIMC), 300 paired seeds on 6 workers
$PY showdown/batch.py --tag E1 --a agentform --b pimc --seeds 1-300 --workers 6
#   -> $OURBOT_WORK/replays_headless/E1_summary.json (pair score of A, 95% CI, errors, CPU per decision)

# live server: play our bot in the browser
(cd $SHOWDOWN_DIR && node pokemon-showdown start --no-security 8000) &
$PY showdown/run_ourbot.py          # then open http://127.0.0.1:8000 and challenge 'eqbot' to Gen 9 Random Battle
$PY showdown/run_fp_challenger.py 5 # or let stock foul-play challenge it 5 times

# record a saved replay to MP4 (headless Chrome + ffmpeg)
$PY showdown/record_replay.py $OURBOT_WORK/replays/<battle>.html 240 Fast
```

Move-choice methods (`--a/--b`): `agentform` (one LP over K sampled worlds), `pimc` (LP per world, blended by
weights), `stock` (foul-play MCTS), `ours` (= agentform). Each game writes a replay `.html`, the spectator `.log`,
a record `.json` (seed, sides, winner, true teams of both sides, timing, sim errors) and per-player decision logs
(mixture, move, and a shadow solve of the other method on the same worlds).

Measured cost: about 36 CPU-s per head-to-head game (0.30 CPU-s per decision per side), 26–32 s wall per paired seed.
