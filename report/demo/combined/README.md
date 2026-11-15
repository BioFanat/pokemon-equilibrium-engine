# Combined demo: Showdown replay + our decision panel

Output: `../showdown_combined.mp4` (1180x800, 30 fps) and `../showdown_combined.gif` (900 wide, 12 fps, 28 s).
Game: `demo-s602-g1`. Our agent (agentform, K=4) is p1 and plays foul-play stock MCTS (100 ms). OurAgentFormA wins in 27 turns.

Files
- `run_ourbot.py`: the git-HEAD `engine/showdown/run_ourbot.py` plus extra matrix logging (`run_ourbot_logging.patch`).
  Each decision record also gets `rows`/`rows_raw`, `x`, `K`, `w`, `opp_teams`, and for world k* = argmax w
  (k* = 0 when all weights are equal): `cols`/`cols_raw`, `M` (rounded to 1 dp) and `y`, the column player's equilibrium from `nash.lp(M_k*)`.
  The move choice and RNG use are unchanged.
- `record_replay.py`: the git-HEAD recorder plus turn timing (`record_replay_turns.patch`). Every 200 ms it evaluates
  `Replays.battle.turn` / `.ended` in the page through CDP (`Replays` is replay-embed.js's global). A 4th argument
  keeps the frames together with `frames.json` and `turns.json`. Recording stops 3 s after the battle ends.
- `composite.py`: the compositor. Output frame k (t = k/30) uses the latest screencast frame and the latest turn sample at or before t,
  so the panel always shows the same turn as the replay's own counter. Opponent and own actions per turn come from the spectator `.log`.
- `run_seeds.sh`: the seed loop. `demo-s602-g1.*`: the game used (log, replay html, record, decisions, turn timing).

Commands (S = a scratch dir holding headless.py, run_ourbot.py, record_replay.py and composite.py; headless.py is git HEAD, unchanged)
```
cd engine && git show HEAD:showdown/headless.py > $S/headless.py   # (same for run_ourbot.py / record_replay.py, then apply the patches)
OURBOT_WORK=$S/work ENGINE_DIR=.../final/engine ~/sandbox/bot-venv/bin/python $S/headless.py --seed 602 --tag demo --a agentform --b stock
~/sandbox/bot-venv/bin/python $S/record_replay.py $S/work/replays_headless/demo-s602-g1.html 300 Fast $S/rec_fast
python3 $S/composite.py $S/work/replays_headless/demo-s602-g1.log $S/work/replays_headless/demo-s602-g1.A.decisions.jsonl $S/rec_fast $S/out_combined.mp4
ffmpeg -ss 27 -t 28 -i out_combined.mp4 -vf "fps=12,scale=900:-1:flags=lanczos,split[a][b];[a]palettegen=max_colors=128:stats_mode=diff[p];[b][p]paletteuse=dither=bayer:bayer_scale=4" out_combined.gif
```
`python3` needs PIL. bot-venv has no PIL; the run used `final/../evolvability/.venv/bin/python3`.
Normal speed took 209 s for this game, so the recording uses Fast (158 s).
The GIF window (27 to 55 s, turns 4 to 10) was picked to cover the most turns with a mixed strategy (max x < 0.85).
