"""Headless paired games (engine/showdown/headless.py) with a side log of the PER-WORLD matrices our planner solved.

Monkeypatches run_ourbot.avg_matrix (the only place the K row-aligned world matrices and their column labels are
together) to append one JSON line per decision to $JOINT_WORLDS_LOG:
  {game: <tag>, n: <call index in game>, Ms: [K][m][n] (2 dp), cols: [K][n], w: [K], labels: union}
Decision records with worlds > 1 and more than one legal move match these lines in order (checked on `labels`).
Use with player A = hedge_bayes / bayes_safex and player B = stock so only one side calls avg_matrix.

  OURBOT_WORK=<scratch> OURBOT_LOG_ABAR=1 JOINT_WORLDS_LOG=<scratch>/worlds.jsonl \
  ~/sandbox/bot-venv/bin/python research/scripts/joint/headless_logworlds.py --seed 1 --a hedge_bayes --b stock --tag JP
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SHOWDOWN = os.path.abspath(os.path.join(HERE, "..", "..", "..", "engine", "showdown"))
sys.path[:0] = [SHOWDOWN, os.path.abspath(os.path.join(SHOWDOWN, ".."))]
import headless                                   # noqa: E402  (adds foul-play and engine/showdown to sys.path)

STATE = dict(game=None, n=0)


def install(run_ourbot):
    orig = run_ourbot.avg_matrix

    def avg_matrix(Ms, cols, w):
        Abar, labels = orig(Ms, cols, w)
        path = os.environ.get("JOINT_WORLDS_LOG")
        if path:
            rec = dict(game=STATE["game"], n=STATE["n"], Ms=[[[round(float(v), 2) for v in row] for row in M] for M in Ms],
                       cols=[list(c) for c in cols], w=[float(x) for x in w], labels=list(labels))
            with open(path, "a") as f:
                f.write(json.dumps(rec) + "\n")
            STATE["n"] += 1
        return Abar, labels
    run_ourbot.avg_matrix = avg_matrix


orig_setup = headless.setup_foulplay


def setup_foulplay(search_time_ms, k):
    state, run_ourbot = orig_setup(search_time_ms, k)
    install(run_ourbot)
    return state, run_ourbot


orig_play = headless.play_game


async def play_game(seed, game, a_side, fns, run_tag, k, state, run_ourbot, stream=0):
    STATE["game"], STATE["n"] = f"{run_tag}-s{seed}-g{game}", 0
    return await orig_play(seed, game, a_side, fns, run_tag, k, state, run_ourbot, stream)


headless.setup_foulplay = setup_foulplay
headless.play_game = play_game

if __name__ == "__main__":
    headless.main()
