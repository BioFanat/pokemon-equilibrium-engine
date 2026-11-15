"""Run OUR agent (agent-form solve over sampled worlds) on a LOCAL Showdown server through foul-play's client.

foul-play (GPL-3.0, ~/sandbox/foul-play) does the websocket protocol, battle tracking and sampling of the opponent's
hidden sets. We replace only its move choice: each turn, take K sampled battles with weights from foul-play's
random-battle mode, convert each to a poke-engine State, build our full depth-2 payoff matrix per world
(nash.full_matrix), and solve ONE agent-form LP over the worlds (hidden_agents.agent_form_lp). Then sample.

Local-only: login skips play.pokemonshowdown.com (the local server runs with --no-security).

  ~/sandbox/bot-venv/bin/python ~/sandbox/ourbot/run_ourbot.py            # waits for your challenge as 'eqbot'
Then in the browser (http://127.0.0.1:8000): pick a name, find user 'eqbot', challenge to [Gen 9] Random Battle.
Per-turn decisions are logged to ~/sandbox/ourbot/decisions.jsonl.
"""
import json
import os
import random
import sys
import time

HOME = os.path.expanduser("~")
FP = os.environ.get("FOULPLAY_DIR", os.path.join(HOME, "sandbox", "foul-play"))
ENGINE = os.environ.get("ENGINE_DIR", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
WORK = os.environ.get("OURBOT_WORK", os.path.join(HOME, "sandbox", "ourbot"))
LOG = os.path.join(WORK, "decisions.jsonl")
K_WORLDS = int(os.environ.get("OURBOT_K", "4"))
sys.path[:0] = [FP, ENGINE]

import numpy as np  # noqa: E402

import fp.modes.base as fp_base  # noqa: E402
from fp import websocket_client as fp_ws  # noqa: E402
from fp.search.poke_engine_helpers import battle_to_poke_engine_state  # noqa: E402

import hidden_agents as ha  # noqa: E402
import nash  # noqa: E402
import pkbattle as pb  # noqa: E402

RNG = random.Random()


def solve(method, Ms, w):
    """Our mixed strategy over the aligned rows. agentform: one LP max_x Σ_k w_k min_j (M_kᵀ x)_j.
    pimc: LP each world, blend x̄ = Σ_k w_k x_k (the engine's hidden_agents pimc, with foul-play's world weights)."""
    if method == "agentform":
        return ha.agent_form_lp(Ms, w)[0]
    x = sum(wk * ha.lp_x(M)[0] for wk, M in zip(w, Ms))
    return x / x.sum()


def make_move_fn(method: str, K: int, log=None):
    """Move function over K of foul-play's sampled worlds (shared sampling, matrices and alignment), solved by
    method in {"agentform", "pimc"}; returns a poke-engine move string. Logs each decision to `log`, or to the
    module-level LOG (read at call time) when log is None. Each record also carries the other method's solve on
    the same matrices (alt_*; not timed in secs, never consumes RNG) so the two arms can be compared per state."""
    assert method in ("agentform", "pimc"), method
    other = "pimc" if method == "agentform" else "agentform"

    def find_best_move(battle) -> str:
        from copy import deepcopy
        battle = deepcopy(battle)
        if battle.team_preview:                              # not used in random battles; mirror foul-play
            battle.user.active = battle.user.reserve.pop(0)
            battle.opponent.active = battle.opponent.reserve.pop(0)
        t0 = time.time()
        worlds = battle.mode.prepare_battles(battle, K)
        Ms, ws, rows, legal, raw = [], [], [], None, {}
        for b, chance in worlds:
            state = battle_to_poke_engine_state(b)
            r = nash.pe.id(state, 0)                         # raw engine option strings for side one (us)
            M, s1, s2 = nash.full_matrix(state)
            if legal is None:
                legal = s1
                raw = {pb._canon(o): o for o in r.s1}
            Ms.append(M); ws.append(chance); rows.append(s1)
            if len(legal) == 1:
                break
        alt = None
        if len(legal) == 1:
            move = raw.get(legal[0], legal[0]); x = np.ones(1)
            secs = time.time() - t0
        else:
            Ms = [ha.align(M, s1, legal) for M, s1 in zip(Ms, rows)]
            w = np.asarray(ws) / np.sum(ws)
            x = solve(method, Ms, w)
            move = raw.get(choice := legal[RNG.choices(range(len(legal)), weights=x)[0]], choice)
            secs = time.time() - t0
            t1 = time.time()
            alt = solve(other, Ms, w)
            alt_secs = time.time() - t1
        top = np.argsort(-x)[:3]
        rec = dict(method=method, turn=battle.turn, worlds=len(Ms), shape=[int(Ms[0].shape[0]), int(Ms[0].shape[1])],
                   mix={legal[i]: round(float(x[i]), 3) for i in top if x[i] >= 0.01},
                   move=move, secs=round(secs, 2))
        if alt is not None:
            atop = np.argsort(-alt)[:3]
            rec.update(alt_method=other, alt_mix={legal[i]: round(float(alt[i]), 3) for i in atop if alt[i] >= 0.01},
                       argmax_same=bool(np.argmax(x) == np.argmax(alt)), tv=round(float(0.5 * np.abs(x - alt).sum()), 4),
                       alt_secs=round(alt_secs, 3))
        with open(log or LOG, "a") as f:
            f.write(json.dumps(rec) + "\n")
        return move
    return find_best_move


our_find_best_move = make_move_fn("agentform", K_WORLDS)


async def local_login(self):
    """Local --no-security server: no assertion needed, so never contact play.pokemonshowdown.com."""
    await self.get_id_and_challstr()
    await self.send_message("", [f"/trn {self.username},0,"])
    import asyncio
    await asyncio.sleep(1)
    return self.username


REPLAYS = os.path.join(WORK, "replays")
REPLAY_HTML = """<!DOCTYPE html>
<meta charset="utf-8"><title>{tag}</title>
<div class="wrapper replay-wrapper" style="max-width:1180px;margin:0 auto">
<div class="battle"></div><div class="battle-log"></div><div class="replay-controls"></div><div class="replay-controls-2"></div>
<script type="text/plain" class="battle-log-data">{log}</script>
</div>
<script src="https://play.pokemonshowdown.com/js/replay-embed.js"></script>
"""
_buf: dict[str, list[str]] = {}
_orig_receive = fp_ws.PSWebsocketClient.receive_message


async def recording_receive(self):
    """Tee each battle room's public protocol lines; on |win| or |tie| write <tag>.log and a replay <tag>.html
    (the standard Showdown replay format, rendered by play.pokemonshowdown.com's replay-embed.js)."""
    msg = await _orig_receive(self)
    if os.environ.get("OURBOT_RAW"):
        open(os.path.join(WORK, "raw.txt"), "a").write(repr(msg[:300]) + "\n")
    if msg.startswith(">battle-"):
        tag, *lines = msg.split("\n")
        tag = tag[1:].strip()
        out, skip = _buf.setdefault(tag, []), False
        for ln in lines:
            if skip:                                        # |split|pN: the next line is the private copy, drop it
                skip = False
                continue
            if ln.startswith("|split|"):
                skip = True
                continue
            if ln.startswith(("|request|", "|init|", "|title|", "|j|", "|J|", "|l|", "|L|")) or not ln:
                continue
            out.append(ln)
        if any(ln.startswith("|win|") or ln == "|tie" or ln.startswith("|tie|") for ln in lines):
            os.makedirs(REPLAYS, exist_ok=True)
            log = "\n".join(out)
            open(os.path.join(REPLAYS, f"{tag}.log"), "w").write(log + "\n")
            open(os.path.join(REPLAYS, f"{tag}.html"), "w").write(REPLAY_HTML.format(tag=tag, log=log.replace("</", "<\\/")))
            _buf.pop(tag, None)
    return msg


fp_base.find_best_move = our_find_best_move
fp_ws.PSWebsocketClient.login = local_login
fp_ws.PSWebsocketClient.receive_message = recording_receive

if __name__ == "__main__":
    import asyncio
    defaults = ["--websocket-uri", "ws://127.0.0.1:8000/showdown/websocket", "--ps-username", "eqbot",
                "--bot-mode", "accept_challenge", "--pokemon-format", "gen9randombattle", "--run-count", "100",
                "--log-level", "INFO"]
    sys.argv = [sys.argv[0]] + (sys.argv[1:] or defaults)
    os.chdir(FP)                                            # foul-play loads data relative to its root
    from fp.main import run_foul_play
    asyncio.run(run_foul_play())
