"""Ladder mode: OUR agent searches for rated games (/search <format>) through foul-play's client, one game at a time.

Reuses run_ourbot (move function, replay recording, the local no-assertion login) and foul-play (websocket protocol,
battle tracking); neither is edited, everything is applied at runtime. All guards run before any fp / run_ourbot
import, so a refused or --dry-run invocation never opens a socket.

  local  (--local, or a ws URI on 127.0.0.1 / localhost): run_ourbot.local_login, no password.
  public (anything else): foul-play's real login with the password from env PS_PASSWORD (never printed, logged or
         written); requires --i-am-registered (the account is registered as a bot via /ht) and "bot" in the username.

  ~/sandbox/bot-venv/bin/python ladder.py --local --server ws://127.0.0.1:8000/showdown/websocket --username localeqbot
  PS_PASSWORD=... ~/sandbox/bot-venv/bin/python ladder.py --server wss://sim3.psim.us/showdown/websocket \\
      --username cs535eqbot --games 1 --i-am-registered

Per game, one JSON line in <out>/ladder_games.jsonl (utc, battle tag, opponent, result, turns, rating before/after if
Showdown sends it, replay path, mean/max decision wall time, |inactive| timer lines). Decisions go to
<out>/decisions.jsonl, replays to <out>/replays/. Ctrl-C stops after the current game (a second Ctrl-C aborts).
"""
import argparse
import asyncio
import json
import logging
import os
import re
import signal
import sys
import time
import traceback
from datetime import datetime, timezone
from urllib.parse import urlparse

HOME = os.path.expanduser("~")
FP = os.environ.get("FOULPLAY_DIR", os.path.join(HOME, "sandbox", "foul-play"))
LOCAL_HOSTS = ("127.0.0.1", "localhost")
PASSWORD_ENV = "PS_PASSWORD"
RATING_RE = re.compile(r"^(?P<name>.+?)'s rating: (?P<before>\d+(?:\.\d+)?) (?:&rarr;|→|->) (?:<strong>)?"
                       r"(?P<after>\d+(?:\.\d+)?)")


def toid(name):
    return re.sub(r"[^a-z0-9]", "", (name or "").lower())


def parse_args(argv):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--server", required=True, help="websocket URI, e.g. ws://127.0.0.1:8000/showdown/websocket")
    p.add_argument("--username", required=True)
    p.add_argument("--method", default="agentform", help="run_ourbot.METHODS or simple_bots.METHODS")
    p.add_argument("--k", type=int, default=4, help="sampled worlds per decision")
    p.add_argument("--format", default="gen9randombattle")
    p.add_argument("--games", type=int, default=1)
    p.add_argument("--out", default=None, help="default ~/sandbox/ourbot/ladder/<username>/")
    p.add_argument("--local", action="store_true", help="local --no-security server: no password, no assertion")
    p.add_argument("--i-am-registered", action="store_true",
                   help="confirm the (public) account is registered as a bot via /ht")
    p.add_argument("--dry-run", action="store_true", help="print the resolved config and exit without connecting")
    return p.parse_args(argv)


def resolve(args):
    """Resolved config (no secrets: only whether PS_PASSWORD is set)."""
    u = urlparse(args.server)
    host = (u.hostname or "").lower()
    out = os.path.abspath(os.path.expanduser(args.out or os.path.join(HOME, "sandbox", "ourbot", "ladder", args.username)))
    return dict(server=args.server, scheme=u.scheme, host=host, local=args.local or host in LOCAL_HOSTS,
                local_flag=args.local, username=args.username, method=args.method, k=args.k, format=args.format,
                games=args.games, out=out, i_am_registered=args.i_am_registered,
                password_set=bool(os.environ.get(PASSWORD_ENV)))


def refusals(cfg):
    """Reasons not to start; empty means go. Checked before any connection code is imported."""
    why = []
    if cfg["scheme"] not in ("ws", "wss") or not cfg["host"]:
        why.append(f"--server must be a ws:// or wss:// URI with a host (got {cfg['server']!r})")
    if cfg["local_flag"] and cfg["host"] not in LOCAL_HOSTS:
        why.append(f"--local given but the server host {cfg['host']!r} is not one of {LOCAL_HOSTS}")
    if cfg["games"] < 1:
        why.append("--games must be >= 1")
    if not cfg["local"]:
        if not cfg["i_am_registered"]:
            why.append("public server: pass --i-am-registered to confirm the account is registered as a bot (/ht)")
        if not cfg["password_set"]:
            why.append(f"public server: set the password in env {PASSWORD_ENV} (never on the command line)")
        if "bot" not in cfg["username"].lower():
            why.append(f"public server: username {cfg['username']!r} must contain 'bot'")
    return why


def banner(cfg):
    kind = "LOCAL" if cfg["local"] else "PUBLIC"
    print(f"=== ladder.py [{kind}] server={cfg['server']} username={cfg['username']} format={cfg['format']} "
          f"games={cfg['games']} method={cfg['method']} k={cfg['k']} ===", flush=True)


def scrub(text, secret):
    return text.replace(secret, "***") if secret else text


class ScrubFilter(logging.Filter):
    """Replace the secret in every log record (foul-play's CustomFormatter formats record.msg directly)."""

    def __init__(self, secret):
        super().__init__()
        self.secret = secret

    def filter(self, record):
        if self.secret:
            record.msg, record.args = scrub(record.getMessage(), self.secret), None
        return True


# ---------------------------------------------------------------- per-battle observation of the message stream
GAMES = {}                                                 # battle_tag -> observed fields
DECISION_SECS = {}                                         # battle_tag -> [wall seconds per decision]
STOP = {"flag": False, "sigints": 0}


def observe(msg):
    """Record players, turns, winner, rating |raw| lines and |inactive| timer lines of each battle room."""
    if not msg.startswith(">battle-"):
        return
    tag, *lines = msg.split("\n")
    g = GAMES.setdefault(tag[1:].strip(), dict(players={}, turns=0, winner=None, tie=False, ratings={}, timer=[]))
    for ln in lines:
        p = ln.split("|")
        if len(p) < 2:
            continue
        if p[1] == "player" and len(p) > 3 and p[3]:
            g["players"][p[2]] = p[3]
        elif p[1] == "turn" and len(p) > 2 and p[2].isdigit():
            g["turns"] = int(p[2])
        elif p[1] == "win" and len(p) > 2:
            g["winner"] = p[2]
        elif p[1] == "tie":
            g["tie"] = True
        elif p[1] == "raw" and len(p) > 2:
            m = RATING_RE.match("|".join(p[2:]))
            if m:
                g["ratings"][toid(m["name"])] = dict(before=float(m["before"]), after=float(m["after"]))
        elif p[1] in ("inactive", "inactiveoff"):
            g["timer"].append(ln)


def game_record(tag, cfg, replays):
    g = GAMES.get(tag, {})
    me = toid(cfg["username"])
    opp = next((n for n in g.get("players", {}).values() if toid(n) != me), None)
    result = "tie" if g.get("tie") else ("win" if toid(g.get("winner")) == me else "loss") if g.get("winner") else None
    secs = DECISION_SECS.get(tag, [])
    replay = os.path.join(replays, f"{tag}.html")
    return dict(utc=datetime.now(timezone.utc).isoformat(timespec="seconds"), battle_tag=tag, opponent=opp,
                result=result, turns=g.get("turns"), rating=g.get("ratings", {}).get(me),
                replay=replay if os.path.exists(replay) else None, decisions=len(secs),
                decision_secs_mean=round(sum(secs) / len(secs), 3) if secs else None,
                decision_secs_max=round(max(secs), 3) if secs else None, timer=g.get("timer", []),
                method=cfg["method"], k=cfg["k"], format=cfg["format"], server=cfg["host"])


# ---------------------------------------------------------------- runtime (imported only after the guards)
def setup_runtime(cfg):
    """Import foul-play and run_ourbot, then apply: login (local patch or foul-play's real one), our timed move fn,
    replay dir, ledger observer on top of run_ourbot's recording receive. Returns the fp modules needed by run()."""
    os.makedirs(cfg["out"], exist_ok=True)
    os.environ.setdefault("OURBOT_WORK", cfg["out"])
    sys.path[:0] = [FP, os.path.dirname(os.path.abspath(__file__))]
    import fp.websocket_client as fp_ws
    real_login = fp_ws.PSWebsocketClient.login             # run_ourbot replaces it with local_login on import
    import run_ourbot
    import simple_bots
    import fp.modes.base as fp_base
    import fp.run_battle as fp_run

    if not cfg["local"]:
        fp_ws.PSWebsocketClient.login = real_login
    run_ourbot.LOG = os.path.join(cfg["out"], "decisions.jsonl")
    run_ourbot.REPLAYS = os.path.join(cfg["out"], "replays")
    if cfg["method"] in run_ourbot.METHODS:
        inner = run_ourbot.make_move_fn(cfg["method"], cfg["k"], log=run_ourbot.LOG,
                                        state_path=os.path.join(cfg["out"], f"{cfg['method']}_state.json"))
    elif cfg["method"] in simple_bots.METHODS:
        inner = simple_bots.make_move_fn(cfg["method"], seed=int(time.time()))
    else:
        raise SystemExit(f"unknown --method {cfg['method']!r}; choose from "
                         f"{run_ourbot.METHODS + simple_bots.METHODS}")

    def timed_move(battle):
        t0 = time.time()
        move = inner(battle)
        DECISION_SECS.setdefault(battle.battle_tag, []).append(time.time() - t0)
        return move

    fp_base.find_best_move = timed_move
    recv = fp_ws.PSWebsocketClient.receive_message          # run_ourbot.recording_receive

    async def ladder_receive(self):
        msg = await recv(self)
        observe(msg)
        return msg

    fp_ws.PSWebsocketClient.receive_message = ladder_receive
    return fp_ws, fp_run


async def run(cfg, password):
    fp_ws, fp_run = setup_runtime(cfg)
    from fp.config import FoulPlayConfig, init_logging
    from fp.data.mods.apply_mods import apply_mods
    from fp.modes import battle_mode
    from fp.teams import load_team

    os.chdir(FP)                                            # foul-play loads data relative to its root
    sys.argv = [sys.argv[0], "--websocket-uri", cfg["server"], "--ps-username", cfg["username"],
                "--bot-mode", "search_ladder", "--pokemon-format", cfg["format"], "--run-count", str(cfg["games"]),
                "--log-level", "INFO"]                     # never the password: it would show in the argv
    FoulPlayConfig.configure()
    FoulPlayConfig.password = password
    init_logging(FoulPlayConfig.log_level, FoulPlayConfig.log_to_file)
    for h in logging.getLogger().handlers:
        h.addFilter(ScrubFilter(password))
    apply_mods(FoulPlayConfig.format_spec)

    task = asyncio.current_task()

    def on_sigint():
        STOP["sigints"] += 1
        STOP["flag"] = True
        if STOP["sigints"] == 1:
            print("\n[ladder] Ctrl-C: stopping after the current game (Ctrl-C again to abort now)", flush=True)
        else:
            task.cancel()

    asyncio.get_running_loop().add_signal_handler(signal.SIGINT, on_sigint)
    ledger = os.path.join(cfg["out"], "ladder_games.jsonl")
    client = await fp_ws.PSWebsocketClient.create(username=cfg["username"], password=password, address=cfg["server"])
    FoulPlayConfig.user_id = await client.login()
    mode = battle_mode(FoulPlayConfig.format_spec.battle_type)
    try:
        for i in range(cfg["games"]):
            if mode.requires_team:
                team_packed, team_dict, _ = load_team(FoulPlayConfig.team_name)
                await client.update_team(team_packed)
            else:
                team_dict = None
                await client.update_team("None")
            known = set(GAMES)
            await client.search_for_match(cfg["format"])
            await fp_run.pokemon_battle(client, cfg["format"], team_dict)
            new = [t for t in GAMES if t not in known and (GAMES[t]["winner"] or GAMES[t]["tie"])]
            for tag in new:
                rec = game_record(tag, cfg, os.path.join(cfg["out"], "replays"))
                with open(ledger, "a") as f:
                    f.write(json.dumps(rec) + "\n")
                print(f"[ladder] game {i + 1}/{cfg['games']}: {rec['result']} vs {rec['opponent']} in {rec['turns']} "
                      f"turns, rating {rec['rating']}, decisions mean/max {rec['decision_secs_mean']}/"
                      f"{rec['decision_secs_max']}s", flush=True)
            if STOP["flag"]:
                break
    finally:
        await client.close()
    print(f"[ladder] done; ledger {ledger}", flush=True)


def main(argv=None):
    args = parse_args(sys.argv[1:] if argv is None else argv)
    cfg = resolve(args)
    why = refusals(cfg)
    banner(cfg)
    if args.dry_run:
        print(json.dumps(dict(cfg, would_refuse=why), indent=2))
        print("[ladder] --dry-run: not connecting")
        return 0
    if why:
        for w in why:
            print(f"[ladder] REFUSED: {w}", file=sys.stderr)
        return 2
    password = None if cfg["local"] else os.environ[PASSWORD_ENV]
    try:
        asyncio.run(run(cfg, password))
    except (KeyboardInterrupt, asyncio.CancelledError):
        print("[ladder] aborted", file=sys.stderr)
        return 130
    except Exception:
        print(scrub(traceback.format_exc(), password), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
