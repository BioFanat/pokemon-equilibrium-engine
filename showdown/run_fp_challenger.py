"""Stock foul-play (MCTS) challenging our bot on the local server, local login only.
  ~/sandbox/bot-venv/bin/python ~/sandbox/ourbot/run_fp_challenger.py [games]"""
import asyncio, os, sys
FP = os.environ.get("FOULPLAY_DIR", os.path.expanduser("~/sandbox/foul-play")); sys.path.insert(0, FP)
from fp import websocket_client as fp_ws

async def local_login(self):
    await self.get_id_and_challstr()
    await self.send_message("", [f"/trn {self.username},0,"])
    await asyncio.sleep(1)
    return self.username

fp_ws.PSWebsocketClient.login = local_login
if __name__ == "__main__":
    n = sys.argv[1] if len(sys.argv) > 1 else "1"
    sys.argv = [sys.argv[0], "--websocket-uri", "ws://127.0.0.1:8000/showdown/websocket", "--ps-username", "foulplaybot",
                "--bot-mode", "challenge_user", "--user-to-challenge", "eqbot", "--pokemon-format", "gen9randombattle",
                "--run-count", n, "--search-time-ms", "100", "--log-level", "INFO"]
    os.chdir(FP)
    from fp.main import run_foul_play
    asyncio.run(run_foul_play())
