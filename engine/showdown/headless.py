"""Headless paired games through the REAL Showdown simulator (no server, no websocket).

  ~/sandbox/bot-venv/bin/python ~/sandbox/ourbot/headless.py --seed 1 [--k 4] [--search-time-ms 100]   # ours vs stock
  ~/sandbox/bot-venv/bin/python ~/sandbox/ourbot/headless.py --seed 1 --tag E1 --a agentform --b pimc  # head to head

Spawns `node pokemon-showdown simulate-battle` and drives two foul-play battle loops (fp.run_battle.pokemon_battle)
in one asyncio process, each through a FakePSClient that turns sim blocks into the websocket messages foul-play
expects and its `/choose ...|rqid` messages into `>pN ...` sim commands. Move choice is per player A / B, each one of
one of run_ourbot.METHODS (make_move_fn over K worlds; "ours" = agentform) or stock (foul-play MCTS,
fp.search.main.find_best_move). --a/--b pick them; --p1fn/--p2fn are the same flags named by game-1 seat.

Paired: one --seed plays two games with the same battle seed and the same per-player team seeds (p1 always gets
team A, p2 team B); player A is p1 in game 1 and p2 in game 2. Outputs go to ~/sandbox/ourbot/replays_headless/ as
<tag>-s<seed>-g<game>.*: .log (spectator log), .html (replay), .json (record), .raw.txt (all sim output), .fp.log,
and .A.decisions.jsonl / .B.decisions.jsonl for each player that runs our solver. A game where the sim rejects a
choice ("[Invalid choice]") and sends no new request within STALL_S ends as an error record (error field set).

Keep the top level light: stock foul-play spawns MCTS worker processes, which re-import this file as __mp_main__.
"""
import argparse
import asyncio
import json
import os
import resource
import sys
import time

HOME = os.path.expanduser("~")
FP = os.environ.get("FOULPLAY_DIR", os.path.join(HOME, "sandbox", "foul-play"))
SHOWDOWN = os.environ.get("SHOWDOWN_DIR", os.path.join(HOME, "sandbox", "pokemon-showdown"))
WORK = os.environ.get("OURBOT_WORK", os.path.join(HOME, "sandbox", "ourbot"))
GAME_TIMEOUT_S = int(os.environ.get("GAME_TIMEOUT_S", "600"))   # longest E1 game was ~150 s; a hang costs at most this
OUT = os.path.join(WORK, "replays_headless")
HEDGE_STATE = os.path.join(WORK, "hedge_state")      # carried Hedge weights: <tag>-stream<k>-<role>.json
FORMAT = "gen9randombattle"
SELF = "zzself"                     # FoulPlayConfig.username; must not be a substring of either player name
NAMES = {"agentform": "OurAgentForm", "pimc": "OurPimc", "stock": "FoulPlay", "hedge": "OurHedge",
         "hedge_fs": "OurHedgeFS", "bandit_ucb1": "OurBanditUCB", "bandit_exp3": "OurBanditExp",
         "rmplus_avg": "OurRMPlus", "bandit_ucb1_mv": "OurMaxVisitUCB", "bandit_exp3_mv": "OurMaxVisitExp", "hedge_bayes": "OurHedgeBayes", "bayes_safex": "OurBayesSafex",
         "exploit_bandit_ucb1": "OurXplUCB", "exploit_bandit_ucb1_mv": "OurXplUCBmv",
         "exploit_bandit_exp3_mv": "OurXplExpmv", "exploit_rmplus_avg": "OurXplRMPlus", "exploit_top": "OurXplTop",
         "exploit_agentform": "OurXplAgentForm", "top": "OurTop", "style_eq": "OurStyleEq", "style_safe": "OurStyleSafe",
         "style_uniform": "OurStyleUnif"}                                             # + player letter: distinct
FNS = {"ours": "agentform", "stock": "stock", **{m: m for m in NAMES if m != "stock"}}
STALL_S = 60                        # fail-fast after an [Invalid choice] with no new request
sys.path[:0] = [FP, os.path.dirname(os.path.abspath(__file__))]


def seeds(s):
    """Battle PRNG seed and the two per-player team seeds (Gen5 PRNG: four 16-bit numbers each, nonzero)."""
    mk = lambda k: [(s * 7919 + k * 104729 + i * 15485863) % 65521 + 1 for i in range(4)]
    return mk(0), mk(1), mk(2)


def public_lines(lines, side=None):
    """Resolve |split|pN: the next line is pN's private copy, the one after the public copy.
    side=None (spectator) keeps the public copy; side=pN keeps the private copy for its own side."""
    out, i = [], 0
    while i < len(lines):
        ln = lines[i]
        if ln.startswith("|split|"):
            owner = ln.split("|")[2]
            out.append(lines[i + 1] if owner == side else lines[i + 2])
            i += 3
            continue
        out.append(ln)
        i += 1
    return out


class FakePSClient:
    """The slice of fp.websocket_client.PSWebsocketClient that pokemon_battle/start_battle use."""

    def __init__(self, side, tag, opp_name, write_cmd):
        self.side, self.tag, self.opp_name, self.write_cmd = side, tag, opp_name, write_cmd
        self.q = asyncio.Queue()
        self.sent, self.rqid = [], 0
        self.q.put_nowait(f">{tag}\n|init|battle\n|title|{SELF} vs. {opp_name}\n")

    def feed_update(self, lines):
        lines = public_lines(lines, self.side)
        players = [ln for ln in lines if ln.startswith("|player|")]
        rest = [ln for ln in lines if not ln.startswith(("|player|", "|t:|"))]
        for ln in players:                                   # the real server sends each |player| line alone
            self.q.put_nowait(f">{self.tag}\n{ln}")
        if any(ln.strip("|") for ln in rest):
            self.q.put_nowait(f">{self.tag}\n" + "\n".join(rest))

    def feed_side(self, lines):
        for ln in lines:                                     # |request| and |error| each as their own message
            if ln.startswith("|request|"):                   # the server (room-battle), not the sim, adds rqid
                self.rqid += 1
                req = json.loads(ln[len("|request|"):])
                req["rqid"] = self.rqid
                ln = "|request|" + json.dumps(req)
            self.q.put_nowait(f">{self.tag}\n{ln}")

    async def receive_message(self):
        return await self.q.get()

    async def send_message(self, room, message_list):
        msg = message_list[0]
        self.sent.append("|".join([room] + message_list))
        if msg.startswith("/choose "):
            self.write_cmd(f">{self.side} {msg[len('/choose '):]}")
        elif msg.startswith(("/switch ", "/team ")):
            self.write_cmd(f">{self.side} {msg[1:]}")
        elif msg.startswith("/leave"):
            self.q.put_nowait(f">{self.tag}\n|deinit")
        # hf, gg, /timer on: chat and timer, nothing to simulate

    async def save_replay(self, battle_tag):
        pass

    async def leave_battle(self, battle_tag):
        await self.send_message("", [f"/leave {battle_tag}"])
        while True:
            msg = await self.receive_message()
            if battle_tag in msg and "deinit" in msg:
                return


class Timer:
    """Per-side decision timing: wall (perf_counter) and CPU (this thread, plus reaped child processes if
    children=True; only stock foul-play spawns any, so ours must not count the stock workers reaped meanwhile)."""

    def __init__(self, children):
        self.children, self.n, self.wall, self.cpu = children, 0, 0.0, 0.0

    def run(self, fn, battle):
        w0, c0 = time.perf_counter(), time.thread_time()
        ch0 = resource.getrusage(resource.RUSAGE_CHILDREN)
        try:
            return fn(battle)
        finally:
            ch1 = resource.getrusage(resource.RUSAGE_CHILDREN)
            self.n += 1
            self.wall += time.perf_counter() - w0
            self.cpu += time.thread_time() - c0
            if self.children:
                self.cpu += (ch1.ru_utime - ch0.ru_utime) + (ch1.ru_stime - ch0.ru_stime)

    def summary(self):
        return dict(decisions=self.n, wall_s=round(self.wall, 2), cpu_s=round(self.cpu, 2),
                    wall_s_per_decision=round(self.wall / max(self.n, 1), 3),
                    cpu_s_per_decision=round(self.cpu / max(self.n, 1), 3))


def setup_foulplay(search_time_ms, k):
    """Configure foul-play's process-global config without argparse; install the per-player move dispatcher."""
    import logging
    os.environ.setdefault("OURBOT_K", str(k))
    os.chdir(FP)                                             # foul-play loads data relative to its root
    from fp.config import FoulPlayConfig, SaveReplay
    from fp.data.mods.apply_mods import apply_mods
    from fp.data.sets.randbats import RandomBattleTeamDatasets
    import fp.modes.base as fp_base
    import fp.search.main as fp_search
    import run_ourbot                                        # patches fp_base.find_best_move; replaced below
    run_ourbot.K_WORLDS = k

    FoulPlayConfig.username = SELF
    FoulPlayConfig.pokemon_format = FORMAT
    FoulPlayConfig.smogon_stats = None
    FoulPlayConfig.search_time_ms = search_time_ms
    FoulPlayConfig.parallelism = 1
    FoulPlayConfig.team_preview_search_time_ms = search_time_ms
    FoulPlayConfig.team_preview_search_parallelism = 1
    FoulPlayConfig.search_threads = 1
    FoulPlayConfig.save_replay = SaveReplay.never
    FoulPlayConfig.log_to_file = False
    FoulPlayConfig.log_level = "INFO"
    logging.getLogger().setLevel(logging.INFO)
    apply_mods(FoulPlayConfig.format_spec)

    # Both loops share the RandomBattleMode singleton; its start_battle re-initializes the set datasets (empty then
    # refill) while the other side may be searching in a worker thread. The data is identical, so load it once.
    orig_init = RandomBattleTeamDatasets.initialize

    def init_once(self, format_spec, pkmn_names=None):
        if self.pkmn_mode != format_spec.full_name or not self.pkmn_sets:
            orig_init(self, format_spec, pkmn_names)
    RandomBattleTeamDatasets.initialize = init_once

    state = dict(fns=None, timers=None)                      # per game: side (p1/p2) -> move fn, Timer

    def dispatch(battle):
        side = battle.user.name
        return state["timers"][side].run(state["fns"][side], battle)
    fp_base.find_best_move = dispatch
    return state, run_ourbot


async def play_game(seed, game, a_side, fns, run_tag, k, state, run_ourbot, stream=0):
    """One game: player A (method fns["A"]) on a_side, player B (fns["B"]) on the other seat. A hedge player carries its
    weights in HEDGE_STATE/<run_tag>-stream<stream>-<role>.json (the same file in both games of the seed)."""
    import logging
    from fp.run_battle import pokemon_battle
    import fp.search.main as fp_search
    battle_seed, seed_a, seed_b = seeds(seed)
    tag = f"{run_tag}-s{seed}-g{game}"
    os.makedirs(OUT, exist_ok=True)
    b_side = "p2" if a_side == "p1" else "p1"
    side_of = {"A": a_side, "B": b_side}
    role = {a_side: "A", b_side: "B"}
    names = {s: NAMES[fns[role[s]]] + role[s] for s in ("p1", "p2")}
    logs, move_fns, timers = {}, {}, {}
    for r, side in side_of.items():
        if fns[r] == "stock":
            move_fns[side], timers[side] = fp_search.find_best_move, Timer(children=True)
            continue
        logs[r] = os.path.join(OUT, f"{tag}.{r}.decisions.jsonl")
        if os.path.exists(logs[r]):
            os.remove(logs[r])
        move_fns[side] = run_ourbot.make_move_fn(
            fns[r], k, log=logs[r], seed=seed, game=game,
            state_path=os.path.join(HEDGE_STATE, f"{run_tag}-stream{stream}-{r}.json"))
        timers[side] = Timer(children=False)
    state.update(fns=move_fns, timers=timers)

    fh = logging.FileHandler(os.path.join(OUT, tag + ".fp.log"), mode="w")
    fh.setLevel(logging.INFO)
    logging.getLogger().addHandler(fh)

    proc = await asyncio.create_subprocess_exec(
        "node", "pokemon-showdown", "simulate-battle", cwd=SHOWDOWN, limit=2 ** 24,
        stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
    raw = open(os.path.join(OUT, tag + ".raw.txt"), "w")

    def write_cmd(cmd):
        raw.write(f"<<< {cmd}\n")
        if not proc.stdin.is_closing():
            proc.stdin.write((cmd + "\n").encode())

    clients = {s: FakePSClient(s, f"battle-{FORMAT}-{tag}-{s}", names["p2" if s == "p1" else "p1"], write_cmd)
               for s in ("p1", "p2")}
    write_cmd(">start " + json.dumps({"formatid": FORMAT, "seed": battle_seed}))
    write_cmd(">player p1 " + json.dumps({"name": names["p1"], "seed": seed_a}))
    write_cmd(">player p2 " + json.dumps({"name": names["p2"], "seed": seed_b}))

    spectator, errors, first_requests, end = [], [], {}, {}
    stall_since = {}                                         # side -> time of an unanswered [Invalid choice]

    async def read_sim():
        block = []
        while True:
            line = await proc.stdout.readline()
            if not line:
                return
            line = line.decode().rstrip("\n")
            raw.write(line + "\n")
            if line:
                block.append(line)
                continue
            if not block:
                continue
            kind, body = block[0], block[1:]
            block = []
            if kind == "update":
                spectator.extend(public_lines(body))
                for c in clients.values():
                    c.feed_update(body)
            elif kind == "sideupdate":
                side, lines = body[0], body[1:]
                for ln in lines:
                    if ln.startswith("|error|"):
                        errors.append(f"{side} {ln}")
                        if "[Invalid choice]" in ln:
                            stall_since.setdefault(side, time.monotonic())
                    elif ln.startswith("|request|"):
                        stall_since.pop(side, None)
                        if side not in first_requests:
                            first_requests[side] = json.loads(ln[len("|request|"):])
                clients[side].feed_side(lines)
            elif kind == "end":
                end.update(json.loads(body[0]))
                return

    async def watchdog():
        while True:
            await asyncio.sleep(1)
            for side, t in list(stall_since.items()):
                if time.monotonic() - t > STALL_S:
                    raise RuntimeError(f"invalid_choice_stall: {side} got [Invalid choice] and no new request "
                                       f"within {STALL_S}s")

    ch0 = resource.getrusage(resource.RUSAGE_CHILDREN)
    self0 = resource.getrusage(resource.RUSAGE_SELF)
    t0 = time.perf_counter()
    tasks = [asyncio.create_task(pokemon_battle(clients["p1"], FORMAT, None)),
             asyncio.create_task(pokemon_battle(clients["p2"], FORMAT, None)),
             asyncio.create_task(read_sim())]
    dog = asyncio.create_task(watchdog())
    error, results = None, [None, None]
    try:
        deadline, pending, failed = time.monotonic() + GAME_TIMEOUT_S, set(tasks + [dog]), []
        while not failed and not all(t.done() for t in tasks):   # the watchdog never finishes on its own
            done, pending = await asyncio.wait(pending, timeout=max(0.0, deadline - time.monotonic()),
                                               return_when=asyncio.FIRST_COMPLETED)
            if not done:
                break
            failed = [t for t in done if not t.cancelled() and t.exception() is not None]
        if failed:
            error = repr(failed[0].exception())
        elif not all(t.done() for t in tasks):
            error = f"timeout: game exceeded {GAME_TIMEOUT_S}s"
        else:
            results = [tasks[0].result(), tasks[1].result()]
    finally:
        for t in tasks + [dog]:
            if not t.done():
                t.cancel()
        await asyncio.gather(*tasks, dog, return_exceptions=True)
    wall = time.perf_counter() - t0
    self1 = resource.getrusage(resource.RUSAGE_SELF)
    ch_mid = resource.getrusage(resource.RUSAGE_CHILDREN)
    proc.stdin.close()
    if error:
        proc.kill()
    try:
        await asyncio.wait_for(proc.wait(), timeout=10)
    except asyncio.TimeoutError:
        proc.kill()
        await proc.wait()
    ch1 = resource.getrusage(resource.RUSAGE_CHILDREN)
    stderr = (await proc.stderr.read()).decode()
    raw.close()
    logging.getLogger().removeHandler(fh)
    fh.close()

    log = [ln for ln in spectator if not ln.startswith("|t:|")]
    open(os.path.join(OUT, tag + ".log"), "w").write("\n".join(spectator) + "\n")
    replay = "\n".join(ln for ln in log if ln)
    open(os.path.join(OUT, tag + ".html"), "w").write(
        run_ourbot.REPLAY_HTML.format(tag=tag, log=replay.replace("</", "<\\/")))
    decisions = {r: [json.loads(ln) for ln in open(p)] if os.path.exists(p) else [] for r, p in logs.items()}
    winner_name = None if error else end.get("winner")
    winner_side = next((s for s in names if names[s] == winner_name), None)
    cpu = lambda a, b: (b.ru_utime - a.ru_utime) + (b.ru_stime - a.ru_stime)
    record = dict(
        tag=tag, run_tag=run_tag, seed=seed, game=game, format=FORMAT, k=k,
        battle_seed=battle_seed, team_seeds=dict(p1=seed_a, p2=seed_b), sim_seed=end.get("seed"),
        stream=stream, sides=dict(A=a_side, B=b_side, names=names), methods=dict(fns),
        winner=winner_name, winner_role=role.get(winner_side), winner_method=fns.get(role.get(winner_side)),
        fp_winners=dict(p1=results[0], p2=results[1]),
        turns=end.get("turns"), score=end.get("score"),
        wall_s=round(wall, 2), error=error,
        timing=dict(A=timers[a_side].summary(), B=timers[b_side].summary(),
                    process_cpu_s=round(cpu(self0, self1), 2),
                    stock_mcts_workers_cpu_s=round(cpu(ch0, ch_mid), 2),
                    sim_node_cpu_s=round(cpu(ch_mid, ch1), 2)),
        sim_errors=errors, sim_stderr=stderr[-2000:],
        teams=dict(p1=end.get("p1team"), p2=end.get("p2team")),
        first_requests=first_requests,
        input_log=end.get("inputLog"),
        decisions=decisions,
    )
    json.dump(record, open(os.path.join(OUT, tag + ".json"), "w"), indent=1)
    return record


async def main_async(args):
    state, run_ourbot = setup_foulplay(args.search_time_ms, args.k)
    fns = {"A": FNS[args.a], "B": FNS[args.b]}
    recs = []
    for game, a_side in ((1, "p1"), (2, "p2")):
        rec = await play_game(args.seed, game, a_side, fns, args.tag, args.k, state, run_ourbot, args.stream)
        recs.append(rec)
        print(json.dumps(dict(tag=rec["tag"], A=f"{fns['A']}@{a_side}", B=fns["B"], winner=rec["winner"],
                              winner_role=rec["winner_role"], turns=rec["turns"], wall_s=rec["wall_s"],
                              errors=len(rec["sim_errors"]), error=rec["error"], timing=rec["timing"])), flush=True)
    dump = lambda t: json.dumps(t, sort_keys=True)
    same = dump(recs[0]["teams"]["p1"]) == dump(recs[1]["teams"]["p1"]) and \
        dump(recs[0]["teams"]["p2"]) == dump(recs[1]["teams"]["p2"]) and recs[0]["teams"]["p1"] is not None
    species = {s: [p["species"] for p in recs[0]["teams"][s] or []] for s in ("p1", "p2")}
    print(json.dumps(dict(paired_teams_identical=same, team_p1=species["p1"], team_p2=species["p2"],
                          A_played=dict(g1="p1 (team p1)", g2="p2 (team p2)"))), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--a", "--p1fn", dest="a", choices=sorted(FNS), default="ours",
                    help="player A (p1 in game 1, p2 in game 2); ours = agentform")
    ap.add_argument("--b", "--p2fn", dest="b", choices=sorted(FNS), default="stock",
                    help="player B (p2 in game 1, p1 in game 2)")
    ap.add_argument("--tag", default="headless", help="run tag: outputs are <tag>-s<seed>-g<game>.*")
    ap.add_argument("--k", type=int, default=int(os.environ.get("OURBOT_K", "4")), help="worlds for our solver")
    ap.add_argument("--search-time-ms", type=int, default=100, help="stock foul-play MCTS time per state")
    ap.add_argument("--stream", type=int, default=0, help="carried-weights stream (hedge state file) of this seed")
    asyncio.run(main_async(ap.parse_args()))


if __name__ == "__main__":
    # Linux defaults to fork; forking stock foul-play's MCTS workers from this multithreaded asyncio process can
    # deadlock (seen on Modal). spawn is macOS's default and is what all local runs used.
    import multiprocessing
    multiprocessing.set_start_method("spawn", force=True)
    main()
