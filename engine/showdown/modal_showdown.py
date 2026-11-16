"""Modal app for real-Showdown experiments: one container per carried-weights stream runs headless.py seed after seed.

  cd engine && .venv/bin/modal run showdown/modal_showdown.py::check                                  # image smoke, 1 seed
  .venv/bin/modal run showdown/modal_showdown.py::main --tag E2cal --a hedge --b stock --seeds 1-4 --streams 2 --max-usd 1
  .venv/bin/modal run showdown/modal_showdown.py::main --tag E2 --a hedge --b stock --seeds 1-600 --streams 8 --max-usd 10
  .venv/bin/python showdown/modal_showdown.py 1200 [--stock]        # cost estimate only (no Modal call, no auth)

Layout mirrors modal_psro.py (pricing constants, Meter, attached run: the local entrypoint stays attached while the
containers play). Each stream is one `run_stream` call: it runs `headless.py --seed S --stream k` for its seeds IN
ORDER (so hedge players carry their weights, hedge_state/<tag>-stream<k>-<role>.json, from seed to seed), with
OURBOT_WORK on the Volume `showdown-results` (<tag>/replays_headless, <tag>/hedge_state), committed after every
seed: a rerun with the same tag skips seeds whose two game JSONs exist and resumes the carried weights. It returns the
game JSONs, decision logs and hedge state as bytes, plus CPU and wall seconds; the local entrypoint writes them to
showdown/results/<tag>/ and aggregates them with batch.summarize (-> results/<tag>/<tag>_summary.json).

Budget guard: main refuses to launch if estimate_cost exceeds --max-usd. While running, the Meter charges every
launched container for its requested CPU and memory from launch until it returns (plus SCALEDOWN_S); over --max-usd
it raises the stop flag (containers finish their current seed and return what they have), and over 1.25 x --max-usd
it cancels the remaining calls.

Cost model: 36 CPU-s per game measured locally (agent form vs PIMC, showdown/SETUP.md; sim + both players), a Modal
core assumed SLOWDOWN = 2x slower, utilisation UTIL = 0.6: container-s per game = 36 * 2 / 0.6 = 120. Containers
request 1 CPU / 2 GiB; a stream with a stock foul-play player gets 2 CPU (foul-play spawns MCTS worker processes),
billed at 2 cores for the same wall time (conservative: the game is sequential). Calibrate on a small run first and pass the measured --cpu-s-per-game.
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent                    # engine/showdown
ENGINE_LOCAL = HERE.parent
REMOTE = "/root/engine"
SHOWDOWN, FOULPLAY = "/root/pokemon-showdown", "/root/foul-play"
VOL_DIR = "/vol"
SD_COMMIT = "7e743761cd13cdb361fc0f58eb5a866284894f5b"
FP_COMMIT = "8f541f80f91f960603673796559f796b10ccf0fb"
NODE_VERSION = "22.11.0"                                  # Node 22 LTS, official linux-x64 tarball
MAX_CONTAINERS, SCALEDOWN_S = 64, 60
CPU, STOCK_CPU, MEM_MB = 1.0, 2.0, 2048
USD_CORE_S, USD_GIB_S = 0.0000131, 0.00000222             # modal.com/pricing 2026-11-14 (= modal_psro.py)
CPU_S_PER_GAME, SLOWDOWN, UTIL = 36.0, 2.0, 0.6
STREAM_TIMEOUT_S = 24 * 3600
SOURCES = ["nash.py", "hidden_agents.py", "hidden.py", "pkbattle.py", "randbats.py", "exp3_live.py"]
PE_BUILD = '--config-settings build-args="--no-default-features --features=poke-engine/terastallization"'



def _run_capped(cmd, f, env):
    """Run one seed's headless.py with a hard wall cap (two games + slack); a hung seed is killed, exit code -9."""
    import subprocess
    cap = 2 * int(os.environ.get("GAME_TIMEOUT_S", "600")) + 120
    try:
        return subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, env=env, timeout=cap).returncode
    except subprocess.TimeoutExpired:
        f.write(f"\nKILLED: seed exceeded {cap}s wall cap\n")
        return -9

def side_methods(spec: str) -> list:
    """The methods a --a/--b spec can play: itself, or both halves of "switch:<m1>@<T1>:<m2>" (batch.resolve_method)."""
    if not spec.startswith("switch:"):
        return [spec]
    m1, rest = spec[len("switch:"):].split("@", 1)
    return [m1, rest.split(":", 1)[1]]


def has_stock(a: str, b: str) -> bool:
    """A stock foul-play player on either side at any point of the stream (it needs STOCK_CPU)."""
    return "stock" in side_methods(a) + side_methods(b)


def container_cpu(a: str, b: str) -> float:
    """STOCK_CPU when a stock player (MCTS worker processes) may play; every other method, including the exploit_*,
    style_* and top bots (single-process numpy/LP like the E4 bandits), gets CPU = 1."""
    return STOCK_CPU if has_stock(a, b) else CPU


def rate_usd_s(cpu: float) -> float:
    """Modal price of one container second at `cpu` cores and MEM_MB."""
    return cpu * USD_CORE_S + MEM_MB / 1024 * USD_GIB_S


def estimate_cost(games: int, streams: int = 8, stock: bool = True, cpu_s_per_game: float = CPU_S_PER_GAME,
                  slowdown: float = SLOWDOWN, util: float = UTIL) -> dict:
    """Pure cost estimate (no Modal calls). Container-seconds = games x cpu_s_per_game x slowdown / util (a game is
    one sequential process tree, so a 2-CPU stock container is assumed no faster), billed at the container's requested
    cpu and memory, plus SCALEDOWN_S idle tail per stream."""
    cpu = STOCK_CPU if stock else CPU
    container_s = games * cpu_s_per_game * slowdown / util + streams * SCALEDOWN_S
    wall_h = container_s / max(min(streams, MAX_CONTAINERS), 1) / 3600
    return dict(games=games, streams=streams, cpu=cpu, mem_gib=MEM_MB / 1024, container_h=round(container_s / 3600, 2),
                wall_h_per_stream=round(wall_h, 2), usd=round(container_s * rate_usd_s(cpu), 2),
                assumptions=dict(cpu_s_per_game=cpu_s_per_game, slowdown=slowdown, util=util))


def format_estimate(e: dict) -> str:
    return (f"estimate: {e['games']} games on {e['streams']} streams x {e['cpu']} CPU / {e['mem_gib']:.0f} GiB: "
            f"{e['container_h']} container-h, ~{e['wall_h_per_stream']} h wall, ${e['usd']} "
            f"({e['assumptions']['cpu_s_per_game']} CPU-s/game local x {e['assumptions']['slowdown']} slowdown "
            f"/ {e['assumptions']['util']} utilisation)")


if __name__ == "__main__" and "modal" not in sys.modules:   # `python modal_showdown.py GAMES [--stock]`: no modal import
    _g = int(sys.argv[1]) if len(sys.argv) > 1 else 1200
    print(format_estimate(estimate_cost(_g, stock="--stock" in sys.argv)))
    sys.exit(0)

import modal  # noqa: E402

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("curl", "ca-certificates", "xz-utils", "git", "build-essential", "pkg-config")
    .run_commands(
        f"curl -fsSL https://nodejs.org/dist/v{NODE_VERSION}/node-v{NODE_VERSION}-linux-x64.tar.xz"
        " | tar -xJ -C /usr/local --strip-components=1",
        "curl -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal")
    .env({"PATH": "/root/.cargo/bin:/usr/local/bin:/usr/bin:/bin"})
    .run_commands(                                         # Showdown: exact lockfile, no install scripts (SETUP.md)
        f"git clone https://github.com/smogon/pokemon-showdown.git {SHOWDOWN}",
        f"cd {SHOWDOWN} && git checkout {SD_COMMIT} && npm ci --ignore-scripts"
        " && cp config/config-example.js config/config.js && node build",
        f"git clone https://github.com/pmariglia/foul-play.git {FOULPLAY} && git -C {FOULPLAY} checkout {FP_COMMIT}")
    .add_local_file(HERE / "requirements-bot.txt", "/root/requirements-bot.txt", copy=True)   # needed at build time
    .run_commands("pip install -r /root/requirements-bot.txt maturin",
                  f"pip install --no-binary poke-engine poke-engine==0.0.48 {PE_BUILD}")
    .env({"SHOWDOWN_DIR": SHOWDOWN, "FOULPLAY_DIR": FOULPLAY, "ENGINE_DIR": REMOTE, "BOT_PYTHON": "python",
          "OURBOT_WORK": f"{VOL_DIR}/default", "PYTHONUNBUFFERED": "1",
          "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"})
    .add_local_dir(ENGINE_LOCAL / "data", f"{REMOTE}/data")
    .add_local_dir(HERE, f"{REMOTE}/showdown", ignore=["results", "**/__pycache__", "*.pyc"])
)
for _f in SOURCES:                                         # add_local_* (runtime mounts) must come last
    image = image.add_local_file(ENGINE_LOCAL / _f, f"{REMOTE}/{_f}")

app = modal.App("showdown-experiments", image=image)
vol = modal.Volume.from_name("showdown-results", create_if_missing=True)
stop_flags = modal.Dict.from_name("showdown-stop", create_if_missing=True)   # tag -> True: finish current seed, return


def _run_stream(tag: str, a: str, b: str, seeds: list, stream: int, k: int, search_time_ms: int,
                extra_env: dict | None = None) -> dict:
    """Container body: headless.py for each seed in order with OURBOT_WORK=/vol/<tag>. Returns {files: {relpath: bytes}
    (game JSONs, decision logs, per-seed stdout, this stream's hedge state), seeds_run, exit_codes, cpu_s, wall_s}.
    a/b may be switch specs, resolved per seed by its position in this stream's full list (batch.resolve_method);
    extra_env (e.g. OURBOT_SAFEX_EPS, OURBOT_LOG_ABAR) is added to every headless.py's environment."""
    import os
    import resource
    import subprocess
    sys.path.insert(0, f"{REMOTE}/showdown")
    from batch import resolve_method
    work = Path(VOL_DIR) / tag
    out = work / "replays_headless"
    out.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, OURBOT_WORK=str(work), **(extra_env or {}))
    vol.reload()                                           # earlier runs of this tag (resume)
    t0, ch0 = time.time(), resource.getrusage(resource.RUSAGE_CHILDREN)
    exit_codes, ran, cpu_by_seed = {}, [], {}
    for i, seed in enumerate(seeds):
        if stop_flags.get(tag, False):
            break
        if all((out / f"{tag}-s{seed}-g{g}.json").exists() for g in (1, 2)):
            continue
        cmd = ["python", f"{REMOTE}/showdown/headless.py", "--seed", str(seed), "--tag", tag,
               "--a", resolve_method(a, i), "--b", resolve_method(b, i),
               "--k", str(k), "--search-time-ms", str(search_time_ms), "--stream", str(stream)]
        c0 = resource.getrusage(resource.RUSAGE_CHILDREN)
        with open(out / f"{tag}-s{seed}.batch.log", "w") as f:
            exit_codes[seed] = _run_capped(cmd, f, env)
        c1 = resource.getrusage(resource.RUSAGE_CHILDREN)
        cpu_by_seed[seed] = round(c1.ru_utime - c0.ru_utime + c1.ru_stime - c0.ru_stime, 1)
        ran.append(seed)
        vol.commit()
    ch1 = resource.getrusage(resource.RUSAGE_CHILDREN)
    files = {}
    for seed in seeds:
        for p in out.glob(f"{tag}-s{seed}-g*.json"):
            files[f"replays_headless/{p.name}"] = p.read_bytes()
        for p in out.glob(f"{tag}-s{seed}-g*.decisions.jsonl"):
            files[f"replays_headless/{p.name}"] = p.read_bytes()
        p = out / f"{tag}-s{seed}.batch.log"
        if p.exists():
            files[f"replays_headless/{p.name}"] = p.read_bytes()
    for p in (work / "hedge_state").glob(f"{tag}-stream{stream}-*.json"):
        files[f"hedge_state/{p.name}"] = p.read_bytes()
    return dict(stream=stream, files=files, seeds_run=ran, exit_codes=exit_codes, cpu_by_seed=cpu_by_seed,
                cpu_s=round(ch1.ru_utime - ch0.ru_utime + ch1.ru_stime - ch0.ru_stime, 1),
                wall_s=round(time.time() - t0, 1))


@app.function(cpu=CPU, memory=MEM_MB, timeout=STREAM_TIMEOUT_S, max_containers=MAX_CONTAINERS,
              scaledown_window=SCALEDOWN_S, volumes={VOL_DIR: vol})
def run_stream(tag: str, a: str, b: str, seeds: list, stream: int, k: int = 4, search_time_ms: int = 100,
               extra_env: dict | None = None) -> dict:
    """One carried-weights stream of our methods only (1 CPU)."""
    return _run_stream(tag, a, b, seeds, stream, k, search_time_ms, extra_env)


@app.function(cpu=STOCK_CPU, memory=MEM_MB, timeout=STREAM_TIMEOUT_S, max_containers=MAX_CONTAINERS,
              scaledown_window=SCALEDOWN_S, volumes={VOL_DIR: vol})
def run_stream_stock(tag: str, a: str, b: str, seeds: list, stream: int, k: int = 4, search_time_ms: int = 100,
                     extra_env: dict | None = None) -> dict:
    """Same, for streams with a stock foul-play player (2 CPU: its MCTS worker processes)."""
    return _run_stream(tag, a, b, seeds, stream, k, search_time_ms, extra_env)


class Meter:
    """Conservative running cost of the stream containers: each is billed at its requested CPU and memory from
    launch until its result arrives, plus SCALEDOWN_S idle tail (no driver container: the entrypoint runs locally)."""

    def __init__(self, max_usd: float, cpu: float):
        self.max_usd, self.rate, self.t0 = max_usd, rate_usd_s(cpu), time.time()
        self.running, self.done_usd = 0, 0.0

    def launch(self, n: int):
        self.running += n

    def finish(self, t_end: float):
        self.running -= 1
        self.done_usd += ((t_end - self.t0) + SCALEDOWN_S) * self.rate

    def usd(self) -> float:
        return self.done_usd + self.running * ((time.time() - self.t0) + SCALEDOWN_S) * self.rate


def write_results(tag: str, res: dict) -> Path:
    root = HERE / "results" / tag
    for rel, data in res["files"].items():
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(data)
    return root


@app.local_entrypoint()
def main(tag: str, a: str = "hedge", b: str = "stock", seeds: str = "1-4", streams: int = 2, max_usd: float = 1.0,
         k: int = 4, search_time_ms: int = 100, cpu_s_per_game: float = CPU_S_PER_GAME, slowdown: float = SLOWDOWN,
         util: float = UTIL, poll_s: float = 30.0, safex_eps: str = "", log_abar: bool = False):
    """--safex-eps E sets OURBOT_SAFEX_EPS=E (bayes_safex / hedge_bayes q-hat safex) and --log-abar sets OURBOT_LOG_ABAR=1
    in every container's bot processes (the local shell environment does not reach the containers)."""
    import argparse
    import json
    import os
    sys.path.insert(0, str(HERE))
    import batch
    seed_list = batch.parse_seeds(seeds)
    streams = max(1, min(streams, len(seed_list)))
    extra_env = {**({"OURBOT_SAFEX_EPS": str(float(safex_eps))} if safex_eps else {}),
                 **({"OURBOT_LOG_ABAR": "1"} if log_abar else {})}
    if extra_env:
        print(f"container env: {extra_env}")
    est = estimate_cost(2 * len(seed_list), streams, stock=has_stock(a, b), cpu_s_per_game=cpu_s_per_game,
                        slowdown=slowdown, util=util)
    print(format_estimate(est))
    if est["usd"] > max_usd:
        raise SystemExit(f"refusing: estimated ${est['usd']:.2f} > --max-usd {max_usd}")
    if streams > MAX_CONTAINERS:
        raise SystemExit(f"--streams {streams} > MAX_CONTAINERS {MAX_CONTAINERS}: streams would queue")
    fn = run_stream_stock if has_stock(a, b) else run_stream
    stop_flags[tag] = False
    meter = Meter(max_usd, container_cpu(a, b))
    calls = {kk: fn.spawn(tag, a, b, ss, kk, k, search_time_ms, extra_env or None)
             for kk, ss in enumerate(batch.split_streams(seed_list, streams))}
    meter.launch(len(calls))
    results, failed, stopped = {}, {}, False
    while calls:
        for kk, call in list(calls.items()):
            try:
                res = call.get(timeout=poll_s / max(len(calls), 1))
            except modal.exception.FunctionTimeoutError as e:   # subclasses of TimeoutError: the call itself failed
                res = e
            except modal.exception.OutputExpiredError as e:
                res = e
            except (modal.exception.TimeoutError, TimeoutError):   # not finished yet (1.6.4 raises the builtin)
                continue
            except Exception as e:                             # remote exception
                res = e
            meter.finish(time.time())
            del calls[kk]
            if isinstance(res, Exception):
                failed[kk] = repr(res)
                print(f"stream {kk}: FAILED {res!r}", flush=True)
                continue
            results[kk] = res
            write_results(tag, res)
            print(f"stream {kk}: {len(res['seeds_run'])} seeds run, nonzero exits "
                  f"{ {s: rc for s, rc in res['exit_codes'].items() if rc} }, cpu {res['cpu_s']}s, "
                  f"wall {res['wall_s']}s; ~${meter.usd():.2f} so far", flush=True)
        if not stopped and meter.usd() > max_usd:
            stopped = True
            stop_flags[tag] = True
            print(f"over budget (~${meter.usd():.2f} > {max_usd}): stop flag set, streams finish their current seed",
                  flush=True)
        if stopped and meter.usd() > 1.25 * max_usd:
            for kk, call in calls.items():
                call.cancel()
                failed[kk] = "cancelled: over 1.25 x --max-usd"
                meter.finish(time.time())
            calls = {}
    root = HERE / "results" / tag
    root.mkdir(parents=True, exist_ok=True)
    batch.OUT = str(root / "replays_headless")
    os.makedirs(batch.OUT, exist_ok=True)
    exit_codes = {int(s): rc for r in results.values() for s, rc in r["exit_codes"].items()}
    ns = argparse.Namespace(tag=tag, a=a, b=b, k=k, streams=streams)
    summary = batch.summarize(ns, seed_list, exit_codes)
    cpu_s = sum(r["cpu_s"] for r in results.values())
    games_run = 2 * sum(len(r["seeds_run"]) for r in results.values())
    summary["modal"] = dict(usd_estimate=round(meter.usd(), 3), stopped_over_budget=stopped, failed_streams=failed,
                            container_cpu=container_cpu(a, b), stream_cpu_s=cpu_s, games_run=games_run,
                            cpu_s_per_game_measured=round(cpu_s / games_run, 1) if games_run else None,
                            stream_wall_s={kk: r["wall_s"] for kk, r in results.items()},
                            entrypoint_wall_s=round(time.time() - meter.t0, 1))
    if extra_env:
        summary["modal"]["env"] = extra_env
    path = root / f"{tag}_summary.json"
    path.write_text(json.dumps(summary, indent=1))
    ps = summary["pair_score_A"]
    print(f"{tag}: A={a} vs B={b}: pair score of A {ps['mean']} (95% CI {ps['ci95']}) over {summary['n_pairs']} pairs; "
          f"{summary['error_games']} error games; {summary['modal']} -> {path}", flush=True)
    for m, r in summary["learning"].items():
        print(f"  {m}: " + ", ".join(f"{kk}={v}" for kk, v in r.items()), flush=True)


@app.local_entrypoint()
def check():
    """Image smoke: one paired seed of agentform vs stock in a container (a few cents); prints the stream result."""
    res = run_stream_stock.remote("modalcheck", "agentform", "stock", [1], 0)
    print({kk: v for kk, v in res.items() if kk != "files"}, sorted(res["files"]))
