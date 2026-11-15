"""Modal app for psro.py: CPU workers play games through `.map`, a driver container runs PPO updates.

  uv pip install --python .venv/bin/python modal       # once (uv-managed venv has no pip; ~/.modal.toml exists)
  .venv/bin/modal run modal_psro.py::main --mode psro --config smoke --max-usd 1     # calibrate first
  .venv/bin/modal run modal_psro.py::main --mode psro --config first --max-usd 10
  .venv/bin/modal run modal_psro.py::main --mode rl_br --target hedge_experts --config first --max-usd 3
  .venv/bin/modal volume get psro-results <run> results/psro/   # fetch iter_*.json, rl_br_*.json, checkpoints

Budget guard: the local entrypoint refuses to launch if psro_cost.estimate (the same function behind
`psro.py --dry-run`) exceeds --max-usd. While running, the driver charges every `.map` call as
min(inputs, MAX_CONTAINERS) x (elapsed + SCALEDOWN_S) x worker price (containers are billed for
requested CPU while idle waiting for stragglers), plus its own lifetime; once over --max-usd the next
map raises psro.BudgetExceeded and psro.run saves stopped.json and the current checkpoint.

poke_engine: PyPI has only the 0.0.48 sdist (no wheels). The local venv built it from that source with
uv + maturin, config_settings build-args="--no-default-features --features=poke-engine/terastallization"
(.venv/.../poke_engine-0.0.48.dist-info/uv_build.json), so the image builds it the same way with Rust.
Written without the modal SDK installed: API names follow Modal's docs (App, Image.add_local_*,
max_containers, scaledown_window, Volume.from_name); check them on the first `modal run`.
"""
from __future__ import annotations

import time
from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent
REMOTE = "/root/engine"
MAX_CONTAINERS, SCALEDOWN_S = 64, 60   # warm containers across PPO updates (no cold start per map)
WORKER_CPU, WORKER_MEM_MB, DRIVER_CPU, DRIVER_MEM_MB = 1.0, 1024, 2.0, 4096   # = psro_cost constants
USD_CORE_S, USD_GIB_S = 0.0000131, 0.00000222                                  # modal.com/pricing 2026-11-14
SOURCES = ["psro.py", "psro_cost.py", "ppo_pilot.py", "pkbattle.py", "nash.py", "experiments.py",
           "exp3_live.py", "hidden.py", "randbats.py"]
PE_BUILD = '--config-settings build-args="--no-default-features --features=poke-engine/terastallization"'

image = (
    modal.Image.debian_slim(python_version="3.12")                        # venv: Python 3.12.12
    .apt_install("curl", "build-essential", "pkg-config")
    .run_commands("curl -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal")
    .env({"PATH": "/root/.cargo/bin:/usr/local/bin:/usr/bin:/bin", "PSRO_ROOT": REMOTE, "PYTHONPATH": REMOTE})
    .pip_install("numpy==2.5.4", "scipy==1.18.1", "maturin")              # venv versions
    .pip_install("torch==2.15.0", index_url="https://download.pytorch.org/whl/cpu")
    .run_commands(f"pip install --no-binary poke-engine poke-engine==0.0.48 {PE_BUILD}")
    .add_local_dir(HERE / "data", f"{REMOTE}/data")
    .add_local_file(HERE / "results" / "ppo_pilot.pt", f"{REMOTE}/results/ppo_pilot.pt")
)
for _f in SOURCES:                                                         # add_local_* must come last
    image = image.add_local_file(HERE / _f, f"{REMOTE}/{_f}")

app = modal.App("psro-pokeengine", image=image)
vol = modal.Volume.from_name("psro-results", create_if_missing=True)
VOLS = {f"{REMOTE}/results/psro": vol}                                    # psro writes results/psro/<run>/


@app.function(cpu=WORKER_CPU, memory=WORKER_MEM_MB, timeout=1800, max_containers=MAX_CONTAINERS,
              scaledown_window=SCALEDOWN_S, volumes=VOLS)
def run_task(fname: str, arg):
    """Remote map target: psro.<fname>(arg) (eval_task / rollout_task) -> (result, seconds)."""
    import psro
    if fname == "eval_task":
        vol.reload()                                                       # net members' .npz from the driver
    t = time.time()
    return getattr(psro, fname)(arg), time.time() - t


class Meter:
    """Conservative running cost: driver lifetime + worker container-seconds. Each map bills
    min(inputs, MAX_CONTAINERS) containers for its wall time; between maps those containers stay warm and are
    billed for min(gap, SCALEDOWN_S) (and that tail again after the last map)."""

    def __init__(self, max_usd: float):
        self.max_usd, self.t0, self.worker_usd, self.task_s = max_usd, time.time(), 0.0, 0.0
        self.last_end, self.last_n = None, 0

    @staticmethod
    def _rate() -> float:
        return WORKER_CPU * USD_CORE_S + WORKER_MEM_MB / 1024 * USD_GIB_S

    def _tail(self, now: float) -> float:
        return 0.0 if self.last_end is None else self.last_n * min(now - self.last_end, SCALEDOWN_S) * self._rate()

    def usd(self) -> float:
        driver = (time.time() - self.t0) * (DRIVER_CPU * USD_CORE_S + DRIVER_MEM_MB / 1024 * USD_GIB_S)
        return driver + self.worker_usd + self._tail(time.time())

    def charge(self, n_inputs: int, t_start: float, t_end: float, task_s: float):
        self.worker_usd += self._tail(t_start)                       # warm idle gap since the previous map
        n = min(n_inputs, MAX_CONTAINERS)
        self.worker_usd += n * (t_end - t_start) * self._rate()
        self.last_end, self.last_n, self.task_s = t_end, n, self.task_s + task_s

    def check(self):
        import psro
        if self.usd() > self.max_usd:
            raise psro.BudgetExceeded(f"spent ~${self.usd():.2f} > --max-usd {self.max_usd}")


@app.function(cpu=DRIVER_CPU, memory=DRIVER_MEM_MB, timeout=24 * 3600, volumes=VOLS)
def driver(cfg: dict, mode: str, max_usd: float) -> dict:
    """Runs psro.run with map_fn = run_task.map; PPO updates happen here (CPU torch, 256-256 MLP)."""
    import psro
    meter = Meter(max_usd)

    def map_fn(f, tasks):
        tasks = list(tasks)
        meter.check()
        if f.__name__ == "eval_task":
            vol.commit()                                                   # make new checkpoints visible
        t = time.time()
        res = list(run_task.map([f.__name__] * len(tasks), tasks))
        meter.charge(len(tasks), t, time.time(), sum(s for _, s in res))
        meter.check()
        return [r for r, _ in res]

    try:
        psro.run(cfg, mode, map_fn, n_chunks=MAX_CONTAINERS)
    finally:
        vol.commit()
    summary = dict(usd_estimate=round(meter.usd(), 3), worker_task_core_h=round(meter.task_s / 3600, 3),
                   wall_h=round((time.time() - meter.t0) / 3600, 3))
    psro._write(Path(psro.ROOT) / "results/psro" / cfg["run"] / "modal_cost.json", summary)
    vol.commit()
    return summary


@app.local_entrypoint()
def main(mode: str = "psro", config: str = "first", target: str = "", max_usd: float = 10.0,
         slowdown: float = 2.0):
    import psro
    import psro_cost
    cfg = psro.load_config(config)
    if target:
        cfg["rl_br_targets"] = [target]
    est = psro_cost.estimate(cfg, mode, parallelism=MAX_CONTAINERS, slowdown=slowdown)
    print(psro_cost.format_estimate(est))
    if est["usd"] > max_usd:
        raise SystemExit(f"refusing: estimated ${est['usd']:.2f} > --max-usd {max_usd}")
    print(driver.remote(cfg, mode, max_usd))


@app.function(cpu=WORKER_CPU, memory=WORKER_MEM_MB, timeout=600)
def one_game(spec: str):
    """Image smoke check: one game of `spec` vs the PPO pilot net -> (spec, (winner, turns), seconds)."""
    import psro
    t = time.time()
    net = {"kind": "net", "path": "results/ppo_pilot.pt"}
    r = psro.play_game(psro.make_agent(psro.member_from_spec(spec), 1), psro.make_agent(net, 2), seed=7)
    return spec, r, round(time.time() - t, 2)


@app.local_entrypoint()
def check():
    """.venv/bin/modal run modal_psro.py::check  (a few cents; no PSRO, no budget meter needed)."""
    for out in one_game.map(["safest:0", "nash:rm+:0:1000:argmax", "hedge_experts", "mcts_it:2000"],
                            return_exceptions=True):
        print(out)
