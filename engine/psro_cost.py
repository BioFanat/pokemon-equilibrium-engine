"""Cost model for psro.py runs on Modal (CPU only). Plays no games: pure arithmetic on measured rates.

Cost = Σ_phases games × seconds/game (local, measured) × slowdown (Modal core vs M4 Max) / utilization
       × (price per physical core-hour + memory GiB × price per GiB-hour), plus the driver container.

Sources (all local measurements are on the Apple M4 Max laptop, one Python process per core):
  * PPO rollout rate: results/ppo_pilot.jsonl, field `dec_per_s` (median 29,644 over 1,775 iterations,
    12 workers -> 2,470 decisions/s/worker, self-play, both sides decide); `decisions`/`games` at the
    last iteration = 29,851,761 / 340,800 = 87.6 decisions per game (both sides) -> 43.8 per side;
    `upd_s` summed = 244.5 s for 29.85M decisions -> 8.2 µs per decision for the PPO update.
  * Matrix bots: results/experiments/E1fix.jsonl, field `ms_per_move_A` (mean over games, includes forced
    moves): nash:rm+:0:1000 74.3 ms, nash:rm+:0:1000:argmax 72.4 ms, safest:0 67.3 ms; `ms_per_move_B`
    for mcts_it:2000 = 2.7 ms; `turns` mean 52-70 (randbats teams, cap 300). Check: nash vs safest mean
    `wall_s` 8.9 s ~= 60 turns x (74 + 69) ms.
  * hedge_experts: no timing logged (results/exp3/*.jsonl); INFERRED equal to nash:rm+:0:1000 (the same
    calls: one nash.full_matrix + rm_plus(1000) per turn).
  * Modal pricing, fetched from https://modal.com/pricing on 2026-11-14: CPU $0.0000131 / physical core / s
    (= $0.0472 / core-h; 1 physical core = 2 vCPU; min 0.125 core per container), memory
    $0.00000222 / GiB / s (= $0.0080 / GiB-h). Starter plan includes $30/month of compute.
ASSUMPTIONS (labelled; recalibrate after the smoke run): Modal core = SLOWDOWN x slower than an M4 Max
core (default 2.0); 60 turns per game on the hand pool with the 200-turn cap; container UTILIZATION 0.6
(containers are billed while idle waiting for straggler games); during BR training all meta-Nash weight
sits on the slowest member (worst case: PSRO mixtures that favour cheap net members cost less).

  .venv/bin/python psro_cost.py            # table for the smoke / first / full presets
"""
from __future__ import annotations

import math

USD_CORE_S, USD_GIB_S = 0.0000131, 0.00000222          # modal.com/pricing, 2026-11-14
WORKER_CPU, WORKER_GIB = 1.0, 1.0                      # per rollout/eval container (modal_psro.py)
DRIVER_CPU, DRIVER_GIB = 2.0, 4.0                      # driver = PPO learner container
SLOWDOWN, UTILIZATION, PARALLELISM = 2.0, 0.6, 64       # assumptions (see docstring)

TURNS = 60                                             # assumption, E1fix mean turns 52-70
BASE_MS_PER_TURN = 1000 / 2470                         # ms per PPO decision incl. engine step (ppo_pilot.jsonl),
                                                       # charged per turn: an upper-ish proxy, tiny vs 74 ms
BR_DEC_PER_GAME = 87.6 / 2                             # ppo_pilot.jsonl decisions / games / 2 sides
UPDATE_S_PER_DEC = 244.5 / 29_851_761                  # ppo_pilot.jsonl sum(upd_s) / decisions
INPUT_OVERHEAD_S = 0.25                                # assumption: per map input (dispatch, pickled W ~0.6 MB,
                                                       # result), local-equivalent s (x slowdown like games)


def move_ms(spec: str) -> float:
    """Local ms per decision call of a member (bot spec, 'hedge_experts', or 'net:...')."""
    kind, *a = spec.split(":")
    if kind == "nash":
        return 72.4 if "argmax" in a else 74.3
    if kind == "safest":
        return 67.3
    if kind == "hedge_experts":
        return 74.3                                    # inferred, see docstring
    if kind in ("mcts_it", "foulplay_it"):
        return 2.8 * int(a[0]) / 2000                  # E7.jsonl: foulplay_it:2000 2.8-2.9 ms/move
    if kind == "foulplay_ms":
        return float(a[0])
    if kind == "mcts_ms":
        return float(a[0])
    if kind in ("net", "random"):
        return 0.0                                     # included in BASE_MS_PER_TURN
    raise ValueError(spec)


def game_s(a: str, b: str) -> float:
    """Local core-seconds for one game between members a and b."""
    return TURNS * (BASE_MS_PER_TURN + move_ms(a) + move_ms(b)) / 1000


def plan(cfg: dict, mode: str = "psro") -> list[dict]:
    """Phases of a psro.run(cfg, mode) call: games, trained decisions, local core-seconds each.
    Mirrors psro.psro_loop / psro.rl_br; the same function backs --dry-run and the Modal guard."""
    bots, gpc, phases = list(cfg["bots"]), cfg["games_per_cell"], []
    if mode == "psro":
        cells = [(a, b) for i, a in enumerate(bots) for b in bots[i + 1:]]
        phases.append(dict(phase="initial meta matrix", games=gpc * len(cells), decisions=0,
                           core_s=gpc * sum(game_s(a, b) for a, b in cells)))
        pop, worst = list(bots), max(bots, key=move_ms)
        for k in range(1, cfg["iters"] + 1):
            g = cfg["br_decisions"] / BR_DEC_PER_GAME
            phases.append(dict(phase=f"iter {k}: train BR", games=round(g), decisions=cfg["br_decisions"],
                               core_s=g * game_s("net", worst) + cfg["br_decisions"] * UPDATE_S_PER_DEC))
            phases.append(dict(phase=f"iter {k}: extend meta ({len(pop)} cells)", games=gpc * len(pop),
                               decisions=0, core_s=gpc * sum(game_s("net", m) for m in pop)))
            pop.append("net")
    for t in cfg.get("rl_br_targets", []):
        g = cfg["rl_br_decisions"] / BR_DEC_PER_GAME
        gpi = max(1, cfg.get("games_per_input", 1))
        phases.append(dict(phase=f"rl_br {t}: train", games=round(g), decisions=cfg["rl_br_decisions"],
                           core_s=g * game_s("net", t) + cfg["rl_br_decisions"] * UPDATE_S_PER_DEC
                           + math.ceil(g / gpi) * INPUT_OVERHEAD_S))
        n_eval = 1 + n_checkpoints(cfg) + bool(cfg.get("init") and (cfg.get("eval_init")
                                                                     or cfg.get("eval_every_decisions")))
        phases.append(dict(phase=f"rl_br {t}: eval x{n_eval}", games=n_eval * cfg["exploit_games"], decisions=0,
                           core_s=n_eval * cfg["exploit_games"] * (game_s("net", t) + INPUT_OVERHEAD_S)))
    return phases


def n_checkpoints(cfg: dict) -> int:
    """Mid-training evals from cfg['eval_every_decisions'] (int or list of decision thresholds); the init
    eval (curve point 0, shared with eval_init) and the final eval are counted separately."""
    every, total = cfg.get("eval_every_decisions"), cfg["rl_br_decisions"]
    if not every:
        return 0
    return len([m for m in every if m < total] if isinstance(every, (list, tuple)) else range(every, total, every))


def estimate(cfg: dict, mode: str = "psro", parallelism: int = PARALLELISM, slowdown: float = SLOWDOWN,
             utilization: float = UTILIZATION) -> dict:
    """Totals on Modal: billed worker core-hours, wall time, and USD (workers + driver)."""
    ph = plan(cfg, mode)
    local_s = sum(p["core_s"] for p in ph)
    billed_s = local_s * slowdown / utilization        # worker container-seconds (1 core each)
    wall_s = billed_s / parallelism
    usd_w = billed_s * WORKER_CPU * USD_CORE_S + billed_s * WORKER_GIB * USD_GIB_S
    usd_d = wall_s * (DRIVER_CPU * USD_CORE_S + DRIVER_GIB * USD_GIB_S)
    return dict(phases=ph, games=sum(p["games"] for p in ph), decisions=sum(p["decisions"] for p in ph),
                local_core_h=local_s / 3600, modal_core_h=billed_s / 3600, wall_h=wall_s / 3600,
                usd=usd_w + usd_d, usd_workers=usd_w, usd_driver=usd_d, parallelism=parallelism,
                slowdown=slowdown, utilization=utilization)


def format_estimate(e: dict, local_workers: int | None = None) -> str:
    rows = [f"{'phase':44s} {'games':>8s} {'decisions':>10s} {'local core-h':>12s}"]
    for p in e["phases"]:
        rows.append(f"{p['phase']:44s} {p['games']:8d} {p['decisions']:10d} {p['core_s'] / 3600:12.2f}")
    rows.append(f"{'TOTAL':44s} {e['games']:8d} {e['decisions']:10d} {e['local_core_h']:12.2f}")
    rows.append(f"Modal: {e['modal_core_h']:.1f} billed core-h (slowdown {e['slowdown']}, utilization "
                f"{e['utilization']}), ${e['usd']:.2f} (workers ${e['usd_workers']:.2f} + driver "
                f"${e['usd_driver']:.2f}), wall {e['wall_h']:.1f} h at {e['parallelism']} containers")
    if local_workers:
        rows.append(f"Local: wall {e['local_core_h'] / local_workers:.1f} h at {local_workers} workers (M4 Max)")
    return "\n".join(rows)


if __name__ == "__main__":
    import psro
    for name in ("smoke", "first", "full"):
        print(f"=== {name}: {psro.PRESETS[name]}")
        print(format_estimate(estimate(psro.load_config(name))), "\n")
