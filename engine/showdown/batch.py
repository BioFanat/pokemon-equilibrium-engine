"""Parallel batch of headless paired games (headless.py), then aggregate into <tag>_summary.json.

  ~/sandbox/bot-venv/bin/python ~/sandbox/ourbot/batch.py --tag E1 --a agentform --b pimc --seeds 1-300 --workers 6

Each seed is an independent `headless.py --seed S` subprocess (two games, seats swapped, same teams), at most
--workers at a time; seeds whose two JSONs already exist are skipped (resume). Per-seed stdout/stderr goes to
replays_headless/<tag>-s<seed>.batch.log. The summary is aggregated from the game JSONs only:
  pair score of A = mean of A's two game scores (win 1, draw 0.5, loss 0), over seeds whose two games both finished
  without error; mean and 95% CI (t interval) over pairs; error games (error set or any sim |error|); mean turns;
  wall and CPU seconds per decision per method; CPU seconds per game (all processes, and our side per method);
  hedge feedback turns (updated / skipped, by reason) and bandit / RM+ Nash gaps (`learning`).
--streams S: seed list split into S streams (stream k = seeds[k::S]); each stream runs its seeds one after another in
one worker, passing --stream k, so hedge players carry their weights (<OURBOT_WORK>/hedge_state/) from seed to seed.
--a/--b may be a switch spec "switch:<m1>@<T1>:<m2>": the i-th seed of a stream (0-based, position in the stream's full
seed list, so resume does not shift it) plays m1 if i < T1 else m2 (without --streams: i = position in --seeds). The
carried state files are keyed by role, not method, so the other side's weights carry across the switch; the summary
then records the method each side actually played per seed (`active`).
"""
import argparse
import json
import math
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(os.environ.get("OURBOT_WORK", HERE), "replays_headless")
PY = os.environ.get("BOT_PYTHON", os.path.join(os.path.expanduser("~"), "sandbox", "bot-venv", "bin", "python"))



def _run_capped(cmd, f, env):
    """Run one seed's headless.py with a hard wall cap (two games + slack); a hung seed is killed, exit code -9."""
    import subprocess
    cap = 2 * int(os.environ.get("GAME_TIMEOUT_S", "600")) + 120
    try:
        return subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, env=env, timeout=cap).returncode
    except subprocess.TimeoutExpired:
        f.write(f"\nKILLED: seed exceeded {cap}s wall cap\n")
        return -9

def parse_seeds(spec):
    out = []
    for part in spec.split(","):
        lo, _, hi = part.partition("-")
        out.extend(range(int(lo), int(hi or lo) + 1))
    return out


def resolve_method(spec, i):
    """The method a side plays at position i of its stream: spec itself, or for "switch:<m1>@<T1>:<m2>" m1 if i < T1
    else m2."""
    if not spec.startswith("switch:"):
        return spec
    m1, rest = spec[len("switch:"):].split("@", 1)
    t1, m2 = rest.split(":", 1)
    return m1 if i < int(t1) else m2


def spec_methods(spec):
    """All methods a side spec can play (one, or both halves of a switch spec)."""
    if not spec.startswith("switch:"):
        return [spec]
    return [resolve_method(spec, 0), resolve_method(spec, 1 << 30)]


def game_path(tag, seed, game):
    return os.path.join(OUT, f"{tag}-s{seed}-g{game}.json")


def run_seed(args, seed, stream=None, i=0):
    a, b = resolve_method(args.a, i), resolve_method(args.b, i)
    cmd = [PY, os.path.join(HERE, "headless.py"), "--seed", str(seed), "--tag", args.tag, "--a", a, "--b", b,
           "--k", str(args.k), "--search-time-ms", str(args.search_time_ms)]
    if stream is not None:
        cmd += ["--stream", str(stream)]
    env = dict(os.environ)
    for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        env.setdefault(v, "1")                               # one core per worker: no BLAS oversubscription
    t0 = time.time()
    with open(os.path.join(OUT, f"{args.tag}-s{seed}.batch.log"), "w") as f:
        rc = _run_capped(cmd, f, env)
    print(f"seed {seed}{'' if stream is None else f' (stream {stream})'}: exit {rc} in {time.time() - t0:.0f}s",
          flush=True)
    return seed, rc


def split_streams(seeds, n):
    """Stream k gets seeds[k::n]; each stream runs its seeds in order, carrying hedge weights from seed to seed."""
    return [seeds[k::n] for k in range(n)]


def run_stream(args, stream, seeds, todo=None):
    """Run the stream's seeds in order, skipping those not in `todo`; position i in the full list picks a switch half."""
    return [run_seed(args, s, stream, i) for i, s in enumerate(seeds) if todo is None or s in todo]


def learning_stats(per_learn, decisions, method):
    """Accumulate one player's decision log: hedge feedback (updated / skipped by reason) and bandit/RM+ Nash gaps."""
    d = per_learn.setdefault(method, dict(updates=0, skipped=0, reasons={}, gaps=[]))
    for rec in decisions:
        fb = rec.get("feedback")
        if fb:
            if fb["skipped"]:
                d["skipped"] += 1
                d["reasons"][fb["reason"]] = d["reasons"].get(fb["reason"], 0) + 1
            else:
                d["updates"] += 1
        if "gap" in rec:
            d["gaps"].append(rec["gap"])


def learning_summary(per_learn):
    out = {}
    for m, d in per_learn.items():
        r = {}
        n_fb = d["updates"] + d["skipped"]
        if n_fb:
            r.update(feedback_turns=n_fb, updates=d["updates"], skipped=d["skipped"],
                     skipped_frac=round(d["skipped"] / n_fb, 4), skipped_reasons=d["reasons"])
        if d["gaps"]:
            g = sorted(d["gaps"])
            r.update(gap_n=len(g), gap_mean=round(sum(g) / len(g), 5), gap_median=g[len(g) // 2], gap_max=g[-1],
                     gap_frac_gt_005=round(sum(x > 0.05 for x in g) / len(g), 4))
        if r:
            out[m] = r
    return out


def game_score(rec):
    """A's score in one finished game: win 1, draw 0.5, loss 0."""
    return {"A": 1.0, "B": 0.0}.get(rec["winner_role"], 0.5)


def t95(n):
    try:
        from scipy.stats import t
        return float(t.ppf(0.975, n - 1))
    except ImportError:
        return 1.96


def summarize(args, seeds, exit_codes):
    pairs, errors, missing, turns, cpu_game, per_method, per_learn, active = {}, [], [], [], [], {}, {}, {}
    for seed in seeds:
        recs = []
        for game in (1, 2):
            p = game_path(args.tag, seed, game)
            if not os.path.exists(p):
                missing.append(f"s{seed}-g{game}")
                continue
            rec = json.load(open(p))
            recs.append(rec)
            if rec.get("error") or rec["sim_errors"]:
                errors.append(dict(game=rec["tag"], error=rec.get("error"), sim_errors=rec["sim_errors"][:3]))
                continue
            turns.append(rec["turns"])
            t = rec["timing"]
            cpu_game.append(t["process_cpu_s"] + t["sim_node_cpu_s"] + t["stock_mcts_workers_cpu_s"])
            for r in ("A", "B"):
                m = per_method.setdefault(rec["methods"][r], dict(decisions=0, wall_s=0.0, cpu_s=0.0, games=0))
                for key in ("decisions", "wall_s", "cpu_s"):
                    m[key] += t[r][key]
                m["games"] += 1
                learning_stats(per_learn, rec.get("decisions", {}).get(r, []), rec["methods"][r])
        if recs:
            active[str(seed)] = dict(recs[0]["methods"])
        ok = [r for r in recs if not r.get("error")]   # recoverable sim errors (re-requested) still count; reported separately
        if len(ok) == 2:
            pairs[seed] = sum(game_score(r) for r in ok) / 2
    vals = list(pairs.values())
    n = len(vals)
    mean = sum(vals) / n if n else None
    sd = math.sqrt(sum((v - mean) ** 2 for v in vals) / (n - 1)) if n > 1 else None
    half = t95(n) * sd / math.sqrt(n) if n > 1 else None
    method_rates = {m: dict(decisions=d["decisions"], wall_s_per_decision=round(d["wall_s"] / max(d["decisions"], 1), 3),
                            cpu_s_per_decision=round(d["cpu_s"] / max(d["decisions"], 1), 3),
                            cpu_s_per_game=round(d["cpu_s"] / max(d["games"], 1), 2))
                    for m, d in per_method.items()}
    out = dict(
        tag=args.tag, A=args.a, B=args.b, k=args.k, seeds=f"{seeds[0]}-{seeds[-1]}" if seeds else "",
        n_seeds=len(seeds), n_pairs=n,
        pair_score_A=dict(mean=round(mean, 4) if n else None, sd=round(sd, 4) if sd is not None else None,
                          ci95=[round(mean - half, 4), round(mean + half, 4)] if half is not None else None,
                          per_seed={str(s): v for s, v in pairs.items()}),
        error_games=len(errors), errors=errors, missing_games=missing,
        nonzero_exit={str(s): rc for s, rc in exit_codes.items() if rc},
        mean_turns=round(sum(turns) / len(turns), 2) if turns else None,
        per_method=method_rates,
        cpu_s_per_game=round(sum(cpu_game) / len(cpu_game), 2) if cpu_game else None,
        streams=args.streams, learning=learning_summary(per_learn),
    )
    if args.a.startswith("switch:") or args.b.startswith("switch:"):   # per seed: the method each role played
        out["active"] = active
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--a", default="agentform")
    ap.add_argument("--b", default="pimc")
    ap.add_argument("--seeds", required=True, help="e.g. 1-300 or 1-4,10")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--search-time-ms", type=int, default=100)
    ap.add_argument("--streams", type=int, default=0,
                    help="split seeds into S carried-weight streams (sequential within, parallel across); 0 = per seed")
    args = ap.parse_args()
    if args.streams == 0 and set(spec_methods(args.a) + spec_methods(args.b)) & {"hedge", "hedge_fs", "hedge_mcts"} and args.workers > 1:
        sys.exit("hedge players carry weights in one file per stream: use --streams S (or --workers 1)")
    os.makedirs(OUT, exist_ok=True)
    seeds = parse_seeds(args.seeds)
    todo = [s for s in seeds if not all(os.path.exists(game_path(args.tag, s, g)) for g in (1, 2))]
    print(f"{args.tag}: {len(seeds)} seeds, {len(seeds) - len(todo)} done, running {len(todo)} "
          f"with {args.workers} workers", flush=True)
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        if args.streams:
            streams = list(enumerate(split_streams(seeds, args.streams)))
            exit_codes = {s: rc for res in pool.map(lambda kv: run_stream(args, *kv, set(todo)), streams) for s, rc in res}
        else:
            exit_codes = dict(pool.map(lambda s: run_seed(args, s, i=seeds.index(s)), todo))
    summary = summarize(args, seeds, exit_codes)
    summary["batch_wall_s"] = round(time.time() - t0, 1)
    summary["workers"] = args.workers
    path = os.path.join(OUT, f"{args.tag}_summary.json")
    json.dump(summary, open(path, "w"), indent=1)
    ps = summary["pair_score_A"]
    rates = ", ".join(f"{m} {r['cpu_s_per_decision']} cpu-s/dec" for m, r in summary["per_method"].items())
    print(f"{args.tag}: A={args.a} vs B={args.b}: pair score of A {ps['mean']} (95% CI {ps['ci95']}) over "
          f"{summary['n_pairs']} pairs; {summary['error_games']} error games; mean turns {summary['mean_turns']}; "
          f"{rates}; {summary['cpu_s_per_game']} cpu-s/game; wall {summary['batch_wall_s']}s -> {path}", flush=True)
    for m, r in summary["learning"].items():
        print(f"  {m}: " + ", ".join(f"{k}={v}" for k, v in r.items()), flush=True)
    sys.exit(1 if summary["error_games"] or summary["missing_games"] else 0)


if __name__ == "__main__":
    main()
