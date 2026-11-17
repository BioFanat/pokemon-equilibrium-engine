"""Fresh, resumable public evaluation with exact decision logs and a post-run audit.

Run in an interactive Terminal; the account password is never written to disk.
The default is 100 hedge_bayes_mw games, up to five at once, on the existing account.
"""
import argparse
import contextlib
import fcntl
import getpass
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from ladder_metrics import summarize_games

REPO = Path(__file__).resolve().parents[2]
ACCOUNT = "bot_bonp"
MCTS_METHODS = ("mcts_native", "mcts_native_sampled", "hedge_mcts", "smmcts_rm")   # pin mcts_ms and sampler


def records(out):
    path = out / "ladder_games.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()] if path.exists() else []
    if any(row.get("result") not in {"win", "loss", "tie", "draw"} for row in rows):
        raise ValueError("Ledger contains a game without a completed result")
    return {row["battle_tag"]: row for row in rows}


def plan(args):
    if not 1 <= args.games <= 100:
        raise ValueError("Choose 1 to 100 games")
    out = args.out.resolve() if args.out else None
    concurrency = getattr(args, "concurrency", None)
    if concurrency is not None and not 1 <= concurrency <= 5:
        raise ValueError("Choose 1 to 5 concurrent games")
    saved = None
    if out and (out / "evaluation.json").exists():
        saved = json.loads((out / "evaluation.json").read_text())
    concurrency = concurrency if concurrency is not None else saved["concurrency"] if saved else 5
    config = dict(account=ACCOUNT, method=args.method, target=args.games, k=4,
                  concurrency=concurrency,
                  format="gen9randombattle", server="wss://sim3.psim.us/showdown/websocket")
    if args.method in MCTS_METHODS:
        budget = getattr(args, "mcts_ms", 1000)
        if not 4 <= budget <= 10000:
            raise ValueError("MCTS budget must be between 4 and 10000 milliseconds")
        config["mcts_ms"] = budget
        config['sampler'] = getattr(args,'sampler','baseline')
        if config['sampler'] not in {'baseline','item_stratified'}:
            raise ValueError('Unknown sampler')
    elif getattr(args,'sampler','baseline') != 'baseline':
        raise ValueError('The item-stratified experiment currently requires an MCTS-family method: '+', '.join(MCTS_METHODS))
    if args.method=='s1_agentform':
        config['s1_arm']='A1_rep'
    if out and (out / "evaluation.json").exists():
        for key, value in config.items():
            saved_value = saved.get(key,'baseline' if key=='sampler' else None)
            if saved_value != value and key != "concurrency":
                raise ValueError(f"Existing batch has different {key}; resume with its original settings")
    elif out and out.exists() and any(out.iterdir()):
        raise ValueError("Use a fresh output directory or a batch containing evaluation.json")
    completed = len(records(out)) if out else 0
    return out, config, completed, max(0, args.games - completed)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=["hedge_bayes_mw", "agentform", *MCTS_METHODS,'s1_agentform'], default="hedge_bayes_mw")
    parser.add_argument("--mcts-ms", type=int, default=1000, help="total native search milliseconds across four worlds")
    parser.add_argument('--sampler',choices=['baseline','item_stratified'],default='baseline')
    parser.add_argument("--games", type=int, default=100)
    parser.add_argument("--out", type=Path, help="fresh directory, or the same directory to resume")
    parser.add_argument("--concurrency", type=int, default=None, help="1 to 5; new batches default to 5")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    try:
        out, config, completed, remaining = plan(args)
    except ValueError as exc:
        parser.error(str(exc))
    print(json.dumps(dict(config, completed=completed, remaining=remaining, out=str(out) if out else "new timestamped directory"), indent=2), flush=True)
    if args.dry_run or remaining == 0:
        return 0
    lock = (REPO / ".local-play/ladder-bot_bonp.lock").open("a")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        print("Another managed account run is active; no games started.", file=sys.stderr)
        return 2
    if out is None:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        out = REPO / ".local-play/work/ladder" / f"decision_audit_{config['method']}_{stamp}"
    out.mkdir(parents=True, exist_ok=True)
    config_path = out / "evaluation.json"
    if not config_path.exists():
        revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
        config_path.write_text(json.dumps(dict(config, revision=revision, created_utc=time.time()), indent=2) + "\n")
    else:
        saved = json.loads(config_path.read_text())
        if saved["concurrency"] != config["concurrency"]:
            saved.setdefault("concurrency_history", []).append(dict(concurrency=saved["concurrency"],
                completed=len(records(out)), ended_utc=time.time()))
            saved.update(config, learning="independent persistent streams" if config["concurrency"] > 1 else "sequential")
            config_path.write_text(json.dumps(saved, indent=2) + "\n")
    completed = len(records(out))
    remaining = max(0, config["target"] - completed)

    def status(phase, code=None):
        summary = summarize_games(list(records(out).values()), config['target'])
        data = dict(phase=phase, exit_status=code, completed=len(records(out)), target=config["target"],
                    method=config["method"], concurrency=config["concurrency"], updated_utc=time.time(),
                    elo_start=summary['elo_start'], elo_end=summary['elo_end'], elo_change=summary['elo_change'])
        temporary = out / "run_status.tmp"
        temporary.write_text(json.dumps(data, indent=2) + "\n")
        temporary.replace(out / "run_status.json")
        metrics = out / 'evaluation_results.tmp'
        metrics.write_text(json.dumps(dict(summary, method=config['method']), indent=2)+'\n')
        metrics.replace(out / 'evaluation_results.json')

    print(f"Logs and replays: {out}", flush=True)
    (out / "run.pid").write_text(str(os.getpid()) + "\n")
    status("awaiting_login")
    password = os.environ.get("PS_PASSWORD")
    if not password:
        if not sys.stdin.isatty():
            print("Open run_decision_audit.command in Terminal for the secure password prompt.", file=sys.stderr)
            status("awaiting_login", 2)
            return 2
        password = getpass.getpass(f"{ACCOUNT} password (not saved): ")
    if not password:
        status("awaiting_login", 2)
        return 2
    manifest = json.loads((REPO / ".local-play/environment.json").read_text())
    os.environ.update(FOULPLAY_DIR=manifest["foulplay"], SHOWDOWN_DIR=manifest["showdown"],
                      ENGINE_DIR=str(REPO / "engine"), OURBOT_WORK=str(out),
                      OURBOT_BELIEF_SOURCE="foulplay_baseline", OURBOT_BELIEF_SHADOW="1",
                      OURBOT_LOG_STATES="1", OURBOT_LOG_ABAR="1", OURBOT_DET_LEAF="1",
                      OURBOT_LOG_WORLDS="0", OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1",
                      MKL_NUM_THREADS="1", VECLIB_MAXIMUM_THREADS="1", PS_PASSWORD=password)
    os.environ["OURBOT_MCTS_MS"] = str(config.get("mcts_ms", 1000))
    os.environ['OURBOT_WORLD_SAMPLER'] = config.get('sampler','baseline')
    console = sys.stdout
    done = threading.Event()

    def monitor():
        while not done.wait(20):
            try:
                n = len(records(out))
                print(f"Recorded {n}/{config['target']}; running {config['method']}. Logs: {out}", file=console, flush=True)
                status("running")
            except (OSError, ValueError):
                pass

    watcher = threading.Thread(target=monitor, daemon=True)
    awake = subprocess.Popen(["/usr/bin/caffeinate", "-i", "-w", str(os.getpid())]) if Path("/usr/bin/caffeinate").exists() else None
    watcher.start()
    code = 1
    status("running")
    print("Keep this Terminal open. Ctrl-C once finishes the active game and stops new searches.", flush=True)
    try:
        with (out / "run.log").open("a", buffering=1) as log:
            with contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
                import ladder
                code = ladder.main(["--server", config["server"], "--username", ACCOUNT, "--i-am-registered",
                                    "--method", config["method"], "--k", str(config["k"]), "--games", str(remaining),
                                    "--concurrency", str(config["concurrency"]), "--out", str(out)])
    except KeyboardInterrupt:
        code = 130
    finally:
        done.set()
        watcher.join(timeout=2)
        os.environ.pop("PS_PASSWORD", None)
        password = None
        status("completed" if code == 0 and len(records(out)) >= config["target"] else "stopped" if code == 0 else "failed", code)
        if awake:
            awake.terminate()
            awake.wait()
        fcntl.flock(lock, fcntl.LOCK_UN)
        lock.close()
    # Analyze even an interrupted batch, after removing the credential from the
    # child environment. This does not start or schedule any additional games.
    audit = subprocess.run([manifest["python"], str(REPO / "engine/showdown/audit_decisions.py"), str(out),
                            "--username", ACCOUNT, "--write"], cwd=REPO)
    print(f"Run exit {code}; audit exit {audit.returncode}; {len(records(out))}/{config['target']} recorded. {out}", flush=True)
    return code if code else audit.returncode


if __name__ == "__main__":
    raise SystemExit(main())
