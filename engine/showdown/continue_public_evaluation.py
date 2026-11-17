"""Wait for the original 100-game batch, then keep playing until an explicit stop.

One interactive password entry is held in memory while waiting and playing.
The account lock prevents overlap with the original process. Existing stream
checkpoints and logs are retained. Touch STOP_CONTINUOUS to drain active games.
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

from run_public_evaluation import ACCOUNT, REPO, records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    out = args.out.resolve()
    config = json.loads((out / "evaluation.json").read_text())
    stop = out / "STOP_CONTINUOUS"
    if config["account"] != ACCOUNT or config["method"] != "hedge_bayes_mw" or config["concurrency"] != 5:
        parser.error("Continuation expects the existing five-stream hedge_bayes_mw evaluation")
    print(f"First cohort: 100 games. Then continue {ACCOUNT} with up to five games until stopped.\nLogs: {out}", flush=True)
    if args.dry_run:
        return 0
    queued_lock = (REPO / ".local-play/continuous-bot_bonp.lock").open("a")
    try:
        fcntl.flock(queued_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        print("A continuation is already queued or running; no second process started.", file=sys.stderr)
        return 2
    if stop.exists():
        print("Stop request exists; no games started.", flush=True)
        return 0

    def status(phase, code=None):
        data = dict(phase=phase, exit_status=code, completed=len(records(out)), until_stop=True,
                    method=config["method"], concurrency=5, updated_utc=time.time(), stop_file=str(stop))
        temporary = out / "continuous_status.tmp"
        temporary.write_text(json.dumps(data, indent=2) + "\n")
        temporary.replace(out / "continuous_status.json")

    (out / "continuous.pid").write_text(str(os.getpid()) + "\n")
    status("awaiting_login")
    password = os.environ.get("PS_PASSWORD")
    if not password:
        if not sys.stdin.isatty():
            print("Open the continuation launcher in an interactive Terminal.", file=sys.stderr)
            return 2
        password = getpass.getpass(f"{ACCOUNT} password for continuous play (not saved): ")
    if not password:
        return 2
    lock = (REPO / ".local-play/ladder-bot_bonp.lock").open("a")
    status("waiting_for_first_cohort")
    while True:
        if stop.exists():
            status("stopped", 0)
            return 0
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            break
        except BlockingIOError:
            time.sleep(5)
    if len(records(out)) < config["target"]:
        print("The first batch stopped early; continuation requires review before starting.", file=sys.stderr)
        status("first_cohort_incomplete", 2)
        return 2
    if stop.exists():
        status("stopped", 0)
        return 0
    manifest = json.loads((REPO / ".local-play/environment.json").read_text())
    os.environ.update(FOULPLAY_DIR=manifest["foulplay"], SHOWDOWN_DIR=manifest["showdown"],
                      ENGINE_DIR=str(REPO / "engine"), OURBOT_WORK=str(out),
                      OURBOT_BELIEF_SOURCE="foulplay_baseline", OURBOT_BELIEF_SHADOW="1",
                      OURBOT_LOG_STATES="1", OURBOT_LOG_ABAR="1", OURBOT_DET_LEAF="1",
                      OURBOT_LOG_WORLDS="0", OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1",
                      MKL_NUM_THREADS="1", VECLIB_MAXIMUM_THREADS="1", PS_PASSWORD=password)
    done = threading.Event()
    console = sys.stdout

    def monitor():
        while not done.wait(20):
            try:
                status("running")
                print(f"Recorded {len(records(out))} games overall; continuing until stopped.", file=console, flush=True)
            except (OSError, ValueError):
                pass

    watcher = threading.Thread(target=monitor, daemon=True)
    watcher.start()
    awake = subprocess.Popen(["/usr/bin/caffeinate", "-i", "-w", str(os.getpid())]) if Path("/usr/bin/caffeinate").exists() else None
    code = 1
    status("running")
    try:
        with (out / "continuous.log").open("a", buffering=1) as log:
            with contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
                import ladder
                code = ladder.main(["--server", config["server"], "--username", ACCOUNT, "--i-am-registered",
                                    "--method", config["method"], "--k", "4", "--concurrency", "5",
                                    "--until-stop", "--stop-file", str(stop), "--out", str(out)])
    except KeyboardInterrupt:
        code = 130
    finally:
        done.set()
        watcher.join(timeout=2)
        os.environ.pop("PS_PASSWORD", None)
        password = None
        status("stopped" if code == 0 else "failed", code)
        if awake:
            awake.terminate()
            awake.wait()
        fcntl.flock(lock, fcntl.LOCK_UN)
        lock.close()
    print(f"Continuous play ended with code {code}; logs: {out}", flush=True)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
