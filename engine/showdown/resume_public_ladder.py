"""Continue the authorized bot_bonp evaluation from an ordinary Terminal.

Counts recorded results across interrupted segments, then plays only the number
needed for 100 recorded completions. Credentials are prompted for, never saved.
Use --dry-run to inspect the budget without login, matchmaking, or file writes.
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

REPO = Path(__file__).resolve().parents[2]
ROOT = REPO / '.local-play/work/ladder'
PREFIX = 'bot_bonp_20261116_100_c5'


def saved_results():
    results = {}
    for ledger in sorted(ROOT.glob(PREFIX + '*/ladder_games.jsonl')):
        for line in ledger.read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get('result') not in ('win', 'loss', 'draw', 'tie'):
                raise ValueError(f'Unrecognized result in {ledger}')
            results[row['battle_tag']] = row
    return results


def write_status(out, phase, prior, code=None):
    data = dict(phase=phase, exit_status=code, updated_utc=time.time(),
                target=100 - prior, concurrency=5, previous_completed=prior,
                overall_completed_target=100)
    temporary = out / 'run_status.tmp'
    temporary.write_text(json.dumps(data, indent=2) + '\n')
    temporary.replace(out / 'run_status.json')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    previous = len(saved_results())
    remaining = max(0, 100 - previous)
    print(f'{previous} recorded games; {remaining} remaining; at most 5 concurrent.', flush=True)
    print('Interrupted games without recorded results are excluded from this count.', flush=True)
    if args.dry_run or not remaining:
        return 0

    lock = (REPO / '.local-play/ladder-bot_bonp.lock').open('a')
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        print('Another managed bot_bonp process is running. No games started.', file=sys.stderr)
        return 2
    # Recheck after acquiring the shared account lock.
    previous = len(saved_results())
    remaining = max(0, 100 - previous)
    if not remaining:
        return 0
    password = getpass.getpass('bot_bonp password (not saved): ')
    if not password:
        return 2
    manifest = json.loads((REPO / '.local-play/environment.json').read_text())
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    out = ROOT / (PREFIX + '_terminal_' + stamp)
    out.mkdir(parents=True)
    (out / 'run.pid').write_text(str(os.getpid()) + '\n')
    os.environ.update(FOULPLAY_DIR=manifest['foulplay'],
                      SHOWDOWN_DIR=manifest['showdown'], ENGINE_DIR=str(REPO / 'engine'),
                      OURBOT_WORK=str(out), OURBOT_BELIEF_SOURCE='foulplay_baseline',
                      OURBOT_BELIEF_SHADOW='1', OPENBLAS_NUM_THREADS='1',
                      OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', VECLIB_MAXIMUM_THREADS='1')
    os.environ['PS_PASSWORD'] = password
    print(f'Logs and replays: {out}', flush=True)
    print('Keep this Terminal open. Ctrl-C once drains active games; twice aborts.', flush=True)
    console = sys.stdout
    finished = threading.Event()

    def monitor():
        while not finished.wait(15):
            try:
                progress = json.loads((out / 'progress.json').read_text())
                print(f"Recorded: {previous + progress['completed']}/100; "
                      f"active: {len(progress['active'])}; failures: {progress['failed']}",
                      file=console, flush=True)
                if progress.get('error'):
                    print('Runner reports an error; see run.log.', file=console, flush=True)
            except FileNotFoundError:
                print('Connecting; see run.log for details.', file=console, flush=True)
            except (ValueError, KeyError):
                pass

    watcher = threading.Thread(target=monitor, daemon=True)
    watcher.start()
    awake = None
    if Path('/usr/bin/caffeinate').exists():
        awake = subprocess.Popen(['/usr/bin/caffeinate', '-i', '-w', str(os.getpid())])
    code = 1
    write_status(out, 'starting', previous)
    try:
        with (out / 'run.log').open('a', buffering=1) as log:
            with contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
                import ladder
                code = ladder.main([
                    '--server', 'wss://sim3.psim.us/showdown/websocket',
                    '--username', 'bot_bonp', '--i-am-registered',
                    '--method', 'agentform', '--k', '4', '--format', 'gen9randombattle',
                    '--games', str(remaining), '--concurrency', '5', '--out', str(out)])
    except KeyboardInterrupt:
        code = 130
    finally:
        finished.set()
        watcher.join(timeout=2)
        os.environ.pop('PS_PASSWORD', None)
        password = None
        count = len(saved_results())
        write_status(out, 'completed' if code == 0 and count >= 100 else
                     'stopped' if code == 0 else 'failed', previous, code)
        if awake:
            awake.terminate()
            awake.wait()
        fcntl.flock(lock, fcntl.LOCK_UN)
        lock.close()
        print(f'Process ended with code {code}; {count}/100 recorded games. Logs: {out}', flush=True)
    return code


if __name__ == '__main__':
    raise SystemExit(main())
