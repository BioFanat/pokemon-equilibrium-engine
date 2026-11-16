"""Install and run the pinned local Showdown + eqbot environment.

No global packages, shell profile edits, external battle server, or Git changes.
Run through play_bot.command, or `python3.12 local_play.py doctor|setup|play`.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import shutil
import signal
import socket
import subprocess
import sys
import tarfile
import tempfile
import time
import urllib.error
import urllib.request
import webbrowser
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
ENGINE = REPO / "engine"
DEFAULT_ROOT = REPO / ".local-play"
SHOWDOWN_REV = "7e743761cd13cdb361fc0f58eb5a866284894f5b"
FOULPLAY_REV = "8f541f80f91f960603673796559f796b10ccf0fb"
SOURCES = {
    "pokemon-showdown": ("smogon/pokemon-showdown", SHOWDOWN_REV),
    "foul-play": ("pmariglia/foul-play", FOULPLAY_REV),
}
METHODS = ("agentform", "pimc", "top", "hedge_bayes")


class SetupError(RuntimeError):
    pass


def run(args, *, env=None, cwd=None):
    print("Running:", " ".join(map(str, args)), flush=True)
    subprocess.run(list(map(str, args)), env=env, cwd=cwd, check=True)


def download(url, destination):
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(destination.name + ".part")
    print("Downloading", url, flush=True)
    request = urllib.request.Request(url, headers={"User-Agent": "pokemon-equilibrium-local-play/1"})
    try:
        with urllib.request.urlopen(request, timeout=30) as response, partial.open("wb") as output:
            shutil.copyfileobj(response, output)
        partial.replace(destination)
    except (OSError, urllib.error.URLError) as exc:
        partial.unlink(missing_ok=True)
        raise SetupError(f"Cannot download {url}: {exc}. Run bash play_bot.command in your normal Terminal with internet access.") from exc
    return destination


def verify_sha256(path, checksum):
    if not re.fullmatch(r"[0-9a-fA-F]{64}", checksum):
        raise SetupError("Malformed download checksum")
    with Path(path).open("rb") as source:
        digest = hashlib.file_digest(source, "sha256").hexdigest()
    if digest.lower() != checksum.lower():
        raise SetupError(f"SHA-256 mismatch for {path}; refusing to install")


def extract_archive(archive, destination):
    """Reject path escapes and unsafe links, then use Python's data filter."""
    destination = Path(destination).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive) as package:
        for member in package.getmembers():
            target = (destination / member.name).resolve()
            if not target.is_relative_to(destination):
                raise SetupError("Archive path escapes destination")
            if member.issym() or member.islnk():
                base = target.parent if member.issym() else destination
                if not (base / member.linkname).resolve().is_relative_to(destination):
                    raise SetupError("Archive link escapes destination")
        package.extractall(destination, filter="data")


def node_major(executable):
    if not executable:
        return 0
    try:
        version = subprocess.check_output([str(executable), "--version"], text=True, timeout=10).strip()
        return int(version.removeprefix("v").split(".")[0])
    except (OSError, ValueError, subprocess.SubprocessError):
        return 0


def tool_environment(root):
    env = os.environ.copy()
    tools = root / "tools"
    env["CARGO_HOME"] = str(tools / "cargo")
    env["RUSTUP_HOME"] = str(tools / "rustup")
    env["PATH"] = os.pathsep.join([str(tools / "node" / "bin"), str(tools / "cargo" / "bin"), env.get("PATH", "")])
    env["PIP_CACHE_DIR"] = str(root / "cache" / "pip")
    env["npm_config_cache"] = str(root / "cache" / "npm")
    env["PYTHONUNBUFFERED"] = "1"
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        env[name] = "1"
    return env


def ensure_node(root, env):
    existing = shutil.which("node", path=env["PATH"])
    if node_major(existing) >= 22:
        return Path(existing)
    system = {"Darwin": "darwin", "Linux": "linux"}.get(platform.system())
    arch = {"arm64": "arm64", "aarch64": "arm64", "x86_64": "x64", "AMD64": "x64"}.get(platform.machine())
    if not system or not arch:
        raise SetupError("Install Node.js 22+ for this operating system, then rerun")
    index = download("https://nodejs.org/dist/index.json", root / "cache" / "node-index.json")
    releases = json.loads(index.read_text())
    release = next((r for r in releases if r["version"].startswith("v22.") and r.get("lts")), None)
    if release is None:
        raise SetupError("No Node 22 LTS release found in the official index")
    version = release["version"]
    filename = f"node-{version}-{system}-{arch}.tar.gz"
    base = f"https://nodejs.org/dist/{version}"
    sums = download(f"{base}/SHASUMS256.txt", root / "cache" / f"node-{version}-sums.txt")
    checksums = {line.split()[1].lstrip("*"): line.split()[0] for line in sums.read_text().splitlines() if len(line.split()) == 2}
    if filename not in checksums:
        raise SetupError("Node archive is absent from official checksums")
    archive = download(f"{base}/{filename}", root / "cache" / filename)
    verify_sha256(archive, checksums[filename])
    with tempfile.TemporaryDirectory(dir=root, prefix="node-install-") as temporary:
        extract_archive(archive, temporary)
        source = Path(temporary) / filename.removesuffix(".tar.gz")
        target = root / "tools" / "node"
        if target.exists():
            raise SetupError(f"Existing local Node installation needs manual repair: {target}")
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(source), target)
    return root / "tools" / "node" / "bin" / "node"


def ensure_rust(root, env):
    if shutil.which("cargo", path=env["PATH"]) and shutil.which("rustc", path=env["PATH"]):
        try:
            subprocess.run(["rustc", "--version"], env=env, check=True, capture_output=True, timeout=10)
            subprocess.run(["cargo", "--version"], env=env, check=True, capture_output=True, timeout=10)
            return
        except (OSError, subprocess.SubprocessError):
            pass
    system = {"Darwin": "apple-darwin", "Linux": "unknown-linux-gnu"}.get(platform.system())
    arch = {"arm64": "aarch64", "aarch64": "aarch64", "x86_64": "x86_64", "AMD64": "x86_64"}.get(platform.machine())
    if not system or not arch:
        raise SetupError("Install Rust/Cargo for this operating system, then rerun")
    base = f"https://static.rust-lang.org/rustup/dist/{arch}-{system}/rustup-init"
    installer = download(base, root / "cache" / "rustup-init")
    checksum = download(base + ".sha256", root / "cache" / "rustup-init.sha256").read_text().split()[0]
    verify_sha256(installer, checksum)
    installer.chmod(0o700)
    run([installer, "-y", "--no-modify-path", "--profile", "minimal", "--default-toolchain", "stable"], env=env)


def ensure_source(root, name):
    repository, revision = SOURCES[name]
    destination = root / name
    marker = destination / ".local-play-revision"
    if destination.exists():
        if marker.is_file() and marker.read_text().strip() == revision:
            return destination
        raise SetupError(f"Refusing to overwrite an unrecognized dependency directory: {destination}")
    archive = download(f"https://codeload.github.com/{repository}/tar.gz/{revision}", root / "cache" / f"{name}-{revision}.tar.gz")
    with tempfile.TemporaryDirectory(dir=root, prefix="source-install-") as temporary:
        extract_archive(archive, temporary)
        folders = list(Path(temporary).iterdir())
        if len(folders) != 1 or not folders[0].is_dir():
            raise SetupError(f"Unexpected source archive for {repository}")
        shutil.move(str(folders[0]), destination)
    marker.write_text(revision + "\n")
    return destination


def configure_server(showdown):
    # This is a dedicated managed checkout. Explicitly bind only to loopback.
    config = showdown / "config" / "config.js"
    text = (showdown / "config" / "config-example.js").read_text()
    text += "\n// Local-play overrides\nexports.bindaddress = '127.0.0.1';\nexports.watchconfig = false;\nexports.repl = false;\nexports.noipchecks = true;\n"
    config.write_text(text)
    # Upstream performs this optional internet lookup unconditionally at module
    # import, even with --no-security/noipchecks. Disable it only in our managed
    # local checkout; it is unrelated to playing a local Pokémon battle.
    ip_tools = showdown / "server" / "ip-tools.ts"
    source = ip_tools.read_text()
    marker = "// Local play: external Tor-list lookup disabled."
    original = "void IPTools.updateTorRanges();"
    if original in source:
        ip_tools.write_text(source.replace(original, marker))
    elif marker not in source:
        raise SetupError("Pinned Showdown IP-tools layout changed; refusing to start with unverified external lookups")


IMPORT_CHECK = "import numpy, scipy.optimize, websockets, poke_engine; from pathlib import Path; print('Bot dependencies ready')"


def setup(root):
    if sys.version_info[:2] != (3, 12):
        raise SetupError("Use Python 3.12 for the pinned bot environment (play_bot.command selects it automatically)")
    root.mkdir(parents=True, exist_ok=True)
    env = tool_environment(root)
    node = ensure_node(root, env)
    ensure_rust(root, env)
    showdown = ensure_source(root, "pokemon-showdown")
    foulplay = ensure_source(root, "foul-play")
    npm = Path(node).parent / "npm"
    if not npm.exists():
        raise SetupError(f"npm not found alongside {node}")
    stamp = root / "showdown-built.json"
    lock_hash = hashlib.sha256((showdown / "package-lock.json").read_bytes()).hexdigest()
    if not stamp.exists() or json.loads(stamp.read_text()).get("lock_sha256") != lock_hash:
        run([npm, "ci", "--ignore-scripts", "--no-audit", "--no-fund"], cwd=showdown, env=env)
        run([node, "build"], cwd=showdown, env=env)
        stamp.write_text(json.dumps(dict(revision=SHOWDOWN_REV, lock_sha256=lock_hash)))
    configure_server(showdown)
    python = root / "venv" / "bin" / "python"
    if not python.exists():
        run([sys.executable, "-m", "venv", root / "venv"], env=env)
    requirements = Path(__file__).with_name("requirements-bot.txt")
    requirement_hash = hashlib.sha256(requirements.read_bytes()).hexdigest()
    pip_stamp = root / "bot-installed.json"
    if not pip_stamp.exists() or json.loads(pip_stamp.read_text()).get("requirements_sha256") != requirement_hash:
        run([python, "-m", "pip", "install", "-r", requirements], env=env)
        run([python, "-m", "pip", "install", "--no-binary", "poke-engine", "poke-engine==0.0.48",
             "--config-settings", "build-args=--features poke-engine/terastallization --no-default-features"], env=env)
        run([python, "-c", IMPORT_CHECK], env=env)
        pip_stamp.write_text(json.dumps(dict(requirements_sha256=requirement_hash, poke_engine="0.0.48+tera")))
    run([python, "-c", IMPORT_CHECK], env=env)
    manifest = dict(node=str(node), python=str(python), showdown=str(showdown), foulplay=str(foulplay),
                    showdown_revision=SHOWDOWN_REV, foulplay_revision=FOULPLAY_REV)
    (root / "environment.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print("Local play environment is installed.", flush=True)
    return manifest


def port_is_open(port):
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=0.3):
            return True
    except OSError:
        return False


def stop_process(process):
    try:
        os.killpg(process.pid, signal.SIGTERM)
        process.wait(timeout=8)
    except ProcessLookupError:
        return
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=3)


BOT_PROBE = r'''
import asyncio,json,sys
import websockets
async def check():
    async with websockets.connect(sys.argv[1],open_timeout=3) as ws:
        await ws.send('|/cmd userdetails '+sys.argv[2])
        async with asyncio.timeout(5):
            while True:
                message=await ws.recv()
                for line in message.split('\n'):
                    if line.startswith('|queryresponse|userdetails|'):
                        data=json.loads(line.split('|',3)[3])
                        return data.get('userid')==sys.argv[2] and data.get('rooms') is not False
try:
    sys.exit(0 if asyncio.run(check()) else 1)
except Exception:
    sys.exit(1)
'''


def play(root, args):
    manifest_path = root / "environment.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else setup(root)
    if port_is_open(args.port):
        raise SetupError(f"Port {args.port} is already in use. Choose another with --port; no existing server was stopped.")
    configure_server(Path(manifest["showdown"]))
    env = tool_environment(root)
    env.update(FOULPLAY_DIR=manifest["foulplay"], SHOWDOWN_DIR=manifest["showdown"],
               ENGINE_DIR=str(ENGINE), OURBOT_WORK=str(root / "work"), BOT_PYTHON=manifest["python"],
               OURBOT_METHOD=args.method, OURBOT_K=str(args.worlds),
               OURBOT_BELIEF_SOURCE="mechanics_v2" if args.mechanics_shadow else "foulplay_baseline",
               OURBOT_BELIEF_SHADOW="1")
    (root / "work").mkdir(parents=True, exist_ok=True)
    logs = root / "logs"
    logs.mkdir(exist_ok=True)
    run([manifest["python"], "-c", IMPORT_CHECK], env=env)
    url = f"http://127.0.0.1:{args.port}"
    websocket = f"ws://127.0.0.1:{args.port}/showdown/websocket"
    children = []
    try:
        with (logs / "server.log").open("w") as server_log, (logs / "bot.log").open("w") as bot_log:
            server = subprocess.Popen([manifest["node"], "pokemon-showdown", "start", "--no-security", str(args.port)],
                                      cwd=manifest["showdown"], env=env, stdout=server_log,
                                      stderr=subprocess.STDOUT, start_new_session=True)
            children.append(server)
            deadline = time.monotonic() + 60
            while not port_is_open(args.port):
                if server.poll() is not None or time.monotonic() > deadline:
                    raise SetupError(f"Showdown did not start. See {logs / 'server.log'}")
                time.sleep(0.3)
            bot = subprocess.Popen([manifest["python"], str(Path(__file__).with_name("run_ourbot.py")),
                                    "--websocket-uri", websocket, "--ps-username", "eqbot",
                                    "--bot-mode", "accept_challenge", "--pokemon-format", "gen9randombattle",
                                    "--run-count", "10000", "--log-level", "INFO"],
                                   cwd=ENGINE, env=env, stdout=bot_log, stderr=subprocess.STDOUT, start_new_session=True)
            children.append(bot)
            deadline = time.monotonic() + 90
            while True:
                if bot.poll() is not None or server.poll() is not None:
                    raise SetupError(f"A process stopped during startup. See {logs}")
                probe = subprocess.run([manifest["python"], "-c", BOT_PROBE, websocket, "eqbot"],
                                       env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=12)
                if probe.returncode == 0:
                    break
                if time.monotonic() > deadline:
                    raise SetupError(f"Bot did not log in. See {logs / 'bot.log'}")
                time.sleep(1)
            print(f"\nReady: {url}\nChoose an unregistered nickname, find eqbot, and challenge it to Gen 9 Random Battle.", flush=True)
            print(f"Bot method: {args.method}. Logs: {logs}\nKeep this terminal open. Ctrl+C stops both processes.\n", flush=True)
            if not args.no_browser:
                webbrowser.open(url)
            while True:
                if server.poll() is not None or bot.poll() is not None:
                    raise SetupError(f"A process exited. See {logs}")
                time.sleep(1)
    except KeyboardInterrupt:
        print("\nStopping the bot and local Showdown server.", flush=True)
    finally:
        for child in reversed(children):
            stop_process(child)


def doctor(root):
    env = tool_environment(root)
    node = shutil.which("node", path=env["PATH"])
    manifest = root / "environment.json"
    missing = []
    if manifest.exists():
        data = json.loads(manifest.read_text())
        missing = [key for key in ("node", "python", "showdown", "foulplay") if not Path(data[key]).exists()]
    report = dict(python=sys.version.split()[0], node=node, node_major=node_major(node),
                  cargo=shutil.which("cargo", path=env["PATH"]), installed=manifest.exists(), runtime=str(root))
    report["missing_paths"] = missing
    print(json.dumps(report, indent=2))
    return 0 if manifest.exists() and report["node_major"] >= 22 and not missing else 1


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("play", "setup", "doctor"), nargs="?", default="play")
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT, help="Directory for local tools, dependencies and logs")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--method", choices=METHODS, default="agentform")
    parser.add_argument("--worlds", type=int, default=4)
    parser.add_argument("--no-browser", action="store_true")
    parser.add_argument("--mechanics-shadow", action="store_true", help="Observe mechanics updates without changing moves")
    args = parser.parse_args(argv)
    if not 1024 <= args.port <= 65535 or args.worlds < 1:
        parser.error("Use a port from 1024 to 65535 and at least one sampled world")
    root = args.root.expanduser().resolve()
    try:
        if args.command == "doctor":
            return doctor(root)
        if args.command == "setup":
            setup(root)
        else:
            play(root, args)
        return 0
    except (SetupError, OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        print(f"\nSetup stopped: {exc}", file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
