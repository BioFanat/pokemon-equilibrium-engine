"""Shared helpers: stream reconstruction for the real-Showdown logs (batch.split_streams: stream k = sorted seeds[k::S])."""
import glob
import json
import os
import re

RES = os.environ.get("SHOWDOWN_RESULTS", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "showdown", "results"))
GAME_RE = re.compile(r"(?P<tag>.+)-s(?P<seed>\d+)-g(?P<game>[12])\.(?P<role>[AB])\.decisions\.jsonl$")
EXPERTS = ["top", "sampled", "safest", "safex"]
STYLES = ["eq", "br_unif", "br_us", "safe", "uniform"]


def game_files(tag, role="A"):
    files = {}
    for p in glob.glob(os.path.join(RES, tag, "replays_headless", f"{tag}-s*-g*.{role}.decisions.jsonl")):
        m = GAME_RE.search(os.path.basename(p))
        if m:
            files[(int(m["seed"]), int(m["game"]))] = p
    return files


def streams(tag, role="A", n_streams=10):
    """Yield (k, [(seed, game, path), ...]) in play order."""
    files = game_files(tag, role)
    seeds = sorted({s for s, _ in files})
    for k in range(n_streams):
        order = [(s, g, files[(s, g)]) for s in seeds[k::n_streams] for g in (1, 2) if (s, g) in files]
        yield k, order


def records(path):
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)


def summary(tag):
    return json.load(open(os.path.join(RES, tag, f"{tag}_summary.json")))
