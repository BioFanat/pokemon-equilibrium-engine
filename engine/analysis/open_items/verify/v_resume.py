"""Resumed tags: record key sets and config-like fields constant across games (earlier vs final invocation)."""
import collections, json, glob, os
from vlib import RES
for tag in ("LOG", "E3f", "XUmv", "XR", "XEmv", "E5t", "E5s", "SX1", "SX10", "SX30"):
    for role in ("A", "B"):
        fs = glob.glob(os.path.join(RES, tag, "replays_headless", f"{tag}-s*-g*.{role}.decisions.jsonl"))
        if not fs: continue
        keys = collections.Counter(); cfg = collections.Counter()
        for f in fs:
            for l in open(f):
                r = json.loads(l)
                if r.get("final"): continue
                keys[tuple(sorted(k for k in r if k not in ("feedback", "bayes", "abar", "labels", "legal", "pfull", "Q", "qs")))] += 1
                cfg[(r.get("method"), r.get("worlds") if r.get("worlds") is None else min(r["worlds"], 4), r.get("iters"), r.get("played_rule"), r.get("eps"))] += 1
        print(tag, role, "files", len(fs), "keysets", len(keys), "cfg", dict(cfg))
        if len(keys) > 1: print("   ", [(k, v) for k, v in keys.items()][:6])
