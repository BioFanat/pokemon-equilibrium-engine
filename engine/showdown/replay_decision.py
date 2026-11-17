"""Re-solve a recorded decision offline, without importing the network client.

Usage: python engine/showdown/replay_decision.py decisions.jsonl --line 1
Requires a record captured with OURBOT_LOG_STATES=1. Historical mixture-only
logs cannot reproduce their sampled hidden worlds or payoffs exactly.
"""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import hidden_agents as ha


def replay(record):
    if "search" not in record:
        raise ValueError("Record has no search snapshot; capture with OURBOT_LOG_STATES=1")
    search = record["search"]
    legal = search["legal"]
    matrices = [ha.align(np.asarray(M), rows, legal)
                for M, rows in zip(search["matrices"], search["rows"])]
    weights = np.asarray(search["weights"], dtype=float)
    weights /= weights.sum()
    if len(legal) == 1:
        x = np.ones(1)
    elif record["method"] in {"agentform", "top"}:
        x = ha.agent_form_lp(matrices, weights)[0]
        if record["method"] == "top":
            x = np.eye(len(legal))[x.argmax()]
    elif record["method"] == "pimc":
        x = sum(w * ha.lp_x(M)[0] for w, M in zip(weights, matrices))
    else:
        raise ValueError("Offline replay supports agentform, pimc and top")
    return dict(battle_tag=record.get("battle_tag"), turn=record["turn"],
                recorded_move=record["move"], mix=dict(zip(legal, x.tolist())))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log", type=Path)
    parser.add_argument("--line", type=int, required=True, help="one-based JSONL line number")
    args = parser.parse_args()
    try:
        if args.line < 1:
            raise ValueError("--line must be positive")
        with args.log.open() as stream:
            record = next((json.loads(line) for i, line in enumerate(stream, 1) if i == args.line), None)
        if record is None:
            raise ValueError("Requested line is outside the log")
        print(json.dumps(replay(record), indent=2))
    except ValueError as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
