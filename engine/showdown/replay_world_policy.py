"""Replay flagged positions through baseline/current live policy, with frozen memory.

No connection or native re-search. Foul Play is imported for its live adapter,
but matrices come from the recorded snapshots. Posterior and expert weights
are reconstructed from rounded logs; see old-vs-recorded reconstruction error.
--baseline points to a preserved pre-change run_ourbot.py.
"""
from contextlib import ExitStack
import argparse
import importlib.util
import json
import os
from pathlib import Path
import random
import sys
import tempfile
from types import SimpleNamespace as NS
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "engine"), str(ROOT / "engine/showdown"),
               os.environ.get("FOULPLAY_DIR", str(ROOT / ".local-play/foul-play"))]
import numpy as np
import run_ourbot as current
import nash
from inspect_audit_positions import find_record


def replay_policy(module, record):
    search = record["search"]
    worlds = [(NS(M=np.asarray(M), s1=rows, s2=cols), w)
              for M, rows, cols, w in zip(search["matrices"], search["rows"], search["cols"], search["weights"])]
    L = module.Learner(experts=module.HEDGE_EXPERTS, **module.HEDGE_CONFIG[record["method"]])
    ew = np.array([record["weights"][e] for e in module.HEDGE_EXPERTS])
    L.logw = np.log(ew / ew.sum())
    post = module.behavior.WorldPosterior()
    P = np.array([record["bayes"]["posterior"][s] for s in module.behavior.STYLES])
    post.logp = np.log(P / P.sum(), out=np.full(len(P), -np.inf), where=P > 0)
    battle = NS(team_preview=False, turn=record["turn"], _ourlines=[], opponent=NS(name="p2"),
                request_json=None, user=NS(reserve=[]), battle_tag="offline-policy-replay",
                mode=NS(prepare_battles=lambda battle, k: worlds))
    with tempfile.TemporaryDirectory() as tmp, ExitStack() as stack:
        stack.enter_context(patch.object(module, "load_learner", lambda *args: (L, dict(games=0, updates=0, skipped=0))))
        stack.enter_context(patch.object(module.behavior, "load_world_posterior",
                                        lambda *args: (post, dict(games=0, updates=0, skipped=0))))
        stack.enter_context(patch.object(module, "battle_to_poke_engine_state", lambda st: st))
        stack.enter_context(patch.object(module, "request_options", lambda battle: search["legal"]))
        stack.enter_context(patch.object(module, "raw_action", lambda label, battle: label))
        stack.enter_context(patch.object(module, "RNG", random.Random(0)))
        stack.enter_context(patch.object(nash, "full_matrix", lambda st, **kw: (st.M, st.s1, st.s2)))
        stack.enter_context(patch.object(nash, "LEAF_STATS_ON", False))
        stack.enter_context(patch.dict(os.environ, OURBOT_LOG_ABAR="1", OURBOT_LOG_STATES="0",
                                      OURBOT_LOG_WORLDS="0", OURBOT_BELIEF_SOURCE="foulplay_baseline"))
        log = Path(tmp)/"decisions.jsonl"
        module.make_move_fn(record["method"], len(worlds), log=str(log))(battle)
        result = json.loads(log.read_text())
        module._FINISHERS.pop(battle.battle_tag, None)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out", type=Path)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--write", type=Path, required=True)
    args = parser.parse_args()
    spec = importlib.util.spec_from_file_location("baseline_world_policy", args.baseline)
    old = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(old)
    audit = json.loads((args.out / f"tactical_audit_first{args.limit}.json").read_text())
    result = []
    for finding in audit["findings"]:
        if not {"ground_into_flying", "setup_then_ko"} & set(finding["categories"]) or finding["classification"] != "multiple_legal_actions":
            continue
        record = find_record(args.out, finding)
        before, after = replay_policy(old, record), replay_policy(current, record)
        legal = record["search"]["legal"]
        chosen = record["move"].removeprefix("switch ")
        ix = legal.index(chosen)
        result.append(dict(battle_tag=finding["battle_tag"], turn=record["turn"], played=chosen,
            old_safest=before["experts"]["safest"], new_safest=after["experts"]["safest"],
            old_probability=before["pfull"][ix], new_probability=after["pfull"][ix],
            reconstruction_max_error=float(np.max(np.abs(np.array(before["pfull"])-record["pfull"]))),
            old_policy=before, new_policy=after))
    args.write.parent.mkdir(parents=True, exist_ok=True)
    args.write.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps([{k:v for k,v in r.items() if k not in {"old_policy", "new_policy"}} for r in result], indent=2))


if __name__ == "__main__":
    main()
