"""Summarize an evaluation and the legacy-vs-per-world pure-safest expert mismatch."""
from collections import Counter
import argparse
import json
import math
from pathlib import Path

from audit_decisions import audit_directory, modeled_values


def decisions_for(out, tag, shared):
    path = out / "decisions" / (tag + ".jsonl")
    if path.exists():
        return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    return shared.get(tag, [])


def summarize(out, username, limit=None):
    report = audit_directory(out, username, limit)
    rows = {r["battle_tag"]: r for r in (json.loads(line) for line in (out / "ladder_games.jsonl").read_text().splitlines() if line.strip())}
    if limit is not None:
        rows = dict(list(rows.items())[:limit])
    shared = {}
    path = out / "decisions.jsonl"
    if path.exists():
        for line in path.read_text().splitlines():
            r = json.loads(line)
            shared.setdefault(r.get("battle_tag"), []).append(r)
    checked, strict_mismatches, security_gaps, snapshot_count = 0, 0, [], 0
    max_decision = max((r.get("decision_secs_max") or 0 for r in rows.values()), default=0)
    total_decisions = sum(r.get("decisions", 0) for r in rows.values())
    weighted_latency = sum((r.get("decision_secs_mean") or 0) * r.get("decisions", 0) for r in rows.values())
    mismatch_examples = []
    for tag in rows:
        for r in decisions_for(out, tag, shared):
            if "move" not in r:
                continue
            snapshot_count += bool(r.get("search", {}).get("states"))
            legal = r.get("search", {}).get("legal", [])
            old = r.get("experts", {}).get("safest")
            if not old or not legal:
                continue
            values = modeled_values(r, legal)
            if old not in values:
                continue
            checked += 1
            best = max(values, key=lambda label: values[label]["weighted_pure_worst_case"])
            gap = values[best]["weighted_pure_worst_case"] - values[old]["weighted_pure_worst_case"]
            if gap > 0.001:  # values are displayed to four decimals; ignore numerical/logging ties
                strict_mismatches += 1
                security_gaps.append(gap)
                mismatch_examples.append(dict(battle_tag=tag, turn=r["turn"], decision_index=r.get("decision_index"),
                                               old_safest=old, per_world_safest=best, security_gap=gap))
    wins = report["results"].get("win", 0)
    n = report["completed_games"]
    p = wins / n if n else None
    interval = None
    if n:
        z = 1.96
        center = (p + z*z/(2*n)) / (1+z*z/n)
        half = z * math.sqrt(p*(1-p)/n+z*z/(4*n*n)) / (1+z*z/n)
        interval = [center-half, center+half]
    rated = [r for r in rows.values() if r.get("rating")]
    return dict(completed_games=n, results=report["results"], win_rate=p, win_rate_wilson95=interval,
                rating_before=rated[0]["rating"]["before"] if rated else None,
                rating_after=rated[-1]["rating"]["after"] if rated else None,
                decisions=report["decisions"], matched_actions=report["matched_actions"],
                snapshot_decisions=snapshot_count, analyzed_games=report["analyzed_games"], missing_logs=report["missing_logs"],
                counts=report["counts"], opportunities=report["opportunities"],
                latency_mean=weighted_latency/total_decisions if total_decisions else None, latency_max=max_decision,
                games_per_stream=dict(Counter(str(r.get("learning_stream", "sequential")) for r in rows.values())),
                pure_safest_expert=dict(checked_decisions=checked, strict_mismatches=strict_mismatches,
                    comparison_tolerance=0.001,
                    mismatch_fraction=strict_mismatches/checked if checked else None,
                    mean_security_gap_on_mismatches=sum(security_gaps)/len(security_gaps) if security_gaps else None,
                    maximum_security_gap=max(security_gaps, default=0),
                    largest_examples=sorted(mismatch_examples, key=lambda r:-r["security_gap"])[:10]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out", type=Path)
    parser.add_argument("--username", default="bot_bonp")
    parser.add_argument("--write", action="store_true")
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    result = summarize(args.out, args.username, args.limit)
    if args.write:
        suffix = f"_first{args.limit}" if args.limit else ""
        (args.out / f"evaluation_summary{suffix}.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
