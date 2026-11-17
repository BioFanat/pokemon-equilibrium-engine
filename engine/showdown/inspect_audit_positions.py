"""Inspect flagged snapshots offline, including modeled survival against the observed attack."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import poke_engine as pe
from audit_decisions import MOVES, modeled_values, toid


def find_record(out, finding):
    tag = finding["battle_tag"]
    path = out / "decisions" / (tag + ".jsonl")
    if not path.exists():
        path = out / "decisions.jsonl"
    for line in path.read_text().splitlines():
        r = json.loads(line)
        if (r.get("battle_tag") == tag and "move" in r and
                r.get("decision_index") == finding["decision_index"]):
            return r
    raise ValueError("Matching snapshot record not found")


def observed_attack(finding):
    for line in finding["actual_lines"]:
        p = line.split("|")
        if len(p) > 4 and p[1] == "move" and not any(x.startswith("[from]") for x in p[4:]):
            return toid(p[3])
    return None


def conditional_survival(record, own_action, attack):
    """Model outcome conditional on this attack, averaging only supporting worlds.

    Native chance branches use the engine's damage approximation, not Showdown's
    full 16-roll distribution. This is a model diagnostic, not hidden ground truth.
    """
    search = record["search"]
    support, ko, hp = 0.0, 0.0, 0.0
    for state_text, rows, cols, weight in zip(search["states"], search["rows"], search["cols"], search["weights"]):
        if own_action not in rows or attack not in cols:
            continue
        state = pe.State.from_string(state_text)
        index = int(state.side_one.active_index)
        support += weight
        for branch in pe.generate_instructions(state, own_action, attack):
            mon = state.apply_instructions(branch).side_one.pokemon[index]
            p = weight * branch.percentage / 100
            ko += p * (mon.hp <= 0)
            hp += p * mon.hp / mon.maxhp
    return dict(support_weight=support, own_active_ko_probability=ko / support if support else None,
                expected_own_active_hp_fraction=hp / support if support else None)


def inspect(out, finding):
    record = find_record(out, finding)
    search = record.get("search", {})
    legal = search.get("legal", [])
    chosen = record["move"].removeprefix("switch ")
    values = modeled_values(record, legal)
    prediction_ranking = sorted(values, key=lambda x: values[x].get("expected_vs_logged_prediction", float("-inf")), reverse=True)
    candidates = list(dict.fromkeys([chosen] + prediction_ranking[:5]))
    result = {k: finding[k] for k in ("battle_tag", "turn", "decision_index", "categories", "action", "classification")}
    result.update(own_critical_hit=finding.get("own_critical_hit"),
                  played_probability=record.get("pfull", [])[legal.index(chosen)] if chosen in legal and record.get("pfull") else None,
                  weights=record.get("weights"), experts=record.get("experts"),
                  values={x: values[x] for x in candidates if x in values})
    Q = record.get("Q", [])
    weights = record.get("weights", {})
    if chosen in legal and Q and len(Q) == len(weights):
        i = legal.index(chosen)
        result["selected_action_expert_contributions"] = {
            name: round(weight * row[i], 6) for (name, weight), row in zip(weights.items(), Q)}
        best_pure = max(values, key=lambda label: values[label]["weighted_pure_worst_case"])
        result["world_consistent_pure_safest"] = best_pure
        if "safest" in weights:
            j = list(weights).index("safest")
            original = sum(weight * row[i] for weight, row in zip(weights.values(), Q))
            result["shadow_probability_replacing_only_safest_expert"] = round(
                original + weights["safest"] * ((best_pure == chosen) - Q[j][i]), 6)
    attack = observed_attack(finding)
    if attack and search.get("states") and chosen in MOVES:
        result["conditional_on_observed_attack"] = dict(attack=attack, actions={
            action: conditional_survival(record, action, attack) for action in candidates
            if action.removesuffix("-tera") in MOVES})
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out", type=Path)
    parser.add_argument("--write", action="store_true")
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    suffix = f"_first{args.limit}" if args.limit else ""
    report = json.loads((args.out / f"tactical_audit{suffix}.json").read_text())
    details = []
    for finding in report["findings"]:
        try:
            details.append(inspect(args.out, finding))
        except (ValueError, KeyError, IndexError, RuntimeError) as exc:
            details.append(dict(battle_tag=finding["battle_tag"], turn=finding["turn"], error=str(exc)))
    if args.write:
        (args.out / f"tactical_details{suffix}.json").write_text(json.dumps(details, indent=2) + "\n")
    print(json.dumps(details, indent=2))


if __name__ == "__main__":
    main()
