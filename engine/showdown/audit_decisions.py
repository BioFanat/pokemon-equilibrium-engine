"""Find replay-backed tactical warning signs, joined to saved decision snapshots.

These are review candidates, not a win-rate or counterfactual-error estimator.
Only completed ledger games are counted. No sockets or account access are used.
"""
import argparse
from collections import Counter, defaultdict
from copy import deepcopy
import json
from pathlib import Path
import re

ENGINE = Path(__file__).resolve().parents[1]
DEX = json.loads((ENGINE / "data/dex_gen9randbats.json").read_text())
MOVES = json.loads((ENGINE / "data/moves_gen9.json").read_text())
SETUP = {"swordsdance": ("atk",), "nastyplot": ("spa",), "calmmind": ("spa", "spd"),
         "bulkup": ("atk", "def"), "dragondance": ("atk", "spe"),
         "quiverdance": ("spa", "spd", "spe"), "irondefense": ("def",),
         "acidarmor": ("def",), "agility": ("spe",), "growth": ("atk", "spa")}
STAGES = dict(atk="attack_boost", defense="defense_boost", spa="special_attack_boost",
              spd="special_defense_boost", spe="speed_boost", accuracy="accuracy_boost", evasion="evasion_boost")
STAGES["def"] = STAGES.pop("defense")
TITLES = {
    "rocks_reflected": "Stealth Rock reflected by Magic Bounce",
    "rocks_already_present": "Stealth Rock attempted while already present",
    "toxic_into_steel": "Toxic failed into Steel",
    "ground_into_flying": "Ground attack met Flying immunity",
    "setup_at_cap": "Setup at all relevant boost caps",
    "setup_then_ko": "Setup followed by own KO that turn",
    "switch_into_revealed_hurricane": "Switch into previously revealed Hurricane",
    "risky_hurricane_switch": "Revealed Hurricane switch: weak target or at least half HP lost",
    "punch_over_knockoff": "Mach/Drain Punch selected with Knock Off available",
    "punch_into_ghost": "Mach/Drain Punch met Ghost immunity",
}


def toid(value):
    return re.sub(r"[^a-z0-9]", "", value.lower())


def hp_fraction(condition):
    value = condition.split()[0]
    if value == "0":
        return 0.0
    if "/" in value:
        hp, maximum = value.split("/", 1)
        return float(hp) / float(maximum)
    return None


def parse_replay(lines, username):
    """Own primary actions with the public state immediately before execution."""
    sides = {s: dict(active=None, mons={}, boosts={}, volatiles=[], rocks=False) for s in ("p1", "p2")}
    me, turn, actions = None, 0, []
    turn_start = deepcopy(sides)
    for index, line in enumerate(lines):
        p = line.split("|")
        if len(p) < 3:
            continue
        kind = p[1]
        if kind == "player" and len(p) > 3 and toid(p[3]) == toid(username):
            me = p[2]
        if kind == "turn":
            turn = int(p[2])
            turn_start = deepcopy(sides)
            continue
        seat = p[2][:2]
        if seat not in sides:
            continue
        side = sides[seat]
        if kind in {"switch", "drag"}:
            if seat == me and turn and kind == "switch":
                actions.append(dict(kind="switch", turn=turn, index=index, actor=p[2],
                                    move=toid(p[3].split(",")[0]), before=deepcopy(sides), turn_start=turn_start))
            species = toid(p[3].split(",")[0])
            mon = side["mons"].setdefault(p[2], dict(species=species, moves=[], types=[], ability=None))
            mon["species"] = species
            mon["types"] = [x.lower() for x in DEX.get(species, {}).get("types", [])]
            tera = re.search(r"tera:([A-Za-z]+)", p[3])
            if tera:
                mon["tera"] = tera[1].lower()
            if mon.get("tera") and mon["tera"] != "stellar":
                mon["types"] = [mon["tera"]]
            if seat == me and turn and kind == "switch":
                actions[-1]["switch_types"] = list(mon["types"])
                actions[-1]["switch_hp"] = hp_fraction(p[4]) if len(p) > 4 else None
            side.update(active=p[2], boosts={}, volatiles=[])
        elif kind == "move":
            primary = not any(x.startswith("[from]") for x in p[4:])
            if seat == me and primary:
                actions.append(dict(kind="move", turn=turn, index=index, actor=p[2], move=toid(p[3]),
                                    target=p[4] if len(p) > 4 else None, before=deepcopy(sides), turn_start=turn_start))
            mon = side["mons"].get(side["active"])
            if mon is not None and toid(p[3]) not in mon["moves"]:
                mon["moves"].append(toid(p[3]))
        elif kind in {"-boost", "-unboost", "-setboost"} and len(p) > 4:
            old = side["boosts"].get(p[3], 0)
            value = int(p[4])
            side["boosts"][p[3]] = value if kind == "-setboost" else max(-6, min(6, old + value * (1 if kind == "-boost" else -1)))
        elif kind == "-clearboost":
            side["boosts"] = {}
        elif kind == "-sidestart" and "Stealth Rock" in line:
            side["rocks"] = True
        elif kind == "-sideend" and "Stealth Rock" in line:
            side["rocks"] = False
        elif kind in {"-start", "-end"} and len(p) > 3:
            volatile = toid(p[3].removeprefix("move: "))
            if kind == "-start" and volatile not in side["volatiles"]:
                side["volatiles"].append(volatile)
            elif kind == "-end" and volatile in side["volatiles"]:
                side["volatiles"].remove(volatile)
        elif kind == "-terastallize" and len(p) > 3:
            mon = side["mons"].get(side["active"])
            if mon is not None:
                mon["tera"] = p[3].lower()
                if mon["tera"] != "stellar":
                    mon["types"] = [mon["tera"]]
        elif kind == "-ability" and len(p) > 3:
            mon = side["mons"].get(side["active"])
            if mon is not None:
                mon["ability"] = toid(p[3])
    for action in actions:
        stop = next((i for i in range(action["index"] + 1, len(lines)) if lines[i].startswith("|turn|")), len(lines))
        action["following"] = [line.split("|") for line in lines[action["index"] + 1:stop]]
        immediate = []
        for p in action["following"]:
            if len(p) > 1 and p[1] == "move":
                break
            immediate.append(p)
        action["immediate"] = immediate
    return me, actions


def snapshot_context(record):
    search = record.get("search", {})
    legal = search.get("legal", record.get("legal", []))
    out = dict(legal=legal, forced=len(legal) == 1 if legal else record.get("shape", [None])[0] == 1,
               snapshot=bool(search.get("states")))
    if not search.get("states"):
        return out
    try:
        import poke_engine as pe
        state = pe.State.from_string(search["states"][0])
        own, opp = state.side_one, state.side_two
        p, q = own.pokemon[int(own.active_index)], opp.pokemon[int(opp.active_index)]
        qtypes = [q.tera_type.lower()] if q.terastallized and q.tera_type.lower() != "stellar" else [str(t).lower() for t in q.types]
        out.update(own=toid(p.id), opponent=toid(q.id), opponent_types=qtypes,
                   opponent_ability=q.ability.lower(),
                   boosts={stat: getattr(own, attr) for stat, attr in STAGES.items()},
                   ability=p.ability.lower(), rocks=bool(opp.side_conditions.stealth_rock),
                   force_switch=bool(own.force_switch or p.hp <= 0),
                   volatiles=[str(v).lower() for v in own.volatile_statuses])
    except (ImportError, ValueError, AttributeError, RuntimeError) as exc:
        out["snapshot_decode_error"] = type(exc).__name__
    return out


def modeled_values(record, labels):
    search = record.get("search", {})
    matrices, rows, weights = (search.get(x, []) for x in ("matrices", "rows", "weights"))
    qk = record.get("bayes", {}).get("qhat_k", [])
    if not matrices or sum(weights) <= 0:
        return {}
    values = {}
    for label in labels:
        worst, expected = 0.0, 0.0
        for k, (M, names, weight) in enumerate(zip(matrices, rows, weights)):
            row = M[names.index(label)] if label in names else [min(col) for col in zip(*M)]
            weight /= sum(weights)
            worst += weight * min(row)
            if k < len(qk) and len(qk[k]) == len(row):
                expected += weight * sum(a * b for a, b in zip(row, qk[k]))
        values[label] = dict(weighted_pure_worst_case=round(worst, 4))
        if len(qk) == len(matrices):
            values[label]["expected_vs_logged_prediction"] = round(expected, 4)
    return values


def inspect_action(action, record, me):
    opp = "p2" if me == "p1" else "p1"
    own_public, opp_public = action["before"][me], action["before"][opp]
    own_mon = own_public["mons"].get(own_public["active"], {})
    opp_mon = opp_public["mons"].get(opp_public["active"], {})
    context = snapshot_context(record)
    legal = context["legal"]
    base_legal = {x.removesuffix("-tera") for x in legal}
    move = action["move"]
    flags = []
    hurricane = {}
    immediate, following = action["immediate"], action["following"]
    immune = any(len(p) > 2 and p[1] == "-immune" and p[2].startswith(opp) for p in immediate)
    actual_types = opp_mon.get("types", [])
    start_opp = action["turn_start"][opp]
    start_mon = start_opp["mons"].get(start_opp["active"], {})
    planned_opponent = context.get("opponent", start_mon.get("species", opp_mon.get("species")))
    switched = planned_opponent is not None and planned_opponent != opp_mon.get("species")
    if action["kind"] == "move":
        if move == "stealthrock":
            reflected = any(len(p) > 3 and p[1] == "move" and toid(p[3]) == move and "Magic Bounce" in "|".join(p) for p in following)
            if reflected:
                flags.append("rocks_reflected")
            placed = any(len(p) > 3 and p[1] == "-sidestart" and p[2].startswith(opp) and "Stealth Rock" in "|".join(p) for p in immediate)
            if context.get("rocks", opp_public["rocks"]) and not placed:
                flags.append("rocks_already_present")
        if move == "toxic" and immune and "steel" in actual_types:
            flags.append("toxic_into_steel")
        if MOVES.get(move, {}).get("type") == "ground" and immune and "flying" in actual_types:
            flags.append("ground_into_flying")
        if move in SETUP:
            boosts = context.get("boosts", own_public["boosts"])
            boosted = any(len(p) > 4 and p[1] == "-boost" and p[2] == action["actor"] and int(p[4]) > 0 for p in immediate)
            if all(boosts.get(s, 0) >= 6 for s in SETUP[move]) and not boosted and context.get("ability", own_mon.get("ability")) != "contrary":
                flags.append("setup_at_cap")
            if any(len(p) > 2 and p[1] == "faint" and p[2] == action["actor"] for p in following):
                flags.append("setup_then_ko")
        if move in {"machpunch", "drainpunch"}:
            if "knockoff" in base_legal:
                flags.append("punch_over_knockoff")
            if immune and "ghost" in actual_types:
                flags.append("punch_into_ghost")
    elif "hurricane" in opp_mon.get("moves", []):
        if any(len(p) > 4 and p[1] == "move" and p[2] == opp_public["active"] and toid(p[3]) == "hurricane" and p[4] == action["actor"] for p in following):
            flags.append("switch_into_revealed_hurricane")
            effectiveness = 1.0
            for t in action.get("switch_types", []):
                effectiveness *= 2 if t in {"bug", "fighting", "grass"} else 0.5 if t in {"electric", "rock", "steel"} else 1
            current_hp, after_hurricane, loss = action.get("switch_hp"), False, None
            for p in following:
                if len(p) > 4 and p[1] == "move":
                    after_hurricane = p[2] == opp_public["active"] and toid(p[3]) == "hurricane"
                if len(p) > 3 and p[1] in {"-damage", "-heal"} and p[2] == action["actor"]:
                    after = hp_fraction(p[3])
                    if after_hurricane and p[1] == "-damage" and not any(x.startswith("[from]") for x in p[4:]) and current_hp is not None and after is not None:
                        loss = current_hp - after
                        break
                    current_hp = after
            hurricane = dict(hurricane_effectiveness=effectiveness, hurricane_hp_loss=loss)
            if effectiveness > 1 or (loss is not None and loss >= 0.5):
                flags.append("risky_hurricane_switch")
    if not flags:
        return None
    locked = any(v in {"encore", "lockedmove"} for v in context.get("volatiles", own_public["volatiles"]))
    classification = "only_one_legal_action" if context["forced"] else "forced_replacement" if context.get("force_switch") else "move_locked_with_other_actions" if locked else "multiple_legal_actions" if legal else "legality_not_logged"
    chosen = record.get("move", move).removeprefix("switch ")
    alternatives = list(dict.fromkeys([chosen, "knockoff"] + list(record.get("mix", {}))))
    return dict(categories=flags, turn=action["turn"], decision_index=record.get("decision_index"),
                replay_line=action["index"] + 1, action=record.get("move", move),
                actor=action["move"] if action["kind"] == "switch" else context.get("own", own_mon.get("species")), actual_opponent=opp_mon.get("species"),
                planned_opponent=planned_opponent, opponent_switched=switched,
                classification=classification, legal=legal, snapshot=context["snapshot"],
                own_critical_hit=any(len(p) > 2 and p[1] == "-crit" and p[2] == action["actor"] for p in following),
                **hurricane,
                modeled_values=modeled_values(record, [x for x in alternatives if x in legal]),
                actual_lines=["|".join(p) for p in following[:16]])


def audit_game(lines, decisions, username):
    me, actions = parse_replay(lines, username)
    if me is None:
        return dict(error="username_not_found", findings=[], matched=0, decisions=0, snapshots=0)
    consumed, findings, matched, snapshots = set(), [], 0, 0
    opportunities = Counter()
    records = [r for r in decisions if "move" in r]
    for record in records:
        context = snapshot_context(record)
        available = {a.removesuffix("-tera") for a in context["legal"]}
        if "stealthrock" in available:
            opportunities["stealthrock_selectable_decisions"] += 1
            if context.get("rocks"):
                opportunities["stealthrock_selectable_with_rocks_present"] += 1
            if context.get("opponent_ability") == "magicbounce":
                opportunities["stealthrock_selectable_vs_sampled_magicbounce"] += 1
        if "toxic" in available:
            opportunities["toxic_selectable_decisions"] += 1
            if "steel" in context.get("opponent_types", []) and context.get("ability") != "corrosion":
                opportunities["toxic_selectable_vs_steel_without_corrosion"] += 1
        if "knockoff" in available and available & {"machpunch", "drainpunch"}:
            opportunities["punch_and_knockoff_both_selectable"] += 1
        if any(m in SETUP and all(context.get("boosts", {}).get(s, 0) >= 6 for s in SETUP[m]) for m in available) and context.get("ability") != "contrary":
            opportunities["capped_setup_selectable_decisions"] += 1
        label = record["move"].removesuffix("-tera").removeprefix("switch ")
        kind = "switch" if record["move"].startswith("switch ") else "move"
        hit = next((i for i, a in enumerate(actions) if i not in consumed and a["turn"] == record["turn"] and a["kind"] == kind and a["move"] == label), None)
        if hit is None:
            continue  # KO before acting, paralysis, sleep, and cancellations are not executed moves.
        consumed.add(hit)
        matched += 1
        snapshots += bool(record.get("search", {}).get("states"))
        action = actions[hit]
        if action["kind"] == "switch":
            opportunities["switch_actions"] += 1
        else:
            opportunities["move_actions"] += 1
            if action["move"] == "stealthrock":
                opportunities["stealthrock_uses"] += 1
            if action["move"] == "toxic":
                opportunities["toxic_uses"] += 1
            if action["move"] in SETUP:
                opportunities["setup_uses"] += 1
            if MOVES.get(action["move"], {}).get("type") == "ground":
                opportunities["ground_uses"] += 1
        finding = inspect_action(action, record, me)
        if finding:
            findings.append(finding)
    return dict(findings=findings, matched=matched, decisions=len(records), snapshots=snapshots,
                unexecuted_or_unmatched=len(records) - matched, opportunities=dict(opportunities))


def audit_directory(out, username, limit=None):
    ledger = out / "ladder_games.jsonl"
    games = {r["battle_tag"]: r for r in (json.loads(line) for line in ledger.read_text().splitlines() if line.strip())} if ledger.exists() else {}
    if limit is not None:
        games = dict(list(games.items())[:limit])
    shared = defaultdict(list)
    if (out / "decisions.jsonl").exists():
        for line in (out / "decisions.jsonl").read_text().splitlines():
            r = json.loads(line)
            if r.get("battle_tag"):
                shared[r["battle_tag"]].append(r)
    findings, coverage, results, missing = [], [], Counter(), []
    for tag, game in games.items():
        if game.get("result") not in {"win", "loss", "tie", "draw"}:
            continue
        results[game["result"]] += 1
        replay = out / "replays" / (tag + ".log")
        per_game = out / "decisions" / (tag + ".jsonl")
        if not replay.exists() or (not per_game.exists() and tag not in shared):
            missing.append(tag)
            continue
        decisions = [json.loads(line) for line in per_game.read_text().splitlines() if line.strip()] if per_game.exists() else shared[tag]
        result = audit_game(replay.read_text().splitlines(), decisions, username)
        for finding in result.pop("findings"):
            findings.append(dict(battle_tag=tag, **finding))
        coverage.append(dict(battle_tag=tag, **result))
    counts = Counter(c for f in findings for c in f["categories"])
    free = Counter(c for f in findings if f["classification"] == "multiple_legal_actions" for c in f["categories"])
    opportunities = Counter()
    for game in coverage:
        opportunities.update(game.get("opportunities", {}))
    return dict(completed_games=sum(results.values()), results=dict(results), analyzed_games=len(coverage),
                missing_logs=missing, decisions=sum(x["decisions"] for x in coverage),
                matched_actions=sum(x["matched"] for x in coverage), snapshots=sum(x["snapshots"] for x in coverage),
                counts={k: counts[k] for k in TITLES}, multiple_action_counts={k: free[k] for k in TITLES},
                opportunities=dict(opportunities), findings=findings, coverage=coverage)


def markdown(report):
    lines = ["# Tactical decision audit", "", f"Completed games: **{report['completed_games']}**; analyzed: **{report['analyzed_games']}**.",
             f"Results: {report['results']}. Matched executed actions: {report['matched_actions']}/{report['decisions']}; {report['snapshots']} have state snapshots.",
             f"Executed-action opportunities: {report.get('opportunities', {})}.",
             "", "Counts identify review candidates. An immune hit after an opponent switch is a failed prediction; a forced move is separated from a free choice. Punch selection alone may be correct because of priority or healing.",
             "", "| Candidate | Occurrences | Multiple legal actions |", "| --- | ---: | ---: |"]
    for key, title in TITLES.items():
        lines.append(f"| {title} | {report['counts'][key]} | {report['multiple_action_counts'][key]} |")
    for key, title in TITLES.items():
        examples = [f for f in report["findings"] if key in f["categories"]][:8]
        if not examples:
            continue
        lines += ["", "## " + title, ""]
        for f in examples:
            lines += [f"- `{f['battle_tag']}` turn {f['turn']}, replay line {f['replay_line']}: **{f['action']}**; {f['actor']} vs {f['actual_opponent']}. {f['classification']}; opponent switched: {f['opponent_switched']}.",
                      f"  Modeled values: `{json.dumps(f['modeled_values'], sort_keys=True)}`."]
    if report["missing_logs"]:
        lines += ["", "Missing log coverage: " + ", ".join(report["missing_logs"])]
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out", type=Path)
    parser.add_argument("--username", default="bot_bonp")
    parser.add_argument("--write", action="store_true")
    parser.add_argument("--limit", type=int, help="freeze the first N completed ledger games")
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    report = audit_directory(args.out, args.username, args.limit)
    if args.write:
        suffix = f"_first{args.limit}" if args.limit else ""
        (args.out / f"tactical_audit{suffix}.json").write_text(json.dumps(report, indent=2) + "\n")
        (args.out / f"tactical_audit{suffix}.md").write_text(markdown(report))
    print(json.dumps({k: v for k, v in report.items() if k not in {"findings", "coverage"}}, indent=2))


if __name__ == "__main__":
    main()
