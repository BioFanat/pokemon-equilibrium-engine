"""Offline ablation of pure-security selection and action-conditional Hedge credit.

Uses the first N completed ledger games. Choice and feedback comparisons keep
the logged strategies, weights, and observations fixed, isolating each change;
they are not a counterfactual battle rollout or a win-rate estimate. Rounded
logged Q and qhat_k are renormalized. Exact sampled payoff matrices are retained.
"""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import hidden_agents as ha
from exp3_live import expert_rewards
from summarize_tactical_run import decisions_for
import world_learning as wl


def row_distributions(rows):
    rows = np.asarray(rows, dtype=float)
    return rows / rows.sum(axis=1, keepdims=True)


def reward_winner_changes(a, b, tolerance=1e-5):
    """Strictly disjoint best-expert sets, excluding ties and rounding noise."""
    return not bool(set(np.flatnonzero(a >= a.max()-tolerance)) &
                    set(np.flatnonzero(b >= b.max()-tolerance)))


def summarize(out, limit=100):
    ledger = [json.loads(line) for line in (out / "ladder_games.jsonl").read_text().splitlines() if line.strip()]
    tags = list(dict.fromkeys(r["battle_tag"] for r in ledger if r.get("result") in {"win", "loss", "tie", "draw"}))[:limit]
    shared = {}
    path = out / "decisions.jsonl"
    if path.exists():
        for line in path.read_text().splitlines():
            r = json.loads(line)
            shared.setdefault(r.get("battle_tag"), []).append(r)
    counts, changes, choices, credit_examples, mixture_gains = Counter(), [], [], [], []
    for tag in tags:
        pending = None
        for r in decisions_for(out, tag, shared):
            fb = r.get("feedback")
            if fb is not None and pending is not None:
                previous = pending
                counts["feedback_records"] += 1
                if fb.get("skipped"):
                    counts["skipped_" + fb.get("reason", "unknown")] += 1
                else:
                    counts["old_scored_feedback"] += 1
                    search = previous["search"]
                    legal = search["legal"]
                    Ms = [ha.align(np.asarray(M), rows, legal) for M, rows in zip(search["matrices"], search["rows"])]
                    Q = row_distributions(previous["Q"])
                    # Stored prediction is pre-feedback, before the style update.
                    qk = [row_distributions([q])[0] for q in previous["bayes"]["qhat_k"]]
                    cond = wl.condition_on_action(search["cols"], search["weights"], qk, fb["b"])
                    if cond is None:
                        counts["newly_skipped_unsupported_exact_label"] += 1
                    else:
                        counts["exact_supported_feedback"] += 1
                        counts["multiple_supporting_worlds"] += len(cond["indices"]) > 1
                        k = fb["world"]
                        old_label = fb["bayes"]["col"]
                        j = search["cols"][k].index(old_label)
                        old_reward = expert_rewards(Ms[k], Q, j)
                        new_reward, utility = wl.conditional_expert_credit(Ms, Q, cond)
                        delta = float(np.max(np.abs(new_reward-old_reward)))
                        changed = reward_winner_changes(old_reward, new_reward)
                        counts["credit_changed_over_1e_5"] += delta > 1e-5
                        counts["best_credit_expert_strictly_changed"] += changed
                        counts["old_resolved_label_differs_from_exact"] += old_label != fb["b"]
                        counts["old_world_zero_conditional_weight"] += cond["weights"][k] == 0
                        changes.append(delta)
                        clipped, all_one = False, True
                        for kk, jj in cond["indices"].items():
                            raw = (Q @ Ms[kk][:, jj] - Ms[kk].min()) / 400
                            clipped |= bool(np.any(raw > 1 + 1e-8))
                            all_one &= bool(np.all(raw >= 1 - 1e-8))
                        counts["some_supported_world_expert_reward_clipped"] += clipped
                        counts["all_experts_saturated_in_all_supporting_worlds"] += all_one
                        counts["conditional_rewards_all_tied_but_raw_utilities_differ"] += bool(
                            np.ptp(new_reward) < 1e-8 and np.ptp(utility) > 1e-3)
                        if changed:
                            credit_examples.append(dict(battle_tag=tag, turn=previous["turn"],
                                decision_index=previous.get("decision_index"), observed=fb["b"],
                                old_resolved_label=old_label, old_world=k,
                                conditional_world_weights=cond["weights"].tolist(),
                                old_rewards=old_reward.tolist(), new_rewards=new_reward.tolist(),
                                expected_utilities=utility.tolist(), max_reward_change=delta))
            # Only multi-action decisions create a pending learner update.
            pending = r if r.get("Q") and r.get("search") and r.get("bayes", {}).get("qhat_k") else None
            if not pending:
                continue
            counts["decisions_with_expert_snapshots"] += 1
            search = r["search"]
            legal = search["legal"]
            Ms = [ha.align(np.asarray(M), rows, legal) for M, rows in zip(search["matrices"], search["rows"])]
            values = wl.pure_security_values(Ms, search["weights"])
            old = legal.index(r["experts"]["safest"])
            new = int(values.argmax())
            gap = float(values[new]-values[old])
            counts["safest_label_changed"] += old != new
            counts["strict_security_improvement_over_0_001"] += gap > .001
            Q = row_distributions(r["Q"])
            ew = np.asarray(list(r["weights"].values()))
            ew /= ew.sum()
            p_old = ew @ Q
            Q[2] = np.eye(len(legal))[new]
            p_new = ew @ Q
            w = wl.normalized_weights(search["weights"])
            mixture_gain = float(sum(wk * ((p_new @ M).min() - (p_old @ M).min()) for wk, M in zip(w, Ms)))
            mixture_gains.append(mixture_gain)
            counts["frozen_mixture_security_improved_over_0_001"] += mixture_gain > .001
            counts["frozen_mixture_security_worsened_over_0_001"] += mixture_gain < -.001
            if old != new:
                choices.append(dict(battle_tag=tag, turn=r["turn"], decision_index=r.get("decision_index"),
                    old_safest=legal[old], new_safest=legal[new], security_gain=gap,
                    mixture_tv=float(.5 * np.abs(p_new-p_old).sum()), mixture_security_change=mixture_gain,
                    played=r["move"], old_played_probability=float(p_old[legal.index(r["move"].removeprefix("switch "))]),
                    new_played_probability=float(p_new[legal.index(r["move"].removeprefix("switch "))])))
        counts["unpaired_final_pending_decisions"] += pending is not None
    return dict(semantics=wl.SEMANTICS, source=str(out.resolve()), games=len(tags), counts={k: int(v) for k, v in counts.items()},
                assumptions="Frozen logged expert strategies/weights/predictions; Q and qhat_k rounded to 4 decimals; not an on-policy experiment",
                mean_max_reward_change=float(np.mean(changes)) if changes else None,
                max_reward_change=max(changes, default=0),
                mean_frozen_mixture_security_change=float(np.mean(mixture_gains)) if mixture_gains else None,
                choice_changes=choices,
                credit_change_examples=sorted(credit_examples, key=lambda x:-x["max_reward_change"])[:20])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out", type=Path)
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--write", type=Path, required=True)
    args = parser.parse_args()
    if args.limit < 1:
        parser.error("--limit must be positive")
    report = summarize(args.out, args.limit)
    args.write.parent.mkdir(parents=True, exist_ok=True)
    args.write.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k not in {"choice_changes", "credit_change_examples"}}, indent=2))


if __name__ == "__main__":
    main()
