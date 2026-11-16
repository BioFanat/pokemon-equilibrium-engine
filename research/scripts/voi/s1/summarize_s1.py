"""Summarise run_s1.py output: E1, E2, E4 scored literally against research/info_gathering_concealment.md §4, CPU per
decision, and supplementary measures. Values are fractions of the decision's depth-2 payoff range (A0's K matrices,
max - min, as in §1 of the report); decisions with range 0 are dropped.

Usage: python summarize_s1.py [out/s1_eval.jsonl ...] [--json out/s1_summary.json]
"""
from __future__ import annotations

import argparse
import json
import os
from collections import Counter

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
MOVES = json.load(open(os.path.join(HERE, "..", "..", "..", "..", "engine", "data", "moves_gen9.json")))
PIVOT = {"uturn", "voltswitch", "flipturn", "partingshot", "teleport", "shedtail", "batonpass", "chillyreception"}
BUCKETS = [(1, 9, "1-9"), (10, 19, "10-19"), (20, 29, "20-29"), (30, 10 ** 6, "30+")]


def kind(label, own_species):
    if label.endswith("-tera"):
        return "tera"
    if label in own_species or label not in MOVES and label != "none":
        return "switch"
    if label in PIVOT:
        return "pivot"
    return "status" if MOVES.get(label, {}).get("category") == "status" else "attack"


def q(a, p):
    return float(np.quantile(a, p)) if len(a) else float("nan")


def stats(a):
    a = np.asarray(a, float)
    return dict(n=int(len(a)), mean=float(a.mean()) if len(a) else float("nan"), median=q(a, .5), p90=q(a, .9),
                max=float(a.max()) if len(a) else float("nan"), min=float(a.min()) if len(a) else float("nan"),
                frac_pos=float(np.mean(a > 1e-9)) if len(a) else float("nan"))


def load(paths):
    decs, games, pairs, errs = [], [], set(), 0
    for p in paths:
        for line in open(p):
            g = json.loads(line)
            pairs.add(g["pair"])
            games += g["games"]
            decs += [d for d in g["decisions"] if "A1_rep" in d and d.get("error") is None]
            errs += sum(d.get("error") is not None for d in g["decisions"])
    return decs, games, pairs, errs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="*", default=[os.path.join(HERE, "out", "s1_eval.jsonl")])
    ap.add_argument("--json", default=os.path.join(HERE, "out", "s1_summary.json"))
    a = ap.parse_args()
    decs, games, pairs, errs = load(a.paths)
    n_all = len(decs)
    decs = [d for d in decs if d["payoff_range"] > 0]
    S = dict(pairs=len(pairs), games=len(games), game_errors=sum(g["error"] is not None for g in games),
             decisions=n_all, decisions_range_pos=len(decs), decision_errors=errs)
    L = lambda f: np.array([f(d) for d in decs], float)
    am = lambda d, k: int(np.argmax(d[k]["x"]))
    r = lambda d: d["payoff_range"]

    # ---------------- E1
    E1 = {}
    for form in ("rep", "pw"):
        dv = L(lambda d: (d[f"A1_{form}"]["V"][am(d, f"A1_{form}")] - d[f"A2_{form}"]["V"][am(d, f"A1_{form}")]) / r(d))
        dmax = L(lambda d: max(np.array(d[f"A1_{form}"]["V"]) - np.array(d[f"A2_{form}"]["V"])) / r(d))
        dmin = L(lambda d: min(np.array(d[f"A1_{form}"]["V"]) - np.array(d[f"A2_{form}"]["V"])) / r(d))
        turn = L(lambda d: d["turn"])
        byb = {name: stats(dv[(turn >= lo) & (turn <= hi)]) for lo, hi, name in BUCKETS}
        E1[form] = dict(at_astar=stats(dv), max_over_a=stats(dmax), min_over_a=stats(dmin), by_turn=byb,
                        violations_astar=int((dv < -1e-9).sum()), violations_any_a=int((dmin < -1e-9).sum()))
    # per pair (one game per pair; the a_side=2 game replays it): cluster view and the largest game left out
    dv = L(lambda d: (d["A1_rep"]["V"][am(d, "A1_rep")] - d["A2_rep"]["V"][am(d, "A1_rep")]) / r(d))
    pid = np.array([d["pair"] for d in decs])
    sizes = Counter(pid.tolist())
    big = max(sizes, key=sizes.get) if sizes else None
    E1["per_pair"] = {str(k): dict(n=int(v), mean=float(dv[pid == k].mean())) for k, v in sorted(sizes.items())}
    E1["rep_without_largest_pair"] = dict(pair=big, **stats(dv[pid != big]))
    pm = np.array([dv[pid == k].mean() for k in sizes])
    E1["rep_pair_means"] = stats(pm)
    e = E1["rep"]
    early, late = e["by_turn"]["1-9"]["mean"], e["by_turn"]["30+"]["mean"]
    E1["score"] = {
        "nonneg at every decision (rep)": (e["violations_astar"] == 0, f"{e['violations_astar']} violations at a*, "
                                           f"{e['violations_any_a']} decisions with some a violated"),
        "mean in [0.5%, 1.5%]": (0.005 <= e["at_astar"]["mean"] <= 0.015, f"mean {100 * e['at_astar']['mean']:.4f}%"),
        "median <= 0.4%": (e["at_astar"]["median"] <= 0.004, f"median {100 * e['at_astar']['median']:.4f}%"),
        "p90 <= 4%": (e["at_astar"]["p90"] <= 0.04, f"p90 {100 * e['at_astar']['p90']:.4f}%"),
        "turns 1-9 >= 5x turns 30+": ((early >= 5 * late and early > 0), f"means {100 * early:.4f}% vs {100 * late:.4f}%"
                                      + (" (both zero: ratio undefined)" if early == 0 and late == 0 else "")),
    }

    # ---------------- E2
    E2 = {}
    for form in ("rep", "pw"):
        ch, kinds, regret = [], Counter(), []
        for d in decs:
            i1, i2 = am(d, f"A1_{form}"), am(d, f"A2_{form}")
            regret.append((d[f"A1_{form}"]["V"][i1] - d[f"A1_{form}"]["V"][i2]) / r(d))
            if i1 != i2:
                own = {l for l in d["legal"] if l not in MOVES and not l.endswith("-tera")}
                k1, k2 = kind(d["legal"][i1], own), kind(d["legal"][i2], own)
                ch.append((k1, k2))
                kinds["+".join(sorted({k1, k2}))] += 1
        nch = len(ch)
        tera = sum("tera" in c for c in ch)
        swst = sum(any(k in ("switch", "status", "pivot") for k in c) for c in ch)
        E2[form] = dict(changed=nch, frac_changed=nch / max(1, len(decs)), kinds=dict(kinds),
                        tera_involved=tera, switch_status_pivot_involved=swst,
                        regret_gt_1pct=float(np.mean(np.array(regret) > 0.01)))
    e2 = E2["rep"]
    E2["score"] = {
        "A1/A2 argmax differ in 5-12%": (0.05 <= e2["frac_changed"] <= 0.12, f"{100 * e2['frac_changed']:.2f}% "
                                         f"({e2['changed']}/{len(decs)}); regret>1%-weighted {100 * e2['regret_gt_1pct']:.2f}%"),
        "of those < 1% involve -tera": ((e2["tera_involved"] / e2["changed"] < 0.01) if e2["changed"] else None,
                                        f"{e2['tera_involved']}/{e2['changed']}"),
        "most involve switch or status/pivot": ((e2["switch_status_pivot_involved"] > e2["changed"] / 2) if e2["changed"] else None,
                                                f"{e2['switch_status_pivot_involved']}/{e2['changed']}"),
    }

    # ---------------- E4
    E4 = {}
    for kk in ("keys", "keys_report"):
        hist = {"move": Counter(), "switch": Counter()}
        for d in decs:
            for f, name in (("keys_per_cell_move", "move"), ("keys_per_cell_switch", "switch")):
                for h, c in d[kk][f]["hist"].items():
                    hist[name][int(h)] += c
        med = {}
        for name, h in hist.items():
            vals = np.repeat(list(h.keys()), list(h.values())) if h else np.array([])
            med[name] = dict(median=q(vals, .5), mean=float(vals.mean()) if len(vals) else float("nan"),
                             p90=q(vals, .9), n_cells=int(len(vals)))
        E4[kk] = dict(keys_per_cell=med,
                      merged_count=stats(L(lambda d: d[kk]["merged_frac_count"])),
                      merged_mass=stats(L(lambda d: d[kk]["merged_frac_mass"])),
                      strict_support_mass=stats(L(lambda d: d[kk]["strict_support_mass"])),
                      within_world=stats(L(lambda d: d[kk]["within_world_frac"])),
                      cells_missing_worlds=stats(L(lambda d: d[kk]["cells_missing_worlds"])),
                      pooled_merged_count=float(sum(d[kk]["merged_frac_count"] * d[kk]["n_nodes"] for d in decs)
                                                / max(1, sum(d[kk]["n_nodes"] for d in decs))))
    e4 = E4["keys"]
    E4["score"] = {
        "keys/cell median 2-3 for opponent move choices": (2 <= e4["keys_per_cell"]["move"]["median"] <= 3,
                                                           f"median {e4['keys_per_cell']['move']['median']}, mean {e4['keys_per_cell']['move']['mean']:.2f}"),
        "keys/cell median 1 for opponent switches": (e4["keys_per_cell"]["switch"]["median"] == 1,
                                                     f"median {e4['keys_per_cell']['switch']['median']}, mean {e4['keys_per_cell']['switch']['mean']:.2f}"),
        "children merged across worlds > 50%": (e4["pooled_merged_count"] > 0.5,
                                               f"pooled {100 * e4['pooled_merged_count']:.1f}% of children; per-decision mean "
                                               f"{100 * e4['merged_count']['mean']:.1f}% (count), {100 * e4['merged_mass']['mean']:.1f}% (mass)"),
        "conditional: if merged < 20% then E1 mean < 0.2%": (None if e4["pooled_merged_count"] >= 0.2 else
                                                             E1["rep"]["at_astar"]["mean"] < 0.002, "antecedent "
                                                             + ("false (merged >= 20%): not triggered" if e4["pooled_merged_count"] >= 0.2 else "true")),
    }

    # ---------------- CPU
    lp = lambda d, k: d["lp_secs"].get(k, 0.0)
    cpu = dict(
        A0=stats(L(lambda d: d["A0"]["cpu"])),
        A1=stats(L(lambda d: d["expand_cpu"] + lp(d, "shared") + lp(d, "A1") + d["root_secs"]["A1_rep"])),
        A2=stats(L(lambda d: d["expand_cpu"] + lp(d, "shared") + lp(d, "A2") + d["root_secs"]["A2_rep"])),
        expand_shared=stats(L(lambda d: d["expand_cpu"])),
        lp_shared=stats(L(lambda d: lp(d, "shared"))), lp_A1=stats(L(lambda d: lp(d, "A1"))),
        lp_A2=stats(L(lambda d: lp(d, "A2"))), lp_A0L=stats(L(lambda d: lp(d, "A0L"))),
        s1_analysis_all_arms=stats(L(lambda d: d["s1_cpu"])), s1_analysis_wall=stats(L(lambda d: d["s1_wall"])),
        d1_calls=stats(L(lambda d: d["d1_calls"])), nodes=stats(L(lambda d: d["keys"]["n_nodes"])),
        ratio_A1_over_A0=stats(L(lambda d: (d["expand_cpu"] + lp(d, "shared") + lp(d, "A1") + d["root_secs"]["A1_rep"])
                                 / max(d["A0"]["cpu"], 1e-6))))
    leaf = {k: int(sum(d["leaf"][k] for d in decs)) for k in ("calls", "non_d1", "fallback")}

    # ---------------- supplementary
    sup = {}
    sup["clairvoyance_gap_children_pw"] = stats(L(lambda d: (d["A0L"]["V"][am(d, "A1_pw")] - d["A1_pw"]["V"][am(d, "A1_pw")]) / r(d)))
    sup["clairvoyance_gap_root_value_pw"] = stats(L(lambda d: (d["A0L"]["v"] - d["A1_pw"]["v"]) / r(d)))
    sup["argmax_A0L_vs_A1pw"] = float(np.mean(L(lambda d: am(d, "A0L") != am(d, "A1_pw"))))
    sup["argmax_A0L_vs_A1rep"] = float(np.mean(L(lambda d: am(d, "A0L") != am(d, "A1_rep"))))
    sup["regret_gt1pct_A0L_vs_A1pw"] = float(np.mean(L(lambda d: (d["A1_pw"]["V"][am(d, "A1_pw")] - d["A1_pw"]["V"][am(d, "A0L")]) / r(d) > 0.01)))
    sup["argmax_A0_vs_A0L (leaf change)"] = float(np.mean(L(lambda d: am(d, "A0") != am(d, "A0L"))))
    sup["root_value_A0_minus_A0L (leaf change)"] = stats(L(lambda d: (d["A0"]["v"] - d["A0L"]["v"]) / r(d)))
    sup["argmax_A0_vs_A1rep"] = float(np.mean(L(lambda d: am(d, "A0") != am(d, "A1_rep"))))
    sup["argmax_A1rep_vs_A1pw"] = float(np.mean(L(lambda d: am(d, "A1_rep") != am(d, "A1_pw"))))
    sup["root_value_A1rep_minus_A1pw"] = stats(L(lambda d: (d["A1_rep"]["v"] - d["A1_pw"]["v"]) / r(d)))
    sup["groups_multi_world_mean"] = float(np.mean(L(lambda d: d["n_groups_multi"])))
    sup["groups_with_V1_gt_V2_frac"] = float(sum(d["groups_dV_pos"] for d in decs) / max(1, sum(d["n_groups_multi"] for d in decs)))
    sup["groups_with_row_alignment"] = float(np.mean(L(lambda d: d["groups_aligned"])))
    sup["payoff_range"] = stats(L(r))

    S.update(E1=E1, E2=E2, E4=E4, cpu=cpu, leaf=leaf, supplementary=sup)
    json.dump(S, open(a.json, "w"), indent=1, default=lambda o: o if not isinstance(o, np.generic) else o.item())

    pct = lambda x: f"{100 * x:.4f}%"
    print(f"pairs {S['pairs']}, games {S['games']} ({S['game_errors']} errors), decisions {n_all} "
          f"({len(decs)} with range > 0, {errs} analysis errors); leaf {leaf}")
    for form in ("rep", "pw"):
        s = E1[form]["at_astar"]
        print(f"E1[{form}] dV(a*)/range: mean {pct(s['mean'])} median {pct(s['median'])} p90 {pct(s['p90'])} "
              f"max {pct(s['max'])} min {pct(s['min'])} >0 in {100 * s['frac_pos']:.1f}% | violations a*: "
              f"{E1[form]['violations_astar']}, any a: {E1[form]['violations_any_a']}")
        print("   by turn: " + "; ".join(f"{k}: n={v['n']} mean {pct(v['mean'])} p90 {pct(v['p90'])}"
                                       for k, v in E1[form]["by_turn"].items()))
        print(f"   max over a: mean {pct(E1[form]['max_over_a']['mean'])} p90 {pct(E1[form]['max_over_a']['p90'])} max {pct(E1[form]['max_over_a']['max'])}")
        print(f"E2[{form}] {E2[form]}")
    print(f"E1[rep] per pair (n, mean%): " + ", ".join(f"{k}:{v['n']}/{100 * v['mean']:.4f}" for k, v in E1['per_pair'].items()))
    w = E1["rep_without_largest_pair"]
    print(f"E1[rep] without largest pair {w['pair']}: n {w['n']} mean {pct(w['mean'])} p90 {pct(w['p90'])}; "
          f"mean of pair means {pct(E1['rep_pair_means']['mean'])}")
    for kk in ("keys", "keys_report"):
        e = E4[kk]
        print(f"E4[{kk}] keys/cell {e['keys_per_cell']}; merged pooled {100 * e['pooled_merged_count']:.1f}% "
              f"(per-dec mean count {100 * e['merged_count']['mean']:.1f}%, mass {100 * e['merged_mass']['mean']:.1f}%); "
              f"strict-support mass mean {100 * e['strict_support_mass']['mean']:.1f}%; within-world {100 * e['within_world']['mean']:.1f}%; "
              f"cells missing worlds {100 * e['cells_missing_worlds']['mean']:.1f}%")
    print("CPU s/decision: " + "; ".join(f"{k} mean {v['mean']:.3f} med {v['median']:.3f} p90 {v['p90']:.3f}"
                                        for k, v in cpu.items()))
    print("supplementary:", json.dumps(sup, default=float))
    for name, sc in (("E1", E1["score"]), ("E2", E2["score"]), ("E4", E4["score"])):
        for k, (ok, txt) in sc.items():
            print(f"  {name} {'PASS' if ok else ('n/a ' if ok is None else 'FAIL')} {k}: {txt}")


if __name__ == "__main__":
    main()
