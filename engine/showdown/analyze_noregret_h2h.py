"""Worst-case analysis of a hedge_mcts vs mcts_native headless batch: where does the no-regret learner lose?

Reads a batch.py work dir (<work>/replays_headless/<tag>-s<seed>-g<1|2>.{json,A.decisions.jsonl,B.decisions.jsonl})
and writes <work>/analysis_<tag>.json plus a short markdown summary to stdout.

Per decision of the hedge side it compares the played distribution with the Bayesian model's argmax (the mcts_top
expert), flags 'off-argmax' plays and 'low-share' plays (an action the belief-weighted search gave < LOW_SHARE of its
visits), and tags turns where the Bayesian-LP strategy is genuinely mixed (max prob < MIXED, our proxy for 'no single
move is optimal', the turns Proposition 9 charges deterministic play for). Per game it relates these rates to the
result; per stream it tracks Hedge's weights and realised regret against the anytime bound R(S).

Usage: python analyze_noregret_h2h.py <work_dir> <tag> [--hedge-role A]
"""
import argparse
import glob
import json
import math
import os
from collections import defaultdict

import numpy as np

LOW_SHARE = 0.10
MIXED = 0.95
EXPERTS = ["mcts_top", "mcts_sampled", "lp_sampled"]


def anytime_bound(S, n=len(EXPERTS)):
    """Hedge anytime regret bound (paper Thm 2): 1.5*sqrt(S/2*ln n) + sqrt(ln n / 8), in [0,1] reward units."""
    return 1.5 * math.sqrt(S / 2 * math.log(n)) + math.sqrt(math.log(n) / 8) if S else 0.0


def load_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def game_records(work, tag):
    for path in sorted(glob.glob(os.path.join(work, "replays_headless", f"{tag}-s*-g*.json"))):
        g = json.load(open(path))
        if g.get("error"):
            continue
        base = path[:-len(".json")]
        yield g, base


def decision_stats(recs):
    d = dict(n=0, off_argmax_mass=0.0, off_argmax_played=0, low_share_played=0, mixed_lp=0, mixed_lp_off=0,
             entropy=0.0)
    for r in recs:
        if "move" not in r or r.get("final") or not r.get("weights") or r["mcts"].get("forced"):
            continue
        legal, p = r["legal"], np.asarray(r["pfull"], float)
        shares = np.asarray(r["mcts"]["visit_policy"], float)
        Q = np.asarray(r["Q"], float)
        top = int(np.argmax(Q[0]))
        mv = r["move"][len("switch "):] if r["move"].startswith("switch ") else r["move"]   # raw switch -> label
        played = legal.index(mv) if mv in legal else None
        d["n"] += 1
        d["off_argmax_mass"] += float(1 - p[top])
        d["entropy"] += float(-(p[p > 0] * np.log(p[p > 0])).sum())
        if played is not None and played != top:
            d["off_argmax_played"] += 1
        if played is not None and shares[played] < LOW_SHARE:
            d["low_share_played"] += 1
        if Q[2].max() < MIXED:
            d["mixed_lp"] += 1
            if played is not None and played != top:
                d["mixed_lp_off"] += 1
    return d


def stream_regret(recs):
    """Realised regret on credited turns from logged feedback rewards: max_e sum r_e - sum p.r (p = pre-update weights)."""
    cum, got, S = np.zeros(len(EXPERTS)), 0.0, 0
    traj = []
    for r in recs:
        fb = r.get("feedback")
        if not fb or fb.get("skipped") or fb.get("r") is None or fb.get("weights") is None:
            continue
        rr, w = np.asarray(fb["r"], float), np.asarray(fb["weights"], float)
        cum += rr
        got += float(w @ rr)
        S += 1
        traj.append(w.tolist())
    return dict(S=S, regret=float(cum.max() - got) if S else 0.0, bound=anytime_bound(S),
                best_expert=EXPERTS[int(cum.argmax())] if S else None,
                final_weights=traj[-1] if traj else None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("work")
    ap.add_argument("tag")
    ap.add_argument("--hedge-role", default="A")
    a = ap.parse_args()
    games, streams = [], defaultdict(list)
    for g, base in game_records(a.work, a.tag):
        role = a.hedge_role
        hp = f"{base}.{role}.decisions.jsonl"
        if not os.path.exists(hp):
            continue
        recs = load_jsonl(hp)
        st = decision_stats(recs)
        won = g.get("winner_role") == role
        lost = g.get("winner_role") not in (None, role)
        games.append(dict(seed=g["seed"], game=g["game"], stream=g.get("stream"), turns=g.get("turns"),
                          won=won, lost=lost, **st))
        streams[g.get("stream")].append((g["seed"], g["game"], recs))
    if not games:
        raise SystemExit("no finished games found")
    # pair-level (two seat-swapped games per seed)
    by_seed = defaultdict(list)
    for x in games:
        by_seed[x["seed"]].append(x)
    pair_scores = [sum(z["won"] + 0.5 * (not z["won"] and not z["lost"]) for z in v) / len(v)
                   for v in by_seed.values()]
    m = float(np.mean(pair_scores))
    se = float(np.std(pair_scores, ddof=1) / math.sqrt(len(pair_scores))) if len(pair_scores) > 1 else float("nan")
    worst = sorted((s for s, v in by_seed.items() if all(z["lost"] for z in v) and len(v) == 2))

    def rate(rows, key):
        n = sum(r["n"] for r in rows)
        return (sum(r[key] for r in rows) / n) if n else float("nan")

    won_g, lost_g = [x for x in games if x["won"]], [x for x in games if x["lost"]]
    summary = dict(
        games=len(games), pairs=len(by_seed), hedge_score=m, ci95=[m - 1.96 * se, m + 1.96 * se],
        both_lost_seeds=worst,
        rates={k: dict(won=rate(won_g, k), lost=rate(lost_g, k))
               for k in ["off_argmax_played", "low_share_played", "mixed_lp"]},
        off_argmax_mass=dict(won=rate(won_g, "off_argmax_mass"), lost=rate(lost_g, "off_argmax_mass")),
        mixed_lp_off_share=(sum(x["mixed_lp_off"] for x in games) / max(1, sum(x["off_argmax_played"] for x in games))),
        turns=dict(won=float(np.mean([x["turns"] for x in won_g])) if won_g else None,
                   lost=float(np.mean([x["turns"] for x in lost_g])) if lost_g else None),
        streams={},
    )
    for s, lst in streams.items():
        lst.sort(key=lambda t: (t[0], t[1]))
        recs = [r for _, _, rs in lst for r in rs]
        summary["streams"][str(s)] = stream_regret(recs)
    out = os.path.join(a.work, f"analysis_{a.tag}.json")
    json.dump(dict(summary=summary, games=games), open(out, "w"), indent=1)
    s = summary
    print(f"# {a.tag}: hedge_mcts vs mcts_native\n")
    print(f"- {s['games']} games / {s['pairs']} paired seeds; hedge score {s['hedge_score']:.3f} "
          f"[{s['ci95'][0]:.3f}, {s['ci95'][1]:.3f}] (95% CI over pairs; 0.5 = even)")
    print(f"- seeds lost in both seats: {len(worst)} {worst[:12]}")
    for k, v in s["rates"].items():
        print(f"- {k} per decision: won games {v['won']:.3f}, lost games {v['lost']:.3f}")
    print(f"- off-argmax probability mass per decision: won {s['off_argmax_mass']['won']:.3f}, "
          f"lost {s['off_argmax_mass']['lost']:.3f}")
    print(f"- share of off-argmax plays on mixed-LP turns: {s['mixed_lp_off_share']:.3f}")
    print(f"- mean game length (turns): won {s['turns']['won']}, lost {s['turns']['lost']}")
    for k, v in s["streams"].items():
        print(f"- stream {k}: S={v['S']} credited turns, realised regret {v['regret']:.2f} vs bound {v['bound']:.2f}, "
              f"best expert {v['best_expert']}, final weights {v['final_weights']}")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
