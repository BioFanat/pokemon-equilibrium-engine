"""Score foul-play's belief about the opponent's hidden sets on REAL Showdown (hidden-set accuracy and calibration).

Input: headless game records <OURBOT_WORK>/replays_headless/<tag>-s*-g*.json from a batch run with
OURBOT_LOG_WORLDS=1 (run_ourbot.belief_log adds a `belief` field to each of our decisions). The true teams are in
record["teams"]; the opponent is the seat that is not record["sides"]["A"].

  ~/sandbox/bot-venv/bin/python showdown/score_belief.py --work $OURBOT_WORK --tag BELIEF --out results/belief/showdown.json

Belief sources scored:
  fp_exact  count-weighted marginals over foul-play's remaining sets for each revealed mon (the distribution its worlds
            are drawn from; no sampling noise)
  fp_K4     the K worlds our agent actually solved over (weights as returned by prepare_battles: uniform 1/K)
Bench species use the OURBOT_LOG_WORLDS_EXTRA extra worlds (fp_extra) and the K worlds (fp_K4).

Events (one per decision x revealed alive opponent mon x attribute still unknown at that decision):
  item / ability / tera   categorical: p_truth = belief mass on the true value; hit = argmax == truth;
                          Brier = sum_v (p_v - y_v)^2; calibration pairs (p_v, y_v) for every value with p_v > 0
                          plus the truth.
  move                    one event per TRUE move not yet revealed: p_truth = P(move in set); hit = move among the
                          (4 - k) highest-probability unrevealed moves (k = moves revealed); calibration pairs
                          (P(m in set), 1[m in true set]) for every unrevealed candidate m and every true move;
                          Brier = mean binary Brier over those pairs.
  bench                   one event per true opponent mon not yet revealed: p_truth = fraction of worlds whose
                          unrevealed slots contain that species; baseline = unrevealed slots / 509 species.
Log loss = -ln max(p_truth, CLIP). Turn bins and k (moves revealed on that mon) stratify every metric.
The metric functions (summarize, calibration) are shared with engine/score_belief_engine.py.
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import re
from collections import defaultdict

import numpy as np

CLIP = 1e-3
N_SPECIES = 509
TURN_BINS = [(1, 5), (6, 10), (11, 20), (21, 30), (31, 10**9)]
NBINS = 10


def norm(s) -> str:
    return re.sub(r"[^a-z0-9]", "", str(s).lower()) if s is not None else ""


def turn_bin(t: int) -> str:
    for a, b in TURN_BINS:
        if a <= t <= b:
            return f"{a}-{b}" if b < 10**9 else f"{a}+"
    return "?"


# ---------------------------------------------------------------- event construction (shared)

def cat_event(src, attr, dist: dict, truth: str, **meta) -> dict:
    """Categorical attribute event. dist: value -> probability (normalised values)."""
    p = dist.get(truth, 0.0)
    am = max(dist, key=dist.get) if dist else None
    vals = {v for v, q in dist.items() if q > 0} | {truth}
    pairs = [(float(dist.get(v, 0.0)), int(v == truth)) for v in vals]
    brier = sum((q - y) ** 2 for q, y in pairs)
    return dict(src=src, attr=attr, p=float(p), hit=int(am == truth), brier=brier, pairs=pairs, **meta)


def move_events(src, pm: dict, true_moves: set, known: set, **meta) -> list:
    """pm: move -> P(move in set) (normalised ids). One event per true unrevealed move; pairs shared."""
    unk_true = true_moves - known
    if not unk_true:
        return []
    cand = {m for m, q in pm.items() if q > 0 and m not in known} | unk_true
    pairs = [(float(min(1.0, pm.get(m, 0.0))), int(m in true_moves)) for m in cand]
    brier = sum((q - y) ** 2 for q, y in pairs) / len(pairs)
    slots = max(0, 4 - len(known))
    ranked = sorted((m for m in pm if m not in known and pm[m] > 0), key=lambda m: -pm[m])[:slots]
    ev = []
    for i, m in enumerate(sorted(unk_true)):
        ev.append(dict(src=src, attr="move", p=float(min(1.0, pm.get(m, 0.0))), hit=int(m in ranked),
                       brier=brier, pairs=pairs if i == 0 else [], **meta))   # pairs counted once per mon-decision
    return ev


# ---------------------------------------------------------------- metrics (shared)

def calibration(pairs) -> dict:
    if not pairs:
        return dict(n=0, ece=None, bins=[])
    P = np.array([p for p, _ in pairs]); Y = np.array([y for _, y in pairs], float)
    idx = np.minimum((P * NBINS).astype(int), NBINS - 1)
    bins, ece = [], 0.0
    for b in range(NBINS):
        m = idx == b
        if not m.any():
            continue
        mp, fy = float(P[m].mean()), float(Y[m].mean())
        ece += m.sum() / len(P) * abs(mp - fy)
        bins.append(dict(lo=b / NBINS, hi=(b + 1) / NBINS, n=int(m.sum()), mean_p=round(mp, 4), freq=round(fy, 4)))
    return dict(n=len(P), ece=round(float(ece), 4), bins=bins)


def basic(evs) -> dict:
    if not evs:
        return dict(n=0)
    p = np.array([e["p"] for e in evs])
    out = dict(n=len(evs), n_games=len({e["game"] for e in evs}), n_clusters=len({cluster(e) for e in evs}),
               mean_p_truth=round(float(p.mean()), 4),
               acc=round(float(np.mean([e["hit"] for e in evs])), 4),
               logloss=round(float(-np.log(np.maximum(p, CLIP)).mean()), 4),
               brier=round(float(np.mean([e["brier"] for e in evs])), 4),
               frac_p0=round(float((p <= 0).mean()), 4))
    if "base" in evs[0]:
        out["baseline_p"] = round(float(np.mean([e["base"] for e in evs])), 4)
    return out


def cluster(e):
    """Independence unit: the team draw (engine events carry cluster = seed, shared by g1/g2), else the game."""
    return e.get("cluster", e["game"])


def bootstrap(evs, B=1000, seed=0) -> dict:
    """95% CI of acc and log loss, resampling clusters (decisions within a game / team draw are correlated)."""
    by = defaultdict(list)
    for e in evs:
        by[cluster(e)].append(e)
    games = list(by)
    if len(games) < 2:
        return {}
    rng = np.random.default_rng(seed)
    acc, ll = [], []
    for _ in range(B):
        s = [e for g in rng.choice(len(games), len(games)) for e in by[games[g]]]
        p = np.array([e["p"] for e in s])
        acc.append(np.mean([e["hit"] for e in s])); ll.append(-np.log(np.maximum(p, CLIP)).mean())
    q = lambda a: [round(float(np.percentile(a, 2.5)), 4), round(float(np.percentile(a, 97.5)), 4)]
    return dict(acc_ci=q(acc), logloss_ci=q(ll))


def ece_null(evs, attr, R=50, seed=0) -> list:
    """ECE (distinct belief states) of a PERFECTLY calibrated belief with the same states: the truth is redrawn from
    the belief itself (categorical: one value from the state's distribution; moves: each candidate independently
    with its probability). Returns [mean, 95th percentile] over R replicates: the ECE noise floor at this n."""
    rng = np.random.default_rng(seed)
    states, seen = [], set()
    for e in evs:
        if not e["pairs"]:
            continue
        key = (cluster(e), e.get("side"), e["mon"], tuple(sorted((round(p, 6), y) for p, y in e["pairs"])))
        if key not in seen:
            seen.add(key)
            states.append(np.array([p for p, _ in e["pairs"]]))
    out = []
    for _ in range(R):
        pairs = []
        for P in states:
            if attr == "move":
                Y = rng.random(len(P)) < P
            else:
                q = P / P.sum() if P.sum() > 0 else np.full(len(P), 1 / len(P))
                Y = np.eye(len(P), dtype=bool)[rng.choice(len(P), p=q)]
            pairs += list(zip(P.tolist(), Y.astype(int).tolist()))
        out.append(calibration(pairs)["ece"])
    return [round(float(np.mean(out)), 4), round(float(np.percentile(out, 95)), 4)]


def distinct_pairs(evs) -> list:
    """Calibration pairs counting each distinct belief state once per (game, side, mon): a mon's belief is repeated
    unchanged over many decisions, so decision-weighted bins are dominated by a few long-lived mons."""
    seen, out = set(), []
    for e in evs:
        if not e["pairs"]:
            continue
        key = (cluster(e), e.get("side"), e["mon"], tuple(sorted((round(p, 6), y) for p, y in e["pairs"])))
        if key not in seen:
            seen.add(key)
            out += e["pairs"]
    return out


def summarize(events) -> dict:
    """Per (src, attr): overall metrics + game-bootstrap CI + calibration, and by turn bin and by k."""
    groups = defaultdict(list)
    for e in events:
        groups[(e["src"], e["attr"])].append(e)
    out = {}
    for (src, attr), evs in sorted(groups.items()):
        d = basic(evs)
        d.update(bootstrap(evs))
        d["n_mon_decisions"] = len({(e["game"], e["dec"], e["mon"]) for e in evs})
        d["calibration"] = calibration([pr for e in evs for pr in e["pairs"]]) if attr != "bench" else None
        d["calibration_distinct"] = calibration(distinct_pairs(evs)) if attr != "bench" else None
        d["ece_null_distinct"] = ece_null(evs, attr) if attr != "bench" else None
        bt, bk = defaultdict(list), defaultdict(list)
        for e in evs:
            bt[turn_bin(e["turn"])].append(e); bk[e["k"]].append(e)
        order = [turn_bin(a) for a, _ in TURN_BINS]
        d["by_turn"] = {t: dict(basic(bt[t]), ece=calibration([pr for e in bt[t] for pr in e["pairs"]])["ece"])
                        for t in order if t in bt}
        d["by_k"] = {str(k): dict(basic(bk[k]), ece=calibration([pr for e in bk[k] for pr in e["pairs"]])["ece"])
                     for k in sorted(bk)}
        out[f"{src}/{attr}"] = d
    return out


# ---------------------------------------------------------------- Showdown extraction

def truth_lookup(team):
    return {norm(t["species"]): dict(species=norm(t["species"]), sid=norm(t.get("speciesId")), item=norm(t["item"]),
                                     ability=norm(t["ability"]), tera=norm(t.get("teraType")),
                                     moves={norm(m) for m in t["moves"]}) for t in team}


def match(name, truth: dict):
    """Truth entry for a foul-play species name: exact (species or speciesId), else longest prefix either way."""
    if name in truth:
        return truth[name]
    for t in truth.values():
        if t["sid"] == name:
            return t
    c = [t for k, t in truth.items() if name.startswith(k) or k.startswith(name)
         or name.startswith(t["sid"]) or t["sid"].startswith(name)]
    return max(c, key=lambda t: len(t["species"])) if c else None


def world_dists(worlds, name):
    """Weighted marginals over the worlds' copies of mon `name`: item/ability/tera dists, move inclusion."""
    W = sum(w["w"] for w in worlds) or 1.0
    d = dict(item=defaultdict(float), ability=defaultdict(float), tera=defaultdict(float), moves=defaultdict(float))
    found = 0.0
    for w in worlds:
        for q in w["opp"]:
            if q["name"] == name:
                c = w["w"] / W
                found += c
                d["item"][norm(q["item"])] += c
                d["ability"][norm(q["ability"])] += c
                d["tera"][norm(q["tera_type"])] += c
                for m in set(q["moves"]):
                    d["moves"][norm(m)] += c
                break
    return d if found > 0 else None


def exact_dists(ex):
    return dict(item={norm(k): v for k, v in ex["item"].items()}, ability={norm(k): v for k, v in ex["ability"].items()},
                tera={norm(k): v for k, v in ex["tera_type"].items()}, moves={norm(k): v for k, v in ex["moves"].items()})


def game_events(rec):
    opp = "p2" if rec["sides"]["A"] == "p1" else "p1"
    truth = truth_lookup(rec["teams"][opp])
    evs, stats = [], defaultdict(int)
    for di, d in enumerate(rec["decisions"].get("A", [])):
        b = d.get("belief")
        if not b or "error" in b:
            stats["decisions_no_belief"] += 1
            continue
        stats["decisions"] += 1
        meta0 = dict(game=rec["tag"], dec=di, turn=int(d.get("turn") or 0))
        revealed = set()
        for kn in b["known"]:
            t = match(kn["name"], truth)
            if t is None:
                stats["mon_unmatched"] += 1
                continue
            revealed.add(t["species"])
            if kn["fainted"] or any(q["name"] == kn["name"] and q["hp"] <= 0 for w in b["worlds"] for q in w["opp"]):
                continue                                  # dead (foul-play keeps hp 0 without always setting fainted)
            known_moves = {norm(m) for m in kn["moves"]}
            k = len(known_moves & t["moves"])
            meta = dict(meta0, mon=t["species"], k=min(k, 4))
            stats["mon_decisions"] += 1
            sources = [("fp_K4", world_dists(b["worlds"], kn["name"]))]
            if kn["name"] in b["exact"]:
                sources.append(("fp_exact", exact_dists(b["exact"][kn["name"]])))
            item_known = kn["item"] is not None or kn["removed_item"] is not None
            for src, dist in sources:
                if dist is None:
                    stats[f"{src}_mon_missing"] += 1
                    continue
                if not item_known:
                    evs.append(cat_event(src, "item", dist["item"], t["item"], **meta))
                if kn["ability"] is None:
                    evs.append(cat_event(src, "ability", dist["ability"], t["ability"], **meta))
                if not kn["terastallized"]:
                    evs.append(cat_event(src, "tera", dist["tera"], t["tera"], **meta))
                evs += move_events(src, dist["moves"], t["moves"], known_moves, **meta)
        # bench: true mons not yet revealed
        unrev = [s for s in truth if s not in revealed]
        for src, worlds in (("fp_extra", [w for w in b.get("extra", [])]),
                            ("fp_K4", [[(q["name"], q["revealed"]) for q in w["opp"]] for w in b["worlds"]])):
            if not worlds or not unrev:
                continue
            for s in unrev:
                hit = [any(not rv and match(n, {s: truth[s]}) is not None for n, rv in w) for w in worlds]
                p = float(np.mean(hit))
                evs.append(dict(src=src, attr="bench", p=p, hit=int(p >= 0.5), brier=(1 - p) ** 2, pairs=[],
                                base=len(unrev) / N_SPECIES, **dict(meta0, mon=s, k=6 - len(unrev))))
    return evs, stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", default=os.environ.get("OURBOT_WORK", os.path.expanduser("~/sandbox/ourbot")))
    ap.add_argument("--tag", default="BELIEF")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    files = sorted(f for f in glob.glob(os.path.join(a.work, "replays_headless", f"{a.tag}-s*-g*.json")))
    events, stats, games = [], defaultdict(int), 0
    for f in files:
        rec = json.load(open(f))
        if rec.get("error"):
            stats["error_games"] += 1
        e, s = game_events(rec)
        games += 1
        events += e
        for k, v in s.items():
            stats[k] += v
    res = dict(source="showdown/foul-play", tag=a.tag, games=games, stats=dict(stats), clip=CLIP,
               metrics=summarize(events))
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump(res, open(a.out, "w"), indent=1)
    print(f"{games} games, {stats['decisions']} decisions, {len(events)} events -> {a.out}")
    for key, d in res["metrics"].items():
        if d["n"]:
            print(f"  {key:18s} n={d['n']:6d} games={d['n_games']:3d} P(truth)={d['mean_p_truth']:.3f} "
                  f"acc={d['acc']:.3f} logloss={d['logloss']:.3f} brier={d['brier']:.3f} p0={d['frac_p0']:.3f} "
                  f"ECE={(d['calibration'] or {}).get('ece')} ECE_distinct={(d['calibration_distinct'] or {}).get('ece')} null={d['ece_null_distinct']}")


if __name__ == "__main__":
    main()
