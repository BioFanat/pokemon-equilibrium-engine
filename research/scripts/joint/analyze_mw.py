"""Analysis of the pre-registered live test of hedge_bayes_mw (research/prereg_mw.md, §3 predictions, §4 analysis).

Reads only the fields the prereg names: per game the game JSON (winner_role, error, sim_errors, stream, timing.A) and
A's decisions.jsonl (feedback.skipped, feedback.bayes.{p_qhat, p_qhat_mw, forced, posterior}); per stream the final
style posterior <work>/hedge_state/<tag>-stream<k>-A.bayes.json (logp), else the last logged posterior.
  * scored turn: a feedback record with skipped == False and a feedback.bayes dict (forced turns included, counted);
    the feedback in record i+1 belongs to decision i (record order; turn numbers repeat, so they never key it)
  * a game counts only once its JSON parses (a running game has a growing jsonl and no JSON); a jsonl line that fails
    to parse is dropped only when it is the last non-empty line (half-written at EOF), anywhere else it is an error
  * error games (error set) are dropped from every metric; seeds pair only when both arms have both games without error
    (batch.py rule; sim_errors are reported, not dropped)
  * L1/L3: d = -ln p_qhat - (-ln p_qhat_mw) per scored turn; mean, game-clustered SE (CR0 x G/(G-1)), t(G-1) 95% CI
  * L2: mean -ln p_qhat (BHs) - mean -ln p_qhat_mw (MWs), i.e. how much lower the MW loss is (positive = MW lower);
    SE = sqrt(se_MWs^2 + se_BHs^2) from the game-clustered SEs, normal 95% CI (unpaired)
  * R1: final MAP style per MWeq stream (argmax logp); count of eq; stream-bootstrap 95% interval of the count; RQeq ref
  * W1/W2: paired seed score difference (seed score = mean of A's two game scores), normal 95% CI
  * C1: pooled CPU-s per decision of A, sum(timing.A.cpu_s) / sum(timing.A.decisions) over non-error games
Status per prediction: PASS / FAIL by the prereg's refutation clause, UNTESTABLE if an arm it uses is missing or
incomplete, NO_CRITERION where the prereg states no refutation clause (W2, C1); in_interval reports the predicted range.
Usage: ~/sandbox/bot-venv/bin/python research/scripts/joint/analyze_mw.py [--partial] [--mws DIR] [--bhs DIR]
       [--mweq DIR] [--rqeq DIR] [--out-dir DIR]     (DIR = an arm's work dir holding replays_headless/, hedge_state/)
"""
import argparse
import glob
import json
import math
import os
import re
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RESEARCH = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(RESEARCH, "..", "engine", "analysis", "open_items"))
from oi_common import cluster_ci, mean_ci_normal, score  # noqa: E402

SANDBOX = os.path.expanduser("~/sandbox/ourbot")
DEFAULT_DIRS = dict(
    MWs=os.path.join(SANDBOX, "mw_MWs"), BHs=os.path.join(SANDBOX, "mw_BHs"), MWeq=os.path.join(SANDBOX, "mw_MWeq"),
    RQeq="/Users/biofanat/Documents/classes/f26/cs535/final/engine/showdown/results/RQeq")
SEEDS = dict(MWs=(1, 200), BHs=(1, 200), MWeq=(1, 100), RQeq=(1, 100))   # prereg §2 grid, games 1 and 2 per seed
NSTREAMS = 10
P_FLOOR = 1e-6                     # logged p's are rounded to 6 decimals; -ln of a logged 0 is floored here (counted)
FNRE = re.compile(r"^(?P<tag>.+)-s(?P<seed>\d+)-g(?P<game>[12])\.json$")


def read_jsonl(path):
    """Records of a decisions.jsonl; a parse failure on the last non-empty line (half-written) is dropped -> (recs, 0/1)."""
    lines = [l for l in open(path) if l.strip()]
    recs = []
    for i, l in enumerate(lines):
        try:
            recs.append(json.loads(l))
        except json.JSONDecodeError:
            if i == len(lines) - 1:
                return recs, 1
            raise
    return recs, 0


def scored_turns(recs):
    """Scored feedback of A's decisions in record order: list of (decision index, feedback.bayes)."""
    out = []
    for i in range(1, len(recs)):
        fb = recs[i].get("feedback")
        if fb and not fb.get("skipped") and isinstance(fb.get("bayes"), dict):
            out.append((i - 1, fb["bayes"]))
    return out


def load_arm(tag, work):
    """All finished games of an arm -> dict with games {(seed, game): {json, turns}} and loading diagnostics."""
    rdir = os.path.join(work, "replays_headless")
    arm = dict(tag=tag, work=work, exists=os.path.isdir(rdir), games={}, unparsable=[], truncated_jsonl=0,
               missing_jsonl=[], n_expected=2 * (SEEDS[tag][1] - SEEDS[tag][0] + 1), missing=None, complete=False,
               error_games=[], sim_error_games=[])
    if not arm["exists"]:
        return arm
    for p in sorted(glob.glob(os.path.join(rdir, f"{tag}-s*-g[12].json"))):
        m = FNRE.match(os.path.basename(p))
        if not m or m["tag"] != tag:
            continue
        key = (int(m["seed"]), int(m["game"]))
        try:
            rec = json.load(open(p))
        except (json.JSONDecodeError, OSError):
            arm["unparsable"].append(f"s{key[0]}-g{key[1]}")
            continue
        jl = p[:-len(".json")] + ".A.decisions.jsonl"
        recs = []
        if os.path.exists(jl):
            recs, trunc = read_jsonl(jl)
            arm["truncated_jsonl"] += trunc
        else:
            arm["missing_jsonl"].append(f"s{key[0]}-g{key[1]}")
        arm["games"][key] = dict(json=rec, recs=recs)
    lo, hi = SEEDS[tag]
    expect = {(s, g) for s in range(lo, hi + 1) for g in (1, 2)}
    arm["n_expected"] = len(expect)
    arm["missing"] = len(expect - set(arm["games"]))
    arm["complete"] = arm["missing"] == 0 and not arm["missing_jsonl"] and not arm["truncated_jsonl"]
    arm["error_games"] = sorted(f"s{s}-g{g}" for (s, g), d in arm["games"].items() if d["json"].get("error"))
    arm["sim_error_games"] = sorted(f"s{s}-g{g}" for (s, g), d in arm["games"].items() if d["json"].get("sim_errors"))
    return arm


def ok_games(arm):
    return {k: d for k, d in arm["games"].items() if not d["json"].get("error")}


def nll(p, floored):
    p = float(p)
    if p < P_FLOOR:
        floored[0] += 1
        p = P_FLOOR
    return -math.log(p)


def turn_values(arm, fn):
    """fn(bayes, floored) per scored turn of every non-error game -> (values, cluster ids, n forced, n floored)."""
    x, g, forced, floored = [], [], 0, [0]
    for key, d in sorted(ok_games(arm).items()):
        for _, b in scored_turns(d["recs"]):
            x.append(fn(b, floored))
            g.append(f"{key[0]}-{key[1]}")
            forced += bool(b.get("forced"))
    return x, g, forced, floored[0]


def clustered(x, g):
    """Game-clustered mean, SE and t 95% CI (None where undefined)."""
    n, G = len(x), len(set(g))
    if n == 0:
        return dict(mean=None, se=None, ci95=None, n_turns=0, n_games=0)
    if G < 2:
        return dict(mean=float(np.mean(x)), se=None, ci95=None, n_turns=n, n_games=G)
    m, se, lo, hi, _ = cluster_ci(x, g)
    return dict(mean=m, se=se, ci95=[lo, hi], n_turns=n, n_games=G)


def paired_turn_metric(arm):
    x, g, forced, floored = turn_values(arm, lambda b, f: nll(b["p_qhat"], f) - nll(b["p_qhat_mw"], f))
    return dict(clustered(x, g), n_forced=forced, n_floored=floored)


def loss_metric(arm, field):
    x, g, forced, floored = turn_values(arm, lambda b, f: nll(b[field], f))
    return dict(clustered(x, g), n_forced=forced, n_floored=floored)


def seed_scores(arm):
    """seed -> mean of A's two game scores, for seeds with both games present and neither an error game."""
    games = arm["games"]
    out = {}
    for s in sorted({s for s, _ in games}):
        recs = [games[(s, g)]["json"] for g in (1, 2) if (s, g) in games]
        if len(recs) == 2 and not any(r.get("error") for r in recs):
            out[s] = sum(score(r, "A") for r in recs) / 2
    return out


def paired_score_diff(a, b):
    sa, sb = seed_scores(a), seed_scores(b)
    common = sorted(set(sa) & set(sb))
    diffs = [sa[s] - sb[s] for s in common]
    res = dict(n_pairs=len(diffs), error_games={a["tag"]: len(a["error_games"]), b["tag"]: len(b["error_games"])},
               sim_error_games={a["tag"]: len(a["sim_error_games"]), b["tag"]: len(b["sim_error_games"])},
               mean=None, se=None, ci95=None)
    if diffs:
        res["mean"] = float(np.mean(diffs))
    if len(diffs) > 1:
        m, se, lo, hi, _ = mean_ci_normal(diffs)
        res.update(se=se, ci95=[lo, hi])
    return res


def cpu_per_decision(arm):
    cpu = dec = 0.0
    for d in ok_games(arm).values():
        t = d["json"].get("timing", {}).get("A", {})
        cpu += t.get("cpu_s", 0.0)
        dec += t.get("decisions", 0)
    return dict(cpu_s=cpu, decisions=int(dec), cpu_s_per_decision=cpu / dec if dec else None,
                n_games=len(ok_games(arm)))


def final_map(arm):
    """Per stream: final MAP style from hedge_state (argmax logp), else the last logged posterior in play order."""
    out = {}
    streams = {}
    for (s, g), d in arm["games"].items():
        k = d["json"].get("stream")
        if k is not None:
            streams.setdefault(int(k), []).append((s, g))
    for k in range(NSTREAMS):
        p = os.path.join(arm["work"], "hedge_state", f"{arm['tag']}-stream{k}-A.bayes.json")
        try:
            st = json.load(open(p))
            styles = st["config"]["styles"]
            out[k] = dict(map=styles[int(np.argmax(st["logp"]))], source="hedge_state")
            continue
        except (OSError, json.JSONDecodeError, KeyError, ValueError):
            pass
        post = None
        for key in sorted(streams.get(k, [])):
            for _, b in scored_turns(arm["games"][key]["recs"]):
                post = b.get("posterior", post)
        out[k] = dict(map=max(post, key=post.get) if post else None, source="last_logged" if post else "none")
    return out


def eq_count(maps, n_boot=10000, seed=0):
    vals = np.array([m["map"] == "eq" for m in maps.values() if m["map"] is not None], float)
    res = dict(n_streams=int(len(vals)), n_eq=int(vals.sum()), boot95=None)
    if len(vals):
        rng = np.random.default_rng(seed)
        c = vals[rng.integers(0, len(vals), (n_boot, len(vals)))].sum(1)
        res["boot95"] = [float(np.percentile(c, 2.5)), float(np.percentile(c, 97.5))]
    return res


def ready(*arms):
    return all(a["exists"] and a["complete"] for a in arms)


def inside(v, lo, hi):
    return None if v is None else bool(lo <= v <= hi)


def status(arms, refuted):
    """UNTESTABLE unless every arm is complete; else FAIL if the refutation clause holds, PASS otherwise."""
    if not ready(*arms) or refuted is None:
        return "UNTESTABLE"
    return "FAIL" if refuted else "PASS"


def analyse(dirs):
    arms = {t: load_arm(t, dirs[t]) for t in ("MWs", "BHs", "MWeq", "RQeq")}
    MWs, BHs, MWeq, RQeq = (arms[t] for t in ("MWs", "BHs", "MWeq", "RQeq"))
    pred = {}

    l1 = paired_turn_metric(MWs)
    ref = None if l1["ci95"] is None else (l1["mean"] < 0.15 or l1["ci95"][0] <= 0 <= l1["ci95"][1])
    pred["L1"] = dict(l1, arms=["MWs"], interval=[0.25, 0.40], in_interval=inside(l1["mean"], 0.25, 0.40),
                      criterion="refuted if mean < 0.15 or game-clustered 95% CI includes 0",
                      status=status([MWs], ref))

    a, b = loss_metric(MWs, "p_qhat_mw"), loss_metric(BHs, "p_qhat")
    l2 = dict(mws_loss_mw=a, bhs_loss_avg=b, mean=None, se=None, ci95=None)
    if a["mean"] is not None and b["mean"] is not None:
        l2["mean"] = b["mean"] - a["mean"]
        if a["se"] is not None and b["se"] is not None:
            l2["se"] = math.sqrt(a["se"] ** 2 + b["se"] ** 2)
            l2["ci95"] = [l2["mean"] - 1.96 * l2["se"], l2["mean"] + 1.96 * l2["se"]]
    pred["L2"] = dict(l2, arms=["MWs", "BHs"], interval=[0.20, 0.45], in_interval=inside(l2["mean"], 0.20, 0.45),
                      criterion="refuted if BHs avg-loss minus MWs mw-loss < 0.10",
                      status=status([MWs, BHs], None if l2["mean"] is None else l2["mean"] < 0.10))

    l3 = paired_turn_metric(MWeq)
    pred["L3"] = dict(l3, arms=["MWeq"], interval=[0.25, 0.45], in_interval=inside(l3["mean"], 0.25, 0.45),
                      criterion="refuted if mean < 0.15",
                      status=status([MWeq], None if l3["mean"] is None else l3["mean"] < 0.15))

    maps_mw = final_map(MWeq) if MWeq["exists"] else {}
    maps_rq = final_map(RQeq) if RQeq["exists"] else {}
    r1 = dict(eq_count(maps_mw), per_stream={str(k): v for k, v in maps_mw.items()},
              reference_RQeq=dict(eq_count(maps_rq), per_stream={str(k): v for k, v in maps_rq.items()}))
    pred["R1"] = dict(r1, arms=["MWeq", "RQeq"], interval=[0, 4], in_interval=inside(r1["n_eq"], 0, 4)
                      if r1["n_streams"] == NSTREAMS else None,
                      criterion="refuted if eq is the final MAP in >= 8 of 10 MWeq streams",
                      status=status([MWeq, RQeq], None if r1["n_streams"] < NSTREAMS else r1["n_eq"] >= 8))

    w1 = paired_score_diff(MWs, BHs)
    pred["W1"] = dict(w1, arms=["MWs", "BHs"], interval=[-0.03, 0.03], in_interval=inside(w1["mean"], -0.03, 0.03),
                      criterion="refuted only if the 95% CI excludes 0 in the negative direction",
                      status=status([MWs, BHs], None if w1["ci95"] is None else w1["ci95"][1] < 0))

    w2 = paired_score_diff(MWeq, RQeq)
    pred["W2"] = dict(w2, arms=["MWeq", "RQeq"], interval=[-0.05, 0.05], in_interval=inside(w2["mean"], -0.05, 0.05),
                      criterion="none stated (descriptive)",
                      status="NO_CRITERION" if ready(MWeq, RQeq) else "UNTESTABLE")

    cm, cb = cpu_per_decision(MWs), cpu_per_decision(BHs)
    rel = None
    if cm["cpu_s_per_decision"] is not None and cb["cpu_s_per_decision"]:
        rel = cm["cpu_s_per_decision"] / cb["cpu_s_per_decision"] - 1
    pred["C1"] = dict(MWs=cm, BHs=cb, rel_diff=rel, arms=["MWs", "BHs"], interval=[-0.10, 0.10],
                      in_interval=inside(rel, -0.10, 0.10), criterion="none stated (prediction: within 10%)",
                      status="NO_CRITERION" if ready(MWs, BHs) else "UNTESTABLE")

    arm_info = {t: dict(work=a["work"], exists=a["exists"], complete=a.get("complete", False),
                        games_parsed=len(a["games"]), games_expected=a.get("n_expected"), missing=a.get("missing"),
                        unparsable=a["unparsable"], truncated_jsonl=a["truncated_jsonl"],
                        missing_jsonl=a["missing_jsonl"], error_games=a.get("error_games", []),
                        sim_error_games=a.get("sim_error_games", []))
                for t, a in arms.items()}
    return dict(arms=arm_info, predictions=pred)


def fmt(v, nd=4):
    if v is None:
        return "-"
    if isinstance(v, list):
        return "[" + ", ".join(fmt(u, nd) for u in v) + "]"
    if isinstance(v, float):
        return f"{v:.{nd}f}"
    return str(v)


def to_md(res, title):
    L = [f"# {title}", "", "Pre-registration: research/prereg_mw.md. Generated by research/scripts/joint/analyze_mw.py.",
         "", "## Arms", "",
         "| arm | exists | complete | games parsed / expected | unparsable | truncated jsonl | error games | "
         "sim-error games |", "|---|---|---|---|---|---|---|---|"]
    for t, a in res["arms"].items():
        L.append(f"| {t} | {a['exists']} | {a['complete']} | {a['games_parsed']} / {a['games_expected']} | "
                 f"{len(a['unparsable'])} | {a['truncated_jsonl']} | {len(a['error_games'])} | "
                 f"{len(a['sim_error_games'])} |")
    L += ["", "## Predictions", "",
          "| prediction | estimate | SE | 95% CI | n | predicted | in interval | status |",
          "|---|---|---|---|---|---|---|---|"]
    for name, p in res["predictions"].items():
        if name in ("L1", "L2", "L3", "W1", "W2"):
            est, se, ci = p["mean"], p["se"], p["ci95"]
            n = (f"{p['n_turns']} turns ({p['n_forced']} forced) / {p['n_games']} games" if "n_turns" in p else
                 f"{p['n_pairs']} seed pairs" if "n_pairs" in p else
                 f"{p['mws_loss_mw']['n_games']} + {p['bhs_loss_avg']['n_games']} games")
        elif name == "R1":
            est, se, ci = p["n_eq"], None, p["boot95"]
            n = f"{p['n_streams']} streams (RQeq ref: {p['reference_RQeq']['n_eq']} of {p['reference_RQeq']['n_streams']})"
        else:
            est, se, ci = p["rel_diff"], None, None
            n = f"{p['MWs']['decisions']} / {p['BHs']['decisions']} decisions"
        L.append(f"| {name} | {fmt(est)} | {fmt(se)} | {fmt(ci)} | {n} | {fmt(p['interval'], 2)} | "
                 f"{fmt(p['in_interval'])} | {p['status']} |")
    L += ["", "Criteria (prereg §3):", ""] + [f"- {k}: {p['criterion']}" for k, p in res["predictions"].items()]
    return "\n".join(L) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--partial", action="store_true", help="run on whatever exists; output titled PARTIAL")
    for t in ("MWs", "BHs", "MWeq", "RQeq"):
        ap.add_argument(f"--{t.lower()}", default=DEFAULT_DIRS[t], help=f"{t} work dir")
    ap.add_argument("--out-dir", default=RESEARCH)
    args = ap.parse_args(argv)
    dirs = {t: os.path.expanduser(getattr(args, t.lower())) for t in ("MWs", "BHs", "MWeq", "RQeq")}
    res = analyse(dirs)
    incomplete = [t for t, a in res["arms"].items() if a["exists"] and not a["complete"]]
    if incomplete and not args.partial:
        sys.exit(f"incomplete arms {incomplete}: rerun with --partial to analyse what exists")
    title = ("PARTIAL: " if args.partial else "") + "hedge_bayes_mw live test (prereg_mw)"
    res["title"], res["partial"] = title, args.partial
    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, "out_mw.json"), "w") as f:
        json.dump(res, f, indent=1)
    with open(os.path.join(args.out_dir, "out_mw.md"), "w") as f:
        f.write(to_md(res, title))
    print(to_md(res, title))


if __name__ == "__main__":
    main()
