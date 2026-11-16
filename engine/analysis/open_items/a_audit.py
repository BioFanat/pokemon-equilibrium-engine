"""Task A: audit every tag against launch_open_items.sh. Output: out_A_audit.json"""
import json, os, glob
import numpy as np
from oi_common import *
import common as syn

res = {}
for tag, (A, B, s0, s1) in GRID.items():
    S = summary(tag)
    G = game_jsons(tag)
    seeds = list(range(s0, s1 + 1))
    exp_games = 2 * len(seeds)
    missing = [(s, g) for s in seeds for g in (1, 2) if (s, g) not in G]
    extra = [k for k in G if k[0] not in set(seeds)]
    errs = [k for k, d in G.items() if d.get("error") or d.get("sim_errors")]
    errdetail = {f's{k[0]}-g{k[1]}': (G[k].get('error') or '')[:90] for k in errs}
    simerr = [k for k, d in G.items() if d.get("sim_errors")]
    # stream field vs seeds[k::10]
    bad_stream = [k for k, d in G.items() if str(d.get("stream")) != str(stream_of(k[0], s0))]
    # file inventory
    files = os.listdir(rdir(tag))
    n_A = sum(f.endswith(".A.decisions.jsonl") for f in files)
    n_B = sum(f.endswith(".B.decisions.jsonl") for f in files)
    # decision-log integrity: records vs timing decisions; turn resets; duplicates
    trunc, dup, reset, noA = [], [], [], []
    for (seed, game), d in G.items():
        for role in ("A", "B"):
            p = os.path.join(rdir(tag), f"{tag}-s{seed}-g{game}.{role}.decisions.jsonl")
            if not os.path.exists(p):
                if role == "A": noA.append((seed, game))
                continue
            recs = [json.loads(l) for l in open(p) if l.strip()]
            nonfinal = [r for r in recs if not r.get("final")]
            turns = [r["turn"] for r in nonfinal]
            if any(b < a for a, b in zip(turns, turns[1:])):
                reset.append((seed, game, role, len(turns), len(set(turns))))
            td = d["timing"][role]["decisions"] if d.get("timing") else None
            # timing decisions counts every request incl. forced; non-final records include forced ones for hedge etc.
            gd = len(d.get("decisions", {}).get(role, [])) if False else None
            if td is not None and len(nonfinal) != td:
                trunc.append((seed, game, role, len(nonfinal), td))
    # win-rate recomputation
    gs = {k: score(d) for k, d in G.items() if not d.get("error")}
    pair = {}
    for s in seeds:
        if (s, 1) in gs and (s, 2) in gs and not G[(s,1)].get("error") and not G[(s,2)].get("error"):
            pair[s] = (gs[(s, 1)] + gs[(s, 2)]) / 2
    pv = np.array(list(pair.values()))
    gv = np.array([gs[k] for k in sorted(gs)])
    gstream = np.array([stream_of(k[0], s0) for k in sorted(gs)])
    pstream = np.array([stream_of(s, s0) for s in pair])
    summ = S["pair_score_A"]
    out = dict(A=A, B=B, expected_games=exp_games, present_game_json=len(G), missing=missing, extra=extra,
               err_games=errs, err_detail=errdetail, sim_err_games=simerr, bad_stream_field=bad_stream[:5], n_bad_stream=len(bad_stream),
               n_A_logs=n_A, n_B_logs=n_B, noA=noA[:5],
               turn_reset_or_dup_files=reset[:10], n_turn_reset=len(reset),
               n_nonfinal_vs_timing_mismatch=len(trunc), mismatch_examples=trunc[:8],
               summary=dict(mean=summ["mean"], ci95=summ["ci95"], n_pairs=S["n_pairs"], error_games=S["error_games"],
                            missing_games=S["missing_games"], nonzero_exit=S["nonzero_exit"],
                            modal=S.get("modal", {}) and {k: S["modal"][k] for k in ("failed_streams", "stopped_over_budget", "games_run", "usd_estimate") if k in S["modal"]},
                            n_summary_per_seed=len(summ["per_seed"])),
               recomputed_pair_mean=float(pv.mean()), n_pairs_recomputed=len(pv),
               pair_mean_matches=bool(abs(pv.mean() - summ["mean"]) < 6e-5),
               per_seed_matches=bool(all(abs(pair.get(int(s), -9) - v) < 1e-9 for s, v in summ["per_seed"].items())),
               game_level_normal=mean_ci_normal(gv), game_level_stream_cluster=cluster_ci(gv, gstream),
               pair_level_normal=mean_ci_normal(pv), pair_level_stream_cluster=cluster_ci(pv, pstream),
               wins=int((gv == 1).sum()), draws=int((gv == .5).sum()), losses=int((gv == 0).sum()))
    if tag in ("E3a", "E3f"):
        act = S["active"]
        out["active_B"] = {m: sum(1 for v in act.values() if v["B"] == m) for m in {v["B"] for v in act.values()}}
        out["active_B_stock_seeds_max"] = max(int(s) for s, v in act.items() if v["B"] == "stock")
        out["active_B_exploit_seeds_min"] = min(int(s) for s, v in act.items() if v["B"] != "stock")
    # hedge-state game count vs games present
    hs = glob.glob(os.path.join(RES, tag, "hedge_state", "*-A.json")) + glob.glob(os.path.join(RES, tag, "hedge_state", "*-A.bayes.json"))
    hmeta = {}
    for p in sorted(hs):
        d = json.load(open(p))
        k = int(os.path.basename(p).split("stream")[1].split("-")[0])
        m = d.get("meta", {})
        hmeta.setdefault(k, {}).update({("bayes_" if "bayes" in p else "") + "games": m.get("games"), ("bayes_" if "bayes" in p else "") + "updates": m.get("updates")})
    if hmeta:
        gps = {k: sum(1 for (s, g) in G if stream_of(s, s0) == k) for k in range(NSTREAMS)}
        out["hedge_state_games_vs_present"] = {k: (v, gps[k]) for k, v in sorted(hmeta.items())}
        out["hedge_state_games_mismatch_streams"] = [k for k, v in hmeta.items() if v.get("games") != gps[k] and v.get("bayes_games") != gps[k]]
    res[tag] = out
dump("out_A_audit.json", res)
for t, o in res.items():
    print(t, o["expected_games"], o["present_game_json"], "miss", len(o["missing"]), "err", len(o["err_games"]), "reset", o["n_turn_reset"],
          "tmis", o["n_nonfinal_vs_timing_mismatch"], "badstream", o["n_bad_stream"], "sum", o["summary"]["mean"], "recomp", round(o["recomputed_pair_mean"],4),
          o["pair_mean_matches"], o["per_seed_matches"], "gm", round(o["game_level_normal"][0],4), "failed", o["summary"]["modal"].get("failed_streams"),
          o.get("hedge_state_games_mismatch_streams"))
