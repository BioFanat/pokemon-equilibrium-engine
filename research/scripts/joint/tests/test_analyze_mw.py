"""Synthetic check of analyze_mw.py: hand-computed L1 mean + clustered SE, L2, W1, C1, R1, skip rules, status logic."""
import json
import math
import os
import shutil
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import analyze_mw as A  # noqa: E402

LN2 = math.log(2)
STY = ["eq", "br_unif", "br_us", "safe", "uniform"]


def dec(turn, fb=None, bayes=True):
    r = dict(method="x", turn=turn, feedback=fb)
    if bayes:
        r["bayes"] = dict(qhat=[0.5, 0.5], labels=["a", "b"], posterior={s: 0.2 for s in STY})
    return r


def fbk(pq, pmw=None, forced=False, post=None):
    b = dict(col="a", j=0, forced=forced, p_qhat=pq, posterior=post or {s: 0.2 for s in STY})
    if pmw is not None:
        b["p_qhat_mw"] = pmw
    return dict(for_turn=0, b="a", how="forced" if forced else "move", skipped=False, world=0, bayes=b)


SKIP = dict(for_turn=0, b="z", how="switch", skipped=True, reason="not_in_worlds")


def game_records(pairs, mw=True):
    """pairs = list of (p_qhat, p_qhat_mw[, forced]) scored in record order, with a skipped feedback and a no-bayes
    decision in between, ending with a final record carrying the last feedback."""
    recs = [dec(1)]
    for i, pr in enumerate(pairs):
        forced = len(pr) > 2 and pr[2]
        fb = fbk(pr[0], pr[1] if mw else None, forced)
        if i == len(pairs) - 1:
            recs.append(dict(method="x", final=True, feedback=fb))
        else:
            recs.append(dec(i + 2, fb, bayes=not forced))
            recs.append(dec(i + 2, SKIP))                      # skipped feedback: not scored
    if not pairs:
        recs.append(dict(method="x", final=True, feedback=SKIP))
    return recs


def write_game(rdir, tag, seed, game, winner, recs, cpu=2.0, decs=10, error=None, stream=0, json_ok=True,
               tail=None):
    base = os.path.join(rdir, f"{tag}-s{seed}-g{game}")
    with open(base + ".A.decisions.jsonl", "w") as f:
        for r in recs:
            f.write(json.dumps(r) + "\n")
        if tail:
            f.write(tail)
    if json_ok is None:
        return
    with open(base + ".json", "w") as f:
        if json_ok:
            json.dump(dict(tag=f"{tag}-s{seed}-g{game}", winner_role=winner, error=error, sim_errors=[], stream=stream,
                           timing=dict(A=dict(decisions=decs, cpu_s=cpu))), f)
        else:
            f.write('{"tag": "half-writ')


def build(root):
    d = {t: os.path.join(root, t) for t in ("MWs", "BHs", "MWeq", "RQeq")}
    for t in d:
        os.makedirs(os.path.join(d[t], "replays_headless"))
        os.makedirs(os.path.join(d[t], "hedge_state"))
    r = os.path.join(d["MWs"], "replays_headless")
    write_game(r, "MWs", 1, 1, "A", game_records([(0.25, 0.5), (1.0, 1.0, True), (0.5, 0.5)]))   # d = 1,0,0 (x ln2)
    write_game(r, "MWs", 1, 2, "B", game_records([(0.25, 0.5), (0.125, 0.5)]))                     # d = 1,2
    write_game(r, "MWs", 2, 1, "A", game_records([(0.5, 0.5)]))                                    # d = 0
    write_game(r, "MWs", 2, 2, "A", game_records([(0.0001, 1.0)]), cpu=100.0, error="boom")       # error: dropped
    write_game(r, "MWs", 3, 1, "A", game_records([(0.25, 0.5)]), tail='{"method": "x", "fee')     # d = 1; EOF cut
    write_game(r, "MWs", 3, 2, "A", game_records([(0.0001, 1.0)]), json_ok=False)                 # unparsable JSON
    write_game(r, "MWs", 4, 1, "A", game_records([(0.0001, 1.0)]), json_ok=None)                  # running: no JSON
    write_game(r, "MWs", 5, 1, "A", game_records([]))
    write_game(r, "MWs", 5, 2, "A", game_records([]))
    write_game(r, "MWs", 6, 1, None, game_records([]))
    write_game(r, "MWs", 6, 2, None, game_records([]))
    r = os.path.join(d["BHs"], "replays_headless")
    for s, (w1, w2) in {1: ("B", "B"), 2: ("A", "A"), 5: ("A", "B"), 6: ("A", "A")}.items():
        for g, w in ((1, w1), (2, w2)):
            write_game(r, "BHs", s, g, w, game_records([(0.5, None)], mw=False), cpu=1.9, decs=10)
    # MWeq: streams 0-2 eq by logp, 3-8 safe, stream 9 has no state file -> last logged posterior (eq)
    for k in range(9):
        lp = [0.0, -5, -5, -9, -9] if k < 3 else [-9, -5, -5, 0.0, -9]
        json.dump(dict(config=dict(styles=STY), logp=lp), open(os.path.join(d["MWeq"], "hedge_state",
                                                                           f"MWeq-stream{k}-A.bayes.json"), "w"))
        json.dump(dict(config=dict(styles=STY), logp=[-9, -5, -5, 0.0, -9]),
                  open(os.path.join(d["RQeq"], "hedge_state", f"RQeq-stream{k}-A.bayes.json"), "w"))
    json.dump(dict(config=dict(styles=STY), logp=[-9, -5, -5, 0.0, -9]),
              open(os.path.join(d["RQeq"], "hedge_state", "RQeq-stream9-A.bayes.json"), "w"))
    recs = game_records([(0.5, 0.5)])
    recs[-1]["feedback"]["bayes"]["posterior"] = dict(eq=0.7, br_unif=0.1, br_us=0.1, safe=0.1, uniform=0.0)
    write_game(os.path.join(d["MWeq"], "replays_headless"), "MWeq", 10, 1, "A", recs, stream=9)
    return d


def main():
    root = tempfile.mkdtemp(dir=os.path.dirname(os.path.abspath(__file__)))
    try:
        d = build(root)
        res = A.analyse(d)
        P, arms = res["predictions"], res["arms"]
        # L1 by hand (units of ln2): games {11: [1,0,0], 12: [1,2], 21: [0], 31: [1]}; n=7, mean 5/7
        groups = [[1, 0, 0], [1, 2], [0], [1]]
        n, m = 7, 5 / 7
        se = math.sqrt(4 / 3 * sum(sum(v - m for v in g) ** 2 for g in groups)) / n * LN2
        assert abs(P["L1"]["mean"] - m * LN2) < 1e-12, P["L1"]["mean"]
        assert abs(P["L1"]["se"] - se) < 1e-12, (P["L1"]["se"], se)
        assert P["L1"]["n_turns"] == 7 and P["L1"]["n_games"] == 4 and P["L1"]["n_forced"] == 1
        # L2: BHs loss ln2 (8 games, 1 turn each); MWs mw-loss (x ln2) [1,0,1, 1,1, 1, 1] -> 6/7
        assert abs(P["L2"]["mean"] - LN2 / 7) < 1e-12, P["L2"]["mean"]
        assert P["L2"]["bhs_loss_avg"]["n_games"] == 8
        # W1: MWs seed scores {1: .5, 5: 1, 6: .5} (2 error, 3 unparsable g2); BHs {1: 0, 2: 1, 5: .5, 6: 1}
        diffs = [0.5, 0.5, -0.5]
        assert P["W1"]["n_pairs"] == 3 and abs(P["W1"]["mean"] - 1 / 6) < 1e-12
        sd = math.sqrt(sum((x - 1 / 6) ** 2 for x in diffs) / 2)
        assert abs(P["W1"]["se"] - sd / math.sqrt(3)) < 1e-12
        assert P["W1"]["error_games"] == {"MWs": 1, "BHs": 0}
        # C1: 8 non-error MWs games 2.0/10, BHs 1.9/10
        assert abs(P["C1"]["rel_diff"] - (0.2 / 0.19 - 1)) < 1e-12 and P["C1"]["MWs"]["n_games"] == 8
        # R1: 3 eq by state + stream 9 eq by last logged posterior; RQeq 0 of 10
        assert P["R1"]["n_eq"] == 4 and P["R1"]["n_streams"] == 10
        assert P["R1"]["per_stream"]["9"] == dict(map="eq", source="last_logged")
        assert P["R1"]["reference_RQeq"]["n_eq"] == 0
        # skip rules and status
        assert arms["MWs"]["unparsable"] == ["s3-g2"] and arms["MWs"]["truncated_jsonl"] == 1
        assert arms["MWs"]["games_parsed"] == 9 and not arms["MWs"]["complete"]
        assert all(p["status"] == "UNTESTABLE" for p in P.values()), {k: p["status"] for k, p in P.items()}
        # non-partial refuses incomplete arms; partial writes PARTIAL-titled outputs
        try:
            A.main(["--mws", d["MWs"], "--bhs", d["BHs"], "--mweq", d["MWeq"], "--rqeq", d["RQeq"],
                    "--out-dir", root])
            raise AssertionError("non-partial run on incomplete arms should exit")
        except SystemExit as e:
            assert "partial" in str(e.code)
        A.main(["--partial", "--mws", d["MWs"], "--bhs", d["BHs"], "--mweq", d["MWeq"], "--rqeq", d["RQeq"],
                "--out-dir", root])
        out = json.load(open(os.path.join(root, "out_mw.json")))
        assert out["title"].startswith("PARTIAL") and open(os.path.join(root, "out_mw.md")).read().startswith(
            "# PARTIAL")
        # a parse failure that is not at EOF is an error, not a silent skip
        bad = os.path.join(root, "bad.jsonl")
        open(bad, "w").write('{"a": 1}\n{"b": \n{"c": 3}\n')
        try:
            A.read_jsonl(bad)
            raise AssertionError("mid-file parse error should raise")
        except json.JSONDecodeError:
            pass
        # a missing arm (MWeq not started) -> its predictions UNTESTABLE, no crash
        res = A.analyse(dict(d, MWeq=os.path.join(root, "nonexistent")))
        assert not res["arms"]["MWeq"]["exists"]
        assert all(res["predictions"][k]["status"] == "UNTESTABLE" for k in ("L3", "R1", "W2"))
        assert res["predictions"]["L3"]["n_turns"] == 0 and res["predictions"]["W2"]["n_pairs"] == 0
        # status logic on complete arms: shrink the grid to what exists
        A.SEEDS.update(MWs=(1, 1), BHs=(1, 1))
        shutil.rmtree(os.path.join(d["MWs"], "replays_headless"))
        os.makedirs(os.path.join(d["MWs"], "replays_headless"))
        r = os.path.join(d["MWs"], "replays_headless")
        write_game(r, "MWs", 1, 1, "A", game_records([(0.25, 0.5), (0.25, 0.5)]))      # d = ln2, ln2
        write_game(r, "MWs", 1, 2, "A", game_records([(0.125, 0.5), (0.25, 0.5)]))     # d = 2ln2, ln2
        res = A.analyse(d)
        P = res["predictions"]
        assert res["arms"]["MWs"]["complete"] and res["arms"]["BHs"]["complete"]
        # L1: G=2, CI (t with 1 df) = mean +- 12.706*se, se = sqrt(2*(2*(.25ln2)^2))/4 -> includes 0 -> FAIL
        assert P["L1"]["status"] == "FAIL" and P["L1"]["ci95"][0] < 0, P["L1"]
        assert P["W1"]["n_pairs"] == 1 and P["W1"]["status"] == "UNTESTABLE"   # one pair: no CI
        assert P["C1"]["status"] == "NO_CRITERION" and P["L3"]["status"] == "UNTESTABLE"
        assert P["L2"]["status"] in ("PASS", "FAIL")
        # a truncated jsonl in an otherwise complete arm makes it incomplete (the final record may be the cut line)
        open(os.path.join(r, "MWs-s1-g2.A.decisions.jsonl"), "a").write('{"method": "x", "fin')
        res = A.analyse(d)
        assert not res["arms"]["MWs"]["complete"] and res["predictions"]["L1"]["status"] == "UNTESTABLE"
        print("all synthetic checks passed")
        print(f"L1 mean {P['L1']['mean']:.6f}")
    finally:
        shutil.rmtree(root)


if __name__ == "__main__":
    main()
