"""Loader for OURBOT_LOG_ABAR=1 decision logs: joins each decision record (abar, labels, legal, pfull, Q, qs, qhat)
with the feedback that the NEXT record (or the final record) carries for it.

A decision row t (our side; Abar_t is n_t x m_t, rows = our legal options, columns = the union of the worlds' opponent
labels; we maximise, the opponent minimises):
  A      Abar_t (engine units, logged to 2 dp)
  p      pfull_t, the mixture we played
  Q      the 4 Hedge experts (top, sampled, safest, safex) as rows (hedge methods only)
  qs     the 5 style distributions (eq, br_unif, br_us, safe, uniform) over the columns (bayes methods only)
  qhat   the posterior's prediction (bayes methods only)
  j      the index in labels of the opponent's realised action (None if the feedback was skipped)
  how    'move' / 'switch' / 'forced'
  r_log  Hedge's logged reward increments (computed by the learner on the world-k matrix, not on Abar)
Usage: rows = load_tag(results_dir, tag, side='A')  -> list of dicts ordered by (stream, position in stream, game, index).
"""
import glob
import json
import os
import re

import numpy as np

DELTA = 400.0   # Hedge's fixed normaliser (exp3_live.Learner scale)


def game_files(res_dir, tag, side="A"):
    out = []
    for p in sorted(glob.glob(os.path.join(res_dir, f"{tag}-s*-g*.{side}.decisions.jsonl"))):
        m = re.search(rf"{tag}-s(\d+)-g(\d)\.", os.path.basename(p))
        out.append((int(m.group(1)), int(m.group(2)), p))
    return out


def load_game(path, seed, game, side, meta, require_feedback=True):
    recs = [json.loads(l) for l in open(path) if l.strip()]
    rows = []
    # feedback for decision i is carried by record i+1 (or the final record); pair them by position
    for i, r in enumerate(recs):
        if r.get("final") or "abar" not in r:
            continue
        fb = recs[i + 1].get("feedback") if i + 1 < len(recs) else None
        if fb is None and require_feedback:
            continue
        if fb is None:
            fb = dict(skipped=True, reason="no_feedback_logged", how=None, b=None)
        elif fb.get("for_turn") != r["turn"]:
            raise ValueError(f"{path}: record {i} turn {r['turn']} vs feedback for_turn {fb.get('for_turn')}")
        row = dict(seed=seed, game=game, idx=i, turn=r["turn"], method=r["method"], side=side,
                   A=np.asarray(r["abar"], float), labels=r["labels"], legal=r["legal"],
                   p=np.asarray(r["pfull"], float), how=fb.get("how"), skipped=fb.get("skipped", True),
                   reason=fb.get("reason"), b=fb.get("b"), j=None, r_log=None, w_log=None, **meta)
        if "Q" in r:
            row["Q"] = np.asarray(r["Q"], float)
            row["w_hedge"] = r["weights"]
        if "qs" in r:
            row["qs"] = np.asarray(r["qs"], float)
            row["qhat"] = np.asarray(r["bayes"]["qhat"], float)
            row["post"] = r["bayes"]["posterior"]
        if not row["skipped"]:
            b = fb["b"]
            lab = row["labels"]
            if "bayes" in fb and "j" in fb["bayes"]:
                j = fb["bayes"]["j"]
                assert lab[j] == fb["bayes"]["col"], (path, i)
            else:                                            # non-bayes methods: resolve the label like map_column
                cands = [b, b[:-len("-tera")]] if b.endswith("-tera") else [b]
                j = next((lab.index(c) for c in cands if c in lab), None)
            row["j"] = j
            if "r" in fb:
                row["r_log"] = np.asarray(fb["r"], float)
                row["w_log"] = np.asarray(fb["weights"], float)
        rows.append(row)
    return rows


def load_tag(res_dir, tag, side="A", exclude_error_games=True, require_feedback=True):
    files = game_files(res_dir, tag, side)
    rows = []
    for seed, game, path in files:
        gp = os.path.join(res_dir, f"{tag}-s{seed}-g{game}.json")
        g = json.load(open(gp)) if os.path.exists(gp) else {}
        if exclude_error_games and g.get("error"):
            continue
        meta = dict(stream=g.get("stream"), winner_role=g.get("winner_role"), score=g.get("score"))
        rows.extend(load_game(path, seed, game, side, meta, require_feedback))
    # order by stream, then by the seed's position in its stream, then game, then index
    streams = {}
    for r in rows:
        streams.setdefault(r["stream"], set()).add(r["seed"])
    pos = {(k, s): i for k, ss in streams.items() for i, s in enumerate(sorted(ss))}
    rows.sort(key=lambda r: (r["stream"] if r["stream"] is not None else -1, pos[(r["stream"], r["seed"])], r["game"], r["idx"]))
    for i, r in enumerate(rows):
        r["t"] = i
    return rows


if __name__ == "__main__":
    import sys
    res, tag = sys.argv[1], sys.argv[2]
    side = sys.argv[3] if len(sys.argv) > 3 else "A"
    rows = load_tag(res, tag, side, require_feedback=(side == "A"))
    n = len(rows); ns = sum(r["skipped"] for r in rows)
    print(f"{tag}/{side}: {n} decision rows with abar+feedback; skipped {ns} ({ns/n:.3f}); methods {set(r['method'] for r in rows)}")
    r0 = next((r for r in rows if not r["skipped"]), rows[0])
    print("example:", {k: (v.shape if isinstance(v, np.ndarray) else v) for k, v in r0.items() if k not in ("labels", "legal")})
    ok = all(r["A"].shape == (len(r["legal"]), len(r["labels"])) for r in rows)
    print("shape consistent:", ok, "| p sums:", np.allclose([r["p"].sum() for r in rows], 1, atol=2e-3))
    if "qs" in r0:
        print("qs shape ok:", all(r["qs"].shape == (5, len(r["labels"])) for r in rows if "qs" in r))
