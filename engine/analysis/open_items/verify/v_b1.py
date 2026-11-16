"""Independent B1: LP value per decision (own formulation), identity terms, drift stats, logging consistency checks."""
import json, os, math, sys, collections
import numpy as np
from multiprocessing import Pool
from scipy.optimize import linprog
RES = "/Users/biofanat/Documents/classes/f26/cs535/final/engine/showdown/results/LOG/replays_headless"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "b1_rows.json")


def lpval(A):
    A = np.asarray(A, float); n, m = A.shape
    c = np.zeros(n + 1); c[-1] = -1.0                       # maximise v
    Aub = np.hstack([-A.T, np.ones((m, 1))])                # v - (x^T A)_j <= 0
    Aeq = np.zeros((1, n + 1)); Aeq[0, :n] = 1
    r = linprog(c, A_ub=Aub, b_ub=np.zeros(m), A_eq=Aeq, b_eq=[1.0], bounds=[(0, None)] * n + [(None, None)], method="highs")
    assert r.status == 0
    return float(r.x[-1])


def strip(mv):
    return mv[len("switch "):] if mv.startswith("switch ") else mv


def one(args):
    s, k = args
    g = json.load(open(os.path.join(RES, f"LOG-s{s}-g{k}.json")))
    if g.get("error"): return None
    z = 1.0 if g["winner_role"] == "A" else (-1.0 if g["winner_role"] == "B" else 0.0)
    seat = g["sides"]["A"]
    recs = [json.loads(l) for l in open(os.path.join(RES, f"LOG-s{s}-g{k}.A.decisions.jsonl")) if l.strip()]
    chk = collections.Counter()
    rows = []
    for idx, r in enumerate(recs):
        if r.get("final") or "abar" not in r:
            continue
        A = np.array(r["abar"]); legal = r["legal"]; labels = r["labels"]; x = np.array(r["pfull"])
        Q = np.array(r["Q"]); w = np.array([r["weights"][e] for e in ("top", "sampled", "safest", "safex")])
        # consistency checks
        p = np.clip(w @ Q, 0, None); p /= p.sum()
        chk["pfull_vs_wQ_maxdiff>1e-3"] += float(np.abs(p - x).max() > 1e-3)
        for e, q in zip(("top", "sampled", "safest", "safex"), Q):
            chk[f"expert_label_mismatch_{e}"] += float(r["experts"][e] != legal[int(q.argmax())])
        safest = int(A.min(1).argmax()); mins = A.min(1)
        tie = np.sum(mins >= mins.max() - 0.011) > 1
        chk["safest_onehot_mismatch"] += float(int(Q[2].argmax()) != safest and not tie)
        chk["safest_tie"] += float(tie)
        chk["top_is_argmax_Q1"] += float(int(Q[0].argmax()) == int(Q[1].argmax()))
        v = lpval(A)
        afv = float((Q[1] @ A).min())                        # agent-form x evaluated on Abar
        i = legal.index(strip(r["move"])) if strip(r["move"]) in legal else None
        chk["i_missing"] += float(i is None)
        if i is not None: chk["x_i_zero"] += float(x[i] <= 0)
        # opponent column: feedback in the next record
        nxt = recs[idx + 1] if idx + 1 < len(recs) else None
        fb = nxt.get("feedback") if nxt else None
        j = None; reason = None
        if fb is None:
            reason = "no_feedback_rec"
        elif fb.get("for_turn") != int(r["turn"]):
            reason = "turn_mismatch"
        elif fb.get("skipped"):
            reason = fb.get("reason")
        else:
            col = (fb.get("bayes") or {}).get("col", fb.get("b"))
            if col in labels: j = labels.index(col)
            else: reason = "col_not_in_labels"
        rows.append(dict(v=v, afv=afv, xAj=float(x @ A[:, j]) if j is not None else None,
                         Aij=float(A[i, j]) if (j is not None and i is not None) else None, reason=reason,
                         minxA=float((x @ A).min())))
    return dict(s=s, k=k, z=z, seat=seat, rows=rows, chk=dict(chk))


if __name__ == "__main__":
    jobs = [(s, k) for s in range(1, 301) for k in (1, 2)]
    with Pool(8) as P:
        res = [r for r in P.map(one, jobs, chunksize=4) if r is not None]
    json.dump(res, open(OUT, "w"))
    print("games", len(res))
