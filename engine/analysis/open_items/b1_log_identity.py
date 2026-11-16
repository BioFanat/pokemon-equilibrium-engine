"""B1: signed pathwise identity z - phi(v_1) = sum_t(-R_t - S_t + D_t) along the LOG games (hedge_bayes vs stock), our side A.
R_t = phi(v_t) - phi((x_t Abar)_j);  S_t = phi((x_t Abar)_j) - phi(Abar[i,j]);  D_t = phi(v_{t+1}) - phi(Abar[i,j]);  phi(v_{T+1}) := z.
v_t = value of Abar_t (scipy LP on the logged, 2-dp-rounded abar), x_t = logged pfull (Hedge mixture actually played),
i_t = played move, j_t = opponent column read from the next record's feedback.b (looked up in labels).
Decisions whose feedback was skipped (opponent column unmappable) have j unknown: R,S,D are not separable there;
only the lump phi(v_{t+1}) - phi(v_t) = -R-S+D is known. Output: out_B1_log.json"""
import sys
import numpy as np
from oi_common import *

TAG = "LOG"
G = game_jsons(TAG)
rows = []          # per decision (j known)
games = []
def move_label(m):
    return m[len("switch "):] if m.startswith("switch ") else m

for (seed, game), meta in sorted(G.items()):
    if meta.get("error"):
        continue
    R = decisions(TAG, seed, game)
    z = {"A": 1.0, "B": -1.0}.get(meta["winner_role"], 0.0)
    dec = []   # (record index) of decisions with abar
    for n, r in enumerate(R):
        if r.get("final") or "abar" not in r:
            continue
        nxt = R[n + 1] if n + 1 < len(R) else None
        fb = (nxt or {}).get("feedback")
        dec.append((r, fb))
    if not dec:
        continue
    As = [np.array(r["abar"], float) for r, _ in dec]
    v = np.array([game_value_lp(A) for A in As])
    # info per decision
    info = []
    for (r, fb), A in zip(dec, As):
        x = np.array(r["pfull"], float)
        legal, labels = r["legal"], r["labels"]
        ml = move_label(r["move"])
        i = legal.index(ml) if ml in legal else None
        if i is None and ml.endswith(" terastallize"):
            i = legal.index(ml.replace(" terastallize", "-tera")) if ml.replace(" terastallize", "-tera") in legal else None
        j = None
        why = None
        if fb is None or fb.get("for_turn") != r["turn"]:
            why = "no_feedback_record"
        elif fb.get("skipped"):
            why = fb.get("reason", "skipped")
        elif fb["b"] in labels:
            j = labels.index(fb["b"])
        else:
            why = "label_missing"
        info.append((x, i, j, why, A))
    rec = dict(seed=seed, game=game, side=meta["sides"]["A"], z=z, turns=meta["turns"], n_dec=len(dec),
               n_j_known=sum(1 for t in info if t[2] is not None and t[1] is not None))
    for name, phi in (("logit", phi_logit), ("tanh", phi_tanh)):
        pv = phi(v)
        pnext = np.append(pv[1:], z)
        lump = pnext - pv          # = -R-S+D
        Rs, Ss, Ds, known, OG, UG = [], [], [], [], [], []
        for t, (x, i, j, why, A) in enumerate(info):
            if j is None or i is None:
                known.append(False); OG.append(np.nan); UG.append(np.nan); Rs.append(np.nan); Ss.append(np.nan); Ds.append(np.nan); continue
            xa = x @ A
            OG.append(phi(xa[j]) - phi(xa.min())); UG.append(pv[t] - phi(xa.min()))
            cell = A[i, j]
            Rs.append(pv[t] - phi(xa[j])); Ss.append(phi(xa[j]) - phi(cell)); Ds.append(pnext[t] - phi(cell)); known.append(True)
        Rs, Ss, Ds, known = map(np.array, (Rs, Ss, Ds, known))
        # closure: for known decisions -R-S+D == lump ; whole-game telescoping: z - phi(v1) == sum lump
        clo_dec = float(np.nanmax(np.abs((-Rs - Ss + Ds)[known] - lump[known]))) if known.any() else 0.0
        clo_game = float(abs((z - pv[0]) - lump.sum()))
        rec[name] = dict(phi_v1=float(pv[0]), z_minus_phi_v1=float(z - pv[0]), lump_total=float(lump.sum()), closure_dec=clo_dec, closure_game=clo_game,
                         sumR_known=float(np.nansum(Rs)), sumS_known=float(np.nansum(Ss)), sumD_known=float(np.nansum(Ds)),
                         sumabsD_known=float(np.nansum(np.abs(Ds))), sum_opp_gap=float(np.nansum(OG)), sum_our_gap=float(np.nansum(UG)), n_known=int(known.sum()),
                         sum_lump_unknown=float(lump[~known].sum()),
                         D=[None if np.isnan(d) else float(d) for d in Ds], R=[None if np.isnan(d) else float(d) for d in Rs],
                         S=[None if np.isnan(d) else float(d) for d in Ss], lump=lump.tolist())
    rec["whys"] = [t[3] for t in info if t[3]]
    rec["i_unmatched"] = sum(1 for t in info if t[1] is None)
    games.append(rec)
    print(seed, game, rec["n_dec"], rec["n_j_known"], file=sys.stderr) if seed % 50 == 0 and game == 1 else None

import collections
out = dict(source="engine/showdown/results/LOG/replays_headless/LOG-s*-g*.A.decisions.jsonl + LOG-s*-g*.json", n_games=len(games), games=games)
dump("out_B1_log_games.json", out)
