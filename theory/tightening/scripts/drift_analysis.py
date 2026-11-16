"""Along-path drift (Bellman residual of the calibrated depth-2 value) on logged trajectories.

Exact per-game telescoping identity (target perspective, z in {-1,0,+1}, phi maps engine units to [-1,1]):
    z - phi(v_1) = sum_t [ -R_t - S_t + D_t ]
    R_t = phi(v_t) - phi((xA)_j)        realised stage gain of the opponent's column (R_t <= stage gap)
    S_t = phi((xA)_j) - phi(A[i,j])     target's own sampling noise (mean 0 given x)
    D_t = phi(v_{t+1}) - phi(A[i,j])    drift: next root value minus the cell that produced it; v_{T+1} := z
so  -z = sum R + sum S - sum D - phi(v_1): the "drift channel" -sum D is the along-path leaf-error term.
Handles the B12 replay format (per_turn with A, x, v, i, j) and neutral_drift.py's format (v, xa_j, a_ij).
Reports both phi maps: logit with s = 98 (fitted on the B12 exploiter games) and tanh(v/400).
Usage: python drift_analysis.py file.jsonl [more.jsonl ...]
"""
from __future__ import annotations

import json
import sys

import numpy as np

S_FIT = 98.0
PHIS = {"logit98": lambda u: 2 / (1 + np.exp(-np.clip(np.asarray(u, float) / S_FIT, -50, 50))) - 1,
        "tanh400": lambda u: np.tanh(np.asarray(u, float) / 400)}
NAMES = {"argmax": "top move", "sample": "sampled", "safex:0.02": "safex 2%"}


def outcome(g):
    if "z" in g:
        return float(g["z"])
    tside = 3 - g["a_side"]
    return 0.0 if g["winner"] == 0 else (1.0 if g["winner"] == tside else -1.0)


def turns(g):
    """yield (v, xa_j, a_ij, g_range, r) per non-trivial target turn."""
    for t in g["per_turn"]:
        if t.get("trivial"):
            continue
        if "A" in t:
            if len(t["rows"]) == 1:          # forced turn (B12 format): skip, as neutral_drift.py does
                continue
            A, x = np.array(t["A"]), np.array(t["x"])
            r = float(np.ptp(A)) or 1.0
            xa = x @ A
            yield t["v"], float(xa[t["j"]]), float(A[t["i"], t["j"]]), float((t["v"] - xa.min()) / r), r
        else:
            yield t["v"], t["xa_j"], t["a_ij"], t["g"] / (t["r"] or 1.0), t["r"]


def per_game(g, phi):
    z = outcome(g)
    T = list(turns(g))
    out = dict(z=z, n=len(T), R=0.0, S=0.0, D=0.0, Dmid=0.0, Dlast=0.0, absD=0.0, g=0.0, phiv1=float(phi(T[0][0])) if T else 0.0,
               Ds=[])
    for k, (v, xaj, aij, gr, r) in enumerate(T):
        vnext = T[k + 1][0] if k + 1 < len(T) else None
        pn = float(phi(vnext)) if vnext is not None else z
        R = float(phi(v) - phi(xaj)); S = float(phi(xaj) - phi(aij)); D = float(pn - phi(aij))
        out["R"] += R; out["S"] += S; out["D"] += D; out["absD"] += abs(D); out["g"] += gr
        if vnext is None:
            out["Dlast"] += D
        else:
            out["Dmid"] += D
        out["Ds"].append(D)
    out["ident"] = (-out["R"] - out["S"] + out["D"]) - (z - out["phiv1"])
    return out


def summarize(label, games, phi_name):
    phi = PHIS[phi_name]
    P = [per_game(g, phi) for g in games]
    n = len(P)
    arr = lambda k: np.array([p[k] for p in P])
    z, R, S, D, Dmid, Dlast, absD, nt, gsum = (arr(k) for k in ["z", "R", "S", "D", "Dmid", "Dlast", "absD", "n", "g"])
    se = lambda a: a.std(ddof=1) / np.sqrt(len(a)) if len(a) > 1 else float("nan")
    Ds = np.concatenate([p["Ds"] for p in P])
    print(f"[{label} | phi={phi_name}] games {n}, target score {(z.mean()+1)/2:.3f}, "
          f"exploitability 2(s-1/2) = -mean z = {-z.mean():+.3f} (SE {se(z):.3f}), turns/game {nt.mean():.1f}, "
          f"identity max err {max(abs(p['ident']) for p in P):.1e}")
    print(f"   per-game means: sum R {R.mean():+.3f} (SE {se(R):.3f}) | sum S {S.mean():+.3f} (SE {se(S):.3f}) | "
          f"-sum D {-D.mean():+.3f} (SE {se(D):.3f}) = -mid {-Dmid.mean():+.3f} -last {-Dlast.mean():+.3f} | -phi(v1) {-np.mean([p['phiv1'] for p in P]):+.3f}")
    print(f"   per-turn drift: mean D {Ds.mean():+.4f}, mean |D| {np.abs(Ds).mean():.4f}, sd {Ds.std():.4f}, "
          f"quantiles 5/50/95 {np.percentile(Ds,5):+.3f}/{np.percentile(Ds,50):+.3f}/{np.percentile(Ds,95):+.3f}; "
          f"sum |D| per game {absD.mean():.2f}; stage-gap sum (range units) {gsum.mean():.3f}")
    return dict(label=label, phi=phi_name, n=n, score=float((z.mean() + 1) / 2), expl=float(-z.mean()), expl_se=float(se(z)),
                turns=float(nt.mean()), R=float(R.mean()), S=float(S.mean()), negD=float(-D.mean()), negD_se=float(se(D)),
                negDmid=float(-Dmid.mean()), negDlast=float(-Dlast.mean()), D_turn_mean=float(Ds.mean()),
                absD_turn_mean=float(np.abs(Ds).mean()), D_turn_sd=float(Ds.std()), absD_game=float(absD.mean()),
                gsum=float(gsum.mean()))


def main():
    results = []
    for path in sys.argv[1:]:
        G = [json.loads(l) for l in open(path)]
        if "tname" in G[0]:
            groups = {NAMES.get(t, t): [g for g in G if g["tname"] == t] for t in sorted({g["tname"] for g in G})}
            label0 = "B12 exploiter path, target "
        else:
            groups = {f"{G[0]['opp']}": G}
            label0 = "neutral path (sampled target) vs "
        for name, games in groups.items():
            for phi_name in PHIS:
                results.append(summarize(label0 + name, games, phi_name))
    out = sys.argv[1].rsplit("/", 1)[0] + "/drift_summary.json" if len(sys.argv) == 2 else "/dev/null"
    json.dump(results, open("/private/tmp/claude-501/-Users-biofanat-Documents-classes-f26-cs535/2bbe670b-2f97-4d9a-bfad-e1139e0352c6/scratchpad/tightening/drift_summary.json", "w"), indent=1)


if __name__ == "__main__":
    main()
