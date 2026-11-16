"""Independent recomputation of the B12 / neutral drift numbers (bounds_epsv.md sec 2.2-2.3, claims 3-5, 8).
Own implementation (does not import drift_analysis.py). Per group: identity closure, per-game sums, game-clustered
SE of the per-decision drift (ratio estimator sum D / sum n, delta method), last-decision |D|, E phi(v1),
pairing structure for the root term.
python v_drift.py
"""
import json
import numpy as np

S = "/private/tmp/claude-501/-Users-biofanat-Documents-classes-f26-cs535/2bbe670b-2f97-4d9a-bfad-e1139e0352c6/scratchpad"
FILES = {"wb2-12": f"{S}/bridges/b12/replay.jsonl", "wb2-60": f"{S}/tightening/replay_sampled_all.jsonl",
         "mcts": f"{S}/tightening/neutral_mcts.jsonl", "self": f"{S}/tightening/neutral_self.jsonl"}
PHI = {"logit": lambda u: 2 / (1 + np.exp(-np.clip(u / 98.0, -50, 50))) - 1, "tanh": lambda u: np.tanh(u / 400.0)}


def z_of(g):
    if "z" in g:
        return float(g["z"])
    t = 3 - g["a_side"]
    return 0.0 if g["winner"] == 0 else (1.0 if g["winner"] == t else -1.0)


def decisions(g):
    out = []
    for t in g["per_turn"]:
        if t.get("trivial"):
            continue
        if "A" in t:
            if len(t["rows"]) == 1:
                continue
            A, x = np.array(t["A"], float), np.array(t["x"], float)
            out.append((t["v"], float((x @ A)[t["j"]]), float(A[t["i"], t["j"]])))
        else:
            out.append((t["v"], t["xa_j"], t["a_ij"]))
    return out


def game_stats(g, phi):
    d = decisions(g); z = z_of(g)
    v = np.array([a for a, _, _ in d]); xa = np.array([b for _, b, _ in d]); aij = np.array([c for _, _, c in d])
    pv = phi(v); pnext = np.append(pv[1:], z)
    R = pv - phi(xa); S_ = phi(xa) - phi(aij); D = pnext - phi(aij)
    ident = (np.sum(-R - S_ + D)) - (z - pv[0])
    return dict(z=z, n=len(d), R=R.sum(), S=S_.sum(), D=D.sum(), absD=np.abs(D).sum(), Dlast=D[-1], absDlast=abs(D[-1]),
                phiv1=pv[0], ident=abs(ident), Dmid=D[:-1].sum())


def ratio_se(num, den):
    """SE of sum(num)/sum(den) with games as clusters (delta method)."""
    r = num.sum() / den.sum(); n = len(num)
    resid = num - r * den
    return r, np.sqrt(n / (n - 1) * np.sum(resid ** 2)) / den.sum()


res = {}
for name, path in FILES.items():
    G = [json.loads(l) for l in open(path)]
    if name == "wb2-12":
        G = [g for g in G if g.get("tname") == "sample"]
    for pn, phi in PHI.items():
        P = [game_stats(g, phi) for g in G]
        a = {k: np.array([p[k] for p in P], float) for k in P[0]}
        se = lambda x: x.std(ddof=1) / np.sqrt(len(x))
        r, rse = ratio_se(a["D"], a["n"])
        res[(name, pn)] = (r, rse, len(G))
        print(f"{name:7s} {pn:5s} games {len(G):3d} ident max {a['ident'].max():.1e} | -Ez {-a['z'].mean():+.3f} ({se(a['z']):.3f}) "
              f"| sumR {a['R'].mean():+.3f} sumS {a['S'].mean():+.3f} -sumD {-a['D'].mean():+.3f} ({se(a['D']):.3f}) "
              f"mid {-a['Dmid'].mean():+.3f} last {-a['Dlast'].mean():+.3f} | E|D_last| {a['absDlast'].mean():.3f} "
              f"| perdec ED {r:+.4f} (SE {rse:.4f}) | sum|D| {a['absD'].mean():.2f} | dec/game {a['n'].mean():.1f} "
              f"median {np.median(a['n']):.0f} | E phi(v1) {a['phiv1'].mean():+.4f} ({se(a['phiv1']):.4f})")

print("\nper-decision drift differences (logit), z-scores:")
for a_, b_ in [("wb2-60", "mcts"), ("wb2-60", "self"), ("mcts", "self")]:
    for pn in PHI:
        (r1, s1, _), (r2, s2, _) = res[(a_, pn)], res[(b_, pn)]
        print(f"  {a_} vs {b_} [{pn}]: {r1:+.4f} vs {r2:+.4f}, diff {r1-r2:+.4f} +- {np.hypot(s1,s2):.4f} (z={(r1-r2)/np.hypot(s1,s2):+.2f}); "
              f"95% CIs [{r1-1.96*s1:+.4f},{r1+1.96*s1:+.4f}] / [{r2-1.96*s2:+.4f},{r2+1.96*s2:+.4f}]")

# pairing structure / root term on paired data
G = [json.loads(l) for l in open(FILES["wb2-60"])]
by = {}
for g in G:
    by.setdefault(g["seed"], []).append(g)
canc = []
for seed, gs in by.items():
    if len(gs) == 2:
        d1, d2 = decisions(gs[0])[0][0], decisions(gs[1])[0][0]
        canc.append(PHI["logit"](d1) + PHI["logit"](d2))
canc = np.array(canc)
print(f"\nW1 sampled: {len(by)} seeds, {sum(len(v)==2 for v in by.values())} with both seats; "
      f"phi(v1)+phi(v1') per pair: max |.| {np.abs(canc).max():.2e}, mean {canc.mean():+.2e}")
for nm in ["mcts", "self"]:
    G = [json.loads(l) for l in open(FILES[nm])]
    seeds = [g["seed"] for g in G]
    print(f"{nm}: {len(set(seeds))} distinct seeds for {len(G)} games (one game per seed => no seat-paired states)")
