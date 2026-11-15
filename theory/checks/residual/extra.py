"""Extra cuts on residual.json + raw pickles: excl. capped games, terminal mismatch of no-term leaves, per-game H3 table."""
import json, glob, pickle, numpy as np
D = "/private/tmp/claude-501/-Users-biofanat-Documents-classes-f26-cs535/2bbe670b-2f97-4d9a-bfad-e1139e0352c6/scratchpad/residual"
R = json.load(open(f"{D}/residual.json"))
print({k: R[k] for k in ["n_games", "n_states_all", "n_states_nonforced", "n_hasterm_nonforced", "winners", "capped", "sanity", "mid_leaf_abs_quantiles", "term_child_H"]})
games = [pickle.load(open(f, "rb")) for f in sorted(glob.glob(f"{D}/raw/game_*.pkl"))]
capped = {g["seed"] for g in games if g["capped"]}
ps = R["per_state"]
L = ["t40+term", "t80+term", "t160+term", "t320+term", "t640+term", "t160", "t320", "t640", "vnet+term"]
L = [l for l in L if l in ps[0]["delta"]]
sel = [p for p in ps if p["shape"][0] > 1 and p["seed"] not in capped]
print(f"\nnonforced states in UNCAPPED games: {len(sel)} ({sum(p['hasterm'] for p in sel)} with a terminal child)")
for l in L:
    a = np.array([p["delta"][l] for p in sel])
    print(f"{l:10s} mean|d| {np.abs(a).mean():.4f} med {np.median(np.abs(a)):.4f} p90 {np.percentile(np.abs(a),90):.4f} mean {a.mean():+.4f} mean d2 {np.mean(a**2):.5f}")
# capped-game share of states and their residual
cs = [p for p in ps if p["shape"][0] > 1 and p["seed"] in capped]
print(f"\ncapped-game nonforced states: {len(cs)}; t160+term mean|d| there {np.mean([abs(p['delta']['t160+term']) for p in cs]):.4f}")
# terminal mismatch of the no-exact-terminal leaf: |tanh(H/k) - u| over terminal children (prob-weighted within state, averaged over states)
print("\nno-term leaf terminal mismatch |tanh(h_child/k) - u| over terminal children (prob-weighted):")
Hs, Ts, Ws = [], [], []
for g in games:
    for d in g["states"]:
        m = ~np.isnan(d["T"])
        Hs.append(d["H"][m]); Ts.append(d["T"][m]); Ws.append(d["P"][m])
Hs, Ts, Ws = map(np.concatenate, (Hs, Ts, Ws))
print("  n terminal children", Hs.size, "draws", int((Ts == 0).sum()), "sign(h)!=u frac", float(np.mean(np.sign(Hs) != Ts)))
for k in [40, 80, 160, 320, 640]:
    mm = np.abs(np.tanh(Hs / k) - Ts)
    print(f"  k={k}: weighted mean {np.average(mm, weights=Ws):.3f}  median {np.median(mm):.3f}  max {mm.max():.3f}")
# H3 per game
print("\nH3 per game (all states t<tau), t160+term / t320+term / vnet+term sum|d|; signed sum t160+term:")
for g in R["per_game"]["t160+term"]["games"]:
    s = g["seed"]
    o = {l: next(x for x in R["per_game"][l]["games"] if x["seed"] == s) for l in ["t320+term", "vnet+term"]}
    print(f"  seed {s} T={g['T']:3d}{' CAP' if s in capped else '    '} sum|d| {g['sum_abs']:.3f} / {o['t320+term']['sum_abs']:.3f} / {o['vnet+term']['sum_abs']:.3f}  signed {g['sum_signed']:+.3f}")
for l in ["t40+term", "t80+term", "t160+term", "t320+term", "t640+term", "t160", "vnet+term"]:
    G = [x for x in R["per_game"][l]["games"] if x["seed"] not in capped]
    s = np.array([x["sum_abs"] for x in G]); T = np.array([x["T"] for x in G])
    print(f"{l:10s} uncapped (n={len(G)}): sum|d| median {np.median(s):.3f} p90 {np.percentile(s,90):.3f} max {s.max():.3f}; 2*sum median {2*np.median(s):.2f}; len median {np.median(T):.0f} p90 {np.percentile(T,90):.0f}; frac games with 2*sum|d|>=2: {np.mean(2*s>=2):.2f}")
