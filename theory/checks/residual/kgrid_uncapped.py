"""Fine k grid on UNCAPPED games' non-forced states, with a game-level bootstrap of the argmin of mean delta^2. -> kgrid_uncapped.json"""
import sys, glob, pickle, json
sys.path.insert(0, "/private/tmp/claude-501/-Users-biofanat-Documents-classes-f26-cs535/2bbe670b-2f97-4d9a-bfad-e1139e0352c6/scratchpad/residual")
import analyze as A
import numpy as np
from multiprocessing import get_context
if __name__ == "__main__":
    games = [g for g in (pickle.load(open(f, "rb")) for f in sorted(glob.glob(f"{A.D}/raw/game_*.pkl"))) if not g["capped"]]
    flat = [(g["seed"], d) for g in games for d in g["states"] if d["shape"][0] > 1]
    with get_context("spawn").Pool(4) as pool:
        res = pool.map(A.per_state, [d for _, d in flat], chunksize=8)
    seeds = np.array([s for s, _ in flat]); us = np.unique(seeds)
    ks = [k for k in A.KGRID]
    D2 = np.array([[r[f"t{k}+term"] for k in ks] for r in res])
    rng = np.random.default_rng(0); arg = []
    for _ in range(2000):
        pick = rng.choice(us, len(us)); idx = np.concatenate([np.flatnonzero(seeds == s) for s in pick])
        arg.append(ks[int(np.argmin((D2[idx] ** 2).mean(0)))])
    out = {"n_states": len(flat), "n_games": len(us), "k": ks,
           "mean_sq": (D2 ** 2).mean(0).tolist(), "mean_abs": np.abs(D2).mean(0).tolist(),
           "argmin_mean_sq": ks[int(np.argmin((D2 ** 2).mean(0)))],
           "bootstrap_argmin_quantiles": dict(zip(["p5", "p50", "p95"], np.percentile(arg, [5, 50, 95]).tolist()))}
    for k, a, b in zip(ks, out["mean_sq"], out["mean_abs"]):
        print(f"k={k:5d} mean d2 {a:.5f} mean|d| {b:.4f}")
    print("argmin", out["argmin_mean_sq"], "bootstrap", out["bootstrap_argmin_quantiles"])
    json.dump(out, open(f"{A.D}/kgrid_uncapped.json", "w"), indent=1)
