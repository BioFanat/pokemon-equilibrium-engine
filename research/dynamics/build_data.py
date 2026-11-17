"""Rebuild the 1,629 real stage positions (residual/raw/game_*.pkl) with option lists, by deterministic
replay of the recorded (m1, m2) choices. Verified per turn: shape matches and h_s == nash._leaf(st,'mid').
Output: brainstorm/data.pkl = list of dicts {seed, turn, M (raw heuristic, side-one units), o1, o2, i1, j2,
h_s, I, J, P, H, T, shape}.
  cd engine && .venv/bin/python build_data.py
"""
import os, sys, pickle, glob, random
ENGINE = "/Users/biofanat/Documents/classes/f26/cs535/final/engine"
D = "/private/tmp/claude-501/-Users-biofanat-Documents-classes-f26-cs535/2bbe670b-2f97-4d9a-bfad-e1139e0352c6/scratchpad"
sys.path.insert(0, ENGINE); os.chdir(ENGINE)
import numpy as np
from multiprocessing import get_context


def replay(path):
    import nash, pkbattle as pb
    g = pickle.load(open(path, "rb"))
    rng = random.Random(g["seed"])
    st = pb.random_state(rng)
    out, bad = [], 0
    for d in g["states"]:
        o1, o2 = pb.options(st)
        ok = (tuple(d["shape"]) == (len(o1), len(o2))) and d["m1"] in o1 and d["m2"] in o2
        hs = nash._leaf(st, "mid")
        ok = ok and abs(hs - d["h_s"]) < 1e-9
        if not ok:
            bad += 1
            break
        if len(o1) > 1:
            rng.random()           # the one random() consumed by rng.choices(o1, weights=x) in play_collect
        M = np.zeros(d["shape"]); np.add.at(M, (d["I"], d["J"]), d["P"] * d["H"])
        out.append(dict(seed=g["seed"], turn=d["turn"], M=M, o1=o1, o2=o2, i1=o1.index(d["m1"]),
                        j2=o2.index(d["m2"]), h_s=d["h_s"], I=d["I"], J=d["J"], P=d["P"], H=d["H"], T=d["T"],
                        shape=tuple(d["shape"])))
        st = pb.step(st, d["m1"], d["m2"], rng)
    return g["seed"], len(g["states"]), len(out), bad, out


if __name__ == "__main__":
    files = sorted(glob.glob(f"{D}/residual/raw/game_*.pkl"))
    allrecs = []
    with get_context("spawn").Pool(4) as pool:
        for seed, n, got, bad, recs in pool.imap_unordered(replay, files):
            print(f"seed {seed}: states {n} replayed {got} mismatch {bad}", flush=True)
            allrecs += recs
    print("total", len(allrecs))
    pickle.dump(allrecs, open(f"{D}/brainstorm/data.pkl", "wb"))
