"""Play games: agent = 'Equilibrium, sampled' (rm_plus(full_matrix,1000) average, sampled) on side one
vs pb.mcts_policy(2000) on side two, random teams. At EVERY state of the trajectory dump the raw data
needed for the minimax TD residual of several leaves: h(s)=nash._leaf(s,'mid'), vnet(s), and per child
(i, j, p, h(child), terminal value or nan, vnet(child) or nan). One pickle per game.
  cd engine && OMP_NUM_THREADS=1 .venv/bin/python play_collect.py N_GAMES [WORKERS] [SEED0]
"""
import os, sys, pickle, random, time
ENGINE = "/Users/biofanat/Documents/classes/f26/cs535/final/engine"
OUT = "/private/tmp/claude-501/-Users-biofanat-Documents-classes-f26-cs535/2bbe670b-2f97-4d9a-bfad-e1139e0352c6/scratchpad/residual/raw"
sys.path.insert(0, ENGINE); os.chdir(ENGINE)
from multiprocessing import get_context


def collect(st, o1, o2):
    import numpy as np, poke_engine as pe, nash, pkbattle as pb, value_net
    I, J, P, H, T, live = [], [], [], [], [], []
    kids = []
    for i, a in enumerate(o1):
        for j, b in enumerate(o2):
            for br in pe.generate_instructions(st, a, b):
                c = st.apply_instructions(br)
                a1, a2 = pb.alive(c.side_one), pb.alive(c.side_two)
                tv = np.nan if (a1 and a2) else (0.0 if a1 == a2 else (1.0 if a1 else -1.0))
                I.append(i); J.append(j); P.append(br.percentage / 100.0); T.append(tv)
                H.append(nash._leaf(c, "mid")); kids.append(c)
    T = np.array(T); VN = np.full(len(kids), np.nan)
    li = np.flatnonzero(np.isnan(T))
    if li.size:
        VN[li] = value_net.value_many([kids[k] for k in li])
    d = dict(shape=(len(o1), len(o2)), h_s=nash._leaf(st, "mid"), vn_s=float(value_net.value_many([st])[0]),
             I=np.array(I), J=np.array(J), P=np.array(P), H=np.array(H), T=T, VN=VN)
    M = np.zeros(d["shape"]); np.add.at(M, (d["I"], d["J"]), d["P"] * d["H"])
    return d, M


def game(seed):
    import numpy as np, nash, pkbattle as pb
    rng = random.Random(seed)
    st = pb.random_state(rng)
    opp = pb.mcts_policy(2000)
    recs, t0, winner = [], time.perf_counter(), None
    for turn in range(1, 301):
        o1, o2 = pb.options(st)
        d, M = collect(st, o1, o2)
        if seed == SANITY_SEED and turn <= 2:
            assert np.allclose(M, nash.full_matrix(st)[0]), "reconstructed M != full_matrix"
        if len(o1) == 1:
            m1 = o1[0]
        else:
            x, _ = nash.rm_plus(M, 1000); x = np.clip(x, 0, None)
            m1 = rng.choices(o1, weights=x / x.sum())[0]
        m2 = opp(st, o2, 2, rng)
        d.update(turn=turn, m1=m1, m2=m2); recs.append(d)
        st = pb.step(st, m1, m2, rng)
        a1, a2 = pb.alive(st.side_one), pb.alive(st.side_two)
        if a1 == 0 or a2 == 0:
            winner = 0 if a1 == a2 else (1 if a1 else 2); break
    capped = winner is None
    out = dict(seed=seed, winner=0 if capped else winner, capped=capped, turns=len(recs), states=recs,
               secs=time.perf_counter() - t0)
    with open(f"{OUT}/game_{seed}.pkl", "wb") as f:
        pickle.dump(out, f)
    return seed, out["winner"], out["turns"], out["secs"]


SANITY_SEED = 1000

if __name__ == "__main__":
    n = int(sys.argv[1]); w = int(sys.argv[2]) if len(sys.argv) > 2 else 4
    s0 = int(sys.argv[3]) if len(sys.argv) > 3 else 1000
    os.makedirs(OUT, exist_ok=True)
    t0 = time.perf_counter()
    with get_context("spawn").Pool(w) as pool:
        for seed, win, turns, secs in pool.imap_unordered(game, range(s0, s0 + n)):
            print(f"game {seed}: winner {win} turns {turns} {secs:.0f}s (wall {time.perf_counter()-t0:.0f}s)", flush=True)
    print("DONE", flush=True)
