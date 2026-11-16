"""Verifier: is 'TV live' (E_BR[TV_1] > 0) a property of the game or of tera_subgame's tie-breaking among best
responses? For each seed: rebuild with OURBOT_DET_LEAF=1, recompute the A3 agent exactly as ts.analyse does, list
per-type argmin sets of stage-0 beta (tol 1e-9*Delta), enumerate all pure BR selections, report min/max E[TV_1],
and the minimum eta over pre-tera children on the belief grid (Jensen sanity: must be >= -LP tol)."""
import itertools, json, os, random, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "scripts"))
assert os.environ.get("OURBOT_DET_LEAF") == "1"
import tera_subgame as ts  # noqa: E402
import hidden_agents as ha  # noqa: E402
ts._stable = lambda fn, key, tries=12, need=3: fn()


def run(seed):
    own, opp, teras = ts.pick_matchup(random.Random(seed))
    states, A0, B0, info = ts.build(opp, own, teras); ts.stage1(info)
    k = len(teras); b0 = {th: 1 / k for th in teras}
    allu = np.concatenate([d["U"][th].ravel() for d in info.values() for th in d["U"]]); D = float(allu.max() - allu.min())
    Mw = {th: ts.backward_single(info, A0, B0, th)[1] for th in teras}
    x0, _ = ha.agent_form_lp([Mw[th] for th in teras], [b0[th] for th in teras])
    x1 = {}
    for I, d in info.items():
        ths = [th for th in teras if th in d["U"]]
        if d["U"][ths[0]].size == 1: x1[I] = np.array([1.0]); continue
        if len(ths) == 1: x1[I] = ha.lp_x(d["U"][ths[0]])[0]; continue
        w = np.array([b0[th] * d["P"][th] for th in ths]); w /= w.sum()
        x1[I] = ha.agent_form_lp([d["U"][th] for th in ths], w)[0]
    val = {th: {} for th in teras}
    for th in teras:
        for beta in B0:
            val[th][beta] = sum(x0[A0.index(d["a"])] * d["P"][th] * float((d["U"][th].T @ x1[I]).min())
                                for I, d in info.items() if d["beta"] == beta and th in d["U"])
    arg = {th: [b for b in B0 if val[th][b] <= min(val[th].values()) + 1e-9 * D] for th in teras}

    def etv(bs):
        e = 0.0
        for I, d in info.items():
            ths = [th for th in teras if th in d["U"] and bs[th] == d["beta"]]
            if not ths: continue
            pr = sum(b0[th] * x0[A0.index(d["a"])] * d["P"][th] for th in ths)
            if pr <= 0: continue
            allt = [th for th in teras if th in d["U"]]
            post = np.array([b0[th] * d["P"][th] * (th in ths) for th in allt]); post /= post.sum()
            hard = np.array([b0[th] * d["P"][th] for th in allt]); hard /= hard.sum()
            e += pr * 0.5 * np.abs(post - hard).sum()
        return e
    tvs = [etv(dict(zip(teras, combo))) for combo in itertools.product(*[arg[th] for th in teras])]
    expl_V0 = None
    grid = list(ts.simplex_grid(k, 0.1)); emin = 0.0
    for I, d in info.items():
        if len(d["U"]) == k and d["U"][teras[0]].size > 1:
            for b in grid:
                emin = min(emin, ts.eta_at(d, teras, b))
    return dict(seed=seed, opp=opp.species, own=own.species, Delta=D, argmin={th: arg[th] for th in teras},
                n_br=len(tvs), etv_min=min(tvs), etv_max=max(tvs), eta_min_grid=emin,
                br_vals={th: {b: round(v, 4) for b, v in val[th].items()} for th in teras})


if __name__ == "__main__":
    seeds = [int(s) for s in sys.argv[1].split(",")]
    from multiprocessing import get_context
    with get_context("spawn").Pool(2) as pool, open(sys.argv[2], "a") as f:
        for out in pool.imap_unordered(run, seeds):
            f.write(json.dumps(out) + "\n"); f.flush()
            print(out["seed"], out["opp"], "argmin", out["argmin"], "E[TV] min/max", round(out["etv_min"], 3), round(out["etv_max"], 3), "eta_min", out["eta_min_grid"], flush=True)
