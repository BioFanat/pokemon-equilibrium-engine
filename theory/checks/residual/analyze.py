"""Minimax TD residual delta(s) = Vhat(s) - val(Q[Vhat](s)) from raw/game_*.pkl (play_collect.py).
val = nash.lp (scipy linprog, HiGHS). Leaves:
  tK+term : Vhat = tanh(mid/K), terminal children exact (+1/-1/0)
  tK      : Vhat = tanh(mid/K) everywhere (terminal children get tanh of their material leaf)
  vnet+term: value_net, terminal children exact
  cd engine && .venv/bin/python analyze.py  -> residual.json
"""
import os, sys, glob, json, pickle
ENGINE = "/Users/biofanat/Documents/classes/f26/cs535/final/engine"
D = "/private/tmp/claude-501/-Users-biofanat-Documents-classes-f26-cs535/2bbe670b-2f97-4d9a-bfad-e1139e0352c6/scratchpad/residual"
sys.path.insert(0, ENGINE); os.chdir(ENGINE)
import numpy as np
from multiprocessing import get_context

KMAIN = [40, 80, 160, 320, 640]
KGRID = sorted(set(np.round(np.geomspace(10, 5120, 28)).astype(int).tolist()) | set(KMAIN))


def leaves(d):
    """yield (name, Vhat(s), child values)"""
    T, H, term = d["T"], d["H"], ~np.isnan(d["T"])
    for k in KGRID:
        th = np.tanh(H / k)
        yield f"t{k}+term", np.tanh(d["h_s"] / k), np.where(term, T, th)
        if k in KMAIN:
            yield f"t{k}", np.tanh(d["h_s"] / k), th
    yield "vnet+term", d["vn_s"], np.where(term, T, d["VN"])


def per_state(d):
    import nash
    out = {}
    for name, v, cv in leaves(d):
        Q = np.zeros(d["shape"]); np.add.at(Q, (d["I"], d["J"]), d["P"] * cv)
        out[name] = float(v - nash.lp(Q)[2])
    return out


def sanity(d):
    import nash
    Q = np.zeros(d["shape"]); np.add.at(Q, (d["I"], d["J"]), d["P"] * np.where(np.isnan(d["T"]), np.tanh(d["H"] / 160), d["T"]))
    x, y, v = nash.lp(Q); xr, yr = nash.rm_plus(Q, 5000)
    return abs(v - float(xr @ Q @ yr)), nash.exploitability(Q, xr, yr), float(np.abs(d["P"].sum() - d["shape"][0] * d["shape"][1]))


def stats(a):
    a = np.asarray(a)
    return dict(n=int(a.size), mean_abs=float(np.mean(np.abs(a))), median_abs=float(np.median(np.abs(a))),
                p90_abs=float(np.percentile(np.abs(a), 90)), mean=float(a.mean()), mean_sq=float(np.mean(a ** 2)),
                max_abs=float(np.abs(a).max()))


if __name__ == "__main__":
    games = [pickle.load(open(f, "rb")) for f in sorted(glob.glob(f"{D}/raw/game_*.pkl"))]
    flat = [(g["seed"], d) for g in games for d in g["states"]]
    with get_context("spawn").Pool(4) as pool:
        res = pool.map(per_state, [d for _, d in flat], chunksize=8)
        san = pool.map(sanity, [d for _, d in flat[::max(1, len(flat) // 40)]])
    san = np.array(san)
    names = list(res[0])
    nonforced = np.array([d["shape"][0] > 1 for _, d in flat])
    hasterm = np.array([bool(np.any(~np.isnan(d["T"]))) for _, d in flat])
    R = {n: np.array([r[n] for r in res]) for n in names}
    vabs = {f"t{k}": float(np.mean([abs(np.tanh(d["h_s"] / k)) for _, d in flat if d["shape"][0] > 1])) for k in KGRID}
    out = dict(n_games=len(games), n_states_all=len(flat), n_states_nonforced=int(nonforced.sum()),
               n_hasterm_nonforced=int((nonforced & hasterm).sum()),
               winners={w: sum(g["winner"] == w for g in games) for w in (0, 1, 2)}, capped=sum(g["capped"] for g in games),
               game_len=[g["turns"] for g in games],
               sanity=dict(lp_vs_rm5000_value_gap_max=float(san[:, 0].max()), rm5000_nashgap_max=float(san[:, 1].max()),
                           prob_mass_err_max=float(san[:, 2].max()), n=len(san)),
               mid_leaf_abs_quantiles=dict(zip(["p50", "p90", "p99"], np.percentile([abs(d["h_s"]) for _, d in flat], [50, 90, 99]).tolist())),
               term_child_H=dict(zip(["p10", "p50", "p90"], np.percentile(np.concatenate([np.abs(d["H"][~np.isnan(d["T"])]) for _, d in flat if np.any(~np.isnan(d["T"]))] or [np.zeros(1)]), [10, 50, 90]).tolist())),
               mean_abs_vhat_nonforced=vabs, by_leaf={}, per_game={})
    for n in names:
        a = R[n]
        out["by_leaf"][n] = dict(nonforced=stats(a[nonforced]), all=stats(a),
                                 nonforced_hasterm=stats(a[nonforced & hasterm]) if (nonforced & hasterm).any() else None,
                                 nonforced_noterm=stats(a[nonforced & ~hasterm]))
    seeds = np.array([s for s, _ in flat])
    for n in ["t40+term", "t80+term", "t160+term", "t320+term", "t640+term", "t160", "vnet+term"]:
        per = []
        for g in games:
            a = R[n][seeds == g["seed"]]
            per.append(dict(seed=g["seed"], T=g["turns"], sum_abs=float(np.abs(a).sum()), sum_signed=float(a.sum()),
                            bound_2sum_abs=float(2 * np.abs(a).sum())))
        s_abs = np.array([p["sum_abs"] for p in per]); T = np.array([p["T"] for p in per])
        out["per_game"][n] = dict(games=per, sum_abs_median=float(np.median(s_abs)), sum_abs_p90=float(np.percentile(s_abs, 90)),
                                  sum_abs_mean=float(s_abs.mean()), len_median=float(np.median(T)), len_p90=float(np.percentile(T, 90)),
                                  sum_signed_median=float(np.median([p["sum_signed"] for p in per])),
                                  per_turn_mean_abs=float(s_abs.sum() / T.sum()))
    out["per_state"] = [dict(seed=int(s), turn=int(d["turn"]), shape=list(d["shape"]), hasterm=bool(ht), h_s=float(d["h_s"]),
                             delta={k: round(r[k], 6) for k in ["t40+term", "t80+term", "t160+term", "t320+term", "t640+term", "t160", "vnet+term"]})
                        for (s, d), r, ht in zip(flat, res, hasterm)]
    json.dump(out, open(f"{D}/residual.json", "w"), indent=1)
    print(json.dumps({k: v for k, v in out.items() if k not in ("per_state", "per_game", "by_leaf")}, indent=1))
    print(f"\n{'leaf':12s} {'mean|d|':>8s} {'med|d|':>8s} {'p90|d|':>8s} {'mean d':>8s} {'mean d2':>8s} {'max|d|':>7s} | mean|Vhat| | hasterm mean|d| / noterm mean|d| (n={out['n_hasterm_nonforced']})")
    for n in names:
        s = out["by_leaf"][n]["nonforced"]; ht = out["by_leaf"][n]["nonforced_hasterm"]; nt = out["by_leaf"][n]["nonforced_noterm"]
        va = vabs.get(n.split("+")[0], float("nan"))
        print(f"{n:12s} {s['mean_abs']:8.4f} {s['median_abs']:8.4f} {s['p90_abs']:8.4f} {s['mean']:+8.4f} {s['mean_sq']:8.5f} {s['max_abs']:7.3f} | {va:.3f} | {ht['mean_abs'] if ht else float('nan'):.4f} / {nt['mean_abs']:.4f}")
    print("\nper game (all states, t<tau):")
    for n, p in out["per_game"].items():
        print(f"{n:12s} sum|d| median {p['sum_abs_median']:.3f} p90 {p['sum_abs_p90']:.3f} mean {p['sum_abs_mean']:.3f} | 2*sum|d| median {2*p['sum_abs_median']:.2f} vs range 2 | len median {p['len_median']:.0f} p90 {p['len_p90']:.0f} | signed sum median {p['sum_signed_median']:+.3f} | per-turn mean|d| {p['per_turn_mean_abs']:.4f}")
