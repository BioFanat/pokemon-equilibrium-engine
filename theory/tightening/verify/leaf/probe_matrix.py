"""Decision-level impact: nash.full_matrix(root) under CPU load (stock nash._child_value, i.e. pe.id(c, 0)) vs a
deterministic reference (nash._child_value monkeypatched with the depth_searched == 1 retry wrapper, unloaded).
Also times full_matrix with the stock leaf vs the wrapper.

Usage: nice -n 10 engine/.venv/bin/python probe_matrix.py <levels comma-sep> <reps> -> out/matrix.json
No project file is edited; the patch is in-process only.
"""
import json, os, random, sys, time
import numpy as np
ARGV = sys.argv[1:]; sys.argv = sys.argv[:1]
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from probe_load import start_load, stop_load, OUT  # noqa: E402
from probe_load import pe, rb, pb, nash  # noqa: E402

LEVELS = [int(x) for x in ARGV[0].split(",")]; REPS = int(ARGV[1])
STOCK = nash._child_value


def child_value_det(child):
    """Deterministic drop-in for nash._child_value: retry pe.id(child, 0) until depth_searched == 1."""
    while True:
        r = pe.id(child, 0)
        if r.depth_searched == 1:
            break
    M = np.array([np.nan if v is None else v for v in r.matrix], dtype=float).reshape(len(r.s1), len(r.s2))
    with np.errstate(all="ignore"):
        return float(np.nanmax(np.nanmin(M, axis=1)))


def roots(seed=1):
    rng = random.Random(seed); out = []
    while sum(k == "6v6" for k, _ in out) < 10:                 # midgame 6v6
        st = rb.random_state(rng)
        for _ in range(rng.randint(0, 8)):
            o1, o2 = pb.options(st); st = pb.step(st, rng.choice(o1), rng.choice(o2), rng)
        if pb.alive(st.side_one) and pb.alive(st.side_two): out.append(("6v6", st))
    tries = 0
    while sum(k == "endgame" for k, _ in out) < 8 and tries < 400:  # random play until <= 4 mons alive in total
        tries += 1; st = rb.random_state(rng)
        for _ in range(150):
            o1, o2 = pb.options(st); st = pb.step(st, rng.choice(o1), rng.choice(o2), rng)
            a1, a2 = pb.alive(st.side_one), pb.alive(st.side_two)
            if a1 == 0 or a2 == 0: break
            if a1 + a2 <= 4: out.append(("endgame", st)); break
    for _ in range(6):                                            # 1v1
        t = rb.sample_team(rng, n=2)
        out.append(("1v1", pe.State(side_one=pe.Side(pokemon=[rb.to_pokemon(t[0])]),
                                    side_two=pe.Side(pokemon=[rb.to_pokemon(t[1])]))))
    return out


def summarize(M, Mr):
    rng_ = float(np.ptp(Mr)) or 1.0
    x, _, v = nash.lp(M); xr, _, vr = nash.lp(Mr)
    return dict(cells_changed=int((np.abs(M - Mr) > 1e-6).sum()), cells=int(M.size),
                max_dev_over_range=float(np.abs(M - Mr).max() / rng_), lp_tv=float(0.5 * np.abs(x - xr).sum()),
                argmax_changed=bool(np.argmax(x) != np.argmax(xr)), value_dev_over_range=float(abs(v - vr) / rng_))


R = roots()
print({k: sum(kk == k for kk, _ in R) for k in ("6v6", "endgame", "1v1")}, "roots", flush=True)
nash._child_value = child_value_det
ref, t_det = [], []
for kind, st in R:
    t0 = time.perf_counter(); ref.append(nash.full_matrix(st)[0]); t_det.append(time.perf_counter() - t0)
again = [nash.full_matrix(st)[0] for _, st in R]
assert all(np.array_equal(a, b) for a, b in zip(ref, again)), "reference not reproducible"
nash._child_value = STOCK
t_stock = []
for kind, st in R:
    t0 = time.perf_counter(); nash.full_matrix(st); t_stock.append(time.perf_counter() - t0)
res = dict(roots={k: sum(kk == k for kk, _ in R) for k in ("6v6", "endgame", "1v1")},
           ref_range_median={k: float(np.median([np.ptp(M) for (kk, _), M in zip(R, ref) if kk == k])) for k in ("6v6", "endgame", "1v1")},
           ms_full_matrix_unloaded=dict(stock_total=1e3 * sum(t_stock), det_total=1e3 * sum(t_det)), levels={})
print(res, flush=True)
for n in LEVELS:
    procs = start_load(n, secs=300)
    rec = []
    try:
        time.sleep(0.5)
        for rep in range(REPS):
            for (kind, st), Mr in zip(R, ref):
                rec.append(dict(kind=kind, **summarize(nash.full_matrix(st)[0], Mr)))
    finally:
        stop_load(procs)
    agg = {}
    for kind in ("6v6", "endgame", "1v1"):
        g = [r for r in rec if r["kind"] == kind]
        agg[kind] = dict(matrices=len(g), matrices_changed=sum(r["cells_changed"] > 0 for r in g),
                         cells_changed=sum(r["cells_changed"] for r in g), cells=sum(r["cells"] for r in g),
                         max_dev_over_range_max=max(r["max_dev_over_range"] for r in g),
                         lp_tv_max=max(r["lp_tv"] for r in g), lp_tv_mean=float(np.mean([r["lp_tv"] for r in g])),
                         argmax_changed=sum(r["argmax_changed"] for r in g),
                         value_dev_over_range_max=max(r["value_dev_over_range"] for r in g))
    res["levels"][n] = agg
    print("N", n, agg, flush=True)
json.dump(res, open(os.path.join(OUT, "matrix.json"), "w"), indent=1)
