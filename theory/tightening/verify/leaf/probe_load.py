"""Discriminate why pe.id(child, 0) (nash._child_value's engine call) is non-deterministic under CPU load.

H1: depth-1 search cut short by the clock (would show changed values with depth_searched == 1).
H2: thread race / NaN handling (would show depth-1 matrices that differ from the unloaded reference).
H3: harness artifact (would show no change when the *same* in-memory states are re-queried).
H4 (from search.rs): depth 1 runs synchronously and can never be truncated; under load the background thread
    sometimes *completes* depth 2 (or deeper) before the main thread sets running=false (sleep(0) makes no
    syscall), so a complete deeper search is returned with depth_searched >= 2.

Usage: nice -n 10 engine/.venv/bin/python probe_load.py [reps] [levels comma-sep]
Writes out/probe_load.json. All load processes self-terminate at a hard deadline and are killed in `finally`.
"""
import json, os, random, signal, subprocess, sys, time
from collections import Counter, defaultdict
import numpy as np

ENGINE = "/Users/biofanat/Documents/classes/f26/cs535/final/engine"
sys.path.insert(0, ENGINE)
import poke_engine as pe, randbats as rb, pkbattle as pb, nash  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out"); os.makedirs(OUT, exist_ok=True)
PY = sys.executable
REPS = int(sys.argv[1]) if len(sys.argv) > 1 else 20
LEVELS = [int(x) for x in sys.argv[2].split(",")] if len(sys.argv) > 2 else [0, 4, 8, 12, 14, 20, 28]


# ------------------------------------------------------------------ states (same recipe as leaf_nondet.py)
def build_states(seed=0, ntr=12, kpt=40):
    rng = random.Random(seed); rows = []
    for trial in range(ntr):
        if trial % 2 == 0:
            t = rb.sample_team(rng, n=2)
            st = pe.State(side_one=pe.Side(pokemon=[rb.to_pokemon(t[0])]), side_two=pe.Side(pokemon=[rb.to_pokemon(t[1])])); kind = "1v1"
        else:
            st = rb.random_state(rng); kind = "6v6"
            for _ in range(rng.randint(0, 6)):
                o1, o2 = pb.options(st); st = pb.step(st, rng.choice(o1), rng.choice(o2), rng)
                if pb.alive(st.side_one) == 0 or pb.alive(st.side_two) == 0: break
        s1, s2 = pb.options(st)
        kids = [st.apply_instructions(br) for a in s1 for b in s2 for br in pe.generate_instructions(st, a, b)]
        rng.shuffle(kids)
        for c in kids[:kpt]:
            rows.append(dict(kind=kind, trial=trial, root=st, child=c, mirror=nash._mirror(c)))
    return rows


# ------------------------------------------------------------------ one engine call -> record
def call(st, ms=0):
    r = pe.id(st, ms)
    M = np.array([np.nan if v is None else v for v in r.matrix], dtype=float).reshape(len(r.s1), len(r.s2))
    with np.errstate(all="ignore"):
        val = float(np.nanmax(np.nanmin(M, axis=1)))            # == nash._child_value formula
    key = (r.depth_searched, tuple(r.s1), tuple(r.s2), tuple("nan" if v != v else repr(v) for v in M.ravel()))
    return dict(d=r.depth_searched, v=val, nan=int(np.isnan(M).sum()), key=key)


# ------------------------------------------------------------------ load generators
def start_load(n, secs):
    code = f"import time\nd=time.time()+{secs}\nx=0\nwhile time.time()<d:\n    x+=1\n"
    return [subprocess.Popen([PY, "-c", code], start_new_session=True) for _ in range(n)]


def stop_load(procs):
    for p in procs:
        try: os.killpg(p.pid, signal.SIGKILL)
        except ProcessLookupError: pass
    for p in procs: p.wait()


def main():
    t_start = time.time()
    rows = build_states()
    states = [(i, side, r[side]) for i, r in enumerate(rows) for side in ("child", "mirror")]
    print(f"{len(rows)} children, {len(states)} engine states; reps={REPS}; levels={LEVELS}", flush=True)
    res = dict(meta=dict(n_children=len(rows), reps=REPS, levels=LEVELS, cpu="Apple M4 Max 10P+4E",
                         note="pid 16335 (orphan engine-venv mp worker) was burning ~1 core throughout"))

    # A. unloaded reference (3 reps) + check that the formula equals nash._child_value
    ref = {}
    for i, side, st in states:
        recs = [call(st) for _ in range(3)]
        assert len({x["key"] for x in recs}) == 1 and recs[0]["d"] == 1, (i, side, [x["d"] for x in recs])
        ref[(i, side)] = recs[0]
    agree = sum(abs(nash._child_value(st) - ref[(i, side)]["v"]) < 1e-9 for i, side, st in states[:200])
    res["A_reference"] = dict(all_depth1_and_identical_over_3=True, formula_matches_nash_child_value=f"{agree}/200")
    ref_leaf = {i: 0.5 * (ref[(i, "child")]["v"] - ref[(i, "mirror")]["v"]) for i in range(len(rows))}
    d1_range = {}
    for i, side, st in states:
        r = pe.id(st, 0); M = np.array([np.nan if v is None else v for v in r.matrix], float)
        d1_range[(i, side)] = float(np.nanmax(M) - np.nanmin(M))
    # battle_is_over as in state.rs:1359 (-1 if side one all fainted, checked first; +1 if side two all fainted)
    bo = {(i, side): (-1 if pb.alive(st.side_one) == 0 else (1 if pb.alive(st.side_two) == 0 else 0))
          for i, side, st in states}
    term_leaf = {i: bo[(i, "child")] != 0 for i in range(len(rows))}
    res["A_reference"]["terminal_children"] = {k: sum(term_leaf[i] for i, r in enumerate(rows) if r["kind"] == k)
                                               for k in ("1v1", "6v6")}
    res["A_reference"]["terminal_d1_range_max"] = max([d1_range[k] for k in bo if bo[k]] or [0])
    print("A done", res["A_reference"], flush=True)

    # B. load levels
    deeper = defaultdict(set)            # (i, side, depth) -> set of matrix keys seen under load
    B = {}
    for n in LEVELS:
        procs = start_load(n, secs=120)
        try:
            time.sleep(0.5)
            per = []                       # (i, side, rep, rec)
            t0 = time.perf_counter()
            for rep in range(REPS):
                for i, side, st in states:
                    per.append((i, side, rep, call(st)))
            wall = time.perf_counter() - t0
        finally:
            stop_load(procs)
        depth_c = Counter(x[3]["d"] for x in per)
        changed = [(i, side, rep, x) for i, side, rep, x in per if x["key"] != ref[(i, side)]["key"]]
        changed_val = [(i, side, rep, x) for i, side, rep, x in changed if abs(x["v"] - ref[(i, side)]["v"]) > 1e-9]
        d1_changed = [c for c in changed if c[3]["d"] == 1]
        ch_term = [c for c in changed if bo[(c[0], c[1])]]
        ch_nont = [c for c in changed if not bo[(c[0], c[1])]]
        # terminal prediction: v_load - v_ref == 100*(d-1)*bo  (f32 arithmetic -> small tolerance)
        formula_ok = sum(abs((x["v"] - ref[(i, s)]["v"]) - 100.0 * (x["d"] - 1) * bo[(i, s)]) <= 1e-3 * max(1.0, abs(x["v"]))
                         for i, s, _, x in ch_term)
        nont_bad_depth = sum(x["d"] < 2 for *_, x in ch_nont)
        nont_dev = [abs(x["v"] - ref[(i, s)]["v"]) for i, s, _, x in ch_nont]
        nont_rel = [abs(x["v"] - ref[(i, s)]["v"]) / max(d1_range[(i, s)], 1e-9) for i, s, _, x in ch_nont]
        for i, side, rep, x in ch_nont:
            deeper[(i, side, x["d"])].add(x["key"])
        # leaf level (what full_matrix consumes): leaf = 0.5*(v_child - v_mirror) per rep
        vals = {(i, side, rep): x["v"] for i, side, rep, x in per}
        leaf_changed, leaf_dev = 0, []
        leaf_nt = dict(evals=0, changed=0, dev=[])
        for i in range(len(rows)):
            for rep in range(REPS):
                lv = 0.5 * (vals[(i, "child", rep)] - vals[(i, "mirror", rep)])
                hit = abs(lv - ref_leaf[i]) > 1e-9
                if hit:
                    leaf_changed += 1; leaf_dev.append(abs(lv - ref_leaf[i]))
                if not term_leaf[i]:
                    leaf_nt["evals"] += 1
                    if hit: leaf_nt["changed"] += 1; leaf_nt["dev"].append(abs(lv - ref_leaf[i]))
        by_kind = {}
        for kind in ("1v1", "6v6"):
            idx = {i for i, r in enumerate(rows) if r["kind"] == kind}
            ch = {(i, side) for i, side, _, _ in changed if i in idx}
            calls_k = [x for x in per if x[0] in idx]
            chv = [abs(x["v"] - ref[(i, side)]["v"]) for i, side, _, x in changed_val if i in idx]
            by_kind[kind] = dict(calls=len(calls_k), calls_not_depth1=sum(x[3]["d"] != 1 for x in calls_k),
                                 states_ever_changed=len(ch), states=2 * len(idx),
                                 abs_dev_median=float(np.median(chv)) if chv else 0.0, abs_dev_max=float(max(chv)) if chv else 0.0)
        nan_ref = np.mean([ref[(i, s)]["nan"] for i, s, _ in states])
        nan_d1 = np.mean([x["nan"] for *_, x in per if x["d"] == 1])
        top = sorted(changed_val, key=lambda c: -abs(c[3]["v"] - ref[(c[0], c[1])]["v"]))[:5]
        B[n] = dict(wall_s=round(wall, 2), calls=len(per), depth_counts=dict(depth_c),
                    frac_calls_not_depth1=1 - depth_c.get(1, 0) / len(per),
                    calls_changed_matrix=len(changed), calls_changed_value=len(changed_val),
                    changed_with_depth1=len(d1_changed),
                    leaf_evals=len(rows) * REPS, leaf_changed=leaf_changed,
                    leaf_dev_median=float(np.median(leaf_dev)) if leaf_dev else 0.0,
                    leaf_dev_max=float(max(leaf_dev)) if leaf_dev else 0.0,
                    nan_mean_ref=float(nan_ref), nan_mean_depth1_loaded=float(nan_d1), by_kind=by_kind,
                    changed_terminal=len(ch_term), terminal_formula_ok=int(formula_ok),
                    changed_nonterminal=len(ch_nont), nonterminal_changed_with_depth_lt2=int(nont_bad_depth),
                    nonterminal_depths=dict(Counter(x["d"] for *_, x in ch_nont)),
                    nonterminal_abs_dev_median=float(np.median(nont_dev)) if nont_dev else 0.0,
                    nonterminal_abs_dev_max=float(max(nont_dev)) if nont_dev else 0.0,
                    nonterminal_dev_over_d1range_median=float(np.median(nont_rel)) if nont_rel else 0.0,
                    nonterminal_dev_over_d1range_max=float(max(nont_rel)) if nont_rel else 0.0,
                    leaf_nonterminal=dict(evals=leaf_nt["evals"], changed=leaf_nt["changed"],
                                          dev_median=float(np.median(leaf_nt["dev"])) if leaf_nt["dev"] else 0.0,
                                          dev_max=float(max(leaf_nt["dev"])) if leaf_nt["dev"] else 0.0),
                    top_dev=[dict(kind=rows[i]["kind"], side=side, d=x["d"], v=x["v"], ref=ref[(i, side)]["v"],
                                  d1_range=d1_range[(i, side)],
                                  alive=[pb.alive(rows[i][side].side_one), pb.alive(rows[i][side].side_two)])
                             for i, side, rep, x in top])
        print(f"N={n}: wall {wall:.1f}s depth {dict(depth_c)} changed-matrix {len(changed)} changed-value {len(changed_val)} "
              f"changed@depth1 {len(d1_changed)} leaf_changed {leaf_changed}/{len(rows)*REPS} | terminal {len(ch_term)} "
              f"(formula ok {formula_ok}) nonterminal {len(ch_nont)} depths {dict(Counter(x['d'] for *_, x in ch_nont))} "
              f"leaf_nt {leaf_nt['changed']}/{leaf_nt['evals']}", flush=True)
    res["B_load"] = B

    # C. reproduce every loaded depth>=2 result unloaded with pe.id(st, t): bitwise equal -> complete deeper search
    idx_state = {(i, side): st for i, side, st in states}
    C = dict(pairs=len(deeper), reproduced=0, not_reproduced=[], multiple_keys_same_depth=0)
    for (i, side, d), keys in deeper.items():
        if len(keys) > 1: C["multiple_keys_same_depth"] += 1
        got = set()
        for t in (1, 1, 2, 2, 3, 5, 5, 8, 10, 15, 20):
            x = call(idx_state[(i, side)], t)
            if x["d"] == d: got.add(x["key"])
            if keys <= got: break
        bucket = "d2-3" if 2 <= d <= 3 else "d>3_or_wrapped"
        C.setdefault(bucket, dict(pairs=0, reproduced=0))["pairs"] += 1
        if keys <= got: C["reproduced"] += 1; C[bucket]["reproduced"] += 1
        else: C["not_reproduced"].append(dict(i=i, side=side, d=d, n_keys=len(keys), got=len(got), kind=rows[i]["kind"]))
    C["not_reproduced"] = C["not_reproduced"][:20]
    res["C_reproduce"] = C
    print("C", {k: v for k, v in C.items() if k != "not_reproduced"}, "not reproduced:", len(C["not_reproduced"]), flush=True)

    # D. mechanism timing: unloaded depth reached for a budget t ms (subset)
    sub = [s for s in states[::4] if not bo[(s[0], s[1])]]
    D = {}
    for t in (1, 2, 5, 10):
        D[t] = {}
        for kind in ("1v1", "6v6"):
            dc = Counter(call(st, t)["d"] for i, _, st in sub if rows[i]["kind"] == kind)
            D[t][kind] = dict(sorted(dc.items()))
    # time for the worker to finish depth 2 = time for pe.id(c, t) to first return d>=2: estimate depth-2 search time
    # by timing pe.id(st, 0) vs a depth-2 proxy is not exposed; report D instead.
    res["D_depth_vs_budget_ms"] = D
    print("D", D, flush=True)

    # E. retry wrapper under heavy load: accept only depth_searched == 1
    E = {}
    for n in [x for x in LEVELS if x >= 12]:
        procs = start_load(n, secs=120)
        try:
            time.sleep(0.5)
            tries, mism, ts = [], 0, []
            for rep in range(5):
                for i, side, st in states:
                    k = 0; t0 = time.perf_counter()
                    while True:
                        k += 1; x = call(st)
                        if x["d"] == 1: break
                    ts.append(time.perf_counter() - t0); tries.append(k)
                    if x["key"] != ref[(i, side)]["key"]: mism += 1
        finally:
            stop_load(procs)
        E[n] = dict(calls=len(tries), retries=int(sum(tries) - len(tries)), max_tries=int(max(tries)),
                    accepted_mismatch_vs_ref=mism, ms_median=1e3 * float(np.median(ts)), ms_mean=1e3 * float(np.mean(ts)))
        print("E", n, E[n], flush=True)
    res["E_retry_wrapper"] = E
    res["meta"]["total_s"] = round(time.time() - t_start, 1)
    json.dump(res, open(os.path.join(OUT, "probe_load.json"), "w"), indent=1, default=str)
    print("total", res["meta"]["total_s"], "s", flush=True)


if __name__ == "__main__":
    main()
