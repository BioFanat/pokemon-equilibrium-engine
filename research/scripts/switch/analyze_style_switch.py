"""Analysis of the pre-registered style-posterior switching test (research/prereg_style_switch.md, T5 of
theory/tightening/bayes_dynamics.md section 6): per-stream switch metrics, medians with stream-bootstrap intervals,
and the P1-P8 verdicts with the prereg's refutation criteria applied literally.

Per arm: a replays root with <tag>-s<seed>-g<game>.json and .A.decisions.jsonl (batch.py writes
$OURBOT_WORK/replays_headless). The carried state <tag>-stream<k>-A.bayes.json is looked up in <root>/../hedge_state
(optional). Every posterior quantity is recomputed from the logged p_style by the arm's recursion from the uniform prior
(bayes: logP += log p; ptd: z = lam logP + log p; dtp: z = lam (logP + log p)); the logged 4-decimal posteriors are only
the validation target (> 1e-3 at any update: stream unclean, excluded).

Run (dry validation on the RQ controls, treated as no-switch arms):
  ~/sandbox/bot-venv/bin/python research/scripts/switch/analyze_style_switch.py \
      --arm RQsafe=<results>/RQsafe/replays_headless --arm RQunif=<results>/RQunif/replays_headless --out <dir>
Arms of the prereg: --arm SW2=$OURBOT_WORK/replays_headless --arm SW4=... --arm SW4p=...:ptd:0.98 --arm SW4d=...:dtp:0.98
"""
import argparse
import json
import math
import os
import sys
import tempfile

import numpy as np
from scipy import stats

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path[:0] = [os.path.join(ROOT, "engine", "analysis", "open_items"), os.path.join(ROOT, "engine", "analysis", "synthesis")]
import common  # noqa: E402
import oi_common  # noqa: E402
from common import STYLES, records  # noqa: E402
from oi_common import score, mean_ci_normal  # noqa: E402

CONTROLS = "/Users/biofanat/Documents/classes/f26/cs535/final/engine/showdown/results"
PRE_METHOD, PRE, POST = "style_safe", "safe", "uniform"           # the switch pair: style_safe -> style_uniform
IU, IS = STYLES.index(POST), STYLES.index(PRE)
MU_U = dict(eq=1.773, br_unif=0.293, br_us=0.312, safe=0.395, uniform=0.0)   # mu_{u->s}, RQunif (prereg section 3)
LN5 = math.log(len(STYLES))
SUST, BURN = 20, 20                       # sustained-20 rule; false alarms skip the first 20 updates of a fresh prior
VAL_TOL, ID_REC_TOL, ID_LOG_TOL = 1e-3, 1e-3, 5e-3
NBOOT = 2000
NSTREAMS = 10


# ---------------------------------------------------------------- loading
def arm_streams(root, tag):
    """[(k, [(seed, game, decisions path), ...]), ...] in play order, via common.streams (which globs
    RES/<tag>/replays_headless): RES is pointed at a temporary symlink to `root`. Records and game JSONs are read here,
    inside the symlink's lifetime."""
    out = []
    with tempfile.TemporaryDirectory() as d:
        os.makedirs(os.path.join(d, tag))
        os.symlink(os.path.abspath(root), os.path.join(d, tag, "replays_headless"))
        old, common.RES = common.RES, d
        try:
            for k, order in common.streams(tag, "A", NSTREAMS):
                out.append((k, load_stream(order)))
        finally:
            common.RES = old
    return out


def load_stream(order):
    """Feedback records of one stream (not skipped, with feedback.bayes; the b6_rq.stream_traj filter) + game info."""
    pq, ps, post, forced, phase1, games = [], [], [], [], [], []
    for seed, g, p in order:
        gj = json.load(open(p[:-len(".A.decisions.jsonl")] + ".json"))
        is_pre = gj["methods"]["B"] == PRE_METHOD
        n = 0
        for rec in records(p):
            fb = rec.get("feedback")
            if not fb or fb.get("skipped") or "bayes" not in fb:
                continue
            b = fb["bayes"]
            pq.append(b["p_qhat"]); ps.append([b["p_style"][s] for s in STYLES])
            post.append([b["posterior"][s] for s in STYLES]); forced.append(b["forced"]); phase1.append(is_pre)
            n += 1
        games.append(dict(seed=seed, game=g, method_B=gj["methods"]["B"], winner_role=gj.get("winner_role"),
                          error=bool(gj.get("error")), score=score(gj), updates=n))
    return dict(pq=np.array(pq, float), ps=np.array(ps, float).reshape(-1, len(STYLES)),
                post=np.array(post, float).reshape(-1, len(STYLES)), forced=np.array(forced, bool),
                phase1=np.array(phase1, bool), games=games)


# ---------------------------------------------------------------- reconstruction and metrics
def reconstruct(ps, variant, lam):
    """Post-update normalised log posterior L[t] and pre-update probabilities Pprev[t] for every update t."""
    logp = np.full(len(STYLES), -LN5)
    L, Pprev = np.empty_like(ps), np.empty_like(ps)
    for t, x in enumerate(np.log(ps)):
        Pprev[t] = np.exp(logp)
        z = logp + x if variant == "bayes" else (lam * logp + x if variant == "ptd" else lam * (logp + x))
        logp = z - (z.max() + np.log(np.exp(z - z.max()).sum()))
        L[t] = logp
    return L, Pprev


def sustained(cond, lo, hi, first=0):
    """First 1-based tau (relative to lo) with cond true at lo+tau-1 .. lo+tau+SUST-2, the window inside [lo, hi) and
    starting at index >= lo+first. Returns (tau, False), or (hi-lo, True) right-censored."""
    run = 0
    for t in range(lo, hi):
        run = run + 1 if cond[t] else 0
        start = t - SUST + 1
        if run >= SUST and start >= lo + first:
            return start - lo + 1, False
    return hi - lo, True


def b6_sustained(p, thr):
    """b6_rq.py's t_sustained rule: first 1-based t with p >= thr at every update from t to the end."""
    ok = p >= thr
    bad = np.where(~ok)[0]
    t = 0 if len(bad) == 0 else bad[-1] + 1
    return int(t + 1) if t < len(p) else None


def stream_metrics(k, S, tag, root, variant, lam, skip_validate):
    ps, pq, logged = S["ps"], S["pq"], S["post"]
    T = len(ps)
    t0 = int(S["phase1"].sum())
    prefix = bool((S["phase1"][:t0].all()) if T else True)
    L, Pprev = reconstruct(ps, variant, lam)
    P = np.exp(L)
    err = np.abs(P - logged).max(axis=1) if T else np.zeros(0)
    val1 = float(err[:t0].max()) if t0 else None
    val2 = float(err[t0:].max()) if T > t0 else None
    L0 = L[t0 - 1] if t0 else np.full(len(STYLES), -LN5)
    row = dict(stream=k, seeds=sorted({g["seed"] for g in S["games"]}), games=len(S["games"]), T=T, t0=t0,
               phase1_is_prefix=prefix, n_forced=int(S["forced"].sum()),
               val_err_phase1=val1, val_err_post=val2, val_err=float(err.max()) if T else None,
               P_t0=dict(zip(STYLES, np.exp(L0).tolist())), lnP_t0=dict(zip(STYLES, L0.tolist())))
    # carried-state check (optional file)
    sp = os.path.join(os.path.dirname(os.path.abspath(root)), "hedge_state", f"{tag}-stream{k}-A.bayes.json")
    meta = json.load(open(sp)) if os.path.exists(sp) else None
    row["state_file"] = sp if meta else None
    row["meta_updates"] = meta["meta"]["updates"] if meta else None
    row["meta_games"] = meta["meta"]["games"] if meta else None
    row["meta_config"] = meta["config"] if meta else None
    row["meta_ok"] = (meta["meta"]["updates"] == T) if meta else None
    row["clean"] = bool(T and prefix and err.max() <= VAL_TOL and row["meta_ok"] is not False)
    if skip_validate:
        row["clean_unvalidated"], row["clean"] = row["clean"], bool(T and prefix)
    # identification time from a fresh prior (b6 rule, for the RQunif t0 = 0 reference)
    row["t_sustained_P90_b6_logged"] = b6_sustained(logged[:, IU], 0.9) if T else None
    row["t_sustained_P90_b6_recon"] = b6_sustained(P[:, IU], 0.9) if T else None
    # false alarms in phase 1 (MAP = uniform for SUST consecutive updates, first BURN updates excluded)
    mapu = L.argmax(1) == IU
    fa, fa_cens = sustained(mapu, 0, t0, first=BURN)
    row["false_alarm"] = (not fa_cens) if t0 else None
    row["false_alarm_at"] = fa if (t0 and not fa_cens) else None
    # P8 identity (cumulative excess loss of the logged q-hat vs each style = ln5 + ln P_t(s))
    lhs = np.cumsum(-np.log(pq))[:, None] - np.cumsum(-np.log(ps), axis=0)
    if variant == "bayes" and T:
        row["id_err_rec"] = float(np.abs(lhs - (LN5 + L)).max())
        m = logged >= 0.01
        row["id_err_logged"] = float(np.abs(lhs - (LN5 + np.log(np.maximum(logged, 1e-300))))[m].max())
    else:
        row["id_err_rec"] = row["id_err_logged"] = None
    # post-switch segment
    n2 = T - t0
    row["post_updates"] = n2
    if n2 <= 0:
        return row
    for name, cond in (("tau_map", mapu), ("tau_90", P[:, IU] >= 0.9), ("tau_99", P[:, IU] >= 0.99),
                       ("tau_leave", P[:, IS] <= 0.1)):
        v, c = sustained(cond, t0, T)
        row[name], row[name + "_censored"] = v, c
    d = L0 - L0[IU]
    if variant == "bayes":
        row["tau_hat"] = max(max(d[i], 0.0) / MU_U[s] for i, s in enumerate(STYLES) if i != IU)
    else:
        G = d / lam if variant == "dtp" else d
        row["tau_hat"] = max(math.log(1 + (1 - lam) * max(G[i], 0.0) / MU_U[s]) / math.log(1 / lam)
                             for i, s in enumerate(STYLES) if i != IU)
    cands = [(L0[IS] - L0[i]) / (MU_U[PRE] - MU_U[s]) for i, s in enumerate(STYLES) if i != IS and MU_U[PRE] - MU_U[s] > 0]
    row["tau_leave_hat"] = min(cands)
    qrec = (Pprev * ps).sum(1)
    row["X"] = float((-np.log(qrec[t0:]) + np.log(ps[t0:, IU])).sum())          # reconstructed q-hat
    row["X_logged"] = float((-np.log(pq[t0:]) + np.log(ps[t0:, IU])).sum())     # logged p_qhat (6 decimals)
    if variant == "bayes":
        ident = float(L[-1, IU] - L0[IU])
        row["X_identity"] = ident
        row["X_identity_err"] = abs(row["X"] - ident)
        row["X_logged_identity_err"] = abs(row["X_logged"] - ident)
    return row


# ---------------------------------------------------------------- statistics
def med_ci(vals, rng):
    v = np.asarray(vals, float)
    if len(v) == 0:
        return None
    b = np.median(v[rng.integers(len(v), size=(NBOOT, len(v)))], axis=1)
    return dict(median=float(np.median(v)), lo=float(np.percentile(b, 2.5)), hi=float(np.percentile(b, 97.5)), n=len(v))


def med_bounds(vals, cens):
    """Median bounds under right-censoring: censored values are lower bounds (upper bound: +inf)."""
    v = np.asarray(vals, float)
    if len(v) == 0:
        return None
    hi = np.where(np.asarray(cens, bool), np.inf, v)
    return float(np.median(v)), float(np.median(hi))


def summarize(rows, variant, rng):
    sw = [r for r in rows if r["clean"] and r["t0"] > 0 and r.get("post_updates", 0) > 0]
    post = [r for r in rows if r["clean"] and r.get("post_updates", 0) > 0]
    out = dict(n_streams=len(rows), n_clean=sum(r["clean"] for r in rows), n_switch_streams=len(sw),
               n_post_streams=len(post))

    def metric(name, rs, f, cf=lambda r: False):
        vals = [f(r) for r in rs if f(r) is not None]
        cens = [cf(r) for r in rs if f(r) is not None]
        if not vals:
            return
        out[name] = dict(med_ci(vals, rng), n_censored=int(sum(cens)), median_bounds=med_bounds(vals, cens))

    for m in ("tau_map", "tau_90", "tau_99", "tau_leave"):
        metric(m, post, lambda r, m=m: r[m], lambda r, m=m: r[m + "_censored"])
    for m in ("tau_hat", "tau_leave_hat", "X", "X_logged"):
        metric(m, post, lambda r, m=m: r[m])
    metric("tau_map_over_t0", sw, lambda r: r["tau_map"] / r["t0"], lambda r: r["tau_map_censored"])
    metric("tau_map_over_tau_hat", post, lambda r: r["tau_map"] / r["tau_hat"] if r["tau_hat"] > 0 else None,
           lambda r: r["tau_map_censored"])
    # tau_90 - tau_map: exact if both observed; lower bound if only tau_90 censored; unknown if tau_map censored
    d = [(r["tau_90"] - r["tau_map"], r["tau_90_censored"]) for r in post if not r["tau_map_censored"]]
    unk = sum(r["tau_map_censored"] for r in post)
    if post:
        vals = [x for x, _ in d] + [-np.inf] * unk
        cens = [c for _, c in d] + [True] * unk
        out["tau90_minus_tau_map"] = dict(med_ci([x for x, _ in d], rng) or {}, n_unknown=unk, median_bounds=med_bounds(vals, cens))
    th = [(r["tau_hat"], r["tau_map"]) for r in post]
    if len(th) >= 3:
        rho = stats.spearmanr([a for a, _ in th], [b for _, b in th])
        out["spearman_tau_hat_tau_map"] = dict(rho=float(rho.statistic), p=float(rho.pvalue), n=len(th))
    out["false_alarm_streams"] = sum(bool(r["false_alarm"]) for r in rows if r["clean"])
    out["early_flips_le30"] = sum((not r["tau_map_censored"]) and r["tau_map"] <= 30 for r in sw)
    for key in ("val_err", "val_err_phase1", "val_err_post", "id_err_rec", "id_err_logged", "X_identity_err",
                "X_logged_identity_err"):
        v = [r[key] for r in rows if r.get(key) is not None]
        out["max_" + key] = max(v) if v else None
    out["meta_ok"] = [r["meta_ok"] for r in rows]
    return out


def arm_scores(rows_games, ctrl):
    """Pair score of A per seed (mean of the two games; seeds with an error game dropped), split by phase, paired by
    seed with RQsafe (phase 1) / RQunif (post-switch, seeds <= 100)."""
    seeds = {}
    for g in rows_games:
        seeds.setdefault(g["seed"], []).append(g)
    out = dict(pre=[], post=[], dropped_error_seeds=[])
    for s, gs in sorted(seeds.items()):
        if any(g["error"] for g in gs):
            out["dropped_error_seeds"].append(s)
            continue
        ph = "pre" if all(g["method_B"] == PRE_METHOD for g in gs) else "post"
        out[ph].append((s, float(np.mean([g["score"] for g in gs]))))
    res = dict(dropped_error_seeds=out["dropped_error_seeds"])
    for ph, ctag, lim in (("pre", "RQsafe", None), ("post", "RQunif", 100)):
        sc = [x for _, x in out[ph]]
        res[ph] = dict(n_seeds=len(sc), mean=float(np.mean(sc)) if sc else None)
        if ctrl.get(ctag) is None:
            continue
        diffs = [x - ctrl[ctag][s] for s, x in out[ph] if s in ctrl[ctag] and (lim is None or s <= lim)]
        if diffs:
            m, se, lo, hi, n = mean_ci_normal(diffs)
            res[ph]["vs_" + ctag] = dict(mean_diff=float(m), se=float(se), lo=float(lo), hi=float(hi), n_pairs=n)
    return res


def control_scores(root):
    """seed -> pair score of A for RQsafe / RQunif (seeds with an error game dropped), via oi_common.game_jsons."""
    out = {}
    if not root or not os.path.isdir(root):
        return out
    old, oi_common.RES = oi_common.RES, root
    try:
        for tag in ("RQsafe", "RQunif"):
            if not os.path.isdir(oi_common.rdir(tag)):
                continue
            by = {}
            for (s, _), d in oi_common.game_jsons(tag).items():
                by.setdefault(s, []).append(d)
            out[tag] = {s: float(np.mean([score(d) for d in ds])) for s, ds in by.items() if not any(d.get("error") for d in ds)}
    finally:
        oi_common.RES = old
    return out


# ---------------------------------------------------------------- P1-P8
def side(bounds, pred):
    """pred on the median bounds: True / False if decided, None if the censoring leaves it open."""
    if bounds is None:
        return None
    a, b = pred(bounds[0]), pred(bounds[1])
    return a if a == b else None


def inside(bounds, lo, hi):
    """Median in [lo, hi]: True / False if the bounds decide it, None otherwise."""
    if bounds is None:
        return None
    if lo <= bounds[0] and bounds[1] <= hi:
        return True
    if bounds[1] < lo or bounds[0] > hi:
        return False
    return None


def verdict(refuted, required_present, open_checks):
    if any(x is True for x in refuted):
        return "FAIL"
    if not required_present or any(x is None for x in refuted) or any(x is None for x in open_checks):
        return "UNTESTABLE"
    return "PASS"


def get(S, tag, key):
    return (S.get(tag) or {}).get(key)


def predictions(S, arms, scores):
    P = {}
    have = lambda *t: all(x in S for x in t)  # noqa: E731
    # P1
    ref, ivl, det = [], [], {}
    for tag, (lo, hi) in (("SW2", (70, 170)), ("SW4", (130, 330))):
        if tag not in S:
            continue
        rb = get(S, tag, "tau_map_over_t0")
        mb = get(S, tag, "tau_map")
        r1 = side(rb and rb["median_bounds"], lambda x: x < 0.4 or x > 2.0)
        r2 = S[tag]["early_flips_le30"] >= 3
        cens_against = sum(1 for r in arms[tag]["rows"] if r["clean"] and r.get("tau_map_censored") and r["tau_map"] > hi)
        cens_open = sum(1 for r in arms[tag]["rows"] if r["clean"] and r.get("tau_map_censored") and r["tau_map"] <= hi)
        ref += [r1, r2]
        ivl.append(inside(mb and mb["median_bounds"], lo, hi))
        det[tag] = dict(median_tau_map=mb, median_ratio_t0=rb, refuted_ratio=r1, early_flips_le30=S[tag]["early_flips_le30"],
                        refuted_early=r2, interval=[lo, hi], median_in_interval=ivl[-1],
                        censored_counting_against=cens_against, censored_inconclusive=cens_open)
    P["P1"] = dict(status=verdict(ref, have("SW2", "SW4"), []), detail=det)
    # P2
    det = {}
    ref = []
    if have("SW2", "SW4") and get(S, "SW2", "tau_map") and get(S, "SW4", "tau_map"):
        b2, b4 = S["SW2"]["tau_map"]["median_bounds"], S["SW4"]["tau_map"]["median_bounds"]
        rb = (b4[0] / b2[1] if b2[1] > 0 else 0.0, b4[1] / b2[0] if b2[0] > 0 else np.inf)
        ref = [side(rb, lambda x: x < 1.25)]
        det = dict(ratio_point=S["SW4"]["tau_map"]["median"] / S["SW2"]["tau_map"]["median"], ratio_bounds=rb,
                   in_interval_1p5_2p7=inside(rb, 1.5, 2.7), refuted_lt_1p25=ref[0])
    P["P2"] = dict(status=verdict(ref, have("SW2", "SW4") and bool(det), []), detail=det)
    # P3 (pooled lambda = 1 switch streams of SW2 and SW4)
    pool = [r for t in ("SW2", "SW4") if t in arms and arms[t]["variant"] == "bayes"
            for r in arms[t]["rows"] if r["clean"] and r["t0"] > 0 and r.get("post_updates", 0) > 0]
    det, ref = dict(n_streams=len(pool), arms=[t for t in ("SW2", "SW4") if t in arms]), []
    if pool:
        rr = [(r["tau_map"] / r["tau_hat"], r["tau_map_censored"]) for r in pool if r["tau_hat"] > 0]
        rb = med_bounds([a for a, _ in rr], [c for _, c in rr]) if rr else None
        rho = stats.spearmanr([r["tau_hat"] for r in pool], [r["tau_map"] for r in pool]).statistic if len(pool) >= 3 else None
        ref = [side(rb, lambda x: x < 0.5 or x > 2.0), None if rho is None or np.isnan(rho) else bool(rho <= 0)]
        det.update(median_ratio_bounds=rb, n_tau_hat_zero=len(pool) - len(rr), spearman=rho,
                   ratio_in_0p7_1p4=inside(rb, 0.7, 1.4), rho_ge_0p5=None if rho is None else bool(rho >= 0.5),
                   refuted_ratio=ref[0], refuted_rho=ref[1])
    P["P3"] = dict(status=verdict(ref, have("SW2", "SW4") and bool(pool), []), detail=det)
    # P4 (no refutation clause: the interval is the criterion)
    det, chk = {}, []
    for tag in ("SW2", "SW4"):
        if tag in S and S[tag].get("tau90_minus_tau_map"):
            d = S[tag]["tau90_minus_tau_map"]
            chk.append(inside(d["median_bounds"], 0, 25))
            det[tag] = dict(tau90_minus_tau_map=d, in_0_25=chk[-1], tau_leave=S[tag].get("tau_leave"),
                            tau_leave_hat=S[tag].get("tau_leave_hat"))
    P["P4"] = dict(status=verdict([None if c is None else not c for c in chk], have("SW2", "SW4") and len(chk) == 2, []),
                   detail=det, note="no refutation clause in the prereg; the [0, 25] interval is used as the criterion")
    # P5
    det, ref = {}, []
    for tag, (lo, hi) in (("SW2", (20, 60)), ("SW4", (45, 110))):
        x = get(S, tag, "X")
        if x:
            ref.append(x["median"] < 10)
            det[tag] = dict(X=x, interval=[lo, hi], in_interval=lo <= x["median"] <= hi, refuted_lt_10=ref[-1])
    P["P5"] = dict(status=verdict(ref, have("SW2", "SW4") and len(ref) == 2, []), detail=det)
    # P6
    det, ref, chk = {}, [], []
    fa = {t: S[t]["false_alarm_streams"] for t in S}
    ref.append(any(v >= 2 for v in fa.values()) if fa else None)
    det["false_alarm_streams"] = fa
    if have("SW4p") and get(S, "SW4p", "tau_map"):
        bp = S["SW4p"]["tau_map"]["median_bounds"]
        ref.append(side(bp, lambda x: x > 100))
        det["SW4p_median_tau_map"] = S["SW4p"]["tau_map"]
        det["SW4p_in_15_70"] = inside(bp, 15, 70)
        rr = get(S, "SW4p", "tau_map_over_tau_hat")
        ref.append(side(rr and rr["median_bounds"], lambda x: x < 0.5 or x > 2.0))
        det["SW4p_ratio_tau_hat"] = rr
        if get(S, "SW4", "tau_map"):
            b4 = S["SW4"]["tau_map"]["median_bounds"]
            rb = (bp[0] / b4[1] if b4[1] > 0 else 0.0, bp[1] / b4[0] if b4[0] > 0 else np.inf)
            ref.append(side(rb, lambda x: x > 0.7))
            det["ratio_SW4p_SW4_bounds"] = rb
            det["ratio_le_0p4"] = side(rb, lambda x: x <= 0.4)
        else:
            chk.append(None)
    if have("SW4d") and get(S, "SW4d", "X"):
        xd = S["SW4d"]["X"]["median"]
        det["SW4d_X"] = S["SW4d"]["X"]
        det["SW4d_X_in_5_30"] = 5 <= xd <= 30
        if get(S, "SW4", "X"):
            det["SW4_over_SW4d_X_ge_2p5"] = S["SW4"]["X"]["median"] / xd >= 2.5 if xd > 0 else None
        if get(S, "SW4p", "X"):
            det["SW4p_X_ge_SW4d_X"] = S["SW4p"]["X"]["median"] >= xd
    P["P6"] = dict(status=verdict(ref, have("SW4p", "SW4"), chk), detail=det)
    # P7 (descriptive)
    det, chk = {}, []
    for tag, sc in scores.items():
        for ph, ctag, b in (("post", "RQunif", 0.05), ("pre", "RQsafe", 0.10)):
            c = sc.get(ph, {}).get("vs_" + ctag)
            if c:
                chk.append(-b <= c["mean_diff"] <= b)
                det[f"{tag}_{ph}_vs_{ctag}"] = dict(c, in_interval=chk[-1], bound=b)
    P["P7"] = dict(status=verdict([not c for c in chk], bool(chk), []), detail=det,
                   note="descriptive, no refutation in the prereg; FAIL here only means outside the stated interval")
    # P8
    det, ref = {}, []
    for tag, a in arms.items():
        if a["variant"] != "bayes":
            continue
        rs = [r for r in a["rows"] if r["clean"] and r["id_err_rec"] is not None]
        if not rs:
            continue
        er, el = max(r["id_err_rec"] for r in rs), max(r["id_err_logged"] for r in rs)
        ref.append(er > ID_REC_TOL or el > ID_LOG_TOL)
        det[tag] = dict(max_id_err_rec=er, max_id_err_logged=el, n_streams=len(rs), violated=ref[-1])
    P["P8"] = dict(status=verdict(ref, bool(ref), []), detail=det)
    return P


# ---------------------------------------------------------------- output
def fmt(v, c=False, nd=1):
    if v is None:
        return "-"
    s = f"{v:.{nd}f}" if isinstance(v, float) else str(v)
    return (">=" + s) if c else s


def md_arm(tag, a):
    rows, S = a["rows"], a["summary"]
    h = ["stream", "T", "t0", "P_t0(u)", "val err", "clean", "tau_map", "tau_90", "tau_99", "tau_leave", "tau_hat",
         "tau_leave_hat", "X", "X_logged", "FA", "P90_b6", "id err rec", "meta"]
    out = [f"### {tag} ({a['variant']}, lambda={a['lam']}, root {a['root']})", "", "| " + " | ".join(h) + " |",
           "|" + "---|" * len(h)]
    for r in rows:
        out.append("| " + " | ".join([
            str(r["stream"]), str(r["T"]), str(r["t0"]), f"{r['P_t0'][POST]:.2e}", fmt(r["val_err"], nd=6),
            "y" if r["clean"] else "n",
            *[fmt(r.get(m), r.get(m + "_censored", False), 0) for m in ("tau_map", "tau_90", "tau_99", "tau_leave")],
            fmt(r.get("tau_hat")), fmt(r.get("tau_leave_hat")), fmt(r.get("X"), nd=2), fmt(r.get("X_logged"), nd=2),
            "-" if r["false_alarm"] is None else ("yes@" + str(r["false_alarm_at"]) if r["false_alarm"] else "no"),
            fmt(r["t_sustained_P90_b6_logged"]), fmt(r["id_err_rec"], nd=6),
            "-" if r["meta_ok"] is None else f"{r['meta_updates']}{'' if r['meta_ok'] else ' MISMATCH'}"]) + " |")
    out.append("")
    for m in ("tau_map", "tau_90", "tau_99", "tau_leave", "tau_hat", "tau_leave_hat", "X", "X_logged",
              "tau_map_over_t0", "tau_map_over_tau_hat", "tau90_minus_tau_map"):
        if S.get(m) and "median" in S[m]:
            v = S[m]
            extra = f", median bounds under censoring {v['median_bounds'][0]:.2f}..{v['median_bounds'][1]:.2f}" \
                if v.get("median_bounds") and v["median_bounds"][1] != v["median_bounds"][0] else ""
            out.append(f"- {m}: median {v['median']:.2f} [{v['lo']:.2f}, {v['hi']:.2f}] (n={v['n']}"
                       f"{', censored ' + str(v['n_censored']) if v.get('n_censored') else ''}{extra})")
    if S.get("spearman_tau_hat_tau_map"):
        sp = S["spearman_tau_hat_tau_map"]
        out.append(f"- Spearman rho(tau_hat, tau_map) = {sp['rho']:.3f} (p={sp['p']:.3g}, n={sp['n']})")
    out.append(f"- clean {S['n_clean']}/{S['n_streams']}; false-alarm streams {S['false_alarm_streams']}; "
               f"flips <= 30 updates {S['early_flips_le30']}; max val err {fmt(S['max_val_err'], nd=6)} "
               f"(phase 1 {fmt(S['max_val_err_phase1'], nd=6)}, post {fmt(S['max_val_err_post'], nd=6)}); "
               f"max id err rec {fmt(S['max_id_err_rec'], nd=6)}, logged {fmt(S['max_id_err_logged'], nd=6)}; "
               f"max |X - identity| {fmt(S['max_X_identity_err'], nd=6)} (logged q-hat {fmt(S['max_X_logged_identity_err'], nd=6)})")
    sc = a["scores"]
    for ph in ("pre", "post"):
        p = sc.get(ph, {})
        diffs = "; ".join(f"{k}: {v['mean_diff']:+.3f} [{v['lo']:+.3f}, {v['hi']:+.3f}] (n={v['n_pairs']})"
                          for k, v in p.items() if k.startswith("vs_"))
        out.append(f"- score {ph}: {p.get('n_seeds', 0)} seeds, mean {fmt(p.get('mean'), nd=3)}{'; ' + diffs if diffs else ''}")
    if sc.get("dropped_error_seeds"):
        out.append(f"- seeds dropped from scores (error game): {sc['dropped_error_seeds']}")
    return "\n".join(out) + "\n"


def md_preds(P):
    out = ["### Predictions", "", "| P | status | detail |", "|---|---|---|"]
    for k, v in P.items():
        out.append(f"| {k} | {v['status']} | {json.dumps(v['detail'], default=float)[:600]}"
                   f"{(' NOTE: ' + v['note']) if v.get('note') else ''} |")
    return "\n".join(out) + "\n"


def parse_arm(s):
    tag, rest = s.split("=", 1)
    parts = rest.split(":")
    root = parts[0]
    variant = parts[1] if len(parts) > 1 and parts[1] else "bayes"
    if variant not in ("bayes", "ptd", "dtp"):
        raise SystemExit(f"unknown variant {variant} (bayes / ptd / dtp)")
    lam = float(parts[2]) if len(parts) > 2 else (1.0 if variant == "bayes" else 0.98)
    return tag, root, variant, lam


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--arm", action="append", required=True, help="TAG=ROOT[:variant[:lambda]], variant bayes|ptd|dtp")
    ap.add_argument("--controls", default=CONTROLS, help="results dir holding RQsafe/ and RQunif/replays_headless")
    ap.add_argument("--out", required=True)
    ap.add_argument("--skip-validate", action="store_true",
                    help="do not exclude streams failing the 1e-3 posterior validation (synthetic inputs only)")
    ap.add_argument("--label", default="", help="prefix for the markdown title (e.g. SYNTHETIC)")
    args = ap.parse_args()
    rng = np.random.default_rng(0)
    ctrl = control_scores(args.controls)
    arms, S, scores = {}, {}, {}
    for spec in args.arm:
        tag, root, variant, lam = parse_arm(spec)
        st = arm_streams(root, tag)
        rows = [stream_metrics(k, d, tag, root, variant, lam, args.skip_validate) for k, d in st]
        sc = arm_scores([g for _, d in st for g in d["games"]], ctrl)
        S[tag] = summarize(rows, variant, rng)
        scores[tag] = sc
        arms[tag] = dict(root=os.path.abspath(root), variant=variant, lam=lam, rows=rows, summary=S[tag], scores=sc)
        print(tag, variant, lam, "streams", len(rows), "clean", S[tag]["n_clean"], "max val err", S[tag]["max_val_err"])
    P = predictions(S, arms, scores)
    os.makedirs(args.out, exist_ok=True)
    res = dict(label=args.label, skip_validate=args.skip_validate, controls=args.controls, arms=arms, predictions=P)
    with open(os.path.join(args.out, "out_style_switch.json"), "w") as f:
        json.dump(res, f, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else
                  (bool(o) if isinstance(o, np.bool_) else str(o)))
    md = [f"# {args.label + ' ' if args.label else ''}Style-switch analysis (research/prereg_style_switch.md)", ""]
    if args.skip_validate:
        md.append("**--skip-validate: streams failing the posterior validation are NOT excluded.**\n")
    md += [md_arm(t, a) for t, a in arms.items()] + [md_preds(P)]
    open(os.path.join(args.out, "out_style_switch.md"), "w").write("\n".join(md))
    print("\n".join(md))


if __name__ == "__main__":
    main()
