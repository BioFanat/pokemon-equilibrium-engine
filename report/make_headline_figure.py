"""Headline figure: regret goes to zero, under the proved bounds, at the three places we use no-regret learning.

Run:  engine/.venv/bin/python report/make_headline_figure.py [--refresh] [--e2 PATH] [--streams S] [--role A]
Writes report/figures/fig_headline.{pdf,png}. Caches go to report/figures/data/.

  A  inside a turn   RM+ self-play on real stage matrices. Data: figures/data/rmplus_gap.json (as fig_rmplus).
  B  across games    carried anytime Hedge over 4 experts vs reset-per-game Hedge, regret per turn R_T/T.
                     Real data: E2 decision logs (--e2 = directory with <tag>-s<seed>-g<game>.<role>.decisions.jsonl,
                     or that directory joined with the tag prefix, e.g. .../replays_headless/E2). Without --e2 the panel
                     is a PLACEHOLDER built from real logged stage matrices (data.pkl) and the opponent's recorded
                     column, with the shipped reward model; cache figures/data/headline_hedge_placeholder.json.
  C  search node     Nash gap of decoupled self-play at one node vs N: full-information RM+ (nash.rm_plus) vs bandit
                     Exp3 / UCB1 (copy of run_ourbot.bandit_selfplay, the E4 code). Rewards (1+tanh(M/160))/2,
                     Bernoulli draws for the bandits. Cache figures/data/headline_node.json.
  D  opponent model  Bayesian style posterior = MWU with eta=1 on log loss over K=5 styles: cumulative log-loss regret
                     R_T vs the best fixed style, per stream, against the bound ln 5 (theory/synthesis.md §3.2).
                     Data: figures/data/bh_regret.json, the 8 count-clean BH streams (0 and 6 excluded: log/state update
                     counts disagree), curves downsampled to <= 500 points (argmax kept), built once from the synthesis
                     pass's out_bh_regret.json (scratchpad synthesis/bh_regret.py over the BH decision logs).
"""
import argparse
import glob
import json
import os
import re
import sys
from multiprocessing import get_context
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "report/figures"
DATA = OUT / "data"
sys.path.insert(0, str(ROOT / "engine"))
sys.path.insert(0, str(ROOT / "report"))
sys.path.insert(0, "/Users/biofanat/.claude/skills/figure-style")
from figcheck import check_figure  # noqa: E402
from make_figures import COL, INK, GRAY  # noqa: E402  (one colour per method, shared with every figure)

plt.style.use("/Users/biofanat/.claude/skills/figure-style/figure.mplstyle")

SCRATCH = Path("/private/tmp/claude-501/-Users-biofanat-Documents-classes-f26-cs535/"
               "2bbe670b-2f97-4d9a-bfad-e1139e0352c6/scratchpad/brainstorm/data.pkl")
BH_SRC = SCRATCH.parents[1] / "synthesis/out_bh_regret.json"
BH_EXCLUDED = {0, 6}                 # streams whose log update count != hedge_state count (synthesis.md §3.2)
CAPPED = {1010, 1012, 1014}          # games that hit the 300-turn cap (excluded, as in brainstorm/exp_*.py)
C_RMPLUS = COL["eq_top"]             # RM+ = the equilibrium agent's solver (accent)
C_UCB1 = COL["mcts"]                 # UCB1 = the MCTS selection rule
C_EXP3 = "#666666"                   # Exp3: comparator, gray
N_GRID = [100, 300, 1000, 3000, 10000]
TAU = 160.0                          # win-probability squash (run_ourbot.BANDIT_TAU)
K_HEDGE, SCALE = 4, 400.0            # experts top / sampled / safest / safex; reward normaliser (HEDGE_CONFIG)
DELTA = 0.05                         # confidence level for the bandit high-probability bound


def slope(x, y):
    return np.polyfit(np.log(x), np.log(y), 1)[0]


def records():
    import pickle
    return pickle.load(open(SCRATCH, "rb"))


# ================================================================ panel A
def panel_a_data():
    d = json.loads((DATA / "rmplus_gap.json").read_text())
    T = np.array(d["T"], float)
    G = np.array([r["gap"] for r in d["rows"]])
    nm = np.array([r["shape"] for r in d["rows"]], float)
    B = 2 * (np.sqrt(nm[:, :1]) + np.sqrt(nm[:, 1:])) / np.sqrt(T[None])  # 2(√n+√m)/√T, payoff range 1
    return dict(T=T, med=np.median(G, 0), p90=np.percentile(G, 90, 0), bound=np.median(B, 0),
                n=len(G), worst_ratio=float((G / B).max()))


# ================================================================ panel C
def bandit_selfplay(A01, kind, N, rng, marks):
    """Copy of engine/showdown/run_ourbot.py:bandit_selfplay (E4), returning the visit shares at every N in `marks`
    (both rules are anytime and the uniforms are drawn per step, so a run to max(marks) read at N equals a run of N)."""
    m, n = A01.shape
    cnt = [np.zeros(m), np.zeros(n)]
    u = rng.random((N, 3))
    out = {}
    if kind == "ucb1":
        rew = [np.zeros(m), np.zeros(n)]

        def pick(s, t):
            c = cnt[s]
            if (c == 0).any():
                return int(np.argmin(c > 0))
            return int(np.argmax(rew[s] / c + np.sqrt(2 * np.log(t) / c)))
        for t in range(1, N + 1):
            i, j = pick(0, t), pick(1, t)
            r = float(u[t - 1, 0] < A01[i, j])
            cnt[0][i] += 1; rew[0][i] += r
            cnt[1][j] += 1; rew[1][j] += 1 - r
            if t in marks:
                out[t] = (cnt[0] / t, cnt[1] / t)
    else:
        loss = [np.zeros(m), np.zeros(n)]
        for t in range(1, N + 1):
            ps = []
            for s, k in ((0, m), (1, n)):
                z = -np.sqrt(np.log(k) / (k * t)) * loss[s]
                p = np.exp(z - z.max())
                ps.append(p / p.sum())
            i = min(int(np.searchsorted(np.cumsum(ps[0]), u[t - 1, 1] * ps[0].sum())), m - 1)
            j = min(int(np.searchsorted(np.cumsum(ps[1]), u[t - 1, 2] * ps[1].sum())), n - 1)
            r = float(u[t - 1, 0] < A01[i, j])
            loss[0][i] += (1 - r) / ps[0][i]
            loss[1][j] += r / ps[1][j]
            cnt[0][i] += 1; cnt[1][j] += 1
            if t in marks:
                out[t] = (cnt[0] / t, cnt[1] / t)
    return out


def node_one(args):
    import nash
    idx, M = args
    A = 0.5 * (1 + np.tanh(np.asarray(M, float) / TAU))
    gap = lambda x, y: float((A @ y).max() - (x @ A).min())
    row = {"idx": int(idx), "shape": list(A.shape), "rmplus": [gap(*nash.rm_plus(A, N)) for N in N_GRID]}
    for s, kind in enumerate(("exp3", "ucb1")):
        sh = bandit_selfplay(A, kind, max(N_GRID), np.random.default_rng([int(idx), s]), set(N_GRID))
        row[kind] = [gap(*sh[N]) for N in N_GRID]
    return row


def panel_c_data(refresh=False, n_mat=100, seed=0):
    path = DATA / "headline_node.json"
    if path.exists() and not refresh:
        return json.loads(path.read_text())
    recs = [r for r in records() if r["shape"][0] > 1 and r["shape"][1] > 1 and r["seed"] not in CAPPED]
    assert len(recs) == 601, len(recs)
    pick = np.sort(np.random.default_rng(seed).choice(len(recs), n_mat, replace=False))
    with get_context("spawn").Pool(2) as pool:
        rows = pool.map(node_one, [(i, recs[i]["M"]) for i in pick], chunksize=5)
    d = {"N": N_GRID, "seed": seed, "pool": len(recs), "rows": rows,
         "note": "data.pkl, 601 non-capped two-sided states; A=(1+tanh(M/160))/2; gap in reward units; "
                 "RM+ = nash.rm_plus (linear averaging); exp3/ucb1 = run_ourbot.bandit_selfplay visit shares"}
    path.write_text(json.dumps(d, indent=1))
    return d


def node_bounds(shapes, N):
    """Median over matrices of: full-information tuned Hedge, sqrt(ln n/2N)+sqrt(ln m/2N) (theory_map B3(c));
    Exp3-IX (Neu 2015 Thm 1) per player 2sqrt(2kN ln k)+(sqrt(2kN/ln k)+1)ln(2/δ), summed and divided by N."""
    nm = np.array(shapes, float)
    full = np.median(np.sqrt(np.log(nm[:, :1]) / (2 * N)) + np.sqrt(np.log(nm[:, 1:]) / (2 * N)), 0)
    ix = lambda k: (2 * np.sqrt(2 * k * N * np.log(k)) + (np.sqrt(2 * k * N / np.log(k)) + 1) * np.log(2 / DELTA)) / N
    bandit = np.median(ix(nm[:, :1]) + ix(nm[:, 1:]), 0)
    return full, bandit


# ================================================================ panel B
def regret_curves(R, W, games):
    """R: [T, k] expert rewards per non-forced turn, W: [T, k] weights the learner played, games: [T] game index.
    Returns carried R_T/T (logged weights), reset R_T/T (weights replayed from 0 at each game) and the max abs
    difference between W and the weights replayed from the rewards under carried play (a log consistency check)."""
    T, k = R.shape
    cumE = np.cumsum(R, 0).max(1)                     # max_e sum_{s<=t} r_{e,s}
    t = np.arange(1, T + 1)
    reg_car = (cumE - np.cumsum((W * R).sum(1))) / t
    wr, wc = np.zeros((T, k)), np.zeros((T, k))
    L_r, n_r, L_c = np.zeros(k), 0, np.zeros(k)
    for s in range(T):
        if s > 0 and games[s] != games[s - 1]:
            L_r, n_r = np.zeros(k), 0
        for L, n, out in ((L_r, n_r, wr), (L_c, s, wc)):
            z = np.sqrt(8 * np.log(k) / n) * L if n > 0 else np.zeros(k)   # w_t ∝ exp(sqrt(8 ln k/(t-1)) L_{t-1})
            e = np.exp(z - z.max()); out[s] = e / e.sum()
        L_r = L_r + R[s]; n_r += 1; L_c = L_c + R[s]
    reg_res = (cumE - np.cumsum((wr * R).sum(1))) / t
    return reg_car, reg_res, float(np.abs(wc - W).max())


def placeholder_streams(refresh=False, n_streams=8, seed=0):
    """Stand-in for E2: real logged stage matrices (data.pkl, two-sided states of all 20 games), the four shipped
    experts on M (top/sampled = LP, safest = pure maximin row, safex 0.3%), the opponent's recorded column j2, and the
    shipped reward r_e = clip((q_e·M[:,j2] - min M)/400, 0, 1). Streams = random orders of the games."""
    path = DATA / "headline_hedge_placeholder.json"
    if path.exists() and not refresh:
        d = json.loads(path.read_text())
    else:
        import nash
        games = {}
        for r in records():
            M = np.asarray(r["M"], float)
            if min(M.shape) < 2 or np.ptp(M) == 0:
                continue
            x = np.clip(nash.lp(M)[0], 0, None); x /= x.sum()
            sx = np.clip(nash.safex_strategy(M, 0.003), 0, None); sx /= sx.sum()
            eye = np.eye(M.shape[0])
            Q = np.stack([eye[int(x.argmax())], x, eye[int(M.min(1).argmax())], sx])
            rew = np.clip((Q @ M[:, r["j2"]] - M.min()) / SCALE, 0, 1)
            games.setdefault(int(r["seed"]), []).append((int(r["turn"]), rew.round(5).tolist()))
        d = {"games": {g: [v for _, v in sorted(t)] for g, t in games.items()}, "n_streams": n_streams, "seed": seed}
        path.write_text(json.dumps(d))
    rng = np.random.default_rng(d["seed"])
    keys = sorted(d["games"])
    streams = []
    for _ in range(d["n_streams"]):
        order = rng.permutation(keys)
        R = np.array([v for g in order for v in d["games"][g]])
        games = np.array([i for i, g in enumerate(order) for _ in d["games"][g]])
        W = carried_weights(R)   # no log exists: the carried learner's weights, replayed with the shipped rule
        streams.append((R, W, games))
    return streams, len(keys)


def carried_weights(R):
    T, k = R.shape
    W, L = np.zeros((T, k)), np.zeros(k)
    for s in range(T):
        z = np.sqrt(8 * np.log(k) / s) * L if s > 0 else np.zeros(k)
        e = np.exp(z - z.max()); W[s] = e / e.sum()
        L = L + R[s]
    return W


GAME_RE = re.compile(r"(?P<tag>.+)-s(?P<seed>\d+)-g(?P<game>[12])\.(?P<role>[AB])\.decisions\.jsonl$")


def read_game_log(path):
    """One game's decision log -> list of (r [k], w [k]) for its non-forced, non-skipped turns.
    A decision record with `weights` (pre-update, HEDGE_EXPERTS order) is paired with the next record's `feedback`
    (for_turn must match; a game's last feedback sits in a `final` record). Forced turns carry no weights."""
    from_order = ["top", "sampled", "safest", "safex"]
    out, pend = [], None
    for line in open(path):
        rec = json.loads(line)
        fb = rec.get("feedback")
        if fb and pend is not None and fb.get("for_turn") == pend[0] and not fb.get("skipped"):
            out.append((np.array(fb["r"], float), pend[1]))
        if fb is not None or rec.get("final"):
            pend = None
        if "weights" in rec and isinstance(rec["weights"], dict):
            pend = (rec["turn"], np.array([rec["weights"][e] for e in from_order], float))
    return out


def e2_streams(spec, n_streams=None, role="A"):
    """spec = directory, or directory/tag-prefix. Stream k = sorted seeds[k::S] (batch.split_streams), games g1, g2."""
    base, prefix = (spec, "") if os.path.isdir(spec) else (os.path.dirname(spec), os.path.basename(spec))
    files = {}
    for p in glob.glob(os.path.join(base, f"{prefix}*-s*-g*.{role}.decisions.jsonl")):
        m = GAME_RE.search(os.path.basename(p))
        if m:
            files[(int(m["seed"]), int(m["game"]))] = p
    if not files:
        raise FileNotFoundError(f"no {role}.decisions.jsonl logs under {spec}")
    if n_streams is None:
        summ = glob.glob(os.path.join(base, f"{prefix}*summary.json"))
        n_streams = json.load(open(summ[0])).get("streams") or 1 if summ else 1
    seeds = sorted({s for s, _ in files})
    streams = []
    for k in range(n_streams):
        R, W, games = [], [], []
        for gi, (s, g) in enumerate((s, g) for s in seeds[k::n_streams] for g in (1, 2) if (s, g) in files):
            for r, w in read_game_log(files[(s, g)]):
                R.append(r); W.append(w); games.append(gi)
        if R:
            streams.append((np.array(R), np.array(W), np.array(games)))
    return streams, len(files)


def panel_b_data(e2=None, n_streams=None, role="A", refresh=False):
    if e2:
        streams, n_games = e2_streams(e2, n_streams, role)
    else:
        streams, n_games = placeholder_streams(refresh)
    Tmin = min(len(s[0]) for s in streams)
    car, res, chk = zip(*[regret_curves(*s) for s in streams])
    car = np.array([c[:Tmin] for c in car]); res = np.array([c[:Tmin] for c in res])
    t = np.arange(1, Tmin + 1)
    bound = 1.5 * np.sqrt(np.log(K_HEDGE) / (2 * t)) + np.sqrt(np.log(K_HEDGE) / 8) / t
    return dict(t=t, car=car, res=res, bound=bound, n_streams=len(streams), n_games=n_games,
                T_per_stream=[len(s[0]) for s in streams], weight_check=max(chk), placeholder=not e2)


# ================================================================ panel D
def panel_d_data(n_pts=500):
    path = DATA / "bh_regret.json"
    if not path.exists():
        src = json.loads(BH_SRC.read_text())
        streams = {}
        for r in src["rows"]:
            k = r["stream"]
            if k in BH_EXCLUDED:
                continue
            R = np.array(src["curves"][str(k)])
            idx = np.unique(np.r_[np.linspace(0, len(R) - 1, n_pts - 1).round().astype(int), R.argmax()])
            streams[str(k)] = dict(T=(idx + 1).tolist(), R=R[idx].round(6).tolist(), n_turns=len(R),
                                   R_final=float(R[-1]), R_max=float(R.max()), best_style=r["best_style"])
        d = dict(ln5=src["ln5"], excluded=sorted(BH_EXCLUDED), streams=streams,
                 note="R_T = sum_t -ln p_qhat(b_t) - min_s sum_t -ln q_s(b_t), feedback turns only (BH run, "
                      "feedback.bayes, 6 dp); from synthesis/bh_regret.py; curves downsampled, argmax kept")
        path.write_text(json.dumps(d))
    return json.loads(path.read_text())


# ================================================================ figure
def band(ax, x, lo, hi, c):
    ax.fill_between(x, lo, hi, color=c, alpha=0.18, lw=0)


def label(ax, x, y, s, c, **kw):
    ax.text(x, y, s, color=c, **{"va": "center", **kw})


def draw(a, b, c, d):
    fig, axs = plt.subplots(2, 2, figsize=(7.09, 5.0), layout="constrained")
    axs = axs.ravel()
    for ax, L in zip(axs, "ABCD"):
        ax.text(-0.02, 1.13, L, transform=ax.transAxes, fontsize=12, fontweight="bold", va="top", ha="right")
        ax.set_xscale("log"); ax.set_yscale("log")

    # ---- A
    ax = axs[0]
    T = a["T"]
    ax.plot(T, a["bound"], color=C_RMPLUS, lw=1.2, ls=(0, (4, 2)))
    ax.fill_between(T, a["med"], a["p90"], color=C_RMPLUS, alpha=0.18, lw=0)
    ax.plot(T, a["med"], color=C_RMPLUS, lw=1.5)
    fit = T >= 10
    sA, sB = slope(T[fit], a["med"][fit]), slope(T[fit], a["bound"][fit])
    label(ax, 60, 0.25, f"proved bound\nslope {sB:.1f}".replace("-", "−"), C_RMPLUS, va="top")
    label(ax, 2.2, a["med"][list(T).index(20)] * 0.25, f"median, slope {sA:.1f}".replace("-", "−"), C_RMPLUS)
    ax.set_xlim(1, 2000); ax.set_ylim(1e-7, 2)
    ax.set_xticks([1, 10, 100, 1000])
    ax.set_xlabel("Iterations $T$")
    ax.set_ylabel("Nash gap (share of payoff range)")
    ax.set_title("Inside a turn: RM$^{\\mathregular{+}}$", pad=10)

    # ---- B
    ax = axs[1]
    t = b["t"]
    pos = lambda v: np.where(v > 0, v, np.nan)
    ax.plot(t, b["bound"], color=GRAY, lw=1.2, ls=(0, (4, 2)))
    for key, col in (("res", COL["hedge_reset"]), ("car", COL["hedge_carried"])):
        V = b[key]
        q1, med, q3 = np.percentile(V, [25, 50, 75], 0)
        band(ax, t, pos(q1), pos(q3), col)
        ax.plot(t, pos(med), color=col, lw=1.5)
    tl = t[-1]
    label(ax, 25, b["bound"][24] * 2.4, "proved bound", GRAY)
    medr = np.median(b["res"], 0); medc = np.median(b["car"], 0)
    label(ax, tl * 0.5, medr[-1] * 1.9, "reset", COL["hedge_reset"], ha="center")
    label(ax, tl * 0.5, max(medc[-1], 1e-4) * 0.42, "carried", COL["hedge_carried"], ha="center")
    ax.set_xlim(1, tl); ax.set_ylim(1e-4, 2)
    ax.set_xlabel("Non-forced turns $T$")
    ax.set_ylabel("Average regret per turn")
    ax.set_title("Across games: Hedge", pad=10)
    if b["placeholder"]:
        ax.text(0.03, 0.04, "PLACEHOLDER", transform=ax.transAxes, color="#B00020", fontsize=10,
                fontweight="bold", alpha=0.85)

    # ---- C
    ax = axs[2]
    N = np.array(c["N"], float)
    full, bandit = node_bounds([r["shape"] for r in c["rows"]], N)
    ax.plot(N, full, color=C_RMPLUS, lw=1.2, ls=(0, (4, 2)))
    ax.plot(N, bandit, color=GRAY, lw=1.2, ls=(0, (4, 2)))
    floor = 1e-7
    for key, col in (("exp3", C_EXP3), ("ucb1", C_UCB1), ("rmplus", C_RMPLUS)):
        G = np.maximum(np.array([r[key] for r in c["rows"]]), floor)
        q1, med, q3 = np.percentile(G, [25, 50, 75], 0)
        band(ax, N, q1, q3, col)
        ax.plot(N, med, color=col, lw=1.5, marker="o", ms=3)
    ex, uc, rp = (np.median([r[k] for r in c["rows"]], 0) for k in ("exp3", "ucb1", "rmplus"))
    label(ax, 115, 0.035, "Exp3", C_EXP3)       # below the two bandit curves (full-info bound is above them here)
    label(ax, 2500, 0.25, "UCB1", C_UCB1, ha="center")   # above: UCB1 ends highest
    label(ax, 115, max(rp[0], floor) * 3.5, "RM$^{\\mathregular{+}}$ (full info)", C_RMPLUS)
    label(ax, 9000, full[-1] * 0.4, "full-info bound", C_RMPLUS, ha="right")
    label(ax, 115, 0.4, "bandit bound (above 1: vacuous)", GRAY)
    ax.set_xlim(N[0], N[-1]); ax.set_ylim(1e-7, 2)    # win-prob gap <= 1; the bandit bound (5.1 at N=100) enters from above
    ax.set_xlabel("Iterations $N$")
    ax.set_ylabel("Nash gap (win probability)")
    ax.set_title("At a search node", pad=10)
    for ax in axs[:3]:
        ax.yaxis.set_major_locator(matplotlib.ticker.LogLocator(numticks=5))
        ax.yaxis.set_minor_locator(matplotlib.ticker.NullLocator())

    # ---- D  (linear axes: the bound is a constant)
    ax = axs[3]
    ax.set_xscale("linear"); ax.set_yscale("linear")
    Tmax = max(v["n_turns"] for v in d["streams"].values())
    grid = np.arange(1, Tmax + 1)
    curves = np.array([np.where(grid <= v["n_turns"], np.interp(grid, v["T"], v["R"]), np.nan)
                       for v in d["streams"].values()])
    for v in d["streams"].values():
        ax.plot(v["T"], v["R"], color=COL["bayes"], lw=0.6, alpha=0.45)
    ax.plot(grid, np.nanmedian(curves, 0), color=COL["bayes"], lw=1.8)   # median over the streams still running
    ax.axhline(d["ln5"], color=INK, lw=1.2, ls=(0, (4, 2)), zorder=3)   # on top: the curves sit on it
    label(ax, Tmax * 0.97, d["ln5"] + 0.2, "proved bound ln 5", INK, ha="right")
    label(ax, Tmax * 0.97, 0.55, f"{len(d['streams'])} streams; median on ln 5", COL["bayes"], ha="right")
    ax.set_xlim(0, Tmax); ax.set_ylim(0, 2)
    ax.set_xlabel("Feedback turns $T$")
    ax.set_ylabel("Log-loss regret (nats)")
    ax.set_title("Opponent model: regret vs ln 5", pad=10)
    return fig, dict(sA=sA, sB=sB, full=full, bandit=bandit)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--refresh", action="store_true", help="recompute panel B placeholder and panel C caches")
    ap.add_argument("--e2", default=None, help="E2 decision-log directory (or directory/tag prefix) for panel B")
    ap.add_argument("--streams", type=int, default=None, help="carried-weight streams in the E2 run (batch --streams)")
    ap.add_argument("--role", default="A", help="player letter of the hedge learner in the E2 logs")
    args = ap.parse_args()
    DATA.mkdir(parents=True, exist_ok=True)
    a = panel_a_data()
    b = panel_b_data(args.e2, args.streams, args.role, args.refresh)
    c = panel_c_data(args.refresh)
    d = panel_d_data()
    fig, s = draw(a, b, c, d)

    # numbers for the report / caption
    print(f"[A] n={a['n']} matrices; slope median (T>=10) {s['sA']:.2f}, bound {s['sB']:.2f}; at T={a['T'][-1]:.0f}: "
          f"median {a['med'][-1]:.2e}, p90 {a['p90'][-1]:.2e}, bound {a['bound'][-1]:.3f}, "
          f"median/bound {a['med'][-1] / a['bound'][-1]:.2e}, p90/bound {a['p90'][-1] / a['bound'][-1]:.2e}; "
          f"max over matrices and T of gap/own bound {a['worst_ratio']:.3f}")
    mc, mr = np.median(b["car"], 0), np.median(b["res"], 0)
    tt = b["t"]; fitb = tt >= 30
    print(f"[B] {'PLACEHOLDER' if b['placeholder'] else 'E2'}: {b['n_streams']} streams, {b['n_games']} games, "
          f"T per stream {b['T_per_stream']}; at T={tt[-1]}: carried R/T median {mc[-1]:.4f} "
          f"(IQR {np.percentile(b['car'][:, -1], 25):.4f}-{np.percentile(b['car'][:, -1], 75):.4f}), reset {mr[-1]:.4f}, "
          f"bound {b['bound'][-1]:.4f}, carried/bound {mc[-1] / b['bound'][-1]:.3f}, reset/bound {mr[-1] / b['bound'][-1]:.3f}; "
          f"max over streams,T of carried R/T minus bound {(b['car'] - b['bound']).max():.4f}; "
          f"share of (stream,T) with carried R_T<=0 {(b['car'] <= 0).mean():.3f}; "
          f"weight-replay check max|w_log - w_replay| {b['weight_check']:.4f}")
    if (mc[fitb] > 0).all():
        print(f"[B] slope of median carried R/T (T>=30): {slope(tt[fitb], mc[fitb]):.2f}; bound "
              f"{slope(tt[fitb], b['bound'][fitb]):.2f}")
    N = np.array(c["N"])
    for key in ("rmplus", "exp3", "ucb1"):
        G = np.array([r[key] for r in c["rows"]])
        med = np.median(G, 0)
        sl = slope(N, med) if (med > 0).all() else float("nan")
        print(f"[C] {key}: n={len(G)}; median " + " ".join(f"{v:.2e}" for v in med) +
              f" | IQR@{N[-1]} {np.percentile(G[:, -1], 25):.2e}-{np.percentile(G[:, -1], 75):.2e} | slope {sl:.2f}"
              f" | median/full-info bound @N={N[-1]}: {med[-1] / s['full'][-1]:.3g}")
    print("[C] full-info bound (median):", np.round(s["full"], 4), "| Exp3-IX bound (median):", np.round(s["bandit"], 3))

    for k, v in d["streams"].items():
        print(f"[D] stream {k}: {v['n_turns']} feedback turns, final R_T {v['R_final']:.4f}, anytime max "
              f"{v['R_max']:.4f}, best style {v['best_style']}")
    print(f"[D] ln 5 = {d['ln5']:.4f}; max over clean streams of anytime max R_T - ln 5: "
          f"{max(v['R_max'] for v in d['streams'].values()) - d['ln5']:.1e}; excluded streams {d['excluded']}")

    issues = check_figure(fig, min_pt=8, max_words=12)
    print("[fig_headline] figcheck:", issues if issues else "clean")
    fig.savefig(OUT / "fig_headline.pdf")
    fig.savefig(OUT / "fig_headline.png", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    main()
