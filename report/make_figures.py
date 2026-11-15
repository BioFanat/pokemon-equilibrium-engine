"""Figures for the CS535 report (report.tex).

Run: engine/.venv/bin/python report/make_figures.py [--refresh]
Writes report/figures/fig_*.pdf (+ .png snapshots). Every number is read from a file:
  fig_rmplus   report/figures/data/rmplus_gap.json      (computed here once from engine/nash.py; --refresh recomputes)
  fig_exploit  engine/results/experiments/{E7,E7t,E6f,E6ft,E6q,E1fix}.jsonl
  fig_hedge    engine/results/experiments/hedge_stream_seed{0..3}[_reset].jsonl, fits from
               theory/checks/rt3_weight_dynamics.py
  fig_hedge_half, fig_exploit_half   same data, drawn at their exact print width in final.tex
               (side by side, included unscaled; see HALF_FRAC)
  fig_leaves   report/figures/data/endgame_leaf_*.json  (copied from the scratch leaf harness);
               current-leaf exploitability from engine/results/endgame/eps.jsonl
"""
import collections
import json
import random
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "report"
OUT = REPORT / "figures"
DATA = OUT / "data"
EXP = ROOT / "engine/results/experiments"
sys.path.insert(0, str(ROOT / "engine"))
sys.path.insert(0, str(ROOT / "theory/checks"))
sys.path.insert(0, "/Users/biofanat/.claude/skills/figure-style")
from figcheck import check_figure  # noqa: E402

plt.style.use("/Users/biofanat/.claude/skills/figure-style/figure.mplstyle")

# One color per method, shared by every figure (Okabe-Ito). Names as in the report text.
COL = {
    "mcts": "#D55E00",        # MCTS / foul-play
    "eq_top": "#0072B2",      # Equilibrium, top move
    "eq_sampled": "#56B4E9",  # Equilibrium, sampled
    "hedge_carried": "#009E73",  # Hedge, weights carried across games
    "hedge_reset": "#E69F00",    # Hedge, weights reset each game
}
INK, GRAY, LIGHT = "#222222", "#888888", "#EDEDED"

# Side-by-side variants for final.tex (fig:main): drawn at the exact width they print at, so the
# 8 pt text is not rescaled. Width = minipage fraction x \linewidth; keep in sync with final.tex.
LINEWIDTH_IN = 8.5 - 2 * 0.65  # letter paper, geometry margin=0.65in -> \linewidth = 7.2 in
HALF_FRAC = {"fig_hedge": 0.51, "fig_exploit": 0.46}  # \begin{minipage}[b]{<frac>\linewidth}
HALF_H_IN = 1.95  # fixed height for both, so the [b]-aligned minipages line up


def finish(fig, name, exact=False):
    """exact=True keeps the page at figsize (no tight bbox), for figures included unscaled."""
    issues = check_figure(fig, min_pt=8, max_words=12)
    print(f"[{name}] figcheck:", issues if issues else "clean")
    # bbox_inches=None falls back to the style's savefig.bbox=tight, so override the rcParam.
    with plt.rc_context({"savefig.bbox": "standard"} if exact else {}):
        fig.savefig(OUT / f"{name}.pdf")
        fig.savefig(OUT / f"{name}.png", dpi=200)
    plt.close(fig)


def half_size(name):
    return (HALF_FRAC[name] * LINEWIDTH_IN, HALF_H_IN)


# ---------------------------------------------------------------- fig 1: RM+ gap vs T
T_GRID = [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000]


def rmplus_data(refresh=False, n_pos=35, seed=0):
    """Same position generator as nash.solver_report: random state, 0-14 random turns, skip
    finished games and one-option sides. Gap = Nash gap of RM+'s average / payoff range."""
    path = DATA / "rmplus_gap.json"
    if path.exists() and not refresh:
        return json.loads(path.read_text())
    import nash
    import pkbattle as pb
    rng = random.Random(seed)
    rows = []
    while len(rows) < n_pos:
        st = pb.random_state(rng)
        for _ in range(rng.randrange(0, 15)):
            o1, o2 = pb.options(st)
            st = pb.step(st, rng.choice(o1), rng.choice(o2), rng)
            if not pb.alive(st.side_one) or not pb.alive(st.side_two):
                break
        if not pb.alive(st.side_one) or not pb.alive(st.side_two):
            continue
        M, _, _ = nash.full_matrix(st)
        if min(M.shape) < 2 or np.ptp(M) == 0:
            continue
        gaps = [nash.exploitability(M, *nash.rm_plus(M, T)) / np.ptp(M) for T in T_GRID]
        rows.append({"shape": list(M.shape), "gap": gaps})
    d = {"T": T_GRID, "seed": seed, "rows": rows,
         "note": "nash.full_matrix on solver_report-style positions; gap = exploitability / ptp(M)"}
    path.write_text(json.dumps(d, indent=1))
    return d


def fig_rmplus(d):
    T = np.array(d["T"], float)
    G = np.array([r["gap"] for r in d["rows"]])
    nm = np.array([r["shape"] for r in d["rows"]], float)
    med, p90 = np.median(G, 0), np.percentile(G, 90, axis=0)
    # proved bound for the code's rm_plus, as a share of the payoff range: 2(sqrt n + sqrt m)/sqrt T
    bound = np.median(2 * (np.sqrt(nm[:, :1]) + np.sqrt(nm[:, 1:])) / np.sqrt(T[None]), 0)
    fit = T >= 10
    slope = np.polyfit(np.log(T[fit]), np.log(med[fit]), 1)[0]
    ps = [np.polyfit(np.log(T[fit]), np.log(np.maximum(g[fit], 1e-300)), 1)[0] for g in G]
    print(f"[fig_rmplus] n={len(G)}; median gap at T=10,100,1000,2000:",
          [f"{med[list(T).index(t)]:.2g}" for t in (10, 100, 1000, 2000)],
          "| p90:", [f"{p90[list(T).index(t)]:.2g}" for t in (10, 100, 1000, 2000)],
          f"| median slope (T>=10) {slope:.2f}; per-position quartiles",
          np.round(np.percentile(ps, [25, 50, 75]), 2),
          "| gap/bound median range", f"{(med / bound).min():.2g}-{(med / bound).max():.2g}",
          "| max over matrices,T of gap/own bound",
          f"{(G / (2 * (np.sqrt(nm[:, :1]) + np.sqrt(nm[:, 1:])) / np.sqrt(T[None]))).max():.3f}")
    fig, ax = plt.subplots(figsize=(3.6, 2.7), layout="constrained")
    ax.loglog(T, bound, color=GRAY, lw=1.2)
    ax.loglog(T, p90, color=INK, lw=1.0, alpha=0.45, marker="o", ms=3)
    ax.loglog(T, med, color=INK, lw=1.5, marker="o", ms=4)
    ax.text(T[-1] * 1.25, bound[-1], "worst-case bound\n$\\propto T^{-1/2}$", va="center", color=GRAY)
    ax.text(T[-1] * 1.25, p90[-1] * 1.5, "90th pct.", va="center", color=INK, alpha=0.7)
    ax.text(T[-1] * 1.25, med[-1] * 0.5, f"median\nslope {slope:.2f}".replace("-", "\u2212"), va="center", color=INK)
    ax.set_xlabel("RM$^+$ iterations $T$ (log)")
    ax.set_ylabel("Nash gap / payoff range (log)")
    ax.set_xlim(0.8, 3e4)
    ax.set_xticks([1, 10, 100, 1000])
    ax.xaxis.set_minor_locator(matplotlib.ticker.FixedLocator(
        [k * 10 ** e for e in range(4) for k in range(2, 10) if k * 10 ** e <= 2000]))
    ax.spines["bottom"].set_bounds(1, 2000)
    finish(fig, "fig_rmplus")
    return slope


# ---------------------------------------------------------------- fig 2: strength vs exploitation
def matchups(exp):
    d = collections.defaultdict(lambda: collections.defaultdict(list))
    for line in (EXP / f"{exp}.jsonl").read_text().splitlines():
        r = json.loads(line)
        if not r.get("error"):
            d[(r["A"], r["B"])][r["pair"]].append(r["score_A"])
    return d


def score_ci(exp, a, b, flip=False):
    """Mean score of A over seat-swapped pairs, and 95% half-width 1.96 sd/sqrt(pairs)."""
    s = np.array([np.mean(v) for v in matchups(exp)[(a, b)].values()])
    m, h = s.mean(), 1.96 * s.std(ddof=1) / np.sqrt(len(s))
    return (1 - m if flip else m), h


def fig_exploit(half=False):
    top, smp = "nash:rm+:0:1000:argmax", "nash:rm+:0:1000"
    bots = [  # label, color key, marker, baseline (score, half-width), vs LBR (score, half-width)
        ("foul-play, 2,000 it", "mcts", "o", score_ci("E7", "foulplay_it:2000", top),
         score_ci("E6f", "foulplay_it:2000", "lbr:foulplay_it:2000")),
        ("foul-play, 75 ms", "mcts", "o", score_ci("E7", "foulplay_ms:75", top),
         score_ci("E6ft", "foulplay_ms:75", "lbr:foulplay_ms:75")),
        ("MCTS, 2,000 it", "mcts", "s", score_ci("E1fix", top, "mcts_it:2000", flip=True),
         score_ci("E6f", "mcts_it:2000", "lbr:mcts")),
        ("Equilibrium, top move", "eq_top", "D", (0.5, 0.0), score_ci("E6q", top, "lbr:argmax")),
        ("Equilibrium, sampled", "eq_sampled", "D", (0.5, 0.0), score_ci("E6q", smp, "lbr:sample")),
    ]
    off = {"foul-play, 2,000 it": (6, 4, "left"), "foul-play, 75 ms": (-6, 4, "right"),
           "MCTS, 2,000 it": (0, 43, "center"), "Equilibrium, top move": (6, 0, "left"),
           "Equilibrium, sampled": (6, -2, "left")}
    if half:  # label offsets retuned for the narrower, shorter canvas
        off = {"foul-play, 2,000 it": (-5, 0, "right"), "foul-play, 75 ms": (5, 0, "left"),
               "MCTS, 2,000 it": (0, 34, "center"), "Equilibrium, top move": (5, 18, "left"),
               "Equilibrium, sampled": (5, -14, "left")}
    fig, ax = plt.subplots(figsize=half_size("fig_exploit") if half else (3.6, 2.7), layout="constrained")
    ax.axhline(0, color=GRAY, lw=0.6)
    for lab, c, mk, (x, hx), (l, hl) in bots:
        g, h = x - l, np.hypot(hx, hl)
        print(f"[fig_exploit] {lab}: x={x:.3f}  vs LBR={l:.3f}  gain={g:.3f} [{g - h:.2f}, {g + h:.2f}]")
        ax.errorbar(x, g, yerr=h, fmt=mk, color=COL[c], ms=5, elinewidth=1.0, capsize=0)
        dx, dy, ha = off[lab]
        txt = lab.replace(", ", ",\n", 1) if half and lab.startswith("foul-play") else lab
        ax.annotate(txt, (x, g), xytext=(dx, dy), textcoords="offset points", ha=ha, va="center", color=INK)
    ax.set_xlim(*((0.13, 0.8) if half else (0.2, 0.75)))
    ax.set_ylim(-0.12, 0.34 if half else 0.3)
    ax.set_xlabel("Score vs same-strength non-exploiter")
    ax.set_ylabel("Exploitation gain")
    finish(fig, "fig_exploit_half" if half else "fig_exploit", exact=half)


# ---------------------------------------------------------------- fig 3: Hedge weight trajectory
def fig_hedge(half=False):
    import rt3_weight_dynamics as rt3
    F, seeds = rt3.F, range(4)
    obs = {k: np.array([[r["weights"][F] for r in rt3.load(s, k == "reset")] for s in seeds])
           for k in ("carried", "reset")}
    # Cor. R3b, as in rt3_weight_dynamics.py: per-seed LS slopes d1 (vs MCTS, through the origin)
    # and d2 (vs exploiter, from the switch), pooled over seeds, then per-seed predicted w_F.
    fits, ts = {}, {}
    for s in seeds:
        R = rt3.load(s, False)
        t = np.array([r["t_learner"] for r in R], float); G = np.array([rt3.G_of(r) for r in R])
        i1 = np.where(np.array([r["opponent"] for r in R]) == "mcts")[0][-1]
        d1 = (t[:i1 + 1] @ G[:i1 + 1]) / (t[:i1 + 1] @ t[:i1 + 1])
        tt, gg = t[i1 + 1:] - t[i1], G[i1 + 1:] - G[i1]
        fits[s], ts[s] = (d1, (tt @ gg) / (tt @ tt), t[i1], i1), t
    d1m, d2m = np.mean([fits[s][0] for s in seeds]), np.mean([fits[s][1] for s in seeds])
    pred = np.array([[rt3.wF(d1m * ts[s][j] if j <= fits[s][3] else
                             d1m * fits[s][2] + d2m * (ts[s][j] - fits[s][2]), ts[s][j])
                      for j in range(60)] for s in seeds]).mean(0)
    mc, mr = obs["carried"].mean(0), obs["reset"].mean(0)
    err = pred - mc
    print(f"[fig_hedge] d1={d1m:.4f} d2={d2m:.4f}; carried mean w_F games 0/29/59: "
          f"{mc[0]:.3f}/{mc[29]:.3f}/{mc[59]:.3f}; reset mean phase1 {mr[:30].mean():.3f} phase2 {mr[30:].mean():.3f}; "
          f"prediction max err {np.abs(err).max():.3f}, RMS {np.sqrt((err ** 2).mean()):.3f}")
    g = np.arange(60)
    fig, ax = plt.subplots(figsize=half_size("fig_hedge") if half else (4.4, 2.6), layout="constrained")
    ax.axvspan(29.5, 59.5, color=LIGHT, lw=0)
    ax.text(14.5, 0.62, "vs MCTS", ha="center", color=INK)
    ax.text(44.5, 0.62, "vs exploiter", ha="center", color=INK)
    for k, c in (("carried", COL["hedge_carried"]), ("reset", COL["hedge_reset"])):
        for row in obs[k]:
            ax.plot(g, row, color=c, lw=0.6, alpha=0.35)
        ax.plot(g, obs[k].mean(0), color=c, lw=1.6)
    ax.plot(g, pred, color=INK, lw=1.0, ls=(0, (3, 2)))
    ax.text(60.5, mr[-3:].mean(), "reset", va="center", color=COL["hedge_reset"])
    ax.text(60.5, mc[-1] + 0.02, "carried", va="center", color=COL["hedge_carried"])
    ax.text(60.5, mc[-1] - 0.05, "prediction", va="center", color=INK)
    ax.set_xlim(-1, 76)
    ax.set_xticks([0, 10, 20, 30, 40, 50, 60])
    ax.spines["bottom"].set_bounds(0, 60)
    ax.set_ylim(0, 0.66)
    ax.set_xlabel("Game index")
    ax.set_ylabel("MCTS-expert weight $w_F$")
    finish(fig, "fig_hedge_half" if half else "fig_hedge", exact=half)


# ---------------------------------------------------------------- fig 4: endgame leaves
LEAF_SRC = Path("/private/tmp/claude-501/-Users-biofanat-Documents-classes-f26-cs535/"
                "2bbe670b-2f97-4d9a-bfad-e1139e0352c6/scratchpad/leaf")
EPS = ROOT / "engine/results/endgame/eps.jsonl"
CASES = [("greattusk_vs_gholdengo", "greattusk vs. gholdengo"),
         ("gholdengo_vs_heatran", "gholdengo vs. heatran"),
         ("garchomp_vs_rotomwash", "garchomp vs. rotomwash")]
VARIANTS = [("mid", "Current leaf"), ("tanh160+term", "Terminals + tanh(h/160)"),
            ("mcts1000", "MCTS-1,000 leaf"), ("vnet", "Value net")]


def leaf_data():
    out = {}
    for case, _ in CASES:
        dst = DATA / f"endgame_leaf_{case}_100.json"
        if not dst.exists():
            dst.write_text((LEAF_SRC / dst.name).read_text())
        out[case] = {r["cand"]: r for r in json.loads(dst.read_text())["rows"]}
    # Current-leaf ("mid") exploitability comes from the main endgame run, not the leaf harness:
    # on greattusk vs gholdengo seat 1 the harness gives 8.8e-6, while eps.jsonl and two independent
    # recomputations give 0.215 (the value in the report's tables). The other cells agree.
    for line in EPS.read_text().splitlines():
        r = json.loads(line)
        if r.get("agent") != "equilibrium sampled":
            continue
        for case, _ in CASES:
            if r["case"].startswith(case.replace("_vs_", " vs ") + " @ 100%"):
                out[case]["mid"]["expl"][f"equilibrium sampled/seat{r['seat']}"] = r["expl"]
    return out


def fig_leaves(d):
    fig, axs = plt.subplots(2, 3, figsize=(7.2, 3.3), sharex="row", sharey=True, layout="constrained")
    y = np.arange(len(VARIANTS))[::-1]
    for j, (case, title) in enumerate(CASES):
        ev = [d[case][v]["eps_v_cal_mean"] for v, _ in VARIANTS]
        ex = np.array([[d[case][v]["expl"][f"equilibrium sampled/seat{s}"] for s in (1, 2)] for v, _ in VARIANTS])
        print(f"[fig_leaves] {case}: eps_v", np.round(ev, 3), "expl s1/s2", np.round(ex, 3).tolist())
        a, b = axs[0, j], axs[1, j]
        a.hlines(y, 0, ev, color=LIGHT, lw=1.0)
        a.plot(ev, y, "o", color=INK, ms=5)
        b.hlines(y, ex.min(1), ex.max(1), color=GRAY, lw=1.2)
        b.plot(ex.mean(1), y, "o", color=INK, ms=5)
        a.set_title(title)
        for ax in (a, b):
            ax.grid(axis="x")
            ax.tick_params(axis="y", length=0)
    axs[0, 0].set_yticks(y, [lab for _, lab in VARIANTS])
    axs[0, 0].set_xlim(0, 0.6)
    axs[1, 0].set_xlim(0, 0.9)
    axs[0, 1].set_xlabel("Mean calibrated $\\epsilon_v$")
    axs[1, 1].set_xlabel("Exploitability, equilibrium sampled")
    finish(fig, "fig_leaves")


if __name__ == "__main__":
    DATA.mkdir(parents=True, exist_ok=True)
    fig_rmplus(rmplus_data(refresh="--refresh" in sys.argv))
    fig_exploit()
    fig_hedge()
    fig_exploit(half=True)
    fig_hedge(half=True)
    fig_leaves(leaf_data())
