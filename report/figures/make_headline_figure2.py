"""Headline figure of final5.tex (fig_main): exploitability and regret, four panels in one row at 7.0 in.

  a  exploitability  exact endgames (engine/results/endgame/eps.jsonl, engine/endgame_eps.py): exact exploitability
                     of each matrix bot in the abstracted endgame (x) against the refined Theorem 3 bound
                     E[sum_t eps_s(s_t) + 2 eps_v(s_t)] along the (bot, best response) chain (y), log-log, one dot per
                     row with exploitability > 1e-3 and a finite bound (22 rows; as in engine/RESULTS.md).
  b-d regret         the three panels of fig_regret (make_regret_figure.py, whose data functions are reused here):
                     RM+ inside a turn (band min-max over 35 matrices), Hedge across games, opponent memory
                     (band 25-75% over 9 matches).

Run:  python3 report/figures/make_headline_figure2.py
Writes report/figures/fig_main.{pdf,png} and report/figures/fig_main_numbers.json (panel a; panels b-d
numbers are in fig_regret_numbers.json, written by make_regret_figure.py).
"""
import json
import math
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from make_regret_figure import (OUT, ROOT, COL, GRAY, INK, LN5, MEM_EXCLUDED, RM_FIT_LO, check_figure,
                                load_streams, analyse, rm_data, hedge_bound, slope)

# One clean style for paper and deck: series keep their colours; all text (labels, annotations) is black Helvetica.
INK = "#111111"
plt.rcParams.update({"font.sans-serif": ["Helvetica", "Arial"], "text.color": INK, "axes.labelcolor": INK,
                     "xtick.color": INK, "ytick.color": INK, "axes.edgecolor": INK})

EPS = ROOT / "engine/results/endgame/eps.jsonl"
EXPL_MIN = 1e-3                                        # rows with exploitability <= 1e-3 have no defined ratio


# ================================================================ data: exact endgames
def endgame_rows():
    rows = [json.loads(l) for l in open(EPS) if l.strip()]
    rows = [r for r in rows if "expl" in r]
    keep = [r for r in rows if r["expl"] > EXPL_MIN and math.isfinite(r["refined_bound"])]
    out = [dict(case=r["case"], agent=r["agent"], seat=r["seat"], expl=r["expl"], bound=r["refined_bound"],
                ratio=r["refined_bound"] / r["expl"]) for r in keep]
    dropped = [dict(case=r["case"], agent=r["agent"], seat=r["seat"], expl=r["expl"], bound=r["refined_bound"])
               for r in rows if r not in keep]
    return out, dropped, len(rows)


# ================================================================ figure
def letters_and_titles(fig, axs, letters):
    """Freeze the constrained layout, then start each panel's letter and title at the panel's left edge (its y label),
    so a title may run over the y-label column instead of overflowing the narrow axes to the right."""
    fig.canvas.draw()
    fig.set_layout_engine("none")
    r = fig.canvas.get_renderer()
    for ax, L in zip(axs, letters):
        inv = ax.transAxes.inverted()
        left = min(t.get_window_extent(r).x0 for t in [ax.yaxis.label] + ax.get_yticklabels() if t.get_text())
        tb = ax._left_title.get_window_extent(r)
        x0, y0 = inv.transform((left, tb.y0))
        lt = ax.text(x0, y0, L, transform=ax.transAxes, fontsize=12, fontweight="bold", va="bottom", ha="left")
        lw = lt.get_window_extent(r).width
        ax.set_title(ax.get_title(loc="left"), loc="left", x=inv.transform((left + lw + 5, 0))[0] )


def log_axis(ax, numticks=5):
    ax.yaxis.set_major_locator(matplotlib.ticker.LogLocator(numticks=numticks))
    ax.yaxis.set_minor_locator(matplotlib.ticker.NullLocator())


def panel_exploit(ax, eg):
    lim = (0.01, 100)
    ax.loglog(lim, lim, color=GRAY, lw=1.2, ls=(0, (4, 2)))
    ax.scatter([r["expl"] for r in eg], [r["bound"] for r in eg], s=18, color=INK, alpha=0.55, lw=0, zorder=3)
    lo, hi = min(r["ratio"] for r in eg), max(r["ratio"] for r in eg)
    ax.text(0.3, 0.12, "bound = measured", color=INK, rotation=0, va="center")
    ax.text(0.013, 150, f"all above the line: bound holds", color=INK, va="center")
    ax.set_xlim(0.01, 3); ax.set_ylim(0.05, 300)
    ax.set_xticks([0.01, 0.1, 1])
    ax.set_xticklabels(["0.01", "0.1", "1"])
    log_axis(ax, 5)
    ax.set_xlabel("Measured exploitability (exact)")
    ax.set_ylabel("Our upper bound")
    ax.set_title("Exploitability bound")


def panel_rm(ax, rm):
    T, G, nm, B = rm
    c = COL["eq_top"]
    med, lo, hi, bnd = np.median(G, 0), G.min(0), G.max(0), np.median(B, 0)
    ax.loglog(T, bnd, color=GRAY, lw=1.2, ls=(0, (4, 2)))
    ax.fill_between(T, lo, hi, color=c, alpha=0.18, lw=0)
    ax.loglog(T, med, color=c, lw=1.5)
    fit = T >= RM_FIT_LO
    s_med = slope(T[fit], med[fit])
    ax.text(30, bnd[list(T).index(20)] * 3.5, "bound", color=INK, va="center")
    ax.text(1.4, 2e-6, f"median, slope {s_med:.1f}".replace("-", "−"), color=INK, va="center")
    ax.set_xlim(1, 2000); ax.set_ylim(1e-8, 20)
    ax.set_xticks([1, 10, 100, 1000])
    log_axis(ax, 5)
    ax.set_xlabel("RM$^{\\mathregular{+}}$ iterations $T$")
    ax.set_ylabel("Gap to equilibrium\n(payoff range)")
    ax.set_title("RM$^{\\mathregular{+}}$ in a turn")
    return s_med


def panel_hedge(ax, hedge_curves):
    c = COL["hedge_carried"]
    Tmax = max(len(v) for v in hedge_curves.values())
    t = np.arange(1, Tmax + 1)
    for v in hedge_curves.values():
        tt = np.arange(1, len(v) + 1)
        ax.plot(tt, np.where(v > 0, v / tt, np.nan), color=c, lw=0.6, alpha=0.5)
    Tmin = min(len(v) for v in hedge_curves.values())
    tm = np.arange(1, Tmin + 1)
    medc = np.median(np.array([v[:Tmin] for v in hedge_curves.values()]), 0)
    ax.plot(tm, np.where(medc > 0, medc / tm, np.nan), color=c, lw=1.8)
    ax.plot(t, hedge_bound(t) / t, color=GRAY, lw=1.2, ls=(0, (4, 2)))
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.text(3, hedge_bound(3) / 3 * 1.9, "anytime bound", color=INK, va="center")
    ax.text(1.5, 5e-4, "regret, 10 matches", color=INK, va="center")
    ax.set_xlim(1, Tmax); ax.set_ylim(3e-4, 5)
    ax.set_xticks([1, 10, 100, 1000])
    log_axis(ax, 5)
    ax.set_xlabel("Counted turns $t$")
    ax.set_ylabel("Regret per counted turn")
    ax.set_title("Hedge over games")


def panel_memory(ax, mem_curves):
    c = COL["bayes"]
    M = [v for s, v in mem_curves.items() if s not in MEM_EXCLUDED]
    Tm = min(len(v) for v in M)
    A = np.array([v[:Tm] for v in M])
    tc = np.arange(1, Tm + 1)
    q25, q75 = np.percentile(A, [25, 75], axis=0)
    ax.fill_between(tc, q25, q75, color=c, alpha=0.18, lw=0)
    ax.plot(tc, np.median(A, 0), color=c, lw=1.5, zorder=3)
    ax.axhline(LN5, color=GRAY, lw=1.4, ls=(0, (4, 2)), zorder=4)
    ax.set_xscale("log")
    ax.text(1.5, LN5 + 0.22, "bound ln 5", color=INK, va="center")
    ax.text(Tm * 0.9, 0.3, f"median, {len(M)} matches", color=INK, ha="right", va="center")
    ax.set_xlim(1, Tm); ax.set_ylim(0, 2)
    ax.set_xticks([1, 10, 100, 1000])
    ax.set_xlabel("Counted turns $t$")
    ax.set_ylabel("Regret vs best style\n(nats)")
    ax.set_title("Memory over games")
    return len(M), Tm


def draw(eg, rm, hedge_curves, mem_curves):
    fig, axs = plt.subplots(1, 4, figsize=(7.0, 2.1), layout="constrained")
    panel_exploit(axs[0], eg)
    s_med = panel_rm(axs[1], rm)
    panel_hedge(axs[2], hedge_curves)
    n_mem, T_mem = panel_memory(axs[3], mem_curves)
    letters_and_titles(fig, axs, "abcd")
    return fig, s_med, n_mem, T_mem


def draw_deck(eg, rm, hedge_curves, mem_curves):
    """Slide version: same data, titles that state each panel's conclusion, plain-language axes, slide-size text."""
    with plt.rc_context({"font.size": 14, "axes.labelsize": 14, "axes.titlesize": 15, "xtick.labelsize": 12,
                         "ytick.labelsize": 12}):
        fig, axs = plt.subplots(1, 4, figsize=(14.0, 4.1), layout="constrained")
        panel_exploit(axs[0], eg)
        panel_rm(axs[1], rm)
        panel_hedge(axs[2], hedge_curves)
        panel_memory(axs[3], mem_curves)
        spec = [("Bound always holds", "Actual exploitability", "Our bound"),
                ("Solver reaches equilibrium", "Solver iterations", "Distance from equilibrium"),
                ("Hedge's regret shrinks", "Turns played", "Regret per turn"),
                ("Prediction error is capped", "Turns played", "Total prediction regret")]
        for ax, (t, xl, yl) in zip(axs, spec):
            ax.set_title(t, loc="left")
            ax.set_xlabel(xl)
            ax.set_ylabel(yl)
        letters_and_titles(fig, axs, "abcd")
        fig.savefig(OUT / "fig_main_deck.png", dpi=200)
        plt.close(fig)


def main():
    eg, dropped, n_rows = endgame_rows()
    streams = load_streams()
    _, hedge_curves, mem_curves = analyse(streams)
    rm = rm_data()
    fig, s_med, n_mem, T_mem = draw(eg, rm, hedge_curves, mem_curves)
    issues = check_figure(fig, min_pt=8, max_words=12)
    print("[fig_main] figcheck:", issues if issues else "clean")
    fig.savefig(OUT / "fig_main.pdf")
    fig.savefig(OUT / "fig_main.png", dpi=200)
    plt.close(fig)
    draw_deck(eg, rm, hedge_curves, mem_curves)

    ratios = [r["ratio"] for r in eg]
    out = dict(
        exploitability=dict(
            source=str(EPS.relative_to(ROOT)), script="engine/endgame_eps.py",
            bound="refined Theorem 3 bound: E[sum_t eps_s(s_t) + 2 eps_v(s_t)] along the (bot, best response) chain",
            scale="values on the [-1, 1] scale (side-one perspective)",
            selection=f"exploitability > {EXPL_MIN} and finite refined bound",
            n_rows_total=n_rows, n_plotted=len(eg), n_endgames_plotted=len({r["case"] for r in eg}),
            n_endgames_total=len({r["case"] for r in eg} | {r["case"] for r in dropped}),
            all_bound_ge_expl=all(r["bound"] >= r["expl"] for r in eg),
            ratio_min=min(ratios), ratio_max=max(ratios), ratio_median=float(np.median(ratios)),
            rows=eg, not_plotted=dropped),
        regret=dict(rmplus_slope_median_T10_2000=s_med, rmplus_band="min-max over 35 matrices",
                    memory_band="25-75% over matches", memory_matches=n_mem, memory_T_plotted=T_mem,
                    numbers="report/figures/fig_regret_numbers.json"))
    (OUT / "fig_main_numbers.json").write_text(json.dumps(out, indent=1, default=float))
    e = out["exploitability"]
    print(f"exploitability: {e['n_plotted']}/{e['n_rows_total']} rows, {e['n_endgames_plotted']} endgames; "
          f"bound>=expl {e['all_bound_ge_expl']}; ratio {e['ratio_min']:.2f}-{e['ratio_max']:.2f} "
          f"(median {e['ratio_median']:.1f})")
    for r in eg:
        print(f"  {r['case'][:44]:44s} {r['agent']:22s} seat {r['seat']}  expl {r['expl']:.4f}  "
              f"bound {r['bound']:.3f}  ratio {r['ratio']:.1f}")


if __name__ == "__main__":
    main()
