"""Deck figures in one clean style: black Helvetica text, grey marks, one accent (blue) for the Bayesian model.

Run: <python with matplotlib> report/figures/make_deck_figures.py  ->  report/figures/deck_inplay.{png,pdf}

Numbers (all from engine/RESULTS.md, section "No-regret over the Bayesian model, and an adversary for it"):
  local Elo vs the preliminary model (Bradley-Terry fit, 95% bootstrap over pairs): Hedge over the Bayesian model
  +27 [-44, +105], Bayesian model +203 [+135, +283]; gain from predicting the move, in-engine (ADV, 100 paired seeds):
  MCTS argmax +0.487 [+0.399, +0.576], MCTS sampling its visit shares +0.010 [-0.039, +0.059].
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
INK, GREY, LIGHT, ACCENT = "#000000", "#8C8C8C", "#D9D9D9", "#0072B2"
plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Helvetica", "Arial"], "font.size": 16,
    "axes.labelsize": 17, "axes.titlesize": 18, "xtick.labelsize": 15, "ytick.labelsize": 16,
    "text.color": INK, "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK, "axes.edgecolor": INK,
    "axes.linewidth": 1.0, "axes.spines.top": False, "axes.spines.right": False, "axes.titlelocation": "left",
    "axes.titleweight": "bold", "xtick.major.width": 1.0, "ytick.major.width": 1.0, "ytick.major.size": 0,
    "savefig.dpi": 200, "pdf.fonttype": 42,
})


def dot_ci(ax, rows, ref=None):
    """rows: (label, value, lo, hi, colour); drawn top to bottom."""
    n = len(rows)
    for i, (lab, v, lo, hi, col) in enumerate(rows):
        y = n - 1 - i
        if lo is not None:
            ax.plot([lo, hi], [y, y], color=col, lw=3, solid_capstyle="butt")
        ax.plot([v], [y], "o", ms=11, color=col, zorder=3)
    ax.set_yticks(range(n))
    ax.set_yticklabels([r[0] for r in rows][::-1])
    ax.set_ylim(-0.6, n - 0.4)
    if ref is not None:
        ax.axvline(ref, color=LIGHT, lw=1.5, zorder=0)


def inplay():
    fig, (a, b) = plt.subplots(1, 2, figsize=(12.4, 3.9), gridspec_kw=dict(wspace=0.75))
    dot_ci(a, [("Bayesian model", 203, 135, 283, ACCENT),
               ("Hedge over it", 27, -44, 105, GREY),
               ("Preliminary model", 0, None, None, INK)], ref=0)
    a.set_xlabel("Elo vs preliminary model")
    a.set_title("Strength")
    a.set_xlim(-80, 320)
    dot_ci(b, [("argmax\n(Bayesian rule)", 0.487, 0.399, 0.576, ACCENT),
               ("sample\nvisit shares", 0.010, -0.039, 0.059, GREY)], ref=0)
    b.set_xlabel("Adversary's gain from predicting")
    b.set_title("Exploitability (in-engine)")
    b.set_xlim(-0.1, 0.65)
    for ax in (a, b):
        ax.tick_params(axis="y", length=0)
    fig.savefig(os.path.join(HERE, "deck_inplay.png"), bbox_inches="tight")
    fig.savefig(os.path.join(HERE, "deck_inplay.pdf"), bbox_inches="tight")
    return fig


def strength():
    fig, a = plt.subplots(figsize=(6.6, 3.6))
    dot_ci(a, [("Bayesian model", 203, 135, 283, ACCENT),
               ("Hedge over it", 27, -44, 105, GREY),
               ("Preliminary model", 0, None, None, INK)], ref=0)
    a.set_xlabel("Local Elo vs preliminary model")
    a.set_xlim(-80, 320)
    a.tick_params(axis="y", length=0)
    fig.savefig(os.path.join(HERE, "deck_strength.png"), bbox_inches="tight")
    return fig


HEDGE_ROW = ("Hedge\nmodel", (0.370, 0.282, 0.458), (0.560, 0.475, 0.645), "+0.19", GREY)   # ADVP, sampled equilibrium
BAYES_ROW = ("Bayesian\nmodel", (0.213, 0.155, 0.270), (0.700, 0.641, 0.759), "+0.49", ACCENT)  # ADV


def _exploit_plot(rows, fname, height, width=6.4):
    """Opponent's score (1 - target score; RESULTS.md ADV / ADVP) guessing blindly (open) vs knowing our rule (filled)."""
    fig, b = plt.subplots(figsize=(width, height))
    for k, (lab, blind, known, d, col) in enumerate(rows):
        y = len(rows) - 1 - k
        for (v, lo, hi), filled in ((blind, False), (known, True)):
            b.plot([lo, hi], [y, y], color=col, lw=2.5, solid_capstyle="butt")
            b.plot([v], [y], "o", ms=11, color=col, mfc=col if filled else "white", mew=2, zorder=3)
        b.annotate("", xy=(known[0] - 0.03, y + 0.28), xytext=(blind[0] + 0.03, y + 0.28),
                   arrowprops=dict(arrowstyle="->", color=INK, lw=1.3))
        b.text((blind[0] + known[0]) / 2, y + 0.34, d, ha="center", va="bottom", color=INK)
    b.set_yticks(range(len(rows)))
    b.set_yticklabels([r[0] for r in rows][::-1])
    b.set_ylim(-0.5, len(rows) - 0.15)
    b.axvline(0.5, color=LIGHT, lw=1.5, zorder=0)
    b.set_xlim(0, 1)
    b.set_xlabel("Opponent's score (0.5 = even)")
    b.plot([], [], "o", ms=9, mfc="white", mec=INK, mew=2, ls="", label="guessing blindly")
    b.plot([], [], "o", ms=9, color=INK, ls="", label="knowing our rule")
    b.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, frameon=False, handletextpad=0.3)
    b.tick_params(axis="y", length=0)
    fig.savefig(os.path.join(HERE, fname), bbox_inches="tight")
    return fig


def exploit():
    _exploit_plot([HEDGE_ROW, BAYES_ROW], "deck_exploit.png", 3.0)
    with plt.rc_context({"font.size": 24, "axes.labelsize": 24, "xtick.labelsize": 22, "ytick.labelsize": 24}):
        return _exploit_plot([HEDGE_ROW], "deck_exploit_hedge.png", 2.2, width=6.4)


def delta_parts():
    """Contributions to the per-turn shortfall delta = eps_s + 2 eps_v on the exactly solved endgames (final5.tex,
    'Exact endgames'): eps_s <= 0.16 on reached states; largest eps_v (affine) 0.7-1.0, so 2 eps_v = 1.4-2.0
    (bar at the midpoint, line = range)."""
    fig, ax = plt.subplots(figsize=(4.6, 3.0))
    ax.bar([0, 1], [0.16, 1.7], color=["#8C8C8C", "#404040"], width=0.55)
    ax.plot([1, 1], [1.4, 2.0], color=INK, lw=1.5)
    ax.text(0, 0.16 + 0.07, "≤ 0.16", ha="center", va="bottom", color=INK)
    ax.text(1, 2.0 + 0.07, "1.4–2.0", ha="center", va="bottom", color=INK)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Solver error\n$\\varepsilon_s$", "Matrix error\n$2\\varepsilon_v$"])
    ax.set_ylim(0, 2.35)
    ax.set_ylabel("Contribution to δ")
    ax.tick_params(axis="x", length=0)
    fig.savefig(os.path.join(HERE, "deck_delta.png"), bbox_inches="tight")
    return fig


if __name__ == "__main__":
    import sys
    sys.path.insert(0, "/Users/biofanat/.claude/skills/figure-style")
    fig = inplay()
    strength(); exploit(); delta_parts()
    try:
        from figcheck import check_figure
        check_figure(fig, min_pt=8, max_words=12)
    except ImportError:
        pass
    print("wrote deck_inplay.{png,pdf}")
