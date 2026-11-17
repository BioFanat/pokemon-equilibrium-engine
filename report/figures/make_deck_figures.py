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
INK, GREY, LIGHT, ACCENT = "#111111", "#8C8C8C", "#D9D9D9", "#0072B2"
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


def exploit():
    fig, b = plt.subplots(figsize=(6.6, 3.0))
    dot_ci(b, [("argmax\n(Bayesian rule)", 0.487, 0.399, 0.576, ACCENT),
               ("sample\nvisit shares", 0.010, -0.039, 0.059, GREY)], ref=0)
    b.set_xlabel("Gain from predicting our move")
    b.set_xlim(-0.1, 0.65)
    b.tick_params(axis="y", length=0)
    fig.savefig(os.path.join(HERE, "deck_exploit.png"), bbox_inches="tight")
    return fig


if __name__ == "__main__":
    import sys
    sys.path.insert(0, "/Users/biofanat/.claude/skills/figure-style")
    fig = inplay()
    strength(); exploit()
    try:
        from figcheck import check_figure
        check_figure(fig, min_pt=8, max_words=12)
    except ImportError:
        pass
    print("wrote deck_inplay.{png,pdf}")
