"""Figure 2 (fig_models): the preliminary model vs the Bayesian model, strength and exploitability.

(a) Score of the named model [95% CI over seat-swapped pairs], local headless Showdown, 30 pairs each, seeds 201-230
    (vs Foul Play) / 101-130 (Bayesian vs preliminary). Sources, engine/RESULTS.md: "Final bot vs Foul Play",
    "Preliminary model vs stock Foul Play", "Local Elo (round robin)" (preliminary vs Bayesian 0.167 -> 0.833).
(b) Opponent's score [95% CI] when it guesses our move blindly (open) vs runs our decision rule to predict it (filled),
    in-engine, full information. Sources: RESULTS.md "Adversary" (ADV, Bayesian model, 100 pairs) and "Preliminary
    model's exploitability" (ADVP, 50 pairs per cell).

Run: <python with matplotlib> report/figures/make_models_figure.py -> report/figures/fig_models.{pdf,png}
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path(__file__).resolve().parent
sys.path.insert(0, "/Users/biofanat/.claude/skills/figure-style")
plt.style.use("/Users/biofanat/.claude/skills/figure-style/figure.mplstyle")
INK, GREY, LIGHT, BAYES = "#000000", "#8C8C8C", "#D9D9D9", "#0072B2"
plt.rcParams.update({"font.sans-serif": ["Helvetica", "Arial"], "text.color": INK, "axes.labelcolor": INK,
                     "xtick.color": INK, "ytick.color": INK, "axes.edgecolor": INK})

STRENGTH = [  # label, score, lo, hi, colour
    ("Preliminary vs Foul Play", 0.333, 0.231, 0.435, GREY),
    ("Bayesian vs Foul Play", 0.600, 0.481, 0.719, BAYES),
    ("Bayesian vs Foul Play, equal time", 0.483, 0.346, 0.620, BAYES),
    ("Bayesian vs preliminary", 0.833, 0.748, 0.919, BAYES),
]
EXPLOIT = [  # label, (blind score, lo, hi), (knowing score, lo, hi), gain label, colour
    ("Preliminary model", (0.370, 0.282, 0.458), (0.560, 0.475, 0.645), "+0.19", GREY),
    ("Bayesian model", (0.213, 0.155, 0.270), (0.700, 0.641, 0.759), "+0.49", BAYES),
]


def strength(ax):
    n = len(STRENGTH)
    for i, (lab, v, lo, hi, c) in enumerate(STRENGTH):
        y = n - 1 - i
        ax.plot([lo, hi], [y, y], color=c, lw=1.6, solid_capstyle="butt")
        ax.plot([v], [y], "o", ms=5, color=c, zorder=3)
    ax.set_yticks(range(n))
    ax.set_yticklabels([r[0] for r in STRENGTH][::-1])
    ax.axvline(0.5, color=LIGHT, lw=1, zorder=0)
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.6, n - 0.4)
    ax.set_xlabel("Score of first-named model")
    ax.set_title("Strength", loc="left")
    ax.tick_params(axis="y", length=0)


def exploit(ax):
    n = len(EXPLOIT)
    for i, (lab, blind, known, d, c) in enumerate(EXPLOIT):
        y = n - 1 - i
        for (v, lo, hi), filled in ((blind, False), (known, True)):
            ax.plot([lo, hi], [y, y], color=c, lw=1.6, solid_capstyle="butt")
            ax.plot([v], [y], "o", ms=5, color=c, mfc=c if filled else "white", mew=1.2, zorder=3)
        ax.annotate("", xy=(known[0] - 0.02, y + 0.3), xytext=(blind[0] + 0.02, y + 0.3),
                    arrowprops=dict(arrowstyle="->", color=INK, lw=0.8))
        ax.text((blind[0] + known[0]) / 2, y + 0.36, d, ha="center", va="bottom", color=INK)
    ax.set_yticks(range(n))
    ax.set_yticklabels([r[0] for r in EXPLOIT][::-1])
    ax.axvline(0.5, color=LIGHT, lw=1, zorder=0)
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.6, n - 0.05)
    ax.set_xlabel("Opponent's score (0.5 = even)")
    ax.set_title("Exploitability", loc="left")
    ax.tick_params(axis="y", length=0)


def main():
    fig, (a, b) = plt.subplots(2, 1, figsize=(3.46, 2.7), layout="constrained",
                               gridspec_kw=dict(height_ratios=[3.4, 2.6]))
    strength(a)
    exploit(b)
    a.set_title("a  Strength", loc="left")
    b.set_title("b  Exploitability", loc="left")
    from figcheck import check_figure
    check_figure(fig, min_pt=7, max_words=12)
    fig.savefig(OUT / "fig_models.pdf")
    fig.savefig(OUT / "fig_models.png", dpi=300)


if __name__ == "__main__":
    main()
