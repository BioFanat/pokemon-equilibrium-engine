"""Win rates of the key head-to-head comparisons (fig_wins): score of the first-named bot [95% CI], one dot per comparison.

Sources:
  tournament rows   engine/showdown/results/gauntlet/G1_ratings.json ("pairwise": agentform vs stock@{25,100,400,1000},
                    agentform vs pimc), 95% CIs over seat-swapped pairs.
  other rows        the values reported in final5.tex before the tables were folded into this figure (former Table 1,
                    in-engine; former Table 2, real Showdown), 95% CIs over pairs.

Run:  python3 report/figures/make_wins_figure.py   ->  report/figures/fig_wins.{pdf,png}
"""
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]                                  # final-integration/
GAUNTLET = ROOT / "engine/showdown/results/gauntlet/G1_ratings.json"
sys.path.insert(0, str(ROOT / "report"))
sys.path.insert(0, "/Users/biofanat/.claude/skills/figure-style")
from figcheck import check_figure  # noqa: E402
from make_figures import COL, INK, GRAY  # noqa: E402

plt.style.use("/Users/biofanat/.claude/skills/figure-style/figure.mplstyle")


def tournament():
    d = json.loads(GAUNTLET.read_text())
    pw = {(p["a"], p["b"]): p for p in d["pairwise"]}
    rows = []
    for ms in (25, 100, 400, 1000):
        p = pw[("agentform", f"stock@{ms}")]
        rows.append((f"Bayesian LP vs Foul Play, {ms} ms", p["score_a"], *p["ci95"], COL["eq_sampled"]))
    p = pw[("agentform", "pimc")]
    rows.append(("Bayesian LP vs PIMC averaging", p["score_a"], *p["ci95"], COL["eq_sampled"]))
    return rows


def rows():
    # Color follows the first-named bot; the Bayesian-LP agent ("agentform") plays its LP solution sampled.
    show = [
        ("Bayesian LP vs Foul Play", 0.314, 0.279, 0.350, COL["eq_sampled"]),
        ("Hedge vs Foul Play", 0.311, 0.280, 0.341, COL["hedge_carried"]),
        ("Hedge vs Bayesian LP", 0.458, 0.421, 0.496, COL["hedge_carried"]),
        ("Bayesian LP vs PIMC averaging", 0.559, 0.521, 0.597, COL["eq_sampled"])]
    engine = [
        ("MCTS vs Bayesian LP, sampled", 0.760, 0.677, 0.843, COL["mcts"]),
        ("MCTS vs Bayesian LP, top move", 0.650, 0.571, 0.729, COL["mcts"]),
        ("MCTS (2,000 iter.) vs top move", 0.310, 0.235, 0.385, COL["mcts"]),
        ("Bayesian LP vs PIMC, sampled", 0.585, 0.540, 0.630, COL["eq_sampled"]),
        ("Bayesian LP vs PIMC, top move", 0.540, 0.452, 0.628, COL["eq_top"])]
    return [("Real Showdown", show), ("Round-robin tournament", tournament()), ("In-engine", engine)]


def main():
    groups = rows()
    fig, ax = plt.subplots(figsize=(3.3, 2.65), layout="constrained")
    y, ticks, labels, heads = 0, [], [], []
    for head, rs in groups:
        heads.append(y); y -= 1
        for lab, s, lo, hi, c in rs:
            ax.errorbar(s, y, xerr=[[s - lo], [hi - s]], fmt="o", color=c, ms=4, elinewidth=1.2, capsize=0)
            ticks.append(y); labels.append(lab); y -= 1
        y -= 0.3
    for yh, (head, _) in zip(heads, groups):
        ax.text(0.0, yh, head, transform=ax.get_yaxis_transform(), ha="right", va="center", fontweight="bold")
    ax.axvline(0.5, color=GRAY, lw=1.0, ls=(0, (4, 2)), zorder=0)
    ax.set_yticks(ticks, labels)
    ax.yaxis.set_tick_params(length=0)
    ax.spines["left"].set_visible(False)
    ax.set_ylim(y + 0.6, 0.6)
    ax.set_xlim(0.1, 0.9)
    ax.set_xticks([0.1, 0.3, 0.5, 0.7, 0.9])
    ax.set_xlabel("Score of first-named bot (0.5 = even)", loc="right")
    issues = check_figure(fig, min_pt=8, max_words=12)
    print("[fig_wins] figcheck:", issues if issues else "clean")
    fig.savefig(OUT / "fig_wins.pdf")
    fig.savefig(OUT / "fig_wins.png", dpi=200)
    plt.close(fig)
    for head, rs in groups:
        for r in rs:
            print(head, r[:4])


if __name__ == "__main__":
    main()
