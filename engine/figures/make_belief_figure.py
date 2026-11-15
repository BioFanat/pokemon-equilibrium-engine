"""Reliability diagrams of the two hidden-set beliefs (results/belief/{showdown,engine}.json).

  .venv/bin/python figures/make_belief_figure.py   -> figures/fig_belief_calibration.{pdf,png}
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

plt.style.use("/Users/biofanat/.claude/skills/figure-style/figure.mplstyle")
sys.path.insert(0, "/Users/biofanat/.claude/skills/figure-style")
from figcheck import check_figure  # noqa: E402

HERE = Path(__file__).parent
RES = HERE.parent / "results" / "belief"
INK, MUTED = "#222222", "#9A9A9A"
SERIES = [("foul-play, Showdown", "showdown.json", "fp_exact", MUTED, "s"),
          ("Ours, engine", "engine.json", "ours", INK, "o")]
ATTRS = [("item", "Item"), ("ability", "Ability"), ("tera", "Tera type"), ("move", "Moves")]

fig, axes = plt.subplots(1, 4, figsize=(180 / 25.4, 2.3), sharex=True, sharey=True)
for ax, (attr, title) in zip(axes, ATTRS):
    ax.plot([0, 1], [0, 1], color="#CCCCCC", lw=0.8, zorder=1)
    for lab, f, src, c, mk in SERIES:
        d = json.load(open(RES / f))["metrics"][f"{src}/{attr}"]
        bins = [b for b in d["calibration_distinct"]["bins"] if b["n"] >= 20]
        ax.plot([b["mean_p"] for b in bins], [b["freq"] for b in bins], color=c, marker=mk, ms=4, lw=1.2,
                mfc=c if mk == "o" else "white", zorder=3, label=lab)
    ax.set_title(title)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_xticks([0, 0.5, 1]); ax.set_yticks([0, 0.5, 1])
    ax.set_xlabel("Predicted probability")
axes[0].set_ylabel("Observed frequency")
axes[1].legend(loc="lower right", frameon=False, handlelength=1.5)
fig.tight_layout()
issues = check_figure(fig, min_pt=8, max_words=12)
print("figcheck:", issues)
fig.savefig(HERE / "fig_belief_calibration.png", dpi=200)
fig.savefig(HERE / "fig_belief_calibration.pdf")
