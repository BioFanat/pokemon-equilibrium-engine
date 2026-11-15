"""Figures for the CS535 write-up. Run from final/engine: .venv/bin/python figures/make_figures.py

Fig 1  full information: fixing the payoff matrix, and sampling vs top move, vs MCTS; RL baseline
Fig 2  hidden information: head-to-head scores, sampled play vs top-move play
Fig 3  blend loss on real decisions and the 4·δ_H bound
Terms follow figures/FIGURE_VOCAB.md.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.style.use("/Users/biofanat/.claude/skills/figure-style/figure.mplstyle")
sys.path.insert(0, "/Users/biofanat/.claude/skills/figure-style")
from figcheck import check_figure  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RES, OUT = ROOT / "results", ROOT / "figures"
MM = 1 / 25.4
INK, MUTED = "#222222", "#9A9A9A"

# one entity -> one color, everywhere
COL = {
    "Agent-form": "#0072B2", "PIMC blend": "#D55E00", "Most-likely set": "#009E73", "Oracle": "#555555",
    "Safest move": "#E69F00", "Equilibrium, sampled": "#CC79A7", "Equilibrium, top move": "#56B4E9",
    "Self-play RL": "#000000",
}
NAME = {"agentform": "Agent-form", "pimc": "PIMC blend", "mostlikely": "Most-likely set", "oracle": "Oracle"}


def load(path):
    return [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]


def pair_ci(recs):
    """Mean score of A with a 95% CI over game pairs (a pair = same teams, sides swapped)."""
    pairs = {}
    for r in recs:
        if r.get("score_A") is not None:
            pairs.setdefault(r["pair"], []).append(r["score_A"])
    ps = np.array([np.mean(v) for v in pairs.values()])
    m = ps.mean()
    h = 1.96 * ps.std(ddof=1) / math.sqrt(len(ps))
    return m, h, 2 * len(ps)


def by_matchup(recs):
    d = {}
    for r in recs:
        d.setdefault((r["A"], r["B"]), []).append(r)
    return d


def finish(fig, name):
    issues = check_figure(fig, min_pt=8, max_words=12)
    fig.savefig(OUT / f"{name}.png", dpi=200)
    fig.savefig(OUT / f"{name}.pdf")
    print(name, "figcheck:", issues if issues else "clean")


# ------------------------------------------------------------------ Fig 1
def fig1():
    old = by_matchup(load(RES / "experiments" / "E1.jsonl"))
    new = by_matchup(load(RES / "experiments" / "E1fix.jsonl"))
    rows = [  # label, old spec (pruned NaN->0 matrix), new spec (full matrix)
        ("Safest move", "safest:50", "safest:0"),
        ("Equilibrium, sampled", "nash:rm+:50:1000", "nash:rm+:0:1000"),
        ("Equilibrium, top move", None, "nash:rm+:0:1000:argmax"),
    ]
    fig, (a, b) = plt.subplots(1, 2, figsize=(180 * MM, 62 * MM), gridspec_kw=dict(width_ratios=[1.15, 1]))
    for y, (lab, o, n) in enumerate(rows[::-1]):
        mn, hn, nn = pair_ci(new[(n, "mcts_it:2000")])
        if o:
            mo, ho, _ = pair_ci(old[(o, "mcts_it:2000")])
            a.plot([mo, mn], [y, y], color=MUTED, lw=1, zorder=1)
            a.errorbar(mo, y, xerr=ho, fmt="o", ms=5, mfc="white", mec=MUTED, ecolor=MUTED, lw=1, zorder=2)
        a.errorbar(mn, y, xerr=hn, fmt="o", ms=5, color=COL[lab], lw=1.2, zorder=3)
    a.axvline(0.5, color=MUTED, lw=0.6)
    a.set_yticks(range(len(rows)), [r[0] for r in rows[::-1]])
    a.set_xlim(0.2, 0.7)
    a.set_xlabel("Score vs MCTS (95% CI)")
    a.set_title("a  Full information, vs MCTS")
    a.text(0.434, 1.62, "pruned matrix", color=MUTED, ha="center", fontsize=8)
    a.text(0.561, 1.62, "full matrix", color=INK, ha="center", fontsize=8)

    ev = [r for r in load(RES / "ppo_pilot.jsonl") if "eval" in r and r["decisions"] > 0]
    x = np.array([r["decisions"] for r in ev]) / 1e6
    for key, lab, c in [("random", "vs random", MUTED), ("mcts200", "vs MCTS", COL["Self-play RL"])]:
        yv = np.array([r["eval"][key] for r in ev])
        k = 9
        sm = np.convolve(yv, np.ones(k) / k, mode="valid")
        b.plot(x[k // 2: k // 2 + len(sm)], sm, color=c, lw=1.2)
        b.text(x[-1] * 1.01, sm[-1], lab, color=c, va="center", fontsize=8)
    b.set_ylim(0, 1)
    b.set_xlim(0, x[-1] * 1.25)
    b.set_xlabel("Self-play decisions (millions)")
    b.set_ylabel("Win rate")
    b.set_title("b  Self-play RL, 25-min laptop run")
    fig.tight_layout()
    finish(fig, "fig1_full_information")


# ------------------------------------------------------------------ Fig 2
def fig2():
    samp = by_matchup(load(RES / "hidden" / "E4.jsonl"))
    top = by_matchup(load(RES / "hidden" / "E4am.jsonl")) if (RES / "hidden" / "E4am.jsonl").exists() else {}
    order = [("agentform:4:200", "pimc:4:200"), ("agentform:4:200", "mostlikely:200"),
             ("pimc:4:200", "mostlikely:200"), ("oracle:200", "agentform:4:200")]
    lab = lambda A, B: f"{NAME[A.split(':')[0]]} vs {NAME[B.split(':')[0]]}"
    fig, axes = plt.subplots(1, 2, figsize=(180 * MM, 55 * MM), sharey=True)
    for ax, data, suffix, title in [(axes[0], samp, "", "a  Sampled play"), (axes[1], top, ":argmax", "b  Top-move play")]:
        for y, (A, B) in enumerate(order[::-1]):
            key = (A + suffix, B + suffix)
            if key not in data:
                continue
            m, h, n = pair_ci(data[key])
            c = COL[NAME[A.split(":")[0]]]
            ax.errorbar(m, y, xerr=h, fmt="o", ms=5, color=c, lw=1.2)
            ax.text(0.97, y, f"n={n}", fontsize=8, color=MUTED, va="center", ha="right",
                    transform=ax.get_yaxis_transform())
        ax.axvline(0.5, color=MUTED, lw=0.6)
        ax.set_xlim(0.25, 0.85)
        ax.set_xlabel("Score of first-named agent (95% CI)")
        ax.set_title(title)
    axes[0].set_yticks(range(len(order)), [lab(*o) for o in order[::-1]])
    fig.tight_layout()
    finish(fig, "fig2_hidden_information")


# ------------------------------------------------------------------ Fig 3
def fig3():
    ds = load(RES / "hidden" / "E5_decisions.jsonl")
    loss = np.array([(d["v_AF"] - d["v_blend"]) for d in ds])
    rng_ = np.array([d["payoff_range"] for d in ds])
    bound = np.array([4 * d["delta_H"] for d in ds])
    rel = loss / np.where(rng_ > 0, rng_, 1)
    fig, (a, b) = plt.subplots(1, 2, figsize=(180 * MM, 62 * MM))
    pos = rel[rel > 1e-6] * 100
    xs = np.sort(pos)
    a.plot(xs, 1 - np.arange(len(xs)) / len(ds) - (1 - len(pos) / len(ds)), color=COL["Agent-form"], lw=1.2)
    a.set_xscale("log")
    a.set_xlabel("Blend loss (% of payoff range, log)")
    a.set_ylabel("Share of decisions with loss ≥ x")
    a.set_title(f"a  Blend loss, {len(ds):,} decisions")
    a.set_ylim(0, 0.7)
    for q in (1, 10):
        s = (rel * 100 >= q).mean()
        a.axvline(q, color=MUTED, lw=0.5)
        a.text(q * 1.1, s + 0.03, f"{s:.1%}", fontsize=8, color=INK)
    ok = (loss > 1e-6) & (bound > 0)
    b.scatter(bound[ok], loss[ok], s=3, color=COL["Agent-form"], alpha=0.25, lw=0, rasterized=True)
    lim = [1.0, bound[ok].max() * 1.5]
    b.plot(lim, lim, color=INK, lw=0.8)
    b.text(lim[1] / 3, lim[1] / 1.6, "loss = bound", fontsize=8, ha="right")
    b.set_xscale("log"); b.set_yscale("log")
    b.set_xlim(*lim)
    b.set_xlabel(r"Bound $4\delta_H$ (payoff units, log)")
    b.set_ylabel("Blend loss (payoff units, log)")
    b.set_title("b  Every decision is under the bound")
    fig.tight_layout()
    finish(fig, "fig3_blend_loss")
    print(f"E5: n={len(ds)}, loss>0 {np.mean(loss > 1e-6):.3f}, median rel {np.median(rel):.4f}, "
          f"p90 {np.quantile(rel, .9):.4f}, p99 {np.quantile(rel, .99):.4f}, max {rel.max():.4f}, "
          f">=1% {np.mean(rel >= .01):.3f}, >=10% {np.mean(rel >= .1):.3f}, "
          f"max loss/bound {np.max(loss[bound > 0] / bound[bound > 0]):.3f}, violations {(loss > bound + 1e-6).sum()}, "
          f"argmax differs {1 - np.mean([d['argmax_same'] for d in ds]):.3f}")


if __name__ == "__main__":
    for f in (fig1, fig2, fig3):
        f()
