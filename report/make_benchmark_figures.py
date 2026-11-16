"""Benchmark figures for the report: the Showdown gauntlet ratings and the live style-switch test of the posterior.

Run:  /Users/biofanat/Documents/classes/f26/cs535/final/engine/.venv/bin/python report/make_benchmark_figures.py [--refresh]
Writes report/figures/fig_gauntlet.{pdf,png} and fig_switch.{pdf,png}. Every number is read from a file:
  fig_gauntlet  A  Bradley-Terry ratings (Elo scale, agent-form = 0) with 95% bootstrap CIs:
                   engine/showdown/results/gauntlet/G1_ratings.json ("ratings", "ci95").
                B  agent-form's score against foul-play at each budget, 95% CI over seed pairs: same file, "pairwise".
  fig_switch    A  per-stream realised re-concentration time tau_map vs the predicted tau_hat, arms SW2, SW4 (lambda = 1)
                   and SW4p (lambda = 0.98): research/switch_results/out_style_switch.json ("arms"/<arm>/"rows").
                B  P(uniform) of the posterior around the switch for two example streams, recomputed from the logged
                   p_style by research/scripts/switch/analyze_style_switch.py (arm_streams + reconstruct) over the logs in
                   ~/sandbox/ourbot/sw{,_p}/replays_headless; cached to report/figures/data/switch_traj.json
                   (--refresh rebuilds it from the logs).
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, NullFormatter, NullLocator

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "report/figures"
DATA = OUT / "data"
sys.path.insert(0, str(ROOT / "report"))
from make_figures import COL, INK, GRAY, LINEWIDTH_IN, finish  # noqa: E402  (shared colours, figcheck + save)

plt.style.use("/Users/biofanat/.claude/skills/figure-style/figure.mplstyle")

GAUNTLET = ROOT / "engine/showdown/results/gauntlet/G1_ratings.json"
SWITCH = ROOT / "research/switch_results/out_style_switch.json"
SWITCH_SCRIPTS = ROOT / "research/scripts/switch"
TRAJ = DATA / "switch_traj.json"

C_OURS = COL["eq_top"]               # agent-form: our agent (same hex as "Agent-form" in engine/figures)
C_COMP = "#666666"                   # comparators that are not foul-play: gray
# foul-play budgets: one hue (COL["mcts"] at 400 ms), darker = more time per sampled battle
FOUL = {25: "#EE9A5E", 100: "#E2772E", 400: COL["mcts"], 1000: "#8F3E00"}
NAME = {"agentform": "Agent-form (ours)", "top": "Agent-form, top move", "pimc": "PIMC blend",
        "maxpower": "Max damage", "random": "Uniform random"}
XLIM_A = (-420, 260)                 # rating axis; the random bot (-911) is drawn as an off-scale arrow

C_LAM1, C_FORGET = COL["bayes"], "#7E3566"   # posterior lambda = 1 / forgetting lambda = 0.98 (validated pair)
ARMS = [  # arm, label, colour, marker
    ("SW2", "λ = 1, switch after 4 games", C_LAM1, "o"),
    ("SW4", "λ = 1, switch after 8 games", C_LAM1, "s"),
    ("SW4p", "λ = 0.98, switch after 8 games", C_FORGET, "^"),
]
EXAMPLES = [("SW4", 1), ("SW4p", 1)]  # streams whose realised time is close to the prediction (247 vs 246, 37 vs 31)
LOG_ROOT = {"SW4": Path.home() / "sandbox/ourbot/sw/replays_headless",
            "SW4p": Path.home() / "sandbox/ourbot/sw_p/replays_headless"}
SW_LIM = (3, 1000)


def label(bot):
    ms = bot.partition("@")[2]
    return f"foul-play, {ms} ms" if ms else NAME[bot]


def colour(bot):
    ms = bot.partition("@")[2]
    return FOUL[int(ms)] if ms else (C_OURS if bot == "agentform" else C_COMP)


def panel_letters(fig, *axes):
    """Bold panel letters at the left edge of each panel's full extent (tick and axis labels included)."""
    fig.canvas.draw()
    inv = fig.transFigure.inverted()
    for ax, letter in zip(axes, "ABCDEF"):
        x0 = inv.transform(ax.get_tightbbox().p0)[0]
        y1 = inv.transform(ax.transAxes.transform((0, 1)))[1]
        fig.text(x0, y1 + 0.02, letter, fontsize=12, fontweight="bold", ha="left", va="bottom")


# ================================================================ fig_gauntlet
def fig_gauntlet():
    d = json.loads(GAUNTLET.read_text())
    bots = sorted(d["bots"], key=lambda b: d["ratings"][b])           # bottom -> top
    fig, (a, b) = plt.subplots(1, 2, figsize=(LINEWIDTH_IN, 2.7), layout="constrained",
                               gridspec_kw=dict(width_ratios=[1.55, 1]))
    a.axvline(0, color=GRAY, lw=0.6, zorder=0)
    for y, bot in enumerate(bots):
        r, (lo, hi) = d["ratings"][bot], d["ci95"][bot]
        c = colour(bot)
        if r < XLIM_A[0]:   # off scale: arrow at the left edge, value as a direct label
            a.plot(XLIM_A[0] + 8, y, marker="<", color=c, ms=6, clip_on=False)
            a.annotate(f"{r:.0f} [{lo:.0f}, {hi:.0f}]".replace("-", "−"), (XLIM_A[0] + 8, y),
                       xytext=(7, 0), textcoords="offset points", va="center", color=INK)
            continue
        a.errorbar(r, y, xerr=[[r - lo], [hi - r]], fmt="o", color=c, ms=6 if bot == "agentform" else 5,
                   elinewidth=1.2, capsize=0, zorder=3)
        print(f"[fig_gauntlet] {bot}: {r:.1f} [{lo:.1f}, {hi:.1f}]  games {d['games_used'][bot]}")
    a.set_yticks(range(len(bots)), [label(x) for x in bots])
    for t, bot in zip(a.get_yticklabels(), bots):
        if bot == "agentform":
            t.set_fontweight("bold")
    a.yaxis.set_tick_params(length=0)
    a.spines["left"].set_visible(False)
    a.set_xlim(*XLIM_A)
    a.set_ylim(-0.6, len(bots) - 0.4)
    a.set_xlabel("Rating (Elo, agent-form = 0)")
    a.set_title("Showdown gauntlet ratings")

    rows = {p["b"]: p for p in d["pairwise"] if p["a"] == "agentform" and p["b"].startswith("stock@")}
    ms = sorted(int(k.partition("@")[2]) for k in rows)
    s = np.array([rows[f"stock@{m}"]["score_a"] for m in ms])
    ci = np.array([rows[f"stock@{m}"]["ci95"] for m in ms])
    print("[fig_gauntlet] agent-form vs foul-play:", [(m, rows[f'stock@{m}']['score_a'], rows[f'stock@{m}']['ci95'],
                                                       rows[f'stock@{m}']['n_pairs']) for m in ms])
    b.axhline(0.5, color=GRAY, lw=0.8, ls="--", zorder=0)
    b.plot(ms, s, color=C_OURS, lw=1.0, zorder=2)
    b.errorbar(ms, s, yerr=[s - ci[:, 0], ci[:, 1] - s], fmt="o", color=C_OURS, ms=5, elinewidth=1.2, capsize=0,
               zorder=3)
    b.text(ms[-1], 0.5, "even", ha="right", va="bottom", color=GRAY)
    b.set_xscale("log")
    b.xaxis.set_major_locator(FixedLocator(ms))
    b.xaxis.set_major_formatter(matplotlib.ticker.FixedFormatter([str(m) for m in ms]))
    b.xaxis.set_minor_locator(NullLocator())
    b.xaxis.set_minor_formatter(NullFormatter())
    b.set_xlim(18, 1400)
    b.set_ylim(0, 0.6)
    b.set_xlabel("Foul-play ms per sampled battle (log)")
    b.set_ylabel("Agent-form score")
    b.set_title("Agent-form vs foul-play")
    panel_letters(fig, a, b)
    finish(fig, "fig_gauntlet")


# ================================================================ fig_gauntlet_col
# Single-column panel A for final5.tex, drawn at its print width (included unscaled) in the text's vocabulary.
COL_W_IN = 3.4                       # \includegraphics without width; final5.tex column is 3.525 in
NAME_COL = {"agentform": "Bayesian LP (ours)", "top": "top move", "pimc": "PIMC",
            "maxpower": "max damage", "random": "uniform random"}


def fig_gauntlet_col():
    d = json.loads(GAUNTLET.read_text())
    bots = sorted(d["bots"], key=lambda b: d["ratings"][b])           # bottom -> top
    fig, a = plt.subplots(figsize=(COL_W_IN, 2.25), layout="constrained")
    a.axvline(0, color=GRAY, lw=0.6, zorder=0)
    for y, bot in enumerate(bots):
        r, (lo, hi) = d["ratings"][bot], d["ci95"][bot]
        c = colour(bot)
        if r < XLIM_A[0]:   # off scale: arrow at the left edge, value as a direct label (CI in the caption)
            a.plot(XLIM_A[0] + 8, y, marker="<", color=c, ms=6, clip_on=False)
            a.annotate(f"{r:.0f}".replace("-", "−"), (XLIM_A[0] + 8, y),
                       xytext=(7, 0), textcoords="offset points", va="center", color=INK)
            continue
        a.errorbar(r, y, xerr=[[r - lo], [hi - r]], fmt="o", color=c, ms=6 if bot == "agentform" else 5,
                   elinewidth=1.2, capsize=0, zorder=3)
    ms_lab = lambda bot: f"Foul Play, {bot.partition('@')[2]} ms" if "@" in bot else NAME_COL[bot]
    a.set_yticks(range(len(bots)), [ms_lab(x) for x in bots])
    for t, bot in zip(a.get_yticklabels(), bots):
        if bot == "agentform":
            t.set_fontweight("bold")
    a.yaxis.set_tick_params(length=0)
    a.spines["left"].set_visible(False)
    a.set_xlim(*XLIM_A)
    a.set_ylim(-0.6, len(bots) - 0.4)
    a.set_xlabel("Rating (Elo, Bayesian LP = 0)")
    finish(fig, "fig_gauntlet_col", exact=True)


# ================================================================ fig_switch
def switch_traj(refresh=False):
    """P(uniform) per update for the EXAMPLES streams, recomputed from the logged p_style (cached)."""
    if TRAJ.exists() and not refresh:
        return json.loads(TRAJ.read_text())
    sys.path.insert(0, str(SWITCH_SCRIPTS))
    import analyze_style_switch as A
    arms = json.loads(SWITCH.read_text())["arms"]
    out = {"source": "research/scripts/switch/analyze_style_switch.py arm_streams + reconstruct", "streams": []}
    for tag, k in EXAMPLES:
        meta = arms[tag]
        S = dict(A.arm_streams(str(LOG_ROOT[tag]), tag))[k]
        L, _ = A.reconstruct(S["ps"], meta["variant"], meta["lam"])
        row = meta["rows"][k]
        t0 = int(S["phase1"].sum())
        assert (len(L), t0) == (row["T"], row["t0"]), (tag, k, len(L), t0)
        out["streams"].append(dict(arm=tag, stream=k, lam=meta["lam"], t0=t0, tau_map=row["tau_map"],
                                   tau_hat=row["tau_hat"], p_uniform=np.exp(L[:, A.IU]).round(6).tolist()))
    TRAJ.write_text(json.dumps(out))
    return out


def fig_switch(refresh=False):
    arms = json.loads(SWITCH.read_text())["arms"]
    fig, (a, b) = plt.subplots(1, 2, figsize=(LINEWIDTH_IN, 2.9), layout="constrained",
                               gridspec_kw=dict(width_ratios=[1, 1.3]))
    a.plot(SW_LIM, SW_LIM, color=GRAY, lw=0.8, zorder=0, label="y = x")
    for tag, lab, c, mk in ARMS:
        rows = [r for r in arms[tag]["rows"] if r["clean"] and r["post_updates"] > 0]
        x = np.array([r["tau_hat"] for r in rows])
        y = np.array([r["tau_map"] for r in rows])
        cens = np.array([r["tau_map_censored"] for r in rows])
        a.scatter(x[~cens], y[~cens], marker=mk, color=c, s=24, lw=0, label=lab, zorder=3)
        for xi, yi in zip(x[cens], y[cens]):   # right-censored: open marker + arrow up
            a.scatter(xi, yi, marker=mk, facecolor="white", edgecolor=c, s=24, lw=1.0, zorder=3)
            a.annotate("", (xi, yi * 1.6), xytext=(xi, yi * 1.08),
                       arrowprops=dict(arrowstyle="-|>", color=c, lw=1.0, mutation_scale=7))
            a.annotate("not reached", (xi, yi * 1.6), xytext=(-4, 0), textcoords="offset points", ha="right",
                       va="center", color=INK)
        print(f"[fig_switch] {tag}: n={len(rows)} censored={int(cens.sum())}")
    a.set_xscale("log")
    a.set_yscale("log")
    a.set_xlim(*SW_LIM)
    a.set_ylim(*SW_LIM)
    a.set_aspect("equal")
    for ax_ in (a.xaxis, a.yaxis):
        ax_.set_major_locator(FixedLocator([10, 30, 100, 300, 1000]))
        ax_.set_major_formatter(matplotlib.ticker.FixedFormatter(["10", "30", "100", "300", "1000"]))
        ax_.set_minor_formatter(NullFormatter())
    a.set_xlabel("Predicted time (updates, log)")
    a.set_ylabel("Realised time (updates, log)")
    a.set_title("Re-concentration after the switch")
    h, l = a.get_legend_handles_labels()
    a.legend(h[1:] + h[:1], l[1:] + l[:1], loc="lower right", handletextpad=0.3, borderaxespad=0.1)

    tr = switch_traj(refresh)
    b.axvline(0, color=GRAY, lw=0.6, zorder=0)
    b.text(3, 0.5, "switch", color=GRAY, ha="left", va="center")
    for st in tr["streams"]:
        c = C_LAM1 if st["lam"] == 1.0 else C_FORGET
        p = np.array(st["p_uniform"])
        t = np.arange(len(p)) - st["t0"] + 1          # update index relative to the first post-switch update
        keep = t >= -60
        b.plot(t[keep], p[keep], color=c, lw=1.2)
        i = st["t0"] + st["tau_map"] - 1                # first update of the sustained MAP = uniform run
        b.plot(t[i], p[i], "o", color=c, ms=5, zorder=3)
        lam = "1" if st["lam"] == 1.0 else f"{st['lam']:g}"
        b.text(t[i] + 25, 0.9, f"λ = {lam}", color=c, fontweight="bold", ha="left", va="top")
        print(f"[fig_switch] example {st['arm']} stream {st['stream']}: tau_map={st['tau_map']} "
              f"tau_hat={st['tau_hat']:.1f}")
    b.set_xlim(-60, 330)
    b.set_ylim(-0.03, 1.03)
    b.set_xlabel("Updates since the switch")
    b.set_ylabel("Posterior on uniform random")
    b.set_title("Two example streams")
    panel_letters(fig, a, b)
    finish(fig, "fig_switch")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--refresh", action="store_true", help="rebuild switch_traj.json from the logs")
    args = ap.parse_args()
    fig_gauntlet()
    fig_switch(args.refresh)
