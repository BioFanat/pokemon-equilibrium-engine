"""Regret along real Showdown games against the proved bounds (fig_regret), three panels:

  A  inside a turn   RM+ self-play (alternating updates, linear averaging; engine/nash.rm_plus) on the 35 real stage
                     matrices of report/figures/data/rmplus_gap.json (seed 0; gap = exploitability / ptp(M)), median and
                     range over matrices, against the bound 2(sqrt n + sqrt m)/sqrt T (median over matrices).
  B  across games    Hedge (4 experts, anytime step eta_t = sqrt(8 ln 4 / t)) carried over each LOG match (Hedge+memory
                     vs Foul Play, 10 matches). Reg_t = max_e sum_{j<=t} r_{j,e} - sum_{j<=t} w_j . r_j on the learner's own
                     clipped rewards (feedback.r) and the weights it played (the decision record's weights), per match,
                     plotted per scored turn, against the anytime bound (3/2) sqrt(t ln4/2) + sqrt(ln4/8) of fact:hedge.
  C  opponent memory Bayes over 5 styles: R_t(theta) = sum_{j<=t} (ln p_style_j(theta) - ln p_qhat_j); plotted is
                     max_theta R_t(theta), the regret vs the best style in hindsight up to t (it equals R_T(theta*) at
                     the end, theta* = end-of-match best style), against ln 5 (prop:B1). Identity R_T = ln 5 + ln Pi_T(theta*) checked
                     against the learner's saved final log-posterior (hedge_state/*.bayes.json).

Run:  python3 report/figures/make_regret_figure.py
Writes report/figures/fig_regret.{pdf,png} and report/figures/fig_regret_numbers.json.
"""
import glob
import json
import os
import re
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]                                  # final-integration/
LOG_DIR = Path("/Users/biofanat/Documents/classes/f26/cs535/final/engine/showdown/results/LOG")
RM_DATA = OUT / "data/rmplus_gap.json"
sys.path.insert(0, str(ROOT / "report"))
sys.path.insert(0, "/Users/biofanat/.claude/skills/figure-style")
from figcheck import check_figure  # noqa: E402
from make_figures import COL, INK, GRAY  # noqa: E402  (one colour per method, shared with every figure)

plt.style.use("/Users/biofanat/.claude/skills/figure-style/figure.mplstyle")

EXPERTS = ["top", "sampled", "safest", "safex"]
STYLES = ["eq", "br_unif", "br_us", "safe", "uniform"]
LN4, LN5 = np.log(4), np.log(5)
MEM_EXCLUDED = {9}                                     # match 9 lost 12 logged updates (1928 vs 1940): identity breaks
HEDGE_FIT_LO = 200                                     # growth-exponent fit range: t in [200, T_min]
RM_FIT_LO = 10                                         # RM+ slope fit range: T in [10, 2000] (as fig_rmplus)


# ================================================================ data: LOG decision logs
def load_streams():
    """Per match (stream): the learner's updates in play order. Error games are kept (the learner updated on them)."""
    res = LOG_DIR / "replays_headless"
    games = []
    for p in glob.glob(str(res / "LOG-s*-g*.A.decisions.jsonl")):
        m = re.search(r"LOG-s(\d+)-g(\d)\.", os.path.basename(p))
        seed, game = int(m.group(1)), int(m.group(2))
        g = json.load(open(res / f"LOG-s{seed}-g{game}.json"))
        games.append((g["stream"], seed, game, p))
    streams = {}
    for stream, seed, game, p in sorted(games):        # seeds of a stream are played in increasing order
        recs = [json.loads(l) for l in open(p) if l.strip()]
        for i, r in enumerate(recs[:-1]):
            fb = recs[i + 1].get("feedback")
            if "abar" not in r or not fb or fb.get("skipped") or "r" not in fb:
                continue
            assert fb["for_turn"] == r["turn"], (p, i)
            b = fb["bayes"]
            streams.setdefault(stream, []).append(dict(
                r=np.array(fb["r"], float), w=np.array([r["weights"][e] for e in EXPERTS]),
                w_after=np.array(fb["weights"], float),
                p_qhat=b["p_qhat"], p_style=np.array([b["p_style"][s] for s in STYLES]),
                p_mix=float(np.dot([r["bayes"]["posterior"][s] for s in STYLES], [b["p_style"][s] for s in STYLES]))))
    return streams


def load_state(stream):
    h = json.load(open(LOG_DIR / f"hedge_state/LOG-stream{stream}-A.json"))
    b = json.load(open(LOG_DIR / f"hedge_state/LOG-stream{stream}-A.bayes.json"))
    return h, b


def replay_weights(R):
    """The learner's FTRL weights from its own rewards: p_t ∝ exp(eta_{t-1} sum_{j<t} r_j), eta_t = sqrt(8 ln4 / t)."""
    cum = np.vstack([np.zeros(4), np.cumsum(R, 0)[:-1]])
    t = np.arange(len(R))
    eta = np.sqrt(8 * LN4 / np.maximum(t, 1))
    z = eta[:, None] * cum
    W = np.exp(z - z.max(1, keepdims=True))
    return W / W.sum(1, keepdims=True)


def hedge_bound(t):
    return 1.5 * np.sqrt(t * LN4 / 2) + np.sqrt(LN4 / 8)


def analyse(streams):
    per, hedge_curves, mem_curves = [], {}, {}
    for s, rows in sorted(streams.items()):
        R = np.array([x["r"] for x in rows]); W = np.array([x["w"] for x in rows])
        T = len(R)
        reg = np.cumsum(R, 0).max(1) - np.cumsum((W * R).sum(1))
        hedge_curves[s] = reg
        h, b = load_state(s)
        wrep = replay_weights(R)
        dev = np.abs(wrep - W).max(1)
        first_bad = int(np.argmax(dev > 2e-3)) if (dev > 2e-3).any() else None
        # memory: log-loss regret against each fixed style; theta* = end-of-match best style
        L = np.log(np.array([x["p_style"] for x in rows])) - np.log([x["p_qhat"] for x in rows])[:, None]
        Rth = np.cumsum(L, 0)
        k = int(Rth[-1].argmax())
        mem_curves[s] = Rth.max(1)                      # plotted: regret vs the best style up to t (ends at theta*)
        logp = np.array(b["logp"])
        lnPi = logp - (logp.max() + np.log(np.exp(logp - logp.max()).sum()))
        k_state = int(logp.argmax())
        per.append(dict(
            match=int(s), T=T, T_state=int(h["t"]), counts_match=bool(T == h["t"]),
            reg_T=float(reg[-1]), bound_T=float(hedge_bound(T)), ratio=float(reg[-1] / hedge_bound(T)),
            reg_per_turn=float(reg[-1] / T), bound_per_turn=float(hedge_bound(T) / T),
            bound_per_turn_cbl2=float((2 * np.sqrt(T * LN4 / 2) + np.sqrt(LN4 / 8)) / T),
            reg_max=float(reg.max()), n_nonpos=int((reg <= 0).sum()),
            cum_vs_state_maxabs=float(np.abs(R.sum(0) - np.array(h["cum"])).max()),
            weight_replay_maxdev=float(dev.max()), weight_replay_first_bad=first_bad,
            mem_best_style=STYLES[k], mem_best_style_state=STYLES[k_state],
            mem_R_T=float(Rth[-1, k]), mem_R_max=float(Rth[:, k].max()), mem_R_min_end_style=float(Rth[:, k].min()),
            mem_Rbest_max=float(Rth.max(1).max()), mem_Rbest_min=float(Rth.max(1).min()),
            mem_identity_err=float(abs(Rth[-1, k] - (LN5 + lnPi[k]))),
            mem_lnPi_T=float(lnPi[k]),
            mem_pqhat_vs_mix_maxabs=float(np.max([abs(x["p_qhat"] - x["p_mix"]) for x in rows]))))
    return per, hedge_curves, mem_curves


def rm_data():
    d = json.loads(RM_DATA.read_text())
    T = np.array(d["T"], float)
    G = np.array([r["gap"] for r in d["rows"]])
    nm = np.array([r["shape"] for r in d["rows"]], float)
    B = 2 * (np.sqrt(nm[:, :1]) + np.sqrt(nm[:, 1:])) / np.sqrt(T[None])
    return T, G, nm, B


def slope(x, y):
    return float(np.polyfit(np.log(x), np.log(y), 1)[0])


# ================================================================ figure
def draw(rm, per, hedge_curves, mem_curves):
    T, G, nm, B = rm
    fig, axs = plt.subplots(1, 3, figsize=(7.0, 1.9), layout="constrained")
    for ax, L in zip(axs, "abc"):
        ax.text(-0.02, 1.2, L, transform=ax.transAxes, fontsize=12, fontweight="bold", va="top", ha="right")

    # ---- a: RM+ inside a turn
    ax = axs[0]
    c = COL["eq_top"]
    med, lo, hi, bnd = np.median(G, 0), G.min(0), G.max(0), np.median(B, 0)
    ax.loglog(T, bnd, color=GRAY, lw=1.2, ls=(0, (4, 2)))
    ax.fill_between(T, lo, hi, color=c, alpha=0.18, lw=0)
    ax.loglog(T, med, color=c, lw=1.5)
    fit = T >= RM_FIT_LO
    s_med = slope(T[fit], med[fit])
    ax.text(30, bnd[list(T).index(20)] * 3.5, "bound", color=GRAY, va="center")
    ax.text(1.4, 2e-6, f"median, slope {s_med:.1f}".replace("-", "−"), color=c, va="center")
    ax.set_xlim(1, 2000); ax.set_ylim(1e-8, 20)
    ax.set_xticks([1, 10, 100, 1000])
    ax.yaxis.set_major_locator(matplotlib.ticker.LogLocator(numticks=5))
    ax.yaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    ax.set_xlabel("RM$^{\\mathregular{+}}$ iterations $T$")
    ax.set_ylabel("Duality gap / payoff range")
    ax.set_title("Inside a turn: RM$^{\\mathregular{+}}$")

    # ---- b: Hedge across games (regret per scored turn)
    ax = axs[1]
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
    ax.text(3, hedge_bound(3) / 3 * 1.9, "anytime bound", color=GRAY, va="center")
    ax.text(1.5, 4e-4, "Hedge regret (10 matches)", color=c, va="center")
    ax.set_xlim(1, Tmax); ax.set_ylim(1e-4, 5)
    ax.set_xticks([1, 10, 100, 1000])
    ax.yaxis.set_major_locator(matplotlib.ticker.LogLocator(numticks=5))
    ax.yaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    ax.set_xlabel("Scored turns $t$")
    ax.set_ylabel("Regret per scored turn")
    ax.set_title("Across games: Hedge")

    # ---- c: opponent memory (median and range over matches, up to the shortest plotted match)
    ax = axs[2]
    c = COL["bayes"]
    M = [v for s, v in mem_curves.items() if s not in MEM_EXCLUDED]
    Tm = min(len(v) for v in M)
    A = np.array([v[:Tm] for v in M])
    tc = np.arange(1, Tm + 1)
    ax.axhline(LN5, color=GRAY, lw=1.2, ls=(0, (4, 2)))
    ax.fill_between(tc, A.min(0), A.max(0), color=c, alpha=0.18, lw=0)
    ax.plot(tc, np.median(A, 0), color=c, lw=1.5, zorder=3)
    ax.set_xscale("log")
    ax.text(1.5, LN5 + 0.22, "bound ln 5", color=GRAY, va="center")
    ax.text(Tm * 0.9, 0.3, f"median ({len(M)} matches)", color=c, ha="right", va="center")
    ax.set_xlim(1, Tm); ax.set_ylim(0, 2)
    ax.set_xticks([1, 10, 100, 1000])
    ax.set_xlabel("Scored turns $t$")
    ax.set_ylabel("Log-loss regret (nats)")
    ax.set_title("Opponent memory")
    return fig, s_med


def main():
    streams = load_streams()
    per, hedge_curves, mem_curves = analyse(streams)
    rm = rm_data()
    T, G, nm, B = rm
    fig, s_med = draw(rm, per, hedge_curves, mem_curves)
    issues = check_figure(fig, min_pt=8, max_words=12)
    print("[fig_regret] figcheck:", issues if issues else "clean")
    fig.savefig(OUT / "fig_regret.pdf")
    fig.savefig(OUT / "fig_regret.png", dpi=200)
    plt.close(fig)

    # ---- numbers
    Tmin = min(len(v) for v in hedge_curves.values())
    medc = np.median(np.array([v[:Tmin] for v in hedge_curves.values()]), 0)
    tt = np.arange(1, Tmin + 1)
    fit = tt >= HEDGE_FIT_LO
    R_all = sum(p["reg_T"] for p in per); T_all = sum(p["T"] for p in per)
    i1000 = list(T).index(1000)
    med_g, med_b = np.median(G, 0), np.median(B, 0)
    fitrm = T >= RM_FIT_LO
    fit100 = T >= 100
    per_mat = [slope(T[fitrm], np.maximum(g[fitrm], 1e-300)) for g in G]
    out = dict(
        rmplus=dict(source=str(RM_DATA.relative_to(ROOT)), n_matrices=len(G),
                    n_range=[int(nm[:, 0].min()), int(nm[:, 0].max())], m_range=[int(nm[:, 1].min()), int(nm[:, 1].max())],
                    median_shape=[float(np.median(nm[:, 0])), float(np.median(nm[:, 1]))],
                    T=T.tolist(), median_gap=med_g.tolist(), min_gap=G.min(0).tolist(), max_gap=G.max(0).tolist(),
                    median_bound=med_b.tolist(),
                    slope_median_T10_2000=s_med, slope_median_T100_2000=slope(T[fit100], med_g[fit100]),
                    slope_per_matrix_quartiles_T10_2000=np.percentile(per_mat, [25, 50, 75]).tolist(),
                    gap_T1000_median=float(med_g[i1000]), bound_T1000_median=float(med_b[i1000]),
                    ratio_T1000=float(med_g[i1000] / med_b[i1000]),
                    max_gap_over_own_bound=float((G / B).max())),
        hedge=dict(bound="(3/2) sqrt(T ln4/2) + sqrt(ln4/8)", per_match=per,
                   pooled_reg_per_turn=R_all / T_all, T_total=T_all,
                   ratio_range=[min(p["ratio"] for p in per), max(p["ratio"] for p in per)],
                   growth_exponent_median=slope(tt[fit], np.maximum(medc[fit], 1e-12)),
                   growth_fit_range=[HEDGE_FIT_LO, Tmin],
                   median_reg_per_turn_at_Tmin=float(medc[-1] / Tmin), bound_per_turn_at_Tmin=float(hedge_bound(Tmin) / Tmin)),
        memory=dict(ln5=float(LN5), plotted_matches=[p["match"] for p in per if p["match"] not in MEM_EXCLUDED],
                    max_identity_err=max(p["mem_identity_err"] for p in per),
                    max_identity_err_plotted=max(p["mem_identity_err"] for p in per if p["match"] not in MEM_EXCLUDED),
                    max_identity_err_count_clean=max(p["mem_identity_err"] for p in per if p["counts_match"]),
                    R_T_range=[min(p["mem_R_T"] for p in per), max(p["mem_R_T"] for p in per)],
                    R_max=max(p["mem_R_max"] for p in per)),
        note="LOG run (Hedge+memory vs Foul Play), per match = per stream with carried weights; rewards = feedback.r "
             "(learner's clipped normalised rewards), weights = decision record weights; error game LOG-s4-g2 included.")
    (OUT / "fig_regret_numbers.json").write_text(json.dumps(out, indent=1))
    h = out["hedge"]
    print(f"RM+: n={len(G)} shapes n {out['rmplus']['n_range']} m {out['rmplus']['m_range']}; slope T>=10 {s_med:.2f}, "
          f"T>=100 {out['rmplus']['slope_median_T100_2000']:.2f}; T=1000 gap {med_g[i1000]:.2e} bound {med_b[i1000]:.3f}")
    print(f"Hedge: pooled {h['pooled_reg_per_turn']:.5f}/turn over {T_all}; ratio {h['ratio_range']}; "
          f"exponent {h['growth_exponent_median']:.2f} on {h['growth_fit_range']}")
    for p in per:
        print({k: (round(v, 5) if isinstance(v, float) else v) for k, v in p.items()})
    print("memory:", out["memory"])


if __name__ == "__main__":
    main()
