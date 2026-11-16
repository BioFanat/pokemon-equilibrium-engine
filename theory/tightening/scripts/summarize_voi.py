"""Summaries for voi.md from the jsonl outputs of tera_subgame.py, tera_two_sided.py, eta_hat_engine.py.

Usage: summarize_voi.py --subgame batch.jsonl --two two_sided.jsonl --eta eta_hat.jsonl
"""
from __future__ import annotations

import argparse
import json

import numpy as np


def q(v, ps=(0, .25, .5, .75, .9, 1)):
    return " ".join(f"{np.quantile(v, p):.4g}" for p in ps)


def subgame(path):
    R = [json.loads(l) for l in open(path)]
    E = [r for r in R if "error" in r]
    R = [r for r in R if "error" not in r]
    print(f"# exact 1v1 tera subgames: {len(R)} ok, {len(E)} errors {[e['error'][:60] for e in E][:3]}")
    print("k dist:", dict(zip(*np.unique([r["k"] for r in R], return_counts=True))), " leaf retries total:", sum(r.get("leaf_retries", 0) for r in R))
    print("checks: C1 holds", sum(r["c1_holds"] for r in R), "/", len(R), "| sandwich holds", sum(r["sandwich_holds"] for r in R),
          "| VOI>=0", sum(r["voi_total"] >= -1e-6 for r in R), "| expl>=0", sum(r["expl_a3"] >= -1e-6 for r in R),
          "| expl <= E_BR[eta]+2*Delta*E[TV]", sum(r["expl_a3"] <= r["e_eta_play"] + 2 * r["Delta"] * r["e_tv1"] + 1e-6 for r in R))
    D = np.array([r["Delta"] for r in R])
    print("Delta (payoff range, engine units) quantiles [0 25 50 75 90 100]:", q(D))
    print(f"{'quantity':<14}{'median':>9}{'p75':>9}{'p90':>9}{'max':>10}  | /Delta median  p90    max  | frac>0.1%Delta frac>1%Delta")
    for key in ["voi_total", "eta_NL0", "eta_b0_max", "eta_b0_mean", "e_eta_play", "sandwich_gap", "expl_a3"]:
        v = np.array([r[key] for r in R]); rel = v / D
        print(f"{key:<14}{np.median(v):9.2f}{np.quantile(v, .75):9.2f}{np.quantile(v, .9):9.2f}{v.max():10.2f}  |  {np.median(rel):.4f}  {np.quantile(rel, .9):.4f} {rel.max():.4f} |  {np.mean(rel > 1e-3):.2f}  {np.mean(rel > 1e-2):.2f}")
    print("frac of pre-tera children with eta(b0)>0: median %.2f max %.2f" % (np.median([r["frac_eta_pos"] for r in R]), max(r["frac_eta_pos"] for r in R)))
    print("matchups with VOI_total > 1e-3 Delta:", sum(r["voi_total"] > 1e-3 * r["Delta"] for r in R), "; with expl_a3 > 1e-3 Delta:", sum(r["expl_a3"] > 1e-3 * r["Delta"] for r in R))
    print("E_TV1 > 0 (type-dependent best response at stage 0):", sum(r["e_tv1"] > 1e-9 for r in R))
    ratios = [r["expl_a3"] / r["bound_C1"] for r in R if r["bound_C1"] > 1e-9 and r["expl_a3"] > 1e-9]
    print("expl / (eta_NL0 + 2 Delta E[TV]) over matchups with expl>0: ", [round(x, 3) for x in sorted(ratios)])
    ch = [(r["tera0_eq"][th], r["tera0_full"][th]) for r in R for th in r["teras"] if r["tera0_eq"][th] is not None and r["tera0_full"][th] is not None]
    d = np.array([a - b for a, b in ch])
    print(f"equilibrium P(tera at stage 0 | theta): hidden - known: n={len(d)} mean {d.mean():.3f}; |diff|>0.1 in {np.mean(np.abs(d) > .1):.2f}; hidden<known-0.1: {np.mean(d < -.1):.2f}; hidden>known+0.1: {np.mean(d > .1):.2f}")
    print("stage-0 argmax differs eq vs A3:", sum(max(r["x0_eq"], key=r["x0_eq"].get) != max(r["x0_a3"], key=r["x0_a3"].get) for r in R), "/", len(R))
    print("time per matchup (s): median %.0f max %.0f" % (np.median([r["total_s"] for r in R]), max(r["total_s"] for r in R)))
    top = sorted(R, key=lambda r: -r["expl_a3"] / r["Delta"])[:5]
    print("top-5 by expl/Delta:")
    for r in top:
        print(f"  seed {r['seed']} {r['opp']}({r['k']}) vs {r['own']}: Delta {r['Delta']:.0f} VOI {r['voi_total']:.2f} etaNL0 {r['eta_NL0']:.2f} expl {r['expl_a3']:.2f} E[TV] {r['e_tv1']:.3f} sandwich {r['sandwich_gap']:.2f} x0_eq {r['x0_eq']} x0_a3 {r['x0_a3']}")


def two(path):
    R = [json.loads(l) for l in open(path)]
    E = [r for r in R if "error" in r]
    R = [r for r in R if "error" not in r]
    print(f"\n# two-sided 1v1 tera subgames: {len(R)} ok, {len(E)} errors; leaf retries {sum(r.get('leaf_retries', 0) for r in R)}")
    for r in R:
        print(f"  seed {r['seed']} {r['own']}({len(r['phis'])}) vs {r['opp']}({len(r['thetas'])}) Delta {r['Delta']:.0f} V_real {r['V_real']:.2f} "
              f"conceal {r['conceal_value']:.3f} ({r['conceal_value'] / r['Delta']:.2%}) >=0:{r['conceal_ge0']} voi_opp(2-sided) {r['voi_opp_two_sided']:.3f} voi_opp(1-sided avg) {r['voi_opp_one_sided_avg']:.3f} {r['total_s']}s")
    if R:
        c = np.array([r["conceal_value"] / r["Delta"] for r in R])
        print(f"conceal/Delta: median {np.median(c):.4f} max {c.max():.4f}; all >= 0: {all(r['conceal_ge0'] for r in R)}")


def eta(path):
    G = [json.loads(l) for l in open(path)]
    D = [r for g in G for r in g["decisions"]]
    es = np.array([r["eta_hat_set"] / r["payoff_range"] for r in D if r["payoff_range"] > 0])
    et = np.array([r["eta_hat_tera"] / r["payoff_range_tera"] for r in D if "eta_hat_tera" in r and r["payoff_range_tera"] > 0])
    print(f"\n# engine eta-hat: games {len(G)} decisions {len(D)} with tera worlds {len(et)}; stable-matrix retries total {max(r.get('retries', 0) for r in D)}")
    print("eta_hat_set/range : mean %.4f median %.4f p90 %.4f max %.4f  frac>0.05 %.3f  frac==0 %.3f" % (es.mean(), np.median(es), np.quantile(es, .9), es.max(), (es > .05).mean(), (es < 1e-6).mean()))
    print("eta_hat_tera/range: mean %.4f median %.4f p90 %.4f max %.4f  frac>0.01 %.3f  frac==0 %.3f" % (et.mean(), np.median(et), np.quantile(et, .9), et.max(), (et > .01).mean(), (et < 1e-6).mean()))
    pg = [sum(r["eta_hat_set"] / r["payoff_range"] for r in g["decisions"] if r["payoff_range"] > 0 and r["agent"] == a) for g in G for a in (1, 2)]
    print("sum_t eta_hat_set/range per game per side: mean %.3f median %.3f max %.3f (turns median %d)" % (np.mean(pg), np.median(pg), np.max(pg), np.median([g["turns"] for g in G])))
    pr = np.array([r["payoff_range"] for r in D]); print("payoff_range (engine units): median %.0f p99 %.0f max %.0f" % (np.median(pr), np.quantile(pr, .99), pr.max()))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--subgame"); ap.add_argument("--two"); ap.add_argument("--eta")
    a = ap.parse_args()
    if a.subgame: subgame(a.subgame)
    if a.two: two(a.two)
    if a.eta: eta(a.eta)
