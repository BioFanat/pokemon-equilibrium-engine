"""Posterior oscillation on the BH logs (hedge_bayes vs stock, 10 carried streams; streams 0 and 6 excluded for the
update-count mismatch, see synthesis.md 3.2).

Per clean stream, feedback turns only:
  G_t = L_t(s2) - L_t(s1): cumulative log-loss gap between the two final leaders s1 (best) and s2 (runner-up); its
  increments g_t = l_{s2,t} - l_{s1,t}: mean mu, sd sigma, lag-1 autocorrelation; sign changes of G_t (leader
  switches among the top two) and among all five styles (MAP switches); number of turns the MAP != s1 ("wrong side");
  the Brownian-with-drift prediction sigma^2 / (2 mu^2) for the expected time on the wrong side (and the discrete
  Hoeffding bound sum_t exp(-2 mu^2 t / B^2)).
  Predictors: mixture q-hat (shipped), MAP = FTL (predict with the current leader's q_s, pre-update posterior),
  best fixed style in hindsight; log loss and argmax accuracy of each; FTL regret vs the best style.
Pre-update posterior reconstructed from the post-update one as in bh_regret.py (P_pre ∝ P_post / p_style).
"""
import glob
import json
import math
import os
import re

import numpy as np

RES = "/Users/biofanat/Documents/classes/f26/cs535/final/engine/showdown/results"
STYLES = ["eq", "br_unif", "br_us", "safe", "uniform"]
GAME_RE = re.compile(r"(?P<tag>.+)-s(?P<seed>\d+)-g(?P<game>[12])\.(?P<role>[AB])\.decisions\.jsonl$")
HERE = os.path.dirname(os.path.abspath(__file__))


def streams(tag, role="A", n_streams=10):
    files = {}
    for p in glob.glob(os.path.join(RES, tag, "replays_headless", f"{tag}-s*-g*.{role}.decisions.jsonl")):
        m = GAME_RE.search(os.path.basename(p))
        if m:
            files[(int(m["seed"]), int(m["game"]))] = p
    seeds = sorted({s for s, _ in files})
    for k in range(n_streams):
        yield k, [files[(s, g)] for s in seeds[k::n_streams] for g in (1, 2) if (s, g) in files]


rows = []
for k, paths in streams("BH"):
    if k in (0, 6):
        continue
    L, PQ, PRE, HIT = [], [], [], []
    for p in paths:
        for line in open(p):
            rec = json.loads(line)
            fb = rec.get("feedback")
            if not fb or fb.get("skipped") or "bayes" not in fb:
                continue
            b = fb["bayes"]
            ps = np.array([max(b["p_style"][s], 1e-9) for s in STYLES])
            post = np.array([b["posterior"][s] for s in STYLES])
            pre = post / ps
            pre = pre / pre.sum() if pre.sum() > 0 else np.full(5, 0.2)
            L.append(-np.log(ps)); PQ.append(-math.log(max(b["p_qhat"], 1e-9))); PRE.append(pre)
            HIT.append([b["hit_qhat"]] + [b["hit_style"][s] for s in STYLES])
    L = np.array(L); PQ = np.array(PQ); PRE = np.array(PRE); HIT = np.array(HIT, float)
    T = len(L)
    cum = np.cumsum(L, 0)
    order = np.argsort(cum[-1])
    s1, s2 = order[0], order[1]
    g = L[:, s2] - L[:, s1]                       # increments of the gap runner-up minus best (positive favours s1)
    G = np.cumsum(g)
    mu, sd = g.mean(), g.std(ddof=1)
    ac1 = np.corrcoef(g[:-1], g[1:])[0, 1]
    # leader by cumulative loss (= MAP of the exact posterior, lam = 1)
    leader = cum.argmin(1)
    map_switch = int((np.diff(leader) != 0).sum())
    top2_switch = int((np.diff(np.sign(G[G != 0])) != 0).sum())
    wrong = int((leader != s1).sum())
    wrong_top2 = int((G < 0).sum())
    B = L.max()
    pred_bm = sd ** 2 / (2 * mu ** 2) if mu > 0 else float("inf")
    pred_hoeff = float(sum(math.exp(-2 * mu ** 2 * t / B ** 2) for t in range(1, T + 1)))
    # FTL (MAP of the pre-update posterior) predictor: at turn t predict with the leader of cum[t-1]
    lead_pre = np.r_[0, leader[:-1]]              # turn 0: style 0 (eq); ties irrelevant
    ftl_loss = L[np.arange(T), lead_pre]
    ftl_hit = HIT[np.arange(T), 1 + lead_pre]
    # also the MAP of the logged pre-update posterior (same thing up to rounding)
    map_pre = PRE.argmax(1)
    map_loss = L[np.arange(T), map_pre]
    mix_reg = PQ.sum() - cum[-1].min()
    ftl_reg = ftl_loss.sum() - cum[-1].min()
    # switch-count bound for FTL on bounded losses: regret <= (switches + 1) * B  (crude), and the data value
    rows.append(dict(stream=k, T=T, s1=STYLES[s1], s2=STYLES[s2], gap_final=float(G[-1]), mu=float(mu), sd=float(sd), ac1=float(ac1),
                     map_switches=map_switch, top2_sign_changes=top2_switch, turns_map_wrong=wrong, turns_G_neg=wrong_top2,
                     pred_bm=float(pred_bm), pred_hoeffding=pred_hoeff, B=float(B),
                     ll_mix=float(PQ.mean()), ll_ftl=float(ftl_loss.mean()), ll_map_logged=float(map_loss.mean()), ll_best=float(cum[-1].min() / T),
                     ll_s2=float(cum[-1][s2] / T), acc_mix=float(HIT[:, 0].mean()), acc_ftl=float(ftl_hit.mean()), acc_best=float(HIT[:, 1 + s1].mean()),
                     reg_mix=float(mix_reg), reg_ftl=float(ftl_reg), frac_g_zero=float((np.abs(g) < 1e-6).mean()),
                     avg_post_s1=float(PRE[:, s1].mean()), avg_post_s2=float(PRE[:, s2].mean())))

print(f"{'str':>3} {'T':>5} {'s1/s2':>16} {'gap':>6} {'mu':>7} {'sd':>6} {'ac1':>6} {'MAPsw':>5} {'sgn':>4} {'wrong':>5} {'Gneg':>5} {'BM':>7} {'Hoeff':>7} | {'ll mix':>6} {'ll FTL':>6} {'ll best':>7} {'ll s2':>6} | {'acc mix':>7} {'acc FTL':>7} | {'reg mix':>7} {'reg FTL':>7}")
for r in rows:
    print(f"{r['stream']:>3} {r['T']:>5} {r['s1']+'/'+r['s2']:>16} {r['gap_final']:>6.1f} {r['mu']:>7.4f} {r['sd']:>6.3f} {r['ac1']:>6.3f} {r['map_switches']:>5} {r['top2_sign_changes']:>4} {r['turns_map_wrong']:>5} {r['turns_G_neg']:>5} {r['pred_bm']:>7.0f} {r['pred_hoeffding']:>7.0f} | {r['ll_mix']:>6.4f} {r['ll_ftl']:>6.4f} {r['ll_best']:>7.4f} {r['ll_s2']:>6.4f} | {r['acc_mix']:>7.4f} {r['acc_ftl']:>7.4f} | {r['reg_mix']:>7.3f} {r['reg_ftl']:>7.3f}")
a = lambda key: np.array([r[key] for r in rows])
print(f"\nmeans over {len(rows)} clean streams: turns MAP wrong {a('turns_map_wrong').mean():.0f} (G<0: {a('turns_G_neg').mean():.0f}), BM prediction sigma^2/(2 mu^2) {a('pred_bm').mean():.0f}, Hoeffding {a('pred_hoeffding').mean():.0f}, T {a('T').mean():.0f}")
print(f"ratio measured/BM per stream: {np.round(a('turns_G_neg') / a('pred_bm'), 2)}")
print(f"fraction of turns with identical loss for s1, s2 (|g|<1e-6, e.g. forced turns): {a('frac_g_zero').mean():.3f}")
print(f"log loss: mixture {a('ll_mix').mean():.4f}, FTL/MAP {a('ll_ftl').mean():.4f}, best style {a('ll_best').mean():.4f}, runner-up {a('ll_s2').mean():.4f}")
print(f"regret vs best style, total nats: mixture {a('reg_mix').mean():.3f} (ln5 {math.log(5):.3f}), FTL {a('reg_ftl').mean():.3f} (min {a('reg_ftl').min():.2f}, max {a('reg_ftl').max():.2f}); MAP switches per stream mean {a('map_switches').mean():.1f}")
print(f"argmax accuracy: mixture {a('acc_mix').mean():.4f}, FTL {a('acc_ftl').mean():.4f}, best style {a('acc_best').mean():.4f}")
json.dump(rows, open(os.path.join(HERE, "out_osc_bh.json"), "w"), indent=1)
