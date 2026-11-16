"""B2: E3a (hedge) / E3f (hedge_fs) vs switch:stock@8:exploit_top. Output out_B2_switch.json"""
import json, math
import numpy as np
from oi_common import *
from common import EXPERTS, streams, records
K = 4
T1 = 8  # seed positions 0..7 of each stream play stock
res = {}

def stream_series(tag, k, order):
    """Per non-skipped feedback update: reward vector R, weights played W, game index (0-based position in order), seed."""
    R, W, gi, seeds, gpos = [], [], [], [], []
    for n, (seed, g, path) in enumerate(order):
        pend = None
        for rec in records(path):
            fb = rec.get("feedback")
            if fb and pend is not None and fb.get("for_turn") == pend[0] and not fb.get("skipped") and "r" in fb:
                R.append(np.array(fb["r"], float)); W.append(pend[1]); gi.append(n); seeds.append(seed)
            if fb is not None or rec.get("final"):
                pend = None
            if isinstance(rec.get("weights"), dict):
                pend = (rec["turn"], np.array([rec["weights"][e] for e in EXPERTS], float))
    return np.array(R), np.array(W), np.array(gi), np.array(seeds)

for tag in ("E3a", "E3f"):
    S = summary(tag)
    G = game_jsons(tag)
    s0 = GRID[tag][2]
    seeds_all = list(range(1, 161))
    rows = []
    for k, order in streams(tag):
        sd = sorted({s for s, _, _ in order})
        pre_seeds = set(sd[:T1])
        R, W, gi, sds = stream_series(tag, k, order)
        pre = np.array([s in pre_seeds for s in sds])
        t0 = int(pre.sum())
        cum_pre, cum_post = R[:t0].sum(0), R[t0:].sum(0)
        a, b = int(cum_pre.argmax()), int(cum_post.argmax())
        mu_pre, mu_post = R[:t0].mean(0), R[t0:].mean(0)
        am = W.argmax(1)
        # flips: first update index >= t0 (relative) where argmax W == b ; sustained = stays b for rest (or >= 20 consecutive)
        first = next((t - t0 for t in range(t0, len(W)) if am[t] == b), None)
        sust = next((t - t0 for t in range(t0, len(W)) if (am[t:t + 20] == b).all() and len(am[t:t + 20]) == 20), None)
        # weights of b
        wb = lambda lo, hi: float(W[max(lo, 0):hi, b].mean())
        wa = lambda lo, hi: float(W[max(lo, 0):hi, a].mean())
        # analog of T5(a) (cumulative-reward crossing): tau* = t0 (mu_a^pre - mu_b^pre)/(mu_b^post - mu_a^post)
        dpost = mu_post[b] - mu_post[a]
        tau_a = float(t0 * (mu_pre[a] - mu_pre[b]) / dpost) if a != b and dpost > 0 else None
        # realised crossing of cumulative reward (exact for anytime Hedge where W argmax = cum argmax)
        cum = R.cumsum(0)
        cross = next((t - t0 for t in range(t0, len(W)) if cum[t, b] >= cum[t, a]), None) if a != b else 0
        # fixed-share analog (b): ln(k/alpha)/(eta * dpost) for hedge_fs eta=0.2 alpha=0.002
        tau_fs = float(math.log(K / 0.002) / (0.2 * dpost)) if dpost > 0 else None
        rows.append(dict(stream=k, t0=t0, T=len(W), pre_best=EXPERTS[a], post_best=EXPERTS[b],
                         mu_pre=mu_pre.round(4).tolist(), mu_post=mu_post.round(4).tolist(),
                         w_postbest_pre_last50=wb(t0 - 50, t0), w_postbest_post_first50=wb(t0, t0 + 50), w_postbest_post_last50=wb(len(W) - 50, len(W)),
                         w_prebest_pre_last50=wa(t0 - 50, t0), w_prebest_post_first50=wa(t0, t0 + 50), w_prebest_post_last50=wa(len(W) - 50, len(W)),
                         w_at_switch=W[t0 - 1].round(3).tolist(), w_end=W[-1].round(3).tolist(),
                         argmax_at_switch=EXPERTS[int(am[t0 - 1])], flip_first=first, flip_sustained20=sust,
                         pred_tau_T5a_analog=tau_a, realised_cum_crossing=cross, pred_tau_fs_analog=tau_fs,
                         n_post_updates=len(W) - t0))
    # pre/post scores
    gsc = []
    for (seed, game), d in sorted(G.items()):
        if d.get("error"): continue
        pos = (seed - s0) // NSTREAMS
        gsc.append(dict(seed=seed, game=game, stream=stream_of(seed, s0), pos=pos, post=pos >= T1, score=score(d), turns=d["turns"]))
    def sc(sel):
        x = np.array([g["score"] for g in sel]); st = np.array([g["stream"] for g in sel])
        return dict(n=len(x), mean=float(x.mean()), normal=mean_ci_normal(x)[2:4], cluster=cluster_ci(x, st)[2:4])
    split = dict(pre=sc([g for g in gsc if not g["post"]]), post=sc([g for g in gsc if g["post"]]),
                 post_first4_games=sc([g for g in gsc if g["post"] and g["pos"] < T1 + 2]),
                 post_later=sc([g for g in gsc if g["post"] and g["pos"] >= T1 + 2]))
    res[tag] = dict(streams=rows, split=split, n_flip_needed=sum(1 for r in rows if r["pre_best"] != r["post_best"]))
    print(tag, split["pre"]["mean"], split["post"]["mean"], [(r["pre_best"], r["post_best"], r["flip_first"], r["flip_sustained20"], r["realised_cum_crossing"], None if r["pred_tau_T5a_analog"] is None else round(r["pred_tau_T5a_analog"]), None if r["pred_tau_fs_analog"] is None else round(r["pred_tau_fs_analog"])) for r in rows])
# paired hedge vs hedge_fs by seed
def pairs(tag):
    G = game_jsons(tag); out = {}
    for s in range(1, 161):
        if (s, 1) in G and (s, 2) in G and not G[(s, 1)].get("error") and not G[(s, 2)].get("error"):
            out[s] = (score(G[(s, 1)]) + score(G[(s, 2)])) / 2
    return out
pa, pf = pairs("E3a"), pairs("E3f")
common_seeds = sorted(set(pa) & set(pf))
def diff(sel):
    d = np.array([pf[s] - pa[s] for s in sel]); st = np.array([stream_of(s, 1) for s in sel])
    return dict(n=len(d), mean_fs_minus_hedge=float(d.mean()), normal=mean_ci_normal(d)[2:4], cluster=cluster_ci(d, st)[2:4])
res["fs_minus_hedge"] = dict(all=diff(common_seeds), pre=diff([s for s in common_seeds if (s - 1) // 10 < T1]), post=diff([s for s in common_seeds if (s - 1) // 10 >= T1]))
print(res["fs_minus_hedge"])
dump("out_B2_switch.json", res)
