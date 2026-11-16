"""Does set-style correlation matter? Offline analysis of the per-world logs written by headless_logworlds.py.

For every logged decision with a scored feedback (opponent action b mapped to a column), and every world k that
contains b: q_sigma(b | M_k) for each style (behavior.style_dists on the WORLD matrix, reference mixture = the played
mixture pfull), and the world-averaged prediction the current code uses (style_dists on Abar). Reports:
  * world re-weighting  w_k ∝ Σ_σ P_t(σ) q_σ(b | M_k)   over the worlds containing b: TV from uniform, ESS, max ratio
  * the planner's response: agent-form LP x on uniform vs re-weighted worlds (L1 distance, argmax change rate)
  * predictive log loss of the NEXT scored action: averaged-matrix q-hat (current code) vs per-world mixture
    Σ_k u_k Σ_σ P(σ) q_σ(.|M_k) (joint, uniform worlds) vs marginal product (Σ_k u_k q_σ averaged, then styles)
  * accumulated per-moveset log-likelihood over a game (option (i) importance weights): spread of the accumulated
    weights across the active's sampled movesets.
Usage: engine/.venv/bin/python analyze_worlds.py <work dir with worlds.jsonl and replays_headless/>
"""
import glob
import json
import os
import sys
from collections import defaultdict

import numpy as np

ENGINE = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "engine"))
sys.path[:0] = [ENGINE, os.path.join(ENGINE, "showdown")]
import behavior                                     # noqa: E402
import hidden_agents as ha                          # noqa: E402

STYLES = behavior.STYLES


def load(work):
    worlds = defaultdict(list)
    for l in open(os.path.join(work, "worlds.jsonl")):
        r = json.loads(l)
        worlds[r["game"]].append(r)
    games = {}
    for f in sorted(glob.glob(os.path.join(work, "replays_headless", "*.A.decisions.jsonl"))):
        g = os.path.basename(f).split(".")[0]
        games[g] = [json.loads(l) for l in open(f)]
    return worlds, games


def join(worlds, recs):
    """(decision record, world record) pairs for the multi-option decisions, matched in order and checked on labels."""
    dec = [r for r in recs if r.get("worlds", 0) > 1 and "abar" in r]
    out, i = [], 0
    for w in worlds:                                 # greedy in-order match on labels (the other seat's calls, when
        if i < len(dec) and dec[i]["labels"] == w["labels"]:   # it is also ours, interleave in the same side file)
            out.append((dec[i], w)); i += 1
    assert i == len(dec), (i, len(dec), len(worlds))
    return out


def feedback_for(recs):
    """turn -> feedback dict (from the record that carries it: the next decision, or the final record)."""
    fb = {}
    for r in recs:
        f = r.get("feedback")
        if f:
            fb[f["for_turn"]] = f
    return fb


def main(work):
    worlds, games = load(work)
    post = behavior.Posterior()                      # fresh posterior per stream of games, as in the real runs
    post_j = behavior.Posterior()                    # option (iii): posterior fed the per-world marginal likelihood
    rows = []
    acc_by_game = []
    for g in sorted(games):
        recs = games[g]
        pairs = join(worlds[g], recs)
        fbs = feedback_for(recs)
        acc = defaultdict(float)                     # option (i): accumulated log-lik per active moveset key
        nseen = defaultdict(int)
        for d, w in pairs:
            Ms = [np.array(M) for M in w["Ms"]]
            cols, labels = w["cols"], w["labels"]
            K = len(Ms)
            Abar = np.array(d["abar"])
            p = np.array(d["pfull"])
            P = post.probs()
            Pj = post_j.probs()
            fb = fbs.get(d["turn"])
            if not fb or fb.get("skipped") or fb.get("b") is None:
                continue
            b = fb["b"]
            cands = [b, b[:-5]] if b.endswith("-tera") else [b]
            jk = {}
            for k in range(K):
                for c in cands:
                    if c in cols[k]:
                        jk[k] = cols[k].index(c); break
            if not jk:
                continue
            # per-world style predictions
            qsk = {k: behavior.style_dists(Ms[k], p) for k in range(K)}
            lik_k = np.array([P @ qsk[k][:, jk[k]] for k in jk])       # Σ_σ P(σ) q_σ(b|M_k)
            ks = list(jk)
            wk = lik_k / lik_k.sum()
            u = np.full(len(ks), 1 / len(ks))
            tv = 0.5 * np.abs(wk - u).sum()
            ess = 1 / (wk ** 2).sum() / len(ks)
            # planner response (only meaningful if >1 world contains b)
            x_u = ha.agent_form_lp(Ms, np.full(K, 1 / K))[0]
            bw = np.zeros(K); bw[ks] = wk
            x_w = ha.agent_form_lp(Ms, bw)[0]
            l1 = np.abs(x_u - x_w).sum()
            # predictive log loss of this action (prequential: P before update)
            qs_bar = behavior.style_dists(Abar, p)
            j_bar = labels.index(cands[-1]) if cands[-1] in labels else labels.index(b)
            p_cur = float(P @ qs_bar[:, j_bar])
            # joint per-world mixture with uniform world weights over the worlds containing b: P(b) = Σ_k (1/K) Σ_σ P(σ) q_σ(b|M_k)
            p_joint = float(sum(P @ qsk[k][:, jk[k]] for k in ks) / K)
            qbar = P @ qs_bar                        # independent marginals: set membership x averaged-matrix style
            p_indep = 0.0
            for k in ks:
                idx = [labels.index(c) for c in cols[k]]
                p_indep += qbar[j_bar] / qbar[idx].sum() / K
            # per-style per-world log-lik (for a per-world-updated posterior; marginal over worlds)
            lik_style_joint = np.array([sum(qsk[k][s, jk[k]] for k in ks) / K for s in range(len(STYLES))])
            lik_style_bar = qs_bar[:, j_bar]
            p_joint_pj = float(Pj @ lik_style_joint)  # option (iii) end to end: per-world lik, per-world-updated posterior
            post_j.logp = post_j.logp + np.log(lik_style_joint); post_j.logp -= post_j.logp.max()
            # accumulated moveset weights (option i): key = sorted active moves in world k
            keyed = defaultdict(list)                # option (i): per-turn log-lik by active moveset key
            for k in ks:
                key = tuple(sorted(c for c in cols[k] if not c.endswith("-tera") and c in cols[k][:8]))
                keyed[key].append(float(np.log(P @ qsk[k][:, jk[k]])))
            if len(keyed) >= 2:
                for a in keyed:
                    for bk in keyed:
                        if a < bk:
                            acc[(a, bk)] += np.mean(keyed[a]) - np.mean(keyed[bk]); nseen[(a, bk)] += 1
            n_keys = len(keyed)
            rows.append(dict(game=g, turn=d["turn"], n_contain=len(ks), K=K, tv=tv, ess=ess, maxratio=float(wk.max() / wk.min()),
                             l1=l1, argmax_change=int(x_u.argmax() != x_w.argmax()), p_cur=p_cur, p_joint=p_joint, p_indep=p_indep, p_joint_pj=p_joint_pj,
                             lik_style_bar=lik_style_bar.tolist(), lik_style_joint=lik_style_joint.tolist(), how=fb["how"], n_keys=n_keys,
                             n_union=len(labels), n_world=len(cols[0])))
            post.update(qs_bar, j_bar)               # the real code's update (averaged matrix)
        acc_by_game.append({k: (v, nseen[k]) for k, v in acc.items()})
    R = rows
    n = len(R)
    multi = [r for r in R if r["n_contain"] > 1]
    print(f"scored turns {n}; turns where >1 world contains b: {len(multi)} ({len(multi)/n:.2f})")
    tv = np.array([r["tv"] for r in multi]); ess = np.array([r["ess"] for r in multi]); mr = np.array([r["maxratio"] for r in multi])
    print(f"world re-weighting (turns with >1 containing world): TV from uniform mean {tv.mean():.3f} median {np.median(tv):.3f} "
          f"p90 {np.percentile(tv,90):.3f}; ESS/n mean {ess.mean():.3f}; max/min lik ratio median {np.median(mr):.2f} p90 {np.percentile(mr,90):.2f}")
    l1 = np.array([r["l1"] for r in multi]); ch = np.array([r["argmax_change"] for r in multi])
    print(f"planner: L1(x_u, x_w) mean {l1.mean():.3f} median {np.median(l1):.3f} p90 {np.percentile(l1,90):.3f}; argmax changes {ch.mean():.3f}")
    pc = np.array([r["p_cur"] for r in R]); pj = np.array([r["p_joint"] for r in R]); pi = np.array([r["p_indep"] for r in R])
    se = lambda a: np.std(a) / np.sqrt(len(a))
    print(f"predictive log loss of b: averaged-matrix q-hat (current) {(-np.log(pc)).mean():.4f} | membership x averaged style "
          f"(independent marginals) {(-np.log(pi)).mean():.4f} | per-world joint {(-np.log(pj)).mean():.4f}")
    pq = np.array([r["p_joint_pj"] for r in R])
    print(f"  option (iii) end to end (per-world lik + per-world-updated posterior): {(-np.log(pq)).mean():.4f}; "
          f"joint(bar-posterior) - option(iii) {(-np.log(pj)+np.log(pq)).mean():+.4f} +- {se(-np.log(pj)+np.log(pq)):.4f}")
    print(f"  paired: current - indep {(-np.log(pc)+np.log(pi)).mean():+.4f} +- {se(-np.log(pc)+np.log(pi)):.4f};  "
          f"indep - joint {(-np.log(pi)+np.log(pj)).mean():+.4f} +- {se(-np.log(pi)+np.log(pj)):.4f}")
    # which style wins under each prediction scheme
    for name in ("lik_style_bar", "lik_style_joint"):
        L = np.array([r[name] for r in R])
        print(f"  per-style log loss [{name}]: " + ", ".join(f"{s} {(-np.log(L[:, i])).mean():.3f}" for i, s in enumerate(STYLES)))
    # accumulated per-moveset weights
    nk = np.array([r["n_keys"] for r in R])
    print(f"dilution: union labels mean {np.mean([r['n_union'] for r in R]):.1f} vs per-world columns {np.mean([r['n_world'] for r in R]):.1f}; "
          f"turns whose containing worlds carry >=2 distinct active movesets: {np.mean(nk >= 2):.3f}")
    acc_lr = [(abs(v), c) for acc in acc_by_game for (v, c) in acc.values() if c >= 2]
    if acc_lr:
        a = np.array([v for v, c in acc_lr]); c = np.array([c for v, c in acc_lr])
        print(f"option (i): accumulated |log-lik ratio| between two movesets of the same active over a game (pairs co-sampled on >=2 turns): "
              f"n {len(a)}, co-sampled turns median {np.median(c):.0f}, |LR| median {np.median(a):.2f} p90 {np.percentile(a,90):.2f} nats "
              f"(>= ln 3 = 1.1 in {np.mean(a >= np.log(3)):.2f} of pairs)")
    json.dump(dict(rows=R), open(os.path.join(work, "analyze_worlds.json"), "w"))


if __name__ == "__main__":
    main(sys.argv[1])
