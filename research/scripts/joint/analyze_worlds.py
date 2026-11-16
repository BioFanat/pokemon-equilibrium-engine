"""Does set-style correlation matter? Offline analysis of the per-world logs written by headless_logworlds.py.

For every logged multi-option decision whose feedback is scored (opponent action b mapped to a label exactly as the
deployed map_column does), and every world k that contains that label: q_sigma(b | M_k) for each style
(behavior.style_dists on the WORLD matrix, reference mixture = the played mixture pfull). Reports:
  * world re-weighting  w_k ∝ Σ_σ P_t(σ) q_σ(b | M_k)   over the worlds containing b: TV from uniform, ESS, max ratio
  * the planner's response: agent-form LP x on uniform vs re-weighted worlds (L1 distance, argmax change rate)
  * predictive log loss of the NEXT scored action, every predictor scored on the same label:
      current       the deployed q-hat (logged bayes.qhat: P(σ)-mixture of q_σ on Abar over the label union)
      membership    the same q-hat restricted to each world's columns, renormalised, averaged over worlds
      perworld      per-world style likelihood: Σ_k (1/K) Σ_σ P(σ) q_σ(b|M_k), P = the logged deployed posterior
      perworld_iii  option (iii): same, with a posterior updated by the per-world marginal likelihood
      truejoint     joint over (active set, style) carried across turns within a game: Σ_k (1/K) Σ_σ P(σ|s_k) q_σ(b|M_k)
                    (see TrueJoint), whose only difference from perworld_iii is the set-conditional style belief
  * accumulated per-moveset log-likelihood over a game (option (i) importance weights): spread of the accumulated
    weights across the active's sampled movesets.
Joins: decision <-> worlds.jsonl line in order, checked on labels, row count and the logged Abar reproduced from the
logged Ms; decision i <-> feedback by RECORD ORDER (the feedback in record i+1 belongs to decision i; turn numbers
repeat, so they cannot key it). The sampled set of the active in world k is read from the decision's
belief.worlds[k].opp[0] (OURBOT_LOG_WORLDS=1), which is in the same order as the logged Ms.
Usage: ~/sandbox/bot-venv/bin/python analyze_worlds.py <work dir with worlds.jsonl and replays_headless/> [out.json]
       ~/sandbox/bot-venv/bin/python analyze_worlds.py --pool <rows.json> <rows.json> ...   (pooled paired summaries)
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


def avg_matrix(Ms, cols, w):                         # verbatim copy of run_ourbot.avg_matrix (importing run_ourbot
    labels = list(dict.fromkeys(c for cs in cols for c in cs))   # pulls in foul-play)
    Abar = np.zeros((Ms[0].shape[0], len(labels)))
    for j, lab in enumerate(labels):
        ks = [k for k in range(len(Ms)) if lab in cols[k]]
        wk = np.array([w[k] for k in ks])
        Abar[:, j] = sum(wi * Ms[k][:, cols[k].index(lab)] for wi, k in zip(wk / wk.sum(), ks))
    return Abar, labels


def map_column(b, cols, w):                          # verbatim copy of run_ourbot.map_column
    cands = [b, b[:-len("-tera")]] if b.endswith("-tera") else [b]
    for k in sorted(range(len(cols)), key=lambda k: -w[k]):
        for c in cands:
            if c in cols[k]:
                return k, cols[k].index(c)
    return None


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
    """(record index, decision record, world record) for the multi-option decisions, matched in order on labels, row
    count and the logged Abar (the other seat's calls, when it also averages, interleave in the same side file)."""
    out, it = [], iter(worlds)
    for i, d in enumerate(recs):
        if "abar" not in d or len(d.get("legal", [])) <= 1:
            continue
        for w in it:
            if w["labels"] == d["labels"] and len(w["Ms"][0]) == len(d["legal"]):
                Ab = avg_matrix([np.array(M) for M in w["Ms"]], w["cols"], np.array(w["w"]))[0]
                if np.abs(Ab - np.array(d["abar"])).max() < 0.011:
                    out.append((i, d, w)); break
        else:
            raise AssertionError(("unmatched decision", d.get("turn"), i))
    return out


def set_key(mon):
    """The sampled set of one opponent mon in one world: (species, item, ability, tera type, moves)."""
    return (mon["name"], mon["item"], mon["ability"], mon["tera_type"], tuple(sorted(mon["moves"])))


class TrueJoint:
    """Joint belief over (active set, style) within one game, on top of the option (iii) style posterior.

    Under the factorised model each scored turn t multiplies every style by the per-world marginal likelihood
    m_t(σ) = mean_{k ∋ b} q_σ(b | M_k). Here, for the turns on which mon X was active, set key s of X carries its own
    factor instead: mean over the worlds with X's set = s (and containing b) of q_σ(b | M_k); keys of X not sampled on
    that turn (or sampled only in worlds without b) get m_t(σ) imputed, so all keys of X have one factor per turn.
        P(σ | X = s) ∝ P_start(σ) · exp(L_all(σ) - L_X(σ) + logw[X, s](σ))
    with L_all = Σ_t log m_t over the game, L_X = the same over X's turns, P_start = the option (iii) posterior at game
    start. A key never seen (logw = L_X) gives exactly the option (iii) posterior. Updated after scoring."""

    def __init__(self, p_start):
        self.logp0 = np.log(p_start)
        self.L_all = np.zeros(len(STYLES))
        self.L_mon = defaultdict(lambda: np.zeros(len(STYLES)))
        self.logw = defaultdict(dict)                # mon -> {set key -> log weight per style}

    def probs(self, key):
        mon = key[0]
        lw = self.logw[mon].get(key, self.L_mon[mon])
        z = self.logp0 + self.L_all - self.L_mon[mon] + lw
        e = np.exp(z - z.max())
        return e / e.sum()

    def update(self, keys, lik_k, m):
        """keys: {k: set key} of the active in the worlds containing b; lik_k: {k: q_σ(b|M_k) (5,)}; m: m_t(σ)."""
        mon = next(iter(keys.values()))[0]
        lm = np.log(m)
        by_key = defaultdict(list)
        for k, key in keys.items():
            by_key[key].append(lik_k[k])
        for key in by_key:
            self.logw[mon].setdefault(key, self.L_mon[mon].copy())
        for key in self.logw[mon]:
            self.logw[mon][key] = self.logw[mon][key] + (np.log(np.mean(by_key[key], axis=0)) if key in by_key else lm)
        self.L_mon[mon] = self.L_mon[mon] + lm
        self.L_all = self.L_all + lm


def se_cluster(d, groups):
    """CR1 cluster-robust standard error of the mean of d."""
    d = np.asarray(d); n = len(d); m = d.mean()
    G = sorted(set(groups)); g = len(G)
    s = np.array([sum(d[i] - m for i in range(n) if groups[i] == gg) for gg in G])
    return np.sqrt(g / (g - 1) * (s ** 2).sum()) / n


def paired(R, a, b):
    """Mean log loss of predictors a and b and the paired difference a - b with game- and seed-clustered SEs."""
    la, lb = -np.log([r[a] for r in R]), -np.log([r[b] for r in R])
    d = la - lb
    games = [r["game"] for r in R]; seeds = [r["game"].rsplit("-g", 1)[0] for r in R]
    return dict(a=a, b=b, n=len(R), loss_a=float(la.mean()), loss_b=float(lb.mean()), diff=float(d.mean()),
                se_iid=float(np.std(d, ddof=1) / np.sqrt(len(d))), se_game=float(se_cluster(d, games)),
                se_seed=float(se_cluster(d, seeds)), G=len(set(games)), S=len(set(seeds)))


PAIRS = (("current", "membership"), ("membership", "perworld"), ("perworld", "perworld_iii"),
         ("perworld_iii", "truejoint"), ("perworld", "truejoint"), ("current_recomp", "current"))


def print_pairs(R):
    out = []
    for a, b in PAIRS:
        s = paired(R, a, b); out.append(s)
        print(f"  {a} {s['loss_a']:.4f} vs {b} {s['loss_b']:.4f}: diff {s['diff']:+.4f} ± {s['se_game']:.4f} game-clustered "
              f"(G = {s['G']}; ± {s['se_seed']:.4f} seed-clustered, S = {s['S']}; ± {s['se_iid']:.4f} iid), n = {s['n']}")
    return out


def main(work, out_path=None):
    worlds, games = load(work)
    post = behavior.Posterior()                      # replay of the deployed posterior (check against the logged one)
    post_log = behavior.Posterior()                  # the same replay from the logged qs (4 dp): checks the joins
    post_j = behavior.Posterior()                    # option (iii): posterior fed the per-world marginal likelihood
    rows = []
    acc_by_game = []
    stats = defaultdict(int)
    for g in sorted(games):
        recs = games[g]
        tj = TrueJoint(post_j.probs())
        acc = defaultdict(float)                     # option (i): accumulated log-lik per active moveset key
        nseen = defaultdict(int)
        for i, d, w in join(worlds[g], recs):
            stats["multi"] += 1
            fb = recs[i + 1].get("feedback") if i + 1 < len(recs) else None
            assert fb is not None, (g, i)
            if fb.get("skipped") or fb.get("b") is None:
                stats["skipped"] += 1
                continue
            Ms = [np.array(M) for M in w["Ms"]]
            cols, labels, wts = w["cols"], w["labels"], np.array(w["w"])
            K = len(Ms)
            mc = map_column(fb["b"], cols, wts)
            if mc is None:
                stats["not_in_this_decision"] += 1
                continue
            lab = cols[mc[0]][mc[1]]                 # the one label every predictor is scored on
            assert lab == fb["bayes"]["col"], (g, i, lab, fb["bayes"]["col"])
            ks = [k for k in range(K) if lab in cols[k]]
            jk = {k: cols[k].index(lab) for k in ks}
            jb = labels.index(lab)
            p = np.array(d["pfull"])
            P_rep = post.probs(); P_replog = post_log.probs(); Pj = post_j.probs()
            P = np.array([d["bayes"]["posterior"][s] for s in STYLES]); P = P / P.sum()   # logged deployed posterior
            qhat = np.array(d["bayes"]["qhat"])      # logged deployed q-hat
            # per-world style predictions
            qsk = {k: behavior.style_dists(Ms[k], p) for k in range(K)}
            lik_k = np.array([P @ qsk[k][:, jk[k]] for k in ks])        # Σ_σ P(σ) q_σ(b|M_k)
            wk = lik_k / lik_k.sum()
            u = np.full(len(ks), 1 / len(ks))
            tv = 0.5 * np.abs(wk - u).sum()
            ess = 1 / (wk ** 2).sum() / len(ks)
            # planner response (only meaningful if >1 world contains b)
            x_u = ha.agent_form_lp(Ms, np.full(K, 1 / K))[0]
            bw = np.zeros(K); bw[ks] = wk
            x_w = ha.agent_form_lp(Ms, bw)[0]
            l1 = np.abs(x_u - x_w).sum()
            # predictive probabilities of this action (prequential: every belief before its update)
            qs_bar = behavior.style_dists(np.array(d["abar"]), p)
            p_cur = float(qhat[jb])
            p_cur_recomp = float(P_rep @ qs_bar[:, jb])
            p_memb = float(sum(qhat[jb] / qhat[[labels.index(c) for c in cols[k]]].sum() for k in ks) / K)
            p_pw = float(sum(P @ qsk[k][:, jk[k]] for k in ks) / K)
            lik_style_pw = np.array([sum(qsk[k][s, jk[k]] for k in ks) / K for s in range(len(STYLES))])
            lik_style_bar = qs_bar[:, jb]
            p_iii = float(Pj @ lik_style_pw)
            # true joint over (active set, style)
            bel = d["belief"]["worlds"]
            assert len(bel) == K, (g, i)
            keys = {k: set_key(bel[k]["opp"][0]) for k in ks}
            p_tj = float(sum(tj.probs(keys[k]) @ qsk[k][:, jk[k]] for k in ks) / K)
            Pk_tj = {k: tj.probs(keys[k]) for k in ks}
            tv_tj = float(np.mean([0.5 * np.abs(Pk_tj[k] - Pj).sum() for k in ks]))
            # accumulated moveset weights (option i): key = sorted active moves in world k
            keyed = defaultdict(list)
            for k in ks:
                mkey = tuple(sorted(c for c in cols[k] if not c.endswith("-tera") and c in cols[k][:8]))
                keyed[mkey].append(float(np.log(P @ qsk[k][:, jk[k]])))
            if len(keyed) >= 2:
                for a in keyed:
                    for bk in keyed:
                        if a < bk:
                            acc[(a, bk)] += np.mean(keyed[a]) - np.mean(keyed[bk]); nseen[(a, bk)] += 1
            rows.append(dict(game=g, turn=d["turn"], idx=i, b=fb["b"], lab=lab, n_contain=len(ks), K=K, tv=tv, ess=ess,
                             maxratio=float(wk.max() / wk.min()), l1=l1, argmax_change=int(x_u.argmax() != x_w.argmax()),
                             current=p_cur, current_recomp=p_cur_recomp, membership=p_memb, perworld=p_pw,
                             perworld_iii=p_iii, truejoint=p_tj, tv_truejoint_vs_iii=tv_tj,
                             n_setkeys=len(set(keys.values())), P_logged=P.tolist(), P_replayed=P_rep.tolist(),
                             P_replayed_logqs=P_replog.tolist(),
                             lik_style_bar=lik_style_bar.tolist(), lik_style_perworld=lik_style_pw.tolist(),
                             how=fb["how"], n_keys=len(keyed), n_union=len(labels), n_world=float(np.mean([len(c) for c in cols]))))
            post.update(qs_bar, jb)                  # the real code's update (averaged matrix), for the replay check
            post_log.update(np.array(d["qs"]), jb)
            post_j.logp = post_j.logp + np.log(lik_style_pw); post_j.logp -= post_j.logp.max()
            tj.update(keys, {k: qsk[k][:, jk[k]] for k in ks}, lik_style_pw * K / len(ks))
        acc_by_game.append({k: (v, nseen[k]) for k, v in acc.items()})
    R = rows
    n = len(R)
    print(f"{work}: games {len(games)}; multi-option decisions {stats['multi']}; skipped feedback {stats['skipped']}; "
          f"label not in this decision's worlds {stats['not_in_this_decision']}; scored turns {n}")
    for key, how in (("P_replayed_logqs", "from the logged qs"), ("P_replayed", "from qs recomputed on Abar with pfull")):
        tvp = [0.5 * np.abs(np.array(r[key]) - np.array(r["P_logged"])).sum() for r in R]
        print(f"replayed deployed posterior ({how}) vs logged: mean TV {np.mean(tvp):.4f}, max {np.max(tvp):.4f}")
    multi = [r for r in R if r["n_contain"] > 1]
    print(f"turns where >1 world contains b: {len(multi)} ({len(multi)/n:.2f})")
    tv = np.array([r["tv"] for r in multi]); ess = np.array([r["ess"] for r in multi]); mr = np.array([r["maxratio"] for r in multi])
    print(f"world re-weighting (turns with >1 containing world): TV from uniform mean {tv.mean():.3f} median {np.median(tv):.3f} "
          f"p90 {np.percentile(tv,90):.3f}; ESS/n mean {ess.mean():.3f}; max/min lik ratio median {np.median(mr):.2f} p90 {np.percentile(mr,90):.2f}")
    l1 = np.array([r["l1"] for r in multi]); ch = np.array([r["argmax_change"] for r in multi])
    print(f"planner: L1(x_u, x_w) mean {l1.mean():.3f} median {np.median(l1):.3f} p90 {np.percentile(l1,90):.3f}; argmax changes {ch.mean():.3f}")
    print("predictive log loss (paired, same label for every predictor):")
    summary = print_pairs(R)
    nk = np.array([r["n_setkeys"] for r in R]); tvj = np.array([r["tv_truejoint_vs_iii"] for r in R])
    print(f"true joint: turns whose containing worlds carry >=2 distinct active set keys {np.mean(nk >= 2):.3f}; "
          f"TV(P(σ|s_k), option (iii) P(σ)) mean {tvj.mean():.4f} p90 {np.percentile(tvj, 90):.4f} max {tvj.max():.4f}")
    for name in ("lik_style_bar", "lik_style_perworld"):
        L = np.array([r[name] for r in R])
        print(f"  per-style log loss [{name}]: " + ", ".join(f"{s} {(-np.log(L[:, i])).mean():.3f}" for i, s in enumerate(STYLES)))
    nm = np.array([r["n_keys"] for r in R])
    print(f"dilution: union labels mean {np.mean([r['n_union'] for r in R]):.1f} vs per-world columns {np.mean([r['n_world'] for r in R]):.1f}; "
          f"turns whose containing worlds carry >=2 distinct active movesets: {np.mean(nm >= 2):.3f}")
    acc_lr = [(abs(v), c) for acc in acc_by_game for (v, c) in acc.values() if c >= 2]
    if acc_lr:
        a = np.array([v for v, c in acc_lr]); c = np.array([c for v, c in acc_lr])
        print(f"option (i): accumulated |log-lik ratio| between two movesets of the same active over a game (pairs co-sampled on >=2 turns): "
              f"n {len(a)}, co-sampled turns median {np.median(c):.0f}, |LR| median {np.median(a):.2f} p90 {np.percentile(a,90):.2f} nats "
              f"(>= ln 3 = 1.1 in {np.mean(a >= np.log(3)):.2f} of pairs)")
    json.dump(dict(stats=dict(stats), summary=summary, rows=R), open(out_path or os.path.join(work, "analyze_worlds.json"), "w"))


def pool(paths):
    R = [r for p in paths for r in json.load(open(p))["rows"]]
    print(f"pooled {paths}: {len(R)} scored turns")
    print_pairs(R)


if __name__ == "__main__":
    if sys.argv[1] == "--pool":
        pool(sys.argv[2:])
    else:
        main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None)
