"""Score OUR hidden-set belief (hidden.Tracker + hidden.Belief) in-engine, where the true teams are known.

  .venv/bin/python score_belief_engine.py play --pairs 20 --workers 4     # agentform:4:200 vs pimc:4:200, paired seeds
  .venv/bin/python score_belief_engine.py score                          # -> results/belief/engine.json
  .venv/bin/python score_belief_engine.py a4                             # -> results/belief/a4_tv.json
  .venv/bin/python score_belief_engine.py check                          # closed form vs Belief.sample Monte Carlo

play: hidden_experiments.play_hidden's loop (same team draw: randbats.sample_team twice from the seed's rng), with a
scoring hook before every decision node for BOTH sides' beliefs about the other. Per revealed alive opponent slot, the
posterior marginals are computed in closed form from hidden.role_posterior / hidden._role_lists (exactly the
distribution Belief.sample draws from): item/ability/tera P(x) = sum_r w_r count_r(x)/|list_r|; a pool move m not yet
revealed has P(m in set) = sum_r w_r [m in pool_r] (min(4,n_r) - k_r)/(n_r - k_r). Truth = the generator's PokeSet
(the start-of-game set; a consumed item is scored against the original item, which Belief also targets). Bench: for
each true unrevealed mon, P(its species key is drawn) under Belief.sample_unrevealed (src ours) and under the exact
conditional of randbats.sample_team given the revealed slots (src gen), both by Monte Carlo (see a4). Events go to
results/belief/engine_events.jsonl; metrics are showdown/score_belief.summarize (same definitions as for foul-play).

a4: size of the A4 error (game_model.md §6 item 2). sample_team fills slot i uniformly over keys whose base is not in
slots < i, so P(team) = prod_i 1/(N - F_<i) with F_<i the summed forme counts of earlier bases. Belief.sample_unrevealed
fills unrevealed slots in index order uniformly over keys whose base is not revealed or already drawn. Importance
weights w = P_gen / P_sampler on sampler draws give the generator's conditional. Given the revealed configuration, both
conditionals are exchangeable within forme-count classes, so per-key probabilities are estimated per class (low noise).
TV is between the two average-over-unrevealed-slots marginals over species keys, per revealed configuration seen in the
played games (and a split-half noise floor).
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from collections import Counter, defaultdict
from functools import lru_cache
from multiprocessing import get_context
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
OUT = HERE / "results" / "belief"
EVENTS = OUT / "engine_events.jsonl"
CONFIGS = OUT / "engine_configs.jsonl"
sys.path.insert(0, str(HERE / "showdown"))
import score_belief as sb  # noqa: E402

N_BENCH = 4000      # Monte Carlo draws per revealed configuration for bench presence


# ---------------------------------------------------------------- species universe (fast sampler + generator weights)

@lru_cache(maxsize=None)
def universe():
    import hidden as hd
    import randbats as rb
    keys = sorted(rb.sets())
    base = [hd.base_of(k) for k in keys]
    nf = Counter(base)
    return keys, base, nf, {k: i for i, k in enumerate(keys)}


def draw_bench(revealed: dict, rng: random.Random):
    """One Belief.sample_unrevealed-style draw (uniform over keys whose base is not taken, slots in index order),
    by rejection. revealed: slot -> key index. Returns (slot -> key index, log importance weight gen/sampler)."""
    keys, base, nf, _ = universe()
    N = len(keys)
    taken = {base[i] for i in revealed.values()}
    out, logq = {}, 0.0
    for s in range(6):
        if s in revealed:
            continue
        F = sum(nf[b] for b in taken)
        while True:
            i = rng.randrange(N)
            if base[i] not in taken:
                break
        out[s] = i
        taken.add(base[i])
        logq -= math.log(N - F)
    full = {**revealed, **out}
    logp, seen = 0.0, set()
    for s in range(6):
        logp -= math.log(N - sum(nf[b] for b in seen))
        seen.add(base[full[s]])
    return out, logp - logq


def bench_tables(revealed: dict, n: int, seed: int):
    """Per forme-count class f: per-key presence probability (key appears in any unrevealed slot) and per-key
    average-slot probability, under the sampler (unweighted) and the generator (importance weighted). Also the two
    split halves of the sampler for a noise floor. Returns dict."""
    keys, base, nf, _ = universe()
    rng = random.Random(seed)
    u = 6 - len(revealed)
    taken = {base[i] for i in revealed.values()}
    elig = Counter(nf[b] for b in base if b not in taken)          # class f -> number of eligible keys
    pres = {"ours": Counter(), "gen": Counter(), "h1": Counter(), "h2": Counter()}
    W = {"ours": 0.0, "gen": 0.0, "h1": 0.0, "h2": 0.0}
    lw = []
    draws = [draw_bench(revealed, rng) for _ in range(n)]
    m = max(l for _, l in draws)
    for j, (out, l) in enumerate(draws):
        w = math.exp(l - m)
        h = "h1" if j % 2 == 0 else "h2"
        for name, wt in (("ours", 1.0), ("gen", w), (h, 1.0)):
            W[name] += wt
            for i in out.values():
                pres[name][nf[base[i]]] += wt
        lw.append(w)
    lw = np.array(lw)
    tab = {name: {f: pres[name][f] / (W[name] * elig[f]) for f in elig} for name in pres}   # per-key presence
    ess = float(lw.sum() ** 2 / (lw ** 2).sum())
    # average-slot per-key probability = presence / u (a key appears at most once per team)
    tv = lambda a, b: 0.5 * sum(elig[f] * abs(tab[a][f] - tab[b][f]) / u for f in elig)
    return dict(tab=tab, elig=dict(elig), u=u, tv=tv("ours", "gen"), tv_noise=tv("h1", "h2"), ess=ess, n=n)


# ---------------------------------------------------------------- closed-form marginals of Belief

def marginals(species: str, moves: frozenset, tera):
    import hidden as hd
    import randbats as rb
    d = dict(item=defaultdict(float), ability=defaultdict(float), tera=defaultdict(float), moves=defaultdict(float))
    for r, w in hd.role_posterior(species, moves, tera):
        if w <= 0:
            continue
        pool, abil, items, teras = hd._role_lists(species, r)
        for L, key in ((items, "item"), (abil, "ability"), (teras, "tera")):
            for x in L:
                d[key][rb.to_id(x)] += w / len(L)
        ids = [rb.to_id(m) for m in pool]
        n, kr = len(ids), len(moves & set(ids))
        slots = min(4, n) - kr
        for m in ids:
            if m in moves:
                d["moves"][m] += w
            elif n - kr > 0:
                d["moves"][m] += w * slots / (n - kr)
    if tera is not None:
        d["tera"] = {tera: 1.0}
    return d


# ---------------------------------------------------------------- play + score hook

def truth_of(ps):
    import randbats as rb
    return dict(species=ps.species, item=rb.to_id(ps.item), ability=rb.to_id(ps.ability), tera=ps.tera.lower(),
                moves={rb.to_id(m) for m in ps.moves})


def score_view(view, teams, side, game, dec, turn, opp_alive, bench_cache):
    """Events for `side`'s belief about the opponent (team teams[3 - side])."""
    import hidden as hd
    keys, base, nf, kidx = universe()
    truth = teams[3 - side]
    evs = []
    for slot, sr in view.reveals.items():
        if not opp_alive[slot]:
            continue
        t = truth_of(truth[slot])
        if hd.species_key(rb_id(t["species"])) != sr.species:
            continue                                          # never expected (revealed species is exact in-engine)
        known = set(sr.moves.values())
        meta = dict(game=game, dec=dec, turn=turn, mon=t["species"], k=min(4, len(known & t["moves"])), side=side)
        d = marginals(sr.species, frozenset(known), sr.tera)
        evs.append(sb.cat_event("ours", "item", d["item"], t["item"], **meta))
        evs.append(sb.cat_event("ours", "ability", d["ability"], t["ability"], **meta))
        if sr.tera is None:
            evs.append(sb.cat_event("ours", "tera", d["tera"], t["tera"], **meta))
        evs += sb.move_events("ours", d["moves"], t["moves"], known, **meta)
    revealed = {s: kidx[sr.species] for s, sr in view.reveals.items()}
    if view.unrevealed:
        key = tuple(sorted(revealed.items()))
        if key not in bench_cache:
            bench_cache[key] = bench_tables(revealed, N_BENCH, seed=hash(key) & 0xFFFFFFF)
        bt = bench_cache[key]
        for s in view.unrevealed:
            f = nf[base[kidx[truth[s].species]]]
            meta = dict(game=game, dec=dec, turn=turn, mon=truth[s].species, k=len(revealed), side=side)
            for src, name in (("ours", "ours"), ("gen", "gen")):
                p = bt["tab"][name][f]
                evs.append(dict(src=src, attr="bench", p=p, hit=int(p >= 0.5), brier=(1 - p) ** 2, pairs=[],
                                base=bt["u"] / sb.N_SPECIES, **meta))
    return evs


def rb_id(species):
    import randbats as rb
    return rb.to_id(species)


def play_scored(A, B, seed, game, max_turns=300):
    import hidden as hd
    import hidden_agents as ha
    import pkbattle as pb
    import poke_engine as pe
    import randbats as rb
    rng = random.Random(seed)
    t1, t2 = rb.sample_team(rng), rb.sample_team(rng)          # == randbats.random_state(rng)'s draws
    side = lambda t: pe.Side(pokemon=[rb.to_pokemon(s) for s in t])
    state = pe.State(side_one=side(t1), side_two=side(t2))
    teams = {1: t1, 2: t2}
    agents = {1: ha.Agent(A), 2: ha.Agent(B)}
    tr = {1: hd.Tracker(1, state), 2: hd.Tracker(2, state)}
    evs, cache, configs, dec, winner = [], {}, {}, 0, 0
    for turn in range(1, max_turns + 1):
        opts = dict(zip((1, 2), pb.options(state)))
        ch = {}
        for s in (1, 2):
            view = hd.make_view(state, tr[s])
            opp = state.side_two if s == 1 else state.side_one
            evs += score_view(view, teams, s, game, dec, turn, [p.hp > 0 for p in opp.pokemon], cache)
            dec += 1
            ch[s] = agents[s].act(state, view, hd.Belief(view), opts[s], s, rng)
        branches = pe.generate_instructions(state, ch[1], ch[2])
        pick = rng.choices(branches, weights=[b.percentage for b in branches])[0]
        new = state.apply_instructions(pick)
        for s in (1, 2):
            tr[s].update(state, ch[1], ch[2], pick, new)
        state = new
        a1, a2 = pb.alive(state.side_one), pb.alive(state.side_two)
        if a1 == 0 or a2 == 0:
            winner = 0 if a1 == a2 else (1 if a1 else 2)
            break
    cfg = [dict(game=game, revealed=list(map(list, k)), u=v["u"], tv=v["tv"], tv_noise=v["tv_noise"], ess=v["ess"],
                n=v["n"]) for k, v in cache.items()]
    return evs, cfg, dict(game=game, seed=seed, winner=winner, turns=turn, decisions=dec)


def _task(t):
    A, B, seed, game = t
    try:
        return play_scored(A, B, seed, game)
    except Exception:
        import traceback
        return [], [], dict(game=game, seed=seed, error=traceback.format_exc()[-800:])


def cmd_play(a):
    OUT.mkdir(parents=True, exist_ok=True)
    A, B = "agentform:4:200", "pimc:4:200"
    tasks = []
    for k in range(a.pairs):
        seed = 1_000_003 + k                                  # E4 matchup-0 seeds
        tasks += [(A, B, seed, f"s{seed}-g1"), (B, A, seed, f"s{seed}-g2")]
    t0 = time.time()
    with get_context("spawn").Pool(a.workers) as pool, EVENTS.open("w") as fe, CONFIGS.open("w") as fc, \
            (OUT / "engine_games.jsonl").open("w") as fg:
        for j, (evs, cfg, g) in enumerate(pool.imap_unordered(_task, tasks), 1):
            for e in evs:
                fe.write(json.dumps(e) + "\n")
            for c in cfg:
                fc.write(json.dumps(c) + "\n")
            fg.write(json.dumps(g) + "\n")
            fe.flush(); fc.flush(); fg.flush()
            print(f"  {j}/{len(tasks)} games, {time.time() - t0:.0f}s {g.get('error', '')[:200]}", flush=True)


def cmd_score(a):
    evs = [json.loads(l) for l in EVENTS.read_text().splitlines()]
    for e in evs:                                             # g1/g2 of a seed share both true teams
        e["cluster"] = e["game"].split("-g")[0]
    games = [json.loads(l) for l in (OUT / "engine_games.jsonl").read_text().splitlines()]
    res = dict(source="engine/hidden.py Tracker+Belief", games=len(games),
               errors=sum("error" in g for g in games), decisions=sum(g.get("decisions", 0) for g in games),
               clip=sb.CLIP, metrics=sb.summarize(evs))
    json.dump(res, open(OUT / "engine.json", "w"), indent=1)
    print(f"{res['games']} games ({res['errors']} errors), {res['decisions']} decision nodes, {len(evs)} events")
    for key, d in res["metrics"].items():
        print(f"  {key:14s} n={d['n']:6d} games={d['n_games']:3d} P(truth)={d['mean_p_truth']:.3f} acc={d['acc']:.3f} "
              f"logloss={d['logloss']:.3f} brier={d['brier']:.3f} p0={d['frac_p0']:.3f} "
              f"ECE={(d['calibration'] or {}).get('ece')} ECE_distinct={(d['calibration_distinct'] or {}).get('ece')} null={d['ece_null_distinct']} "
              f"base={d.get('baseline_p')}")


def cmd_a4(a):
    """Toy check of the IS construction, then TV over the revealed configurations seen in play (one row each)."""
    from fractions import Fraction  # noqa: F401
    # toy: keys {A-1, A-2, B, C}, team of 2, slot 1 = B. Exact: slot 0 = (3/8, 3/8, 1/4) vs sampler (1/3, 1/3, 1/3).
    keys = ["A-1", "A-2", "B", "C"]; bs = [k.split("-")[0] for k in keys]; nf = Counter(bs)
    rng = random.Random(0); acc = Counter(); W = 0.0
    for _ in range(200000):
        while True:
            i = rng.randrange(4)
            if bs[i] != "B":
                break
        lq = -math.log(4 - nf["B"])
        lp = -math.log(4) - math.log(4 - nf[bs[i]])
        w = math.exp(lp - lq); acc[keys[i]] += w; W += w
    toy = {k: round(v / W, 4) for k, v in sorted(acc.items())}
    seen = {}
    for l in CONFIGS.read_text().splitlines():               # distinct revealed configurations from play
        c = json.loads(l)
        seen.setdefault(tuple(map(tuple, c["revealed"])), c)
    allc = list(seen)
    isprefix = lambda k: sorted(s for s, _ in k) == list(range(len(k)))
    nonp = [k for k in allc if not isprefix(k)]
    random.Random(0).shuffle(nonp)
    with get_context("spawn").Pool(a.workers) as pool:        # exact for all; Monte Carlo cross-check on a4_configs
        rows = pool.map(_a4_row, [(k, a.a4_draws if i < a.a4_configs else 0) for i, k in enumerate(nonp)])
    cfg = [dict(revealed=k, **r) for k, r in zip(nonp, rows)]
    cfg += [dict(revealed=k, tv=0.0, n=0) for k in allc if isprefix(k)]
    tv = np.array([c["tv"] for c in cfg])
    mc = [(c["tv"], c["tv_mc"], c["tv_noise"]) for c in cfg if c.get("tv_mc") is not None]
    nz = np.array([m[2] for m in mc])
    prefix = np.array([isprefix(c["revealed"]) for c in cfg])
    _, base, nfu, _ = universe()
    multi = sum(1 for b in nfu if nfu[b] > 1)
    res = dict(toy_generator=toy, toy_expected={"A-1": 0.375, "A-2": 0.375, "C": 0.25},
               n_configs=len(cfg), n_distinct_configs_seen=len(allc), n_nonprefix=len(nonp),
               n_draws_per_config=a.a4_draws,
               note="prefix configs (revealed slots = 0..j-1) have TV 0 exactly: all importance weights are equal",
               species_keys=len(base), bases=len(nfu), multi_forme_bases=multi,
               forme_count_hist=dict(Counter(nfu.values())),
               tv_mean=float(tv.mean()), tv_median=float(np.median(tv)), tv_p95=float(np.percentile(tv, 95)),
               tv_max=float(tv.max()), worst_config=[list(x) for x in cfg[int(tv.argmax())]["revealed"]],
               mc_check=dict(n=len(mc), draws=a.a4_draws, mean_exact=float(np.mean([m[0] for m in mc])),
                             mean_mc=float(np.mean([m[1] for m in mc])), mc_split_half_noise=float(nz.mean()),
                             max_abs_exact_minus_mc=float(max(abs(m[0] - m[1]) for m in mc))),
               tv_mean_prefix_configs=float(tv[prefix].mean()) if prefix.any() else None,
               tv_mean_nonprefix_configs=float(tv[~prefix].mean()) if (~prefix).any() else None,
               n_prefix=int(prefix.sum()), ess_min=float(min(c["ess"] for c in cfg if c.get("ess"))))
    json.dump(res, open(OUT / "a4_tv.json", "w"), indent=1)
    print(json.dumps(res, indent=1))


def exact_tables(revealed: dict):
    """Exact version of bench_tables' per-key average-slot probabilities, by enumerating forme-count CLASS sequences
    of the unrevealed slots (both conditionals are exchangeable within a class). Sampler: class f at a slot with
    probability f * (#eligible bases of class f) / (N - F_taken). Generator: P(class seq) proportional to
    prod_{all six slots} 1/(N - F_<i) * prod_{unrevealed picks} (#eligible keys of the class). Returns (tv, u)."""
    keys, base, nf, _ = universe()
    N = len(keys)
    taken0 = {base[i] for i in revealed.values()}
    nb = Counter(nf[b] for b in nf if b not in taken0)          # class f -> eligible bases
    elig = {f: f * c for f, c in nb.items()}                    # class f -> eligible keys
    u = 6 - len(revealed)
    acc = {"ours": Counter(), "gen": Counter()}
    tot = {"ours": 0.0, "gen": 0.0}

    def rec(slot, F_seq, F_taken, nb, q, g, picks):
        if slot == 6:
            for name, w in (("ours", q), ("gen", g)):
                tot[name] += w
                for f in picks:
                    acc[name][f] += w
            return
        g = g / (N - F_seq)
        if slot in revealed:
            f = nf[base[revealed[slot]]]
            rec(slot + 1, F_seq + f, F_taken, nb, q, g, picks)
            return
        for f, c in nb.items():
            if c == 0:
                continue
            nk = f * c
            nb2 = dict(nb); nb2[f] = c - 1
            rec(slot + 1, F_seq + f, F_taken + f, nb2, q * nk / (N - F_taken), g * nk, picks + (f,))

    F_rev = sum(nf[b] for b in taken0)
    rec(0, 0, F_rev, dict(nb), 1.0, 1.0, ())
    p = {name: {f: acc[name][f] / tot[name] / elig[f] / u for f in elig} for name in acc}   # per-key, avg slot
    tv = 0.5 * sum(elig[f] * abs(p["ours"][f] - p["gen"][f]) for f in elig)
    return tv, u


def _a4_row(t):
    k, n = t
    r = bench_tables(dict(k), n, seed=hash(k) & 0xFFFFFFF) if n else dict(tv=None, tv_noise=None, ess=None)
    tv, u = exact_tables(dict(k))
    return dict(u=u, tv=tv, tv_mc=r["tv"], tv_noise=r["tv_noise"], ess=r["ess"], n=n)


def cmd_check(a):
    """Closed-form marginals vs Belief.sample Monte Carlo on a few random reveal patterns."""
    import hidden as hd
    import randbats as rb
    rng = random.Random(1)
    worst = 0.0
    for _ in range(6):
        sp = rng.choice(sorted(rb.sets()))
        ps = rb.sample_set(sp, rng)
        rev = {j: rb.to_id(m) for j, m in enumerate(ps.moves[:rng.randint(0, 2)])}
        sr = hd.SlotReveal(sp, rev, ps.tera.lower() if rng.random() < 0.3 else None)
        view = hd.View(1, None, {}, {0: {}}, {0: sr}, [], {}, False, None)
        bel = hd.Belief(view)
        d = marginals(sp, frozenset(rev.values()), sr.tera)
        cnt = defaultdict(Counter); n = 20000
        for _ in range(n):
            s = bel.sample(0, rng)
            cnt["item"][rb.to_id(s.item)] += 1; cnt["ability"][rb.to_id(s.ability)] += 1
            cnt["tera"][s.tera.lower()] += 1
            for m in s.moves:
                cnt["moves"][rb.to_id(m)] += 1
        for key in d:
            for v in set(d[key]) | set(cnt[key]):
                worst = max(worst, abs(d[key].get(v, 0) - cnt[key][v] / n))
        print(sp, "revealed", list(rev.values()), "tera", sr.tera, "max |closed - MC| so far", round(worst, 4))
    print("OK" if worst < 0.02 else "MISMATCH", worst)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["play", "score", "a4", "check"])
    ap.add_argument("--pairs", type=int, default=20)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--a4-configs", type=int, default=40, help="non-prefix configurations also checked by Monte Carlo")
    ap.add_argument("--a4-draws", type=int, default=100000, help="Monte Carlo draws per configuration")
    a = ap.parse_args()
    dict(play=cmd_play, score=cmd_score, a4=cmd_a4, check=cmd_check)[a.cmd](a)
