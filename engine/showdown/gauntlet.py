"""Round-robin gauntlet of bots through batch.py, then Bradley-Terry / Elo ratings with bootstrap CIs.

  ~/sandbox/bot-venv/bin/python engine/showdown/gauntlet.py --bots random,maxpower,agentform,stock@25,stock@100 \
      --seeds-per-pair 20 [--override 'agentform:stock@100=60'] --workers 10 --prefix G [--dry-run] [--fit-only]

Every unordered pair (a, b), in --bots order, runs `batch.py --tag <prefix>_<a>_<b> --a a --b b --seeds 1-N` ("@" in
a tag becomes "at"); pairings run one after another, each on all --workers, and batch.py's resume skips seeds already
played. The fit reads the pairings' game JSONs: a seed counts when both its games finished without error (the error
seeds are counted); A's game score is win 1 / draw 0.5 / loss 0 (batch.game_score). Bradley-Terry is fit by maximum
likelihood on the games (a draw is half a win each way) with a weak L2 prior on the log-strengths (--l2), which keeps
perfect records finite; ratings are on the Elo scale (400 log10), anchored at random = 0 (else the first bot). 95% CIs:
--bootstrap resamples of seeds within each pairing. With >= 2 stock@<ms> bots, each other bot gets the foul-play
budget whose interpolated rating (piecewise linear in log ms) equals its own, or "out of range".
Meta-game: the pairwise scores define a symmetric zero-sum game, P[i, j] = score of i vs j - 0.5 (P[i, i] = 0); its
maximin mixture sigma (an LP) is reported with bootstrap CIs (same resampled seeds) and each bot's Nash-averaged payoff
(P sigma)_i. Bots whose pairings have no seeds are left out of the game: while one is missing, the bot with the most
missing pairings is dropped (ties: the later one in --bots).
Outputs: engine/showdown/results/gauntlet/<out-prefix>_ratings.json and .md (--out-prefix defaults to --prefix).
"""
import argparse
import itertools
import json
import math
import os
import subprocess
import sys
import time

os.environ.setdefault("OURBOT_WORK", os.path.join(os.path.expanduser("~"), "sandbox", "ourbot"))   # as headless.py
import numpy as np                                                                       # noqa: E402
from scipy.optimize import linprog, minimize                                             # noqa: E402

import batch                                                                             # noqa: E402
from headless import method_spec, stock_ms                                               # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results", "gauntlet")
ELO = 400 / math.log(10)                                     # natural log-strength -> Elo points


def tag_of(prefix, a, b):
    return f"{prefix}_{a}_{b}".replace("@", "at")


def parse_overrides(spec):
    """'a:b=N,c:d=M' -> {frozenset((a, b)): N, ...} (order-insensitive)."""
    out = {}
    for part in filter(None, (spec or "").split(",")):
        pair, _, n = part.partition("=")
        a, _, b = pair.partition(":")
        out[frozenset((a.strip(), b.strip()))] = int(n)
    return out


def pairings(bots, n, overrides):
    return [(a, b, overrides.get(frozenset((a, b)), n)) for a, b in itertools.combinations(bots, 2)]


def batch_cmd(prefix, a, b, n, workers):
    return [batch.PY, os.path.join(HERE, "batch.py"), "--tag", tag_of(prefix, a, b), "--a", a, "--b", b,
            "--seeds", f"1-{n}", "--workers", str(workers)]


def run_all(plan, prefix, workers):
    codes = {}
    for i, (a, b, n) in enumerate(plan, 1):
        print(f"[{i}/{len(plan)}] {a} vs {b}: {n} seeds ({2 * n} games)", flush=True)
        t0 = time.time()
        rc = subprocess.run(batch_cmd(prefix, a, b, n, workers)).returncode   # exit 1 on error/missing games: go on
        codes[tag_of(prefix, a, b)] = rc
        print(f"[{i}/{len(plan)}] {a} vs {b}: exit {rc} in {time.time() - t0:.0f}s", flush=True)
    return codes


def load_pairing(prefix, a, b, n):
    """Per seed (both games finished without error): A's two game scores; plus counts of what was left out."""
    tag = tag_of(prefix, a, b)
    seeds, error_games, sim_error_games, missing = {}, 0, 0, 0
    for s in range(1, n + 1):
        recs = []
        for g in (1, 2):
            p = batch.game_path(tag, s, g)
            if not os.path.exists(p):
                missing += 1
                continue
            recs.append(json.load(open(p)))
        error_games += sum(bool(r.get("error")) for r in recs)
        sim_error_games += sum(bool(r.get("sim_errors")) and not r.get("error") for r in recs)
        if len(recs) == 2 and not any(r.get("error") for r in recs):
            seeds[s] = [batch.game_score(r) for r in recs]
    return dict(tag=tag, seeds=seeds, error_games=error_games, sim_error_games=sim_error_games, missing_games=missing)


def wins_matrix(bots, data, pick=None):
    """W[i, j] = games i won against j (a draw is half each way); pick: pairing tag -> list of seeds to use."""
    idx = {m: i for i, m in enumerate(bots)}
    W = np.zeros((len(bots), len(bots)))
    for (a, b), d in data.items():
        for s in (pick[d["tag"]] if pick else d["seeds"]):
            for sc in d["seeds"][s]:
                W[idx[a], idx[b]] += sc
                W[idx[b], idx[a]] += 1 - sc
    return W


def fit_bt(W, l2, anchor):
    """MAP log-strengths: max sum_ij W_ij log sigmoid(t_i - t_j) - l2/2 |t|^2, shifted so t[anchor] = 0."""
    N = W + W.T

    def f(t):
        d = t[:, None] - t[None, :]
        ll = np.sum(W * -np.logaddexp(0, -d))
        g = np.sum(W - N / (1 + np.exp(-d)), axis=1)
        return -(ll - 0.5 * l2 * t @ t), -(g - l2 * t)
    t = minimize(f, np.zeros(len(W)), jac=True, method="L-BFGS-B").x
    return t - t[anchor]


def meta_bots(bots, data):
    """(kept, excluded): drop bots until every pairing among the kept ones has seeds; each step drops the bot with the
    most seedless pairings among the kept, ties to the later one in --bots."""
    empty = {frozenset(k) for k, d in data.items() if not d["seeds"]}
    kept = list(bots)
    while True:
        miss = {m: sum(frozenset((m, o)) in empty for o in kept if o != m) for m in kept}
        worst = max(reversed(kept), key=lambda m: miss[m])     # max keeps the first maximum: the later bot
        if not miss[worst]:
            return kept, [m for m in bots if m not in kept]
        kept.remove(worst)


def payoff_matrix(bots, data, pick=None):
    """P[i, j] = mean pair score of i vs j - 0.5 (skew-symmetric, P[i, i] = 0); pick as in wins_matrix."""
    idx = {m: i for i, m in enumerate(bots)}
    P = np.zeros((len(bots), len(bots)))
    for (a, b), d in data.items():
        if a in idx and b in idx:
            seeds = pick[d["tag"]] if pick else d["seeds"]
            P[idx[a], idx[b]] = np.mean([np.mean(d["seeds"][s]) for s in seeds]) - 0.5
            P[idx[b], idx[a]] = -P[idx[a], idx[b]]
    return P


def meta_nash(P, tol=1e-7):
    """Maximin mixture of the symmetric zero-sum game P (row player's payoff): max v s.t. (sigma^T P)_j >= v for all
    j, sigma in the simplex, by LP (HiGHS). As P is skew-symmetric the value is 0 and one mixture is the equilibrium.
    check: no pure strategy gains against sigma (max_i (P sigma)_i <= tol) and min_j (sigma^T P)_j >= value - tol.
    The dual (-ineqlin marginals) is the column player's mixture, itself an equilibrium here: if it differs from
    sigma the equilibrium is provably not unique; an off-support bot with zero reduced cost flags possible ties."""
    n = len(P)
    c = np.zeros(n + 1)
    c[-1] = -1                                              # minimize -v
    res = linprog(c, A_ub=np.hstack([-P.T, np.ones((n, 1))]), b_ub=np.zeros(n),
                  A_eq=[[1.0] * n + [0.0]], b_eq=[1.0], bounds=[(0, None)] * n + [(None, None)], method="highs")
    if not res.success:
        raise RuntimeError(f"meta-game LP failed: {res.message}")
    sigma = np.clip(res.x[:n], 0, None)
    sigma /= sigma.sum()
    value = float(res.x[-1])
    payoff = P @ sigma                                      # (P sigma)_i: pure i against the mixture
    col = sigma @ P                                         # (sigma^T P)_j: the mixture against pure j
    y = np.clip(-res.ineqlin.marginals, 0, None)
    y = y / y.sum() if y.sum() > 0 else y
    zero_rc = [i for i in range(n) if sigma[i] <= 1e-6 and abs(res.lower.marginals[i]) <= 1e-9]
    note = None
    if np.abs(y - sigma).max() > 1e-6:
        note = "not unique: the LP dual's mixture is a different equilibrium"
    elif zero_rc:
        note = "possibly not unique: off-support strategies with zero reduced cost (dual degenerate)"
    return dict(sigma=sigma, value=value, support=[i for i in range(n) if sigma[i] > 1e-6], payoff=payoff,
                exploitability=float(payoff.max()), dual_sigma=y, zero_reduced_cost=zero_rc, note=note,
                check=bool(payoff.max() <= tol and col.min() >= value - tol))


def cpu_per_decision(prefix, plan, data):
    """Mean CPU-s per decision per bot over its pairings: batch summaries' per_method, else the game JSONs."""
    tot = {}
    for a, b, n in plan:
        d = data[(a, b)]
        path = os.path.join(batch.OUT, f"{d['tag']}_summary.json")
        if os.path.exists(path):
            per = json.load(open(path))["per_method"]
            rows = [(m, r["decisions"], r["cpu_s_per_decision"] * r["decisions"]) for m, r in per.items()]
        else:
            rows = []
            for s in d["seeds"]:
                for g in (1, 2):
                    rec = json.load(open(batch.game_path(d["tag"], s, g)))
                    rows += [(rec["methods"][r], rec["timing"][r]["decisions"], rec["timing"][r]["cpu_s"])
                             for r in ("A", "B")]
        for m, dec, cpu in rows:
            t = tot.setdefault(m, [0, 0.0])
            t[0] += dec
            t[1] += cpu
    return {m: round(c / dec, 4) if dec else None for m, (dec, c) in tot.items()}


def equivalent_budgets(bots, ratings):
    """For each non-stock bot: the stock@<ms> budget with the same rating (linear in log ms between adjacent budgets)."""
    stock = sorted((stock_ms(m), ratings[m]) for m in bots if stock_ms(m))
    if len(stock) < 2:
        return None
    out = {}
    for m in bots:
        if stock_ms(m) or m == "stock":
            continue
        r, res = ratings[m], None
        for (m0, r0), (m1, r1) in zip(stock, stock[1:]):
            if min(r0, r1) <= r <= max(r0, r1) and r0 != r1:
                f = (r - r0) / (r1 - r0)
                res = round(math.exp(math.log(m0) + f * (math.log(m1) - math.log(m0))), 1)
                break
        out[m] = dict(ms=res) if res is not None else dict(ms=None, note=(
            f"out of range: rating {r:.0f} vs stock@{stock[0][0]}..{stock[-1][0]} ratings "
            f"{min(x[1] for x in stock):.0f}..{max(x[1] for x in stock):.0f}"))
    return out


def fit(args, bots, plan):
    resolved = {m: method_spec(m) for m in bots}             # "ours" -> "agentform": the key the records use
    data = {(a, b): load_pairing(args.prefix, a, b, n) for a, b, n in plan}
    anchor = (bots.index(args.anchor) if args.anchor else
              bots.index("random") if "random" in bots else 0)   # a bot with a perfect record makes a poor anchor
    t = fit_bt(wins_matrix(bots, data), args.l2, anchor) * ELO
    kept, excluded = meta_bots(bots, data)
    nash = meta_nash(payoff_matrix(kept, data))
    rng = np.random.default_rng(args.boot_seed)
    boots, pair_boots, sigma_boots, boot_check_fail = [], {k: [] for k in data}, [], 0
    for _ in range(args.bootstrap):
        pick = {d["tag"]: list(rng.choice(list(d["seeds"]), len(d["seeds"]))) if d["seeds"] else []
                for d in data.values()}
        boots.append(fit_bt(wins_matrix(bots, data, pick), args.l2, anchor) * ELO)
        nb = meta_nash(payoff_matrix(kept, data, pick))
        sigma_boots.append(nb["sigma"])
        boot_check_fail += not nb["check"]
        for k, d in data.items():
            if pick[d["tag"]]:
                pair_boots[k].append(np.mean([np.mean(d["seeds"][s]) for s in pick[d["tag"]]]))
    boots = np.array(boots)
    ci = lambda x: [round(float(v), 1) for v in np.percentile(x, [2.5, 97.5])] if len(x) else None
    ci_score = lambda x: [round(float(v), 4) for v in np.percentile(x, [2.5, 97.5])] if len(x) else None  # scores in [0, 1]
    ratings = {m: round(float(t[i]), 1) for i, m in enumerate(bots)}
    cpu = cpu_per_decision(args.prefix, plan, data)
    games = {m: 0 for m in bots}
    errs = {m: 0 for m in bots}
    pairwise = []
    for (a, b), d in data.items():
        ng = 2 * len(d["seeds"])
        for m in (a, b):
            games[m] += ng
            errs[m] += d["error_games"]
        mean = float(np.mean([np.mean(v) for v in d["seeds"].values()])) if d["seeds"] else None
        pairwise.append(dict(a=a, b=b, tag=d["tag"], score_a=round(mean, 4) if mean is not None else None,
                             ci95=ci_score(pair_boots[(a, b)]), n_pairs=len(d["seeds"]), games_used=ng,
                             error_games=d["error_games"], sim_error_games=d["sim_error_games"],
                             missing_games=d["missing_games"]))
    out = dict(
        prefix=args.prefix, bots=bots, anchor=bots[anchor], scale="Elo (400 log10), Bradley-Terry MAP",
        l2_prior=args.l2, bootstrap=args.bootstrap, boot_seed=args.boot_seed,
        ratings=ratings, ci95={m: ci(boots[:, i]) for i, m in enumerate(bots)},
        cpu_s_per_decision={m: cpu.get(resolved[m]) for m in bots},
        games_used=games, error_games=errs, pairwise=pairwise,
        foulplay_equivalent_ms=equivalent_budgets(bots, ratings),
        meta_game=meta_block(kept, excluded, nash, np.array(sigma_boots).reshape(-1, len(kept)), boot_check_fail),
    )
    os.makedirs(RESULTS, exist_ok=True)
    path = os.path.join(RESULTS, f"{args.out_prefix or args.prefix}_ratings")
    json.dump(out, open(path + ".json", "w"), indent=1)
    open(path + ".md", "w").write(markdown(out))
    print(markdown(out))
    print(f"-> {path}.json, {path}.md", flush=True)


def meta_block(kept, excluded, nash, sb, boot_check_fail):
    """JSON block for the meta-game. nash_avg_payoff is (P sigma)_i, the bot's expected score margin against the
    equilibrium mixture: <= 0, and 0 on the support ("Nash averaging", Balduzzi, Tuyls, Perolat & Graepel 2018,
    "Re-evaluating evaluation", NeurIPS)."""
    r4 = lambda x: round(float(x), 4)
    return dict(
        game="symmetric zero-sum, P[i, j] = score of i vs j - 0.5", bots=kept, excluded=excluded,
        value=r4(nash["value"]), sigma={m: r4(nash["sigma"][i]) for i, m in enumerate(kept)},
        support=[kept[i] for i in nash["support"]],
        nash_avg_payoff={m: r4(nash["payoff"][i]) for i, m in enumerate(kept)},
        exploitability=float(nash["exploitability"]), check_passed=nash["check"], uniqueness_note=nash["note"],
        dual_sigma={m: r4(nash["dual_sigma"][i]) for i, m in enumerate(kept)},
        sigma_boot_mean={m: r4(sb[:, i].mean()) for i, m in enumerate(kept)} if len(sb) else None,
        sigma_ci95={m: [r4(v) for v in np.percentile(sb[:, i], [2.5, 97.5])] for i, m in enumerate(kept)}
        if len(sb) else None,
        support_freq={m: r4(np.mean(sb[:, i] > 0.01)) for i, m in enumerate(kept)} if len(sb) else None,
        boot_check_failures=boot_check_fail,
    )


def markdown(o):
    lines = [f"# Gauntlet {o['prefix']}: Bradley-Terry ratings (Elo scale, {o['anchor']} = 0)", "",
             "| bot | rating | 95% CI | CPU-s / decision | games used | error games |", "|---|---|---|---|---|---|"]
    for m in sorted(o["bots"], key=lambda m: -o["ratings"][m]):
        c = o["ci95"][m]
        lines.append(f"| {m} | {o['ratings'][m]:.0f} | {c[0]:.0f} to {c[1]:.0f} | {o['cpu_s_per_decision'][m]} | "
                     f"{o['games_used'][m]} | {o['error_games'][m]} |")
    lines += ["", "| A | B | score of A | 95% CI | pairs | error games |", "|---|---|---|---|---|---|"]
    for p in o["pairwise"]:
        c = f"{p['ci95'][0]:.3f} to {p['ci95'][1]:.3f}" if p["ci95"] else "-"
        lines.append(f"| {p['a']} | {p['b']} | {p['score_a']} | {c} | {p['n_pairs']} | {p['error_games']} |")
    eq = o["foulplay_equivalent_ms"]
    if eq:
        lines += ["", "| bot | foul-play equivalent budget (ms per state) |", "|---|---|"]
        lines += [f"| {m} | {v['ms'] if v['ms'] is not None else v['note']} |" for m, v in eq.items()]
    g = o["meta_game"]
    lines += ["", f"Meta-game equilibrium (symmetric zero-sum, P = score - 0.5; value {g['value']:.4f}, "
                  f"exploitability {g['exploitability']:.1e}, check {'passed' if g['check_passed'] else 'FAILED'})", "",
              "| bot | sigma | bootstrap mean | 95% CI | P(sigma > 0.01) | Nash-averaged payoff |",
              "|---|---|---|---|---|---|"]
    for m in sorted(g["bots"], key=lambda m: (-g["sigma"][m], -g["nash_avg_payoff"][m])):
        ci_m = f"{g['sigma_ci95'][m][0]:.3f} to {g['sigma_ci95'][m][1]:.3f}" if g["sigma_ci95"] else "-"
        bm = f"{g['sigma_boot_mean'][m]:.3f}" if g["sigma_boot_mean"] else "-"
        sf = f"{g['support_freq'][m]:.3f}" if g["support_freq"] else "-"
        lines.append(f"| {m} | {g['sigma'][m]:.3f} | {bm} | {ci_m} | {sf} | {g['nash_avg_payoff'][m]:+.4f} |")
    if g["excluded"]:
        lines += ["", f"Left out of the meta-game (pairings without data): {', '.join(g['excluded'])}."]
    if g["uniqueness_note"]:
        lines += ["", f"Equilibrium {g['uniqueness_note']}; sigma is the one the LP returned."]
    if g["boot_check_failures"]:
        lines += ["", f"Exploitability check failed in {g['boot_check_failures']} bootstrap resamples."]
    lines += ["", f"Bootstrap: {o['bootstrap']} resamples of seeds within each pairing; L2 prior {o['l2_prior']} "
                  "on natural log-strengths."]
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bots", required=True, help="comma list of headless methods, e.g. random,maxpower,stock@25")
    ap.add_argument("--seeds-per-pair", type=int, default=20)
    ap.add_argument("--override", default="", help="per-pairing seed counts: 'a:b=N,...'")
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--prefix", default="G")
    ap.add_argument("--out-prefix", default=None, help="name the outputs <out-prefix>_ratings.* (default: --prefix)")
    ap.add_argument("--dry-run", action="store_true", help="print the batch commands and the game count only")
    ap.add_argument("--fit-only", action="store_true", help="skip running; fit what is on disk")
    ap.add_argument("--bootstrap", type=int, default=1000)
    ap.add_argument("--boot-seed", type=int, default=0)
    ap.add_argument("--anchor", default=None, help="bot rated 0 (default: random if present, else the first bot)")
    ap.add_argument("--l2", type=float, default=0.01, help="L2 prior on natural log-strengths (keeps sweeps finite)")
    args = ap.parse_args()
    bots = [m.strip() for m in args.bots.split(",") if m.strip()]
    for m in bots:
        try:
            method_spec(m)
        except argparse.ArgumentTypeError as e:
            sys.exit(str(e))
    if len(set(bots)) != len(bots) or len(bots) < 2:
        sys.exit("--bots: need at least two distinct bots")
    plan = pairings(bots, args.seeds_per_pair, parse_overrides(args.override))
    if args.dry_run:
        for a, b, n in plan:
            print(" ".join(batch_cmd(args.prefix, a, b, n, args.workers)))
        print(f"{len(plan)} pairings, {sum(2 * n for _, _, n in plan)} games (before resume skips)")
        return
    if not args.fit_only:
        run_all(plan, args.prefix, args.workers)
    fit(args, bots, plan)


if __name__ == "__main__":
    main()
