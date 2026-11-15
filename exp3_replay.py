"""Offline EXP3 / Hedge over whole-battle experts, replayed from recorded games (no new battles).

Each round t = one game. The learner picks an expert (a bot); its reward is a recorded game score
(win 1, draw 1/2, loss 0) of that expert vs the round's opponent, drawn with replacement. Seeds in
E1fix / X1 differ across matchups, so per-battle counterfactuals don't exist: this is a bootstrap of a
stochastic bandit with i.i.d. rewards (valid for fixed opponents; it drops team-pairing correlation).

EXP3 (Auer et al. 2002): p = (1-γ) w/Σw + γ/k, unbiased estimate r̂_i = r/p_i for the played arm,
w_i ← w_i exp(γ r̂_i / k). With γ = min(1, sqrt(k ln k / ((e-1) N))), E[regret] ≤ 2 sqrt((e-1) N k ln k).
Hedge sees every expert's reward each round (full information): regret ≤ sqrt(N ln k / 2).
EXP3.S / fixed-share Hedge (Auer et al. 2002; Herbster & Warmuth 1998) also mix α = 1/N of the
weight back to uniform each round, so they track the best *sequence* of experts (shifting regret).
Run from final/engine: .venv/bin/python exp3_replay.py
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

RES = Path(__file__).resolve().parent / "results"
NAMES = {"nash:rm+:0:1000:argmax": "Equilibrium, top move", "nash:rm+:0:1000": "Equilibrium, sampled",
         "safest:0": "Safest move"}


def scores(exp: str, opp: str) -> dict[str, np.ndarray]:
    """opp = an opponent spec, or 'lbr' = each expert's own local-best-response exploiter (E6q)."""
    out = {}
    for line in (RES / "experiments" / f"{exp}.jsonl").read_text().splitlines():
        r = json.loads(line)
        hit = r["B"].startswith("lbr:") if opp == "lbr" else r["B"] == opp
        if hit and r["A"] in NAMES and r["score_A"] is not None:
            out.setdefault(NAMES[r["A"]], []).append(r["score_A"])
    return {k: np.array(v) for k, v in out.items()}


def run(schedule, experts, rng, algo):
    """schedule: list of per-round score pools {expert: array}. Returns (total reward, picks)."""
    k, N = len(experts), len(schedule)
    gamma = min(1.0, math.sqrt(k * math.log(k) / ((math.e - 1) * N)))
    eta = math.sqrt(8 * math.log(k) / N)
    alpha = 1.0 / N                               # one expected switch over the horizon
    logw = np.zeros(k)
    total, picks = 0.0, np.empty(N, int)
    for t, pools in enumerate(schedule):
        w = np.exp(logw - logw.max())
        if algo in ("exp3", "exp3s"):
            p = (1 - gamma) * w / w.sum() + gamma / k
            i = rng.choice(k, p=p)
            r = rng.choice(pools[experts[i]])
            logw[i] += gamma * (r / p[i]) / k
            if algo == "exp3s":                   # fixed share (EXP3.S): mix weights toward uniform
                w = np.exp(logw - logw.max()); w = w / w.sum()
                logw = np.log((1 - alpha) * w + alpha / k)
        elif algo == "fixedshare":
            p = w / w.sum()
            i = rng.choice(k, p=p)
            rs = np.array([rng.choice(pools[e]) for e in experts])
            r = rs[i]
            logw += eta * rs
            w = np.exp(logw - logw.max()); w = w / w.sum()
            logw = np.log((1 - alpha) * w + alpha / k)
        elif algo == "hedge":
            p = w / w.sum()
            i = rng.choice(k, p=p)
            rs = np.array([rng.choice(pools[e]) for e in experts])
            r = rs[i]
            logw += eta * rs
        else:                                     # uniform over experts
            i = rng.integers(k)
            r = rng.choice(pools[experts[i]])
        total += r
        picks[t] = i
    return total, picks


def scenario(name, schedule, experts, reps=2000, seed=0):
    N = len(schedule)
    means = np.array([[pools[e].mean() for e in experts] for pools in schedule])   # per-round expected score
    best_fixed = means.sum(0).max()
    rng = np.random.default_rng(seed)
    out = {"scenario": name, "N": N, "experts": experts,
           "expert_mean_score": dict(zip(experts, means.mean(0).round(3).tolist())),
           "best_fixed_mean": round(best_fixed / N, 3)}
    k = len(experts)
    for algo in ("exp3", "exp3s", "hedge", "fixedshare", "uniform"):
        tot, last_share = [], []
        for _ in range(reps):
            T, picks = run(schedule, experts, rng, algo)
            tot.append(T)
            last_share.append(np.bincount(picks[-N // 4:], minlength=k) / (N // 4))
        tot = np.array(tot)
        out[algo] = {"mean_score": round(tot.mean() / N, 3),
                     "regret_vs_best_fixed": round(best_fixed - tot.mean(), 1),
                     "regret_p90": round(np.quantile(best_fixed - tot, 0.9), 1),
                     "last_quarter_share": dict(zip(experts, np.mean(last_share, 0).round(2).tolist()))}
    out["exp3_bound"] = round(2 * math.sqrt((math.e - 1) * N * k * math.log(k)), 1)
    out["hedge_bound"] = round(math.sqrt(N * math.log(k) / 2), 1)
    return out


def main(only_new: bool = False):
    m2000 = scores("E1fix", "mcts_it:2000")
    lbr = scores("E6q", "lbr")
    m75 = scores("X1", "mcts_ms:75")
    three = list(NAMES.values())
    two = ["Equilibrium, top move", "Safest move"]
    new = [scenario("opponent switches: MCTS 2,000 it then exploiter (LBR), 3 experts",
                    [m2000] * 300 + [lbr] * 300, three, reps=1000),
           scenario("exploiter (LBR) only, 3 experts", [lbr] * 400, three, reps=1000)]
    res = new if only_new else new + [scenario("vs MCTS 2,000 iterations, 3 experts", [m2000] * 400, three),
           scenario("vs MCTS 2,000 iterations, 3 experts, long", [m2000] * 4000, three),
           scenario("opponent switches: MCTS 2,000 it then 75 ms", [m2000] * 200 + [m75] * 200, two)]
    out = RES / "exp3"
    out.mkdir(exist_ok=True)
    (out / "replay_vs_exploiter.json").write_text(json.dumps(new, indent=1))
    if not only_new:
        (out / "replay.json").write_text(json.dumps(res, indent=1))
    for r in res:
        print(f"\n== {r['scenario']} (N={r['N']}); expert mean scores {r['expert_mean_score']}; best fixed {r['best_fixed_mean']}")
        for a in ("exp3", "exp3s", "hedge", "fixedshare", "uniform"):
            x = r[a]
            print(f"  {a:7s} score {x['mean_score']:.3f}  regret {x['regret_vs_best_fixed']:6.1f} (p90 {x['regret_p90']:6.1f})"
                  f"  last-quarter picks {x['last_quarter_share']}")
        print(f"  bounds: EXP3 {r['exp3_bound']}, Hedge {r['hedge_bound']}")


if __name__ == "__main__":
    import sys
    main(only_new="--new" in sys.argv)
