"""Cor R3: back out the cumulative surrogate reward gap G_t = C_N(t) - C_F(t) (normalised units) from the
FTRL identity ln(w_N/w_F) = eta_t G_t, eta_t = sqrt(8 ln 2 / t), t = t_learner, using the end-of-game
weights in engine/results/experiments/hedge_stream_seed*.jsonl. Reads data only.
Run: engine/.venv/bin/python theory/checks/rt3_weight_dynamics.py"""
import json
from pathlib import Path

import numpy as np

D = Path(__file__).resolve().parents[2] / "engine/results/experiments"
F, N = "Foul-play MCTS", "Equilibrium, sampled"
E8 = 8 * np.log(2)


def load(seed, reset):
    p = D / f"hedge_stream_seed{seed}{'_reset' if reset else ''}.jsonl"
    return [json.loads(l) for l in p.read_text().splitlines()]


def G_of(r):
    w, t = r["weights"][F], r["t_learner"]
    return np.log((1 - w) / w) / np.sqrt(E8 / t)


def wF(G, t):
    return 1 / (1 + np.exp(np.sqrt(E8 / t) * G))


def mci(x):
    x = np.asarray(x, float); return x.mean(), 1.96 * x.std(ddof=1) / np.sqrt(len(x))


if __name__ == "__main__":
    seeds = range(4)
    print("== persistent stream: G at end of each phase, per-phase slope d (per learner turn) ==")
    inc = {"mcts": [], "exploiter": []}     # per-game (d_game, dt, mean_pF, seed, game)
    fits = {}
    for s in seeds:
        R = load(s, False)
        t = np.array([r["t_learner"] for r in R], float); G = np.array([G_of(r) for r in R])
        ph = np.array([r["opponent"] for r in R]); i1 = np.where(ph == "mcts")[0][-1]
        d1 = (t[:i1 + 1] @ G[:i1 + 1]) / (t[:i1 + 1] @ t[:i1 + 1])          # LS through origin
        tt, gg = t[i1 + 1:] - t[i1], G[i1 + 1:] - G[i1]
        d2 = (tt @ gg) / (tt @ tt)
        fits[s] = (d1, d2, t[i1], G[i1])
        r1 = 1 - ((G[:i1 + 1] - d1 * t[:i1 + 1]) ** 2).sum() / ((G[:i1 + 1] - G[:i1 + 1].mean()) ** 2).sum()
        r2 = 1 - ((gg - d2 * tt) ** 2).sum() / ((gg - gg.mean()) ** 2).sum()
        tprev, Gprev = np.r_[0, t[:-1]], np.r_[0, G[:-1]]
        for j, r in enumerate(R):
            dt = t[j] - tprev[j]
            if dt > 0:
                inc[r["opponent"]].append(((G[j] - Gprev[j]) / dt, dt, r["mean_pF"], s, j))
        pred = np.array([wF(d1 * t[j], t[j]) if j <= i1 else wF(G[i1] + d2 * (t[j] - t[i1]), t[j]) for j in range(len(R))])
        obs = np.array([r["weights"][F] for r in R])
        print(f"seed {s}: G(end ph1)={G[i1]:6.2f} at t={t[i1]:5.0f} | d1={d1:.4f} (R2 {r1:.3f}) d2={d2:.4f} (R2 {r2:.3f})"
              f" | d2/d1={d2 / d1:.2f} | w_F pred-obs RMS {np.sqrt(((pred - obs) ** 2).mean()):.4f}, max {np.abs(pred - obs).max():.4f}")
    D1 = np.array([fits[s][0] for s in seeds]); D2 = np.array([fits[s][1] for s in seeds])
    print(f"slopes over seeds: d1 = {D1.mean():.4f} ± sd {D1.std(ddof=1):.4f}; d2 = {D2.mean():.4f} ± sd {D2.std(ddof=1):.4f}"
          f"  (x400 = {400 * D1.mean():.1f} / {400 * D2.mean():.1f} heuristic units per turn)")
    print("\n== constancy within a phase: per-game slopes (G increment / turns), turn-weighted ==")
    for ph in ("mcts", "exploiter"):
        x = np.array(inc[ph]); d, w = x[:, 0], x[:, 1]
        m = (d * w).sum() / w.sum()
        # trend: weighted LS of per-game slope on game index within phase
        g = x[:, 4] - (0 if ph == "mcts" else 30)
        A = np.c_[np.ones_like(g), g]; W = np.diag(w)
        beta = np.linalg.solve(A.T @ W @ A, A.T @ W @ d)
        res = d - A @ beta; s2 = (w * res ** 2).sum() / (len(d) - 2)
        se = np.sqrt(s2 * np.linalg.inv(A.T @ W @ A)[1, 1])
        # first vs second half of phase
        h1, h2 = d[g < 15], d[g >= 15]; w1, w2 = w[g < 15], w[g >= 15]
        print(f"{ph:9s}: weighted mean {m:.4f}; per-game sd {d.std(ddof=1):.4f} (n={len(d)}); trend per game "
              f"{beta[1]:+.5f} ± {1.96 * se:.5f}; first-half {np.average(h1, weights=w1):.4f} vs second-half {np.average(h2, weights=w2):.4f}")
    print("\n== seed-mean w_F trajectory: logged vs constant-slope prediction (pooled d1, d2) ==")
    d1m, d2m = D1.mean(), D2.mean()
    for j in (0, 9, 19, 29, 39, 49, 59):
        rs = [load(s, False)[j] for s in seeds]
        obs = np.mean([r["weights"][F] for r in rs]); tj = [r["t_learner"] for r in rs]
        pr = []
        for s, r in zip(seeds, rs):
            t1 = fits[s][2]
            G = d1m * r["t_learner"] if j < 30 else d1m * t1 + d2m * (r["t_learner"] - t1)
            pr.append(wF(G, r["t_learner"]))
        print(f"game {j:2d}: mean t={np.mean(tj):6.0f}  logged w_F {obs:.3f}  predicted {np.mean(pr):.3f}")
    print("\n== reset stream: per-game slope G/t (learner restarts each game) ==")
    rr = {"mcts": [], "exploiter": []}
    for s in seeds:
        for r in load(s, True):
            rr[r["opponent"]].append((G_of(r) / r["t_learner"], r["t_learner"], r["mean_pF"], r["weights"][F]))
    for ph in ("mcts", "exploiter"):
        x = np.array(rr[ph]); m, h = mci(x[:, 0])
        print(f"{ph:9s}: mean slope {m:.4f} ± {h:.4f} (turn-weighted {np.average(x[:, 0], weights=x[:, 1]):.4f}); "
              f"mean p_F {x[:, 2].mean():.3f}; end w_F {x[:, 3].mean():.3f}; median turns {np.median(x[:, 1]):.0f}")
    print("\n== does the slope rise with the MCTS share? pooled per-game (persistent increments + reset games) ==")
    for ph in ("mcts", "exploiter"):
        x = np.r_[np.array(inc[ph])[:, [0, 1, 2]], np.array(rr[ph])[:, [0, 1, 2]]]
        d, w, p = x[:, 0], x[:, 1], x[:, 2]
        A = np.c_[np.ones_like(p), p]; W = np.diag(w)
        beta = np.linalg.solve(A.T @ W @ A, A.T @ W @ d)
        res = d - A @ beta; s2 = (w * res ** 2).sum() / (len(d) - 2)
        se = np.sqrt(s2 * np.linalg.inv(A.T @ W @ A)[1, 1])
        print(f"{ph:9s}: slope d = {beta[0]:+.4f} {beta[1]:+.4f} * p_F   (coef 95% ± {1.96 * se:.4f}, n={len(d)})")
    print("\n== surrogate deficit vs the equilibrium expert, per learner turn: w_F * d (normalised; x400 heuristic) ==")
    print("   (identity p.A[:,b] = q_N.A[:,b] - 400 w_F d_t; per game estimated as mean_pF * game slope)")
    for ph in ("mcts", "exploiter"):
        xp = np.array(inc[ph]); xr = np.array(rr[ph])
        dp = np.average(xp[:, 2] * xp[:, 0], weights=xp[:, 1]); dr = np.average(xr[:, 2] * xr[:, 0], weights=xr[:, 1])
        print(f"{ph:9s}: persistent {dp:.4f} ({400 * dp:.2f} units)  reset {dr:.4f} ({400 * dr:.2f} units)  ratio reset/persistent {dr / dp:.1f}")
    print("\n== paired score difference vs predicted deficit difference (exploiter phase, exploratory) ==")
    ds, dd = [], []
    for s in seeds:
        P, Q = load(s, False), load(s, True)
        tprev = np.r_[0, [r["t_learner"] for r in P[:-1]]]; Gp = np.r_[0, [G_of(r) for r in P[:-1]]]
        for j in range(30, 60):
            dtp = P[j]["t_learner"] - tprev[j]
            defp = P[j]["mean_pF"] * (G_of(P[j]) - Gp[j]) / max(dtp, 1)
            defr = Q[j]["mean_pF"] * G_of(Q[j]) / Q[j]["t_learner"]
            ds.append(P[j]["score"] - Q[j]["score"]); dd.append(defr - defp)
    ds, dd = np.array(ds), np.array(dd)
    print(f"mean score diff {ds.mean():+.3f}; mean per-turn deficit diff {dd.mean():+.4f}; corr {np.corrcoef(ds, dd)[0, 1]:+.3f} (n={len(ds)})")
    print("\n== Thm R2 regret bound per learner turn vs realised deficit (normalised units, k = 2) ==")
    b = lambda S: (3 * np.sqrt(S * np.log(2) / 8) + np.sqrt(np.log(2) / 8)) / S
    Sp = np.mean([load(s, False)[-1]["t_learner"] for s in seeds])
    tr = np.mean([r["t_learner"] for s in seeds for r in load(s, True) if r["opponent"] == "exploiter"])
    print(f"persistent: S = {Sp:.0f} turns over 60 games -> bound {b(Sp):.4f}/turn; reset: mean {tr:.1f} turns/game -> bound {b(tr):.4f}/turn")
