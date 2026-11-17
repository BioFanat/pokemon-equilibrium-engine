"""B3: bandit vs full feedback at the per-turn decision, on REAL logged per-turn matrices
(scratchpad/brainstorm/data.pkl: 1629 decisions vs MCTS-2000, heuristic units; we use two-sided ones, normalised to [0,1]).
Solvers compared at MCTS-like budgets:
  full-info RM+ (alternating, linear averaging), 1000 iterations            -> engine's "sampled" expert
  Exp3 self-play, bandit feedback, N in {500, 2000}: one cell sampled per iteration, Bernoulli(A[i,j]) reward
  decoupled UCB1 self-play, same feedback (poke-engine's rule), N in {500, 2000}   [U's territory; small N only]
  OMWU last iterate vs Hedge time-average, full info, N = 1000 (Lecture 7: average vs last iterate)
Metrics: Nash gap of (xbar, ybar) in the normalised matrix (range 1); the pure gap of argmax(xbar) (foul-play plays the top move);
Exp3 realised-share gap (what a tree would use) vs mixed-average gap.
"""
import pickle, time, numpy as np
T0 = time.time()
D = '/private/tmp/claude-501/-Users-biofanat-Documents-classes-f26-cs535/2bbe670b-2f97-4d9a-bfad-e1139e0352c6/scratchpad/brainstorm/data.pkl'
recs = pickle.load(open(D, 'rb'))
mats = [np.asarray(r['M'], float) for r in recs if min(np.asarray(r['M']).shape) >= 2]
mats = [M for M in mats if np.all(np.isfinite(M)) and np.ptp(M) > 0]
rng = np.random.default_rng(0)
idx = rng.choice(len(mats), size=120, replace=False)
mats = [(mats[i] - mats[i].min()) / np.ptp(mats[i]) for i in idx]
print(f"{len(mats)} two-sided real matrices; shapes median {np.median([M.shape[0] for M in mats]):.0f}x{np.median([M.shape[1] for M in mats]):.0f}")

def gap(A, x, y): return (A @ y).max() - (x @ A).min()

def rm_plus(A, T):
    n, m = A.shape; Rx = np.zeros(n); Ry = np.zeros(m); xs = np.zeros(n); ys = np.zeros(m)
    x = np.ones(n) / n; y = np.ones(m) / m
    for t in range(T):
        u = A @ y; Rx = np.maximum(Rx + u - x @ u, 0); x = Rx / Rx.sum() if Rx.sum() > 0 else np.ones(n) / n
        v = x @ A; Ry = np.maximum(Ry - (v - v @ y), 0); y = Ry / Ry.sum() if Ry.sum() > 0 else np.ones(m) / m
        xs += (t + 1) * x; ys += (t + 1) * y
    return xs / xs.sum(), ys / ys.sum()

def exp3_selfplay(A, N, rng):
    n, m = A.shape
    eta_x = np.sqrt(np.log(n) / (N * n)); eta_y = np.sqrt(np.log(m) / (N * m))
    gx = min(1, np.sqrt(n * np.log(n) / N)); gy = min(1, np.sqrt(m * np.log(m) / N))
    Lx = np.zeros(n); Ly = np.zeros(m); xs = np.zeros(n); ys = np.zeros(m); cx = np.zeros(n); cy = np.zeros(m)
    for t in range(N):
        px = np.exp(-eta_x * (Lx - Lx.min())); px = (1 - gx) * px / px.sum() + gx / n
        py = np.exp(-eta_y * (Ly - Ly.min())); py = (1 - gy) * py / py.sum() + gy / m
        i = rng.choice(n, p=px); j = rng.choice(m, p=py)
        r = rng.random() < A[i, j]
        Lx[i] += (1 - r) / px[i]; Ly[j] += r / py[j]
        xs += px; ys += py; cx[i] += 1; cy[j] += 1
    return xs / N, ys / N, cx / N, cy / N

def ucb1_selfplay(A, N, rng):
    n, m = A.shape; cx = np.zeros(n); cy = np.zeros(m); sx = np.zeros(n); sy = np.zeros(m)
    for t in range(1, N + 1):
        ux = np.where(cx > 0, sx / np.maximum(cx, 1) + np.sqrt(2 * np.log(t) / np.maximum(cx, 1)), np.inf)
        uy = np.where(cy > 0, sy / np.maximum(cy, 1) + np.sqrt(2 * np.log(t) / np.maximum(cy, 1)), np.inf)
        i = int(np.argmax(ux)); j = int(np.argmax(uy))
        r = float(rng.random() < A[i, j])
        cx[i] += 1; sx[i] += r; cy[j] += 1; sy[j] += 1 - r
    return cx / N, cy / N

def hedge_omwu(A, N, optimistic):
    n, m = A.shape; eta = 0.1
    Lx = np.zeros(n); Ly = np.zeros(m); xs = np.zeros(n); ys = np.zeros(m); gx_prev = np.zeros(n); gy_prev = np.zeros(m)
    x = np.ones(n) / n; y = np.ones(m) / m
    for t in range(N):
        gx = A @ y; gy = x @ A                                   # row maximises, col minimises
        Lx += gx; Ly += gy
        px = Lx + (gx if optimistic else 0); py = Ly + (gy if optimistic else 0)
        x = np.exp(eta * (px - px.max())); x /= x.sum()
        y = np.exp(-eta * (py - py.min())); y /= y.sum()
        xs += x; ys += y
    return xs / N, ys / N, x, y

res = {k: [] for k in ['rm+1000', 'exp3-500 mix', 'exp3-500 shares', 'exp3-2000 mix', 'exp3-2000 shares', 'ucb1-500', 'ucb1-2000',
                       'hedge avg 1000', 'hedge last 1000', 'omwu avg 1000', 'omwu last 1000']}
pure = {k: [] for k in ['rm+1000', 'exp3-2000 shares', 'ucb1-2000']}
val_cache = {}
for A in mats:
    x, y = rm_plus(A, 1000); v = 0.5 * ((A @ y).max() + (x @ A).min()); val_cache[id(A)] = v
    res['rm+1000'].append(gap(A, x, y)); pure['rm+1000'].append(v - A[int(np.argmax(x))].min())
    for N in [500, 2000]:
        xm, ym, xc, yc = exp3_selfplay(A, N, rng)
        res[f'exp3-{N} mix'].append(gap(A, xm, ym)); res[f'exp3-{N} shares'].append(gap(A, xc, yc))
        if N == 2000: pure['exp3-2000 shares'].append(v - A[int(np.argmax(xc))].min())
        xc, yc = ucb1_selfplay(A, N, rng); res[f'ucb1-{N}'].append(gap(A, xc, yc))
        if N == 2000: pure['ucb1-2000'].append(v - A[int(np.argmax(xc))].min())
    xa, ya, xl, yl = hedge_omwu(A, 1000, False); res['hedge avg 1000'].append(gap(A, xa, ya)); res['hedge last 1000'].append(gap(A, xl, yl))
    xa, ya, xl, yl = hedge_omwu(A, 1000, True); res['omwu avg 1000'].append(gap(A, xa, ya)); res['omwu last 1000'].append(gap(A, xl, yl))
print(f"{'solver':>18} | {'median gap':>10} {'p90 gap':>8} {'frac>0.05':>9}   (Nash gap in the normalised stage matrix, range 1)")
for k, vals in res.items():
    vals = np.array(vals); print(f"{k:>18} | {np.median(vals):10.4f} {np.quantile(vals, .9):8.4f} {np.mean(vals > .05):9.2f}")
print("pure gap of the top move (val - min_b A[argmax xbar, b]) -- what a most-visited rule pays:")
for k, vals in pure.items():
    vals = np.array(vals); print(f"{k:>18} | median {np.median(vals):.4f}  p90 {np.quantile(vals, .9):.4f}")
print("Exp3 pseudo-regret bound for the gap at N=2000, k=9: 2*(sqrt(2 k ln k / N))*2 =", round(4 * np.sqrt(2 * 9 * np.log(9) / 2000), 3),
      "; full-info Hedge bound 2*sqrt(2 ln k / N) =", round(2 * np.sqrt(2 * np.log(9) / 1000), 3))
print(f"done in {time.time()-T0:.1f}s")
