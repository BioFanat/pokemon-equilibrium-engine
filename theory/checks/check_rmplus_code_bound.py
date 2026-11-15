"""Closing the §4 gap: for the CODE's rm_plus (alternating, linear averaging over x_1..x_T, y_1..y_T)
  W*[u(a,ybar)-u(xbar,b)] = sum_t t r^x_{t+1}(a) + sum_t t r^y_t(b) + sum_t t [u(x_t,y_t)-u(x_t,y_{t-1})]
with the last (cross) term <= 0 termwise by the RM+ improvement lemma (Lean: rmplus_improvement),
so gap <= 2*Delta*(sqrt(n(T+1)) + sqrt(mT))/(T+1) <= 2*Delta*(sqrt n + sqrt m)/sqrt T.
Checks: (1) identity holds numerically, (2) each cross step <= 0, (3) gap <= the bound."""
import numpy as np
rng = np.random.default_rng(3)
worst_id, worst_step, worst_ratio = 0.0, -np.inf, 0.0
for rep in range(400):
    n, m = rng.integers(2, 9), rng.integers(2, 9); M = rng.uniform(-1, 1, (n, m)) * rng.choice([1, 0.1, 10]); D = np.ptp(M)
    T = int(rng.choice([1, 2, 5, 17, 60, 200]))
    Rx, Ry = np.zeros(n), np.zeros(m); x, y = np.full(n, 1/n), np.full(m, 1/m)
    xs, ys = [x], [y]
    for t in range(1, T + 2):                       # one extra row update (virtual) for r^x_{T+1}
        u = M @ y; Rx = np.maximum(Rx + u - x @ u, 0); x = Rx / Rx.sum() if Rx.sum() > 0 else np.full(n, 1/n)
        w = -(x @ M); Ry = np.maximum(Ry + w - y @ w, 0); y = Ry / Ry.sum() if Ry.sum() > 0 else np.full(m, 1/m)
        xs.append(x); ys.append(y)
    W = T * (T + 1) / 2
    xb = sum(t * xs[t] for t in range(1, T + 1)) / W; yb = sum(t * ys[t] for t in range(1, T + 1)) / W
    gap = (M @ yb).max() - (xb @ M).min()
    for a in range(n):
        for b in range(m):
            lhs = W * (M[a] @ yb - xb @ M[:, b])
            rx = sum(t * (M[a] @ ys[t] - xs[t] @ M @ ys[t]) for t in range(1, T + 1))
            ry = sum(t * (xs[t] @ M @ ys[t - 1] - xs[t] @ M[:, b]) for t in range(1, T + 1))
            cr = sum(t * (xs[t] @ M @ ys[t] - xs[t] @ M @ ys[t - 1]) for t in range(1, T + 1))
            worst_id = max(worst_id, abs(lhs - (rx + ry + cr)))
    worst_step = max(worst_step, max((xs[t] @ M @ ys[t] - xs[t] @ M @ ys[t - 1]) / D for t in range(1, T + 1)))
    bound = 2 * D * (np.sqrt(n * (T + 1)) + np.sqrt(m * T)) / (T + 1)
    worst_ratio = max(worst_ratio, gap / bound)
print(f"identity max abs error = {worst_id:.2e}")
print(f"max cross step u(x_t,y_t)-u(x_t,y_(t-1)) / Delta = {worst_step:.2e}  (must be <= 0)")
print(f"max gap / [2 Delta (sqrt(n(T+1))+sqrt(mT))/(T+1)] = {worst_ratio:.4f}  (must be <= 1)")
