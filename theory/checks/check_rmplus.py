"""Prop. 4 / Cor. A' (full_game.md §4): does the CODE's rm_plus (alternating, linear averaging,
engine/nash.py:47-61) obey  gap <= 2*Delta*(sqrt n + sqrt m)/sqrt T  ?  Also the simultaneous variant
(the scheme Prop. 4 actually covers), Lemma 3's ||Q_T||_2 <= Delta sqrt(nT) on the code's accumulators,
and the sign of the cross term in (4.1)."""
import sys, numpy as np
sys.path.insert(0, "/Users/biofanat/Documents/classes/f26/cs535/final/engine")
src = open("/Users/biofanat/Documents/classes/f26/cs535/final/engine/nash.py").read()
ns = {"np": np}; start = src.index("def rm_plus"); end = src.index("def omwu")
exec(src[start:end], ns); rm_plus_code = ns["rm_plus"]          # exact code, without importing poke_engine

def gap(M, x, y): return float((M @ y).max() - (x @ M).min())

def rm_sim(M, T):
    m, n = M.shape; Rx, Ry = np.zeros(m), np.zeros(n); x, y = np.full(m, 1/m), np.full(n, 1/n)
    sx, sy = np.zeros(m), np.zeros(n)
    for t in range(1, T + 1):
        sx += t * x; sy += t * y
        u, w = M @ y, -(x @ M)
        Rx = np.maximum(Rx + u - x @ u, 0); Ry = np.maximum(Ry + w - y @ w, 0)
        x = Rx / Rx.sum() if Rx.sum() > 0 else np.full(m, 1/m); y = Ry / Ry.sum() if Ry.sum() > 0 else np.full(n, 1/n)
    return sx / sx.sum(), sy / sy.sum()

def rm_alt_trace(M, T):
    """Code's iteration, also returning ||Rx||_2,||Ry||_2 and the uniform-average cross term of (4.1)."""
    m, n = M.shape; Rx, Ry = np.zeros(m), np.zeros(n); x, y = np.full(m, 1/m), np.full(n, 1/n)
    cross = 0.0; maxratio = 0.0; D = np.ptp(M)
    for t in range(1, T + 1):
        u = M @ y; xold = x
        Rx = np.maximum(Rx + u - x @ u, 0); x = Rx / Rx.sum() if Rx.sum() > 0 else np.full(m, 1/m)
        cross += xold @ M @ y - x @ M @ y          # u(x_{t-1},y_{t-1}) - u(x_t,y_{t-1})
        w = -(x @ M); Ry = np.maximum(Ry + w - y @ w, 0); y = Ry / Ry.sum() if Ry.sum() > 0 else np.full(n, 1/n)
        maxratio = max(maxratio, np.linalg.norm(Rx) / (D * np.sqrt(m * t)), np.linalg.norm(Ry) / (D * np.sqrt(n * t)))
    return cross / T, maxratio

rng = np.random.default_rng(0)
gens = {
    "uniform": lambda n, m: rng.uniform(-1, 1, (n, m)),
    "sign": lambda n, m: rng.choice([-1.0, 1.0], (n, m)),
    "lowrank": lambda n, m: np.outer(rng.normal(size=n), rng.normal(size=m)) + 0.1 * rng.normal(size=(n, m)),
    "rps-like": lambda n, m: (lambda A: A - A.T)(rng.normal(size=(n, n)))[:, :m] if m <= n else rng.uniform(-1, 1, (n, m)),
}
worst_code, worst_sim, worst_l3 = 0, 0, 0; cross_pos = 0; cross_max = -np.inf; cnt = 0
for name, gen in gens.items():
    wc = 0
    for rep in range(60):
        n, m = rng.integers(2, 10), rng.integers(2, 10)
        M = gen(n, m); D = np.ptp(M)
        if D == 0: continue
        for T in (1, 2, 3, 5, 10, 30, 100, 300):
            bound = 2 * D * (np.sqrt(n) + np.sqrt(m)) / np.sqrt(T)
            r = gap(M, *rm_plus_code(M, T)) / bound
            wc = max(wc, r); worst_code = max(worst_code, r)
            worst_sim = max(worst_sim, gap(M, *rm_sim(M, T)) / bound)
        c, l3 = rm_alt_trace(M, 200); worst_l3 = max(worst_l3, l3); cnt += 1
        cross_pos += c > 1e-12; cross_max = max(cross_max, c / D)
    print(f"{name:9s}: max gap/bound (code alternating) = {wc:.4f}")
print(f"overall max gap/bound: code(alternating, linear avg)={worst_code:.4f}; simultaneous linear avg={worst_sim:.4f}")
print(f"Lemma 3 on code accumulators: max ||R_t||_2/(Delta sqrt(dim t)) = {worst_l3:.4f} (must be <= 1)")
print(f"cross term (4.1), uniform avg, T=200: positive in {cross_pos}/{cnt} matrices; max cross/(Delta) = {cross_max:.2e}")

# adversarial-ish small search: maximise gap*sqrt(T)/(Delta(sqrt n+sqrt m)) for the code over 2x2 / 3x3 matrices
best = 0
for rep in range(3000):
    n = m = rng.integers(2, 4); M = rng.uniform(-1, 1, (n, m)); D = np.ptp(M)
    for T in (1, 2, 4, 8, 16):
        best = max(best, gap(M, *rm_plus_code(M, T)) / (2 * D * (np.sqrt(n) + np.sqrt(m)) / np.sqrt(T)))
print(f"random 2x2/3x3 search, T<=16: max gap/bound = {best:.4f}")
