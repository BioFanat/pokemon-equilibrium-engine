"""Checks for search_rl_links.md section L4 (one MCTS node = self-play; visit shares = average strategy).

(1) Proposition L4 identity  NashGap(xbar, ybar) = (Reg1 + Reg2)/N  on random selection sequences, with
    deterministic and Bernoulli rewards (regrets built from the mean matrix by explicit per-step sums).
(2) Decoupled UCB1 (DUCT) self-play at one node, a replica of poke-engine src/mcts.rs:
      ucb_i = total_i/visits_i + sqrt(2 ln N_node / visits_i),  unvisited -> +inf,
      strict '>' scan from f32::MIN -> first index wins ties,  side one credited r, side two 1 - r.
    Reward modes: deterministic r = A[a,b]; Bernoulli r ~ Bern(A[a,b]).  float64 by default, plus a float32
    replicate (the engine accumulates total_score and computes the UCB in f32) on the small families.
    Comparators: full-information regret matching (average of sigma_t), bandit regret matching with
    importance weighting (Lisy et al. style), Exp3 -- both bandit methods with decaying exploration.
    Final-move rules for side one: most-visited move, foul-play's 0.75-threshold rule (fp/search/main.py:16).
(3) Support-restricted gap lower bound (Consequence 4).
(4) Long single-game runs of the worst clean instances (pure-Python replica, also cross-checks the batch sim).

Run:  engine/.venv/bin/python theory/checks/lk4_duct.py           (all; ~10 min on 2 cores)
      engine/.venv/bin/python theory/checks/lk4_duct.py --quick   (N_max = 1e4 smoke run)
Raw arrays go to $LK4_CACHE (default: a temp dir); worst instances to theory/checks/lk4_duct_worst.json.
"""
import json
import math
import os
import sys
import tempfile
import time
from multiprocessing import Pool
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import numpy as np
from scipy.optimize import linprog

HERE = Path(__file__).resolve().parent
K = 10                                   # padded arm count
QUICK = "--quick" in sys.argv
N_MAX = 10_000 if QUICK else 1_000_000
CKPTS = [c for c in (1_000, 3_000, 10_000, 30_000, 100_000, 300_000, 1_000_000) if c <= N_MAX]
CACHE = Path(os.environ.get("LK4_CACHE", tempfile.gettempdir())) / ("lk4_quick" if QUICK else "lk4_full")
N_RANDOM = 200


# ----------------------------------------------------------------------------------------------- games
def rps(w_s, w_p, w_r):
    """Biased RPS on [-1,1] scale then mapped to [0,1]: rock beats scissors by w_r, paper beats rock by w_p,
    scissors beats paper by w_s."""
    M = np.array([[0.0, -w_p, w_r], [w_p, 0.0, -w_s], [-w_r, w_s, 0.0]])
    return (M / np.abs(M).max() + 1.0) / 2.0


def make_games():
    games = [dict(family="MP", name="matching_pennies", A=np.array([[1.0, 0.0], [0.0, 1.0]]))]
    for (ws, wp, wr) in [(1, 1, 1), (1, 1, 2), (1, 1, 3), (1, 2, 3), (1, 1, 5), (1, 1, 10), (2, 3, 7)]:
        games.append(dict(family="RPS", name=f"rps_{ws}_{wp}_{wr}", A=rps(ws, wp, wr)))
    games.append(dict(family="SYM2", name="sym2_dominant_0.9", A=np.array([[0.5, 0.9], [0.1, 0.5]])))
    rng = np.random.default_rng(99)
    for d in [g for g in games if g["family"] == "RPS"]:
        games.append(dict(family="RPSpert", name=d["name"] + "_pert",
                          A=np.clip(d["A"] + rng.uniform(-0.01, 0.01, (3, 3)), 0, 1)))
    for n in (3, 5, 10):
        rng = np.random.default_rng(1000 + n)
        for s in range(N_RANDOM):
            games.append(dict(family=f"rand{n}", name=f"rand{n}_{s}", A=rng.random((n, n))))
    return games


def solve(A):
    """val, x*, y* of max_x min_y x^T A y (A >= 0 not required)."""
    n, m = A.shape
    # max v s.t. A^T x >= v, sum x = 1
    c = np.r_[np.zeros(n), -1.0]
    r = linprog(c, A_ub=np.c_[-A.T, np.ones(m)], b_ub=np.zeros(m), A_eq=np.r_[np.ones(n), 0.0][None],
                b_eq=[1.0], bounds=[(0, None)] * n + [(None, None)], method="highs")
    s = linprog(np.r_[np.zeros(m), 1.0], A_ub=np.c_[A, -np.ones(n)], b_ub=np.zeros(n),
                A_eq=np.r_[np.ones(m), 0.0][None], b_eq=[1.0], bounds=[(0, None)] * m + [(None, None)],
                method="highs")
    return -r.fun, r.x[:n], s.x[:m]


def restricted_maximin(A, S):
    """max_{x in Delta(S)} min_b x^T A e_b."""
    v, x, _ = solve(A[list(S)])
    return v


def pad(games):
    G = len(games)
    Ap = np.zeros((G, K, K)); na = np.zeros(G, int); nb = np.zeros(G, int)
    for g, d in enumerate(games):
        n, m = d["A"].shape
        Ap[g, :n, :m] = d["A"]; na[g] = n; nb[g] = m
    return Ap, na, nb


def gap_side(A, x, y):
    """NashGap = max_a (A y)_a - min_b (x^T A)_b, and side-one exploitability parts."""
    return (A @ y).max() - (x @ A).min()


# ------------------------------------------------------------------------------------- batched DUCT
def run_duct(Ap, na, nb, mode, dtype, seed, n_max=None, ckpts=None):
    n_max = n_max or N_MAX
    ckpts = set(ckpts or CKPTS)
    G = Ap.shape[0]
    dt = np.dtype(dtype)
    A = Ap.astype(dt)
    Af = A.reshape(-1)
    realA = np.arange(K)[None, :] < na[:, None]
    realB = np.arange(K)[None, :] < nb[:, None]
    SA = np.where(realA, 0.0, -1e9).astype(dt); nA = np.where(realA, 0.0, 1.0).astype(dt)
    SB = np.where(realB, 0.0, -1e9).astype(dt); nB = np.where(realB, 0.0, 1.0).astype(dt)
    cntA = np.zeros((G, K), np.int64); cntB = np.zeros((G, K), np.int64)
    sum_mean = np.zeros(G); sum_real = np.zeros(G)
    N = np.zeros(G, dt)
    rng = np.random.default_rng(seed)
    gK = np.arange(G) * K; gKK = np.arange(G) * K * K
    out = dict(ck=[], cntA=[], cntB=[], sum_mean=[], sum_real=[], first=[])
    two = dt.type(2.0); one = dt.type(1.0)
    chunk = 4096; U = None
    for t in range(n_max):
        if t < K:  # some real arm may be unvisited: +inf, padded arms -inf (Rust has no padded arms)
            with np.errstate(divide="ignore", invalid="ignore"):
                lnN = np.log(np.maximum(N, one))
                uA = SA / nA + np.sqrt(two * lnN[:, None] / nA)
                uB = SB / nB + np.sqrt(two * lnN[:, None] / nB)
            uA = np.where(realA & (cntA == 0), np.inf, np.where(realA, uA, -np.inf))
            uB = np.where(realB & (cntB == 0), np.inf, np.where(realB, uB, -np.inf))
        else:
            lnN2 = two * np.log(N)
            uA = SA / nA + np.sqrt(lnN2[:, None] / nA)
            uB = SB / nB + np.sqrt(lnN2[:, None] / nB)
        a = uA.argmax(1); b = uB.argmax(1)      # numpy argmax = first maximiser = Rust strict '>' scan
        p = Af[gKK + a * K + b]
        if mode == "bern":
            if t % chunk == 0:
                U = rng.random((chunk, G))
            r = (U[t % chunk] < p).astype(dt)
        elif mode.startswith("gauss"):          # low-noise model of the engine's sigmoid leaf: clip(A + s*Z)
            if t % chunk == 0:
                U = rng.standard_normal((chunk, G))
            r = np.clip(p + float(mode[5:]) * U[t % chunk], 0, 1).astype(dt)
        else:
            r = p
        if t < 3:
            out["first"].append(np.c_[a, b].tolist())
        ia = gK + a; ib = gK + b
        SA.reshape(-1)[ia] += r; nA.reshape(-1)[ia] += one
        SB.reshape(-1)[ib] += one - r; nB.reshape(-1)[ib] += one
        cntA.reshape(-1)[ia] += 1; cntB.reshape(-1)[ib] += 1
        sum_mean += Ap.reshape(-1)[gKK + a * K + b]; sum_real += r
        N += one
        if t + 1 in ckpts:
            out["ck"].append(t + 1); out["cntA"].append(cntA.copy()); out["cntB"].append(cntB.copy())
            out["sum_mean"].append(sum_mean.copy()); out["sum_real"].append(sum_real.copy())
    return {k: np.array(v) for k, v in out.items()}


# ------------------------------------------------------------------------------- batched comparators
def _sample(p, u, nreal):
    c = np.cumsum(p, 1)
    return np.minimum((c < u[:, None]).sum(1), nreal - 1)


def run_rm_full(Ap, na, nb, seed):
    G = Ap.shape[0]
    realA = (np.arange(K)[None, :] < na[:, None]).astype(float)
    realB = (np.arange(K)[None, :] < nb[:, None]).astype(float)
    x = realA / na[:, None]; y = realB / nb[:, None]
    RA = np.zeros((G, K)); RB = np.zeros((G, K)); sx = np.zeros((G, K)); sy = np.zeros((G, K))
    out = dict(ck=[], avgA=[], avgB=[])
    for t in range(N_MAX):
        sx += x; sy += y
        uA = np.einsum("gab,gb->ga", Ap, y); uB = 1.0 - np.einsum("gab,ga->gb", Ap, x)
        RA += uA - (x * uA).sum(1, keepdims=True); RB += uB - (y * uB).sum(1, keepdims=True)
        pA = np.maximum(RA, 0) * realA; pB = np.maximum(RB, 0) * realB
        zA = pA.sum(1, keepdims=True); zB = pB.sum(1, keepdims=True)
        x = np.where(zA > 0, pA / np.where(zA > 0, zA, 1), realA / na[:, None])
        y = np.where(zB > 0, pB / np.where(zB > 0, zB, 1), realB / nb[:, None])
        if t + 1 in CKPTS:
            out["ck"].append(t + 1); out["avgA"].append(sx / (t + 1)); out["avgB"].append(sy / (t + 1))
    return {k: np.array(v) for k, v in out.items()}


def run_bandit(Ap, na, nb, algo, seed, mode="det"):
    """algo 'rm': sigma = (1-g) RM(R) + g/K, g_t = min(1, t^-1/3); R_i += x~_i - r, x~ = r/sigma on the
    chosen arm (Lisy et al. 2013 RM selection).  algo 'exp3': p = (1-g) softmax(eta Ghat) + g/K,
    g_t = min(1/2, sqrt(K ln K / t)), eta_t = g_t / K, Ghat += r/p on the chosen arm (Auer et al. 2002)."""
    G = Ap.shape[0]
    rng = np.random.default_rng(seed)
    realA = (np.arange(K)[None, :] < na[:, None]).astype(float)
    realB = (np.arange(K)[None, :] < nb[:, None]).astype(float)
    RA = np.zeros((G, K)); RB = np.zeros((G, K))
    cntA = np.zeros((G, K), np.int64); cntB = np.zeros((G, K), np.int64)
    sA = np.zeros((G, K)); sB = np.zeros((G, K)); sum_mean = np.zeros(G)
    gK = np.arange(G) * K; gKK = np.arange(G) * K * K; gi = np.arange(G)
    Af = Ap.reshape(-1)
    out = dict(ck=[], cntA=[], cntB=[], avgA=[], avgB=[], sum_mean=[])
    kA = na[:, None].astype(float); kB = nb[:, None].astype(float)
    chunk = 2048
    for t in range(N_MAX):
        if t % chunk == 0:
            U = rng.random((chunk, 3, G))
        tt = t + 1
        if algo == "rm":
            g = min(1.0, tt ** (-1 / 3))
            def strat(R, real, k):
                q = np.maximum(R, 0) * real; z = q.sum(1, keepdims=True)
                q = np.where(z > 0, q / np.where(z > 0, z, 1), real / k)
                return (1 - g) * q + g * real / k
        else:
            def strat(R, real, k):
                gg = np.minimum(0.5, np.sqrt(k * np.log(k) / tt)); eta = gg / k
                z = np.where(real > 0, eta * R, -np.inf); z = np.exp(z - z.max(1, keepdims=True))
                q = z / z.sum(1, keepdims=True)
                return (1 - gg) * q + gg * real / k
        pA = strat(RA, realA, kA); pB = strat(RB, realB, kB)
        a = _sample(pA, U[t % chunk, 0], na); b = _sample(pB, U[t % chunk, 1], nb)
        mean = Af[gKK + a * K + b]
        r = (U[t % chunk, 2] < mean).astype(float) if mode == "bern" else mean
        qa = pA[gi, a]; qb = pB[gi, b]
        if algo == "rm":
            RA -= r[:, None] * realA; RB -= (1 - r)[:, None] * realB
        RA.reshape(-1)[gK + a] += r / qa; RB.reshape(-1)[gK + b] += (1 - r) / qb
        cntA.reshape(-1)[gK + a] += 1; cntB.reshape(-1)[gK + b] += 1
        sA += pA; sB += pB; sum_mean += mean
        if tt in CKPTS:
            out["ck"].append(tt); out["cntA"].append(cntA.copy()); out["cntB"].append(cntB.copy())
            out["avgA"].append(sA / tt); out["avgB"].append(sB / tt); out["sum_mean"].append(sum_mean.copy())
    return {k: np.array(v) for k, v in out.items()}


# ------------------------------------------------------------------- pure-Python single-game replica
def duct_single(A, n_steps, ckpts, mode="det", seed=0, f32=False):
    """Line-by-line replica of mcts.rs maximize_ucb_for_side / ucb1 / backpropagate at one node."""
    n, m = A.shape
    rnd = np.random.default_rng(seed).random if mode == "bern" else None
    cast = (lambda v: float(np.float32(v))) if f32 else (lambda v: v)
    SA = [0.0] * n; vA = [0] * n; SB = [0.0] * m; vB = [0] * m
    Al = [[cast(v) for v in row] for row in A.tolist()]
    Nn = 0; res = []; sum_mean = 0.0; ub = None
    for t in range(n_steps):
        lnN = math.log(Nn) if Nn > 0 else 0.0
        if f32:
            lnN = cast(math.log(cast(Nn))) if Nn > 0 else 0.0
        best = -3.4e38; a = 0
        for i in range(n):
            if vA[i] == 0:
                u = math.inf
            elif f32:
                u = cast(cast(SA[i] / vA[i]) + cast(math.sqrt(cast(cast(2.0 * lnN) / vA[i]))))
            else:
                u = SA[i] / vA[i] + math.sqrt(2.0 * lnN / vA[i])
            if u > best:
                best = u; a = i
        best = -3.4e38; b = 0
        for j in range(m):
            if vB[j] == 0:
                u = math.inf
            elif f32:
                u = cast(cast(SB[j] / vB[j]) + cast(math.sqrt(cast(cast(2.0 * lnN) / vB[j]))))
            else:
                u = SB[j] / vB[j] + math.sqrt(2.0 * lnN / vB[j])
            if u > best:
                best = u; b = j
        p = Al[a][b]
        if mode == "bern":
            if t % 65536 == 0:
                ub = rnd(65536)
            r = 1.0 if ub[t % 65536] < p else 0.0
        else:
            r = p
        SA[a] = cast(SA[a] + r); vA[a] += 1
        SB[b] = cast(SB[b] + cast(1.0 - r)); vB[b] += 1
        sum_mean += A[a, b]; Nn += 1
        if Nn in ckpts:
            res.append((Nn, list(vA), list(vB), sum_mean))
    return res


# ------------------------------------------------------------------------------------------ claim 1
def claim1():
    rng = np.random.default_rng(7)
    worst = 0.0; worst_real = 0.0; worst_indiv = 0.0
    for trial in range(300):
        n, m = rng.integers(2, 8, 2)
        A = rng.random((n, m)); N = int(rng.integers(1, 3000))
        bias_a = rng.dirichlet(np.ones(n) * 0.3); bias_b = rng.dirichlet(np.ones(m) * 0.3)
        a_seq = rng.choice(n, N, p=bias_a); b_seq = rng.choice(m, N, p=bias_b)
        r_seq = (rng.random(N) < A[a_seq, b_seq]).astype(float)   # Bernoulli realised rewards
        cf1 = np.zeros(n); cf2 = np.zeros(m); joint = 0.0; joint_r = 0.0
        for i in range(N):                                      # explicit per-step sums
            cf1 += A[:, b_seq[i]]; cf2 += 1.0 - A[a_seq[i], :]
            joint += A[a_seq[i], b_seq[i]]; joint_r += r_seq[i]
        reg1 = cf1.max() - joint; reg2 = cf2.max() - (N - joint)
        x = np.bincount(a_seq, minlength=n) / N; y = np.bincount(b_seq, minlength=m) / N
        gap = gap_side(A, x, y)
        worst = max(worst, abs(gap - (reg1 + reg2) / N))
        # realised-reward joint term (comparators still from A): r cancels in the sum as well
        rr1 = cf1.max() - joint_r; rr2 = cf2.max() - (N - joint_r)
        worst_real = max(worst_real, abs(gap - (rr1 + rr2) / N))
        worst_indiv = max(worst_indiv, abs(rr1 - reg1) / N)
    return worst, worst_real, worst_indiv


# ------------------------------------------------------------------------------------------ claim 3
def claim3():
    rng = np.random.default_rng(11)
    viol = 0; checks = 0; min_slack = np.inf
    for trial in range(300):
        n = int(rng.integers(2, 8)); m = int(rng.integers(2, 8))
        A = rng.random((n, m)); val = solve(A)[0]
        k = int(rng.integers(1, n + 1)); S = sorted(rng.choice(n, k, replace=False))
        bound = val - restricted_maximin(A, S)
        for _ in range(50):
            x = np.zeros(n); x[S] = rng.dirichlet(np.ones(k) * rng.choice([0.2, 1, 5]))
            e = val - (x @ A).min()
            checks += 1; min_slack = min(min_slack, e - bound)
            viol += e < bound - 1e-9
        # attainment: the restricted-LP maximiser has exploitability equal to the bound
        v, xs, _ = solve(A[S]); x = np.zeros(n); x[S] = xs
        min_slack = min(min_slack, (val - (x @ A).min()) - bound)
    # randomised support: marginal of a 50/50 mixture of two single-move rules can beat both bounds
    mix_examples = 0
    for trial in range(300):
        A = rng.random((3, 3)); val = solve(A)[0]
        pure = [val - A[a].min() for a in range(3)]
        x = np.array([0.5, 0.5, 0.0])
        if val - (x @ A).min() < min(pure[0], pure[1]) - 1e-9:
            mix_examples += 1
    return checks, viol, min_slack, mix_examples


# ------------------------------------------------------------------------------------------ analysis
def fp_rule(shares):
    keep = shares >= shares.max() * 0.75
    x = np.where(keep, shares, 0.0)
    return x / x.sum()


def analyse(games, sol, res, kind):
    """Per game per checkpoint: gap of (x, y), regrets, side-one exploitabilities of the final rules."""
    rows = []
    ck = res["ck"]
    for g, d in enumerate(games):
        A = d["A"]; n, m = A.shape; val, xs, ys = sol[g]
        traj = []
        for c in range(len(ck)):
            N = ck[c]
            if kind == "avg":
                x = res["avgA"][c, g, :n]; y = res["avgB"][c, g, :m]
            else:
                x = res["cntA"][c, g, :n] / N; y = res["cntB"][c, g, :m] / N
            gap = gap_side(A, x, y)
            e = dict(N=int(N), gap=float(gap), expl1=float(val - (x @ A).min()))
            if "sum_mean" in res and kind != "avg":
                sm = res["sum_mean"][c, g]
                reg1 = N * (A @ y).max() - sm; reg2 = sm - N * (x @ A).min()
                e.update(reg1=float(reg1 / N), reg2=float(reg2 / N))
            if kind == "visits":
                top = int(np.argmax(x))
                e.update(expl_top=float(val - A[top].min()), expl_fp=float(val - (fp_rule(x) @ A).min()))
            traj.append(e)
        rows.append(traj)
    return rows


def pct(v, q):
    return float(np.percentile(v, q))


def main():
    t0 = time.time()
    CACHE.mkdir(parents=True, exist_ok=True)
    print(f"N_MAX={N_MAX}  checkpoints={CKPTS}  cache={CACHE}")

    w, wr, wi = claim1()
    print("\n=== Claim 1: Prop L4 identity on 300 random selection sequences (n,m in 2..7, N<=3000)")
    print(f"  max |gap - (Reg1+Reg2)/N|, mean-reward regrets          = {w:.2e}")
    print(f"  same with realised Bernoulli joint term (A comparators) = {wr:.2e}   (r cancels in the sum)")
    print(f"  but individual regrets differ by up to                 = {wi:.3f} (per step)")

    checks, viol, slack, mixex = claim3()
    print("\n=== Claim 3: support-restricted lower bound, 300 random matrices x 50 strategies on random S")
    print(f"  checks={checks}  violations={viol}  min slack (incl. restricted maximiser) = {slack:.2e}")
    print(f"  3x3 games where a 50/50 marginal over two single-move rules is LESS exploitable than either"
          f" pure move: {mixex}/300  (the bound applies to the marginal's support, not a per-run S)")

    games = make_games()
    sol = [solve(d["A"]) for d in games]
    Ap, na, nb = pad(games)
    fam = np.array([d["family"] for d in games])
    pure_eq = np.array([np.isclose(d["A"].min(1).max(), d["A"].max(0).min(), atol=1e-12) for d in games])

    # cross-check the batch simulator against the line-by-line single-game replica
    small = [0, 1, 3, 8, 9] + [int(np.where(fam == f)[0][0]) for f in ("SYM2", "RPSpert", "rand5", "rand10")]
    ref = run_duct(Ap[small], na[small], nb[small], "det", "float64", 0, n_max=2000, ckpts=[1000, 2000])
    for j, g in enumerate(small):
        sg = duct_single(games[g]["A"], 2000, {1000, 2000})
        for (Nn, vA, vB, _), c in zip(sg, range(2)):
            n, m = games[g]["A"].shape
            assert list(ref["cntA"][c, j, :n]) == vA and list(ref["cntB"][c, j, :m]) == vB, games[g]["name"]
    print(f"\n  batch DUCT == line-by-line replica on {len(small)} games to N=2000: OK;  first selections of rand10_0:",
          [tuple(ref["first"][t][-1]) for t in range(3)])

    f32_idx = np.where(np.isin(fam, ["MP", "SYM2", "RPS", "RPSpert", "rand3"]))[0]
    jobs = {
        "duct_det": (run_duct, (Ap, na, nb, "det", "float64", 1)),
        "duct_bern": (run_duct, (Ap, na, nb, "bern", "float64", 2)),
        "duct_det_f32": (run_duct, (Ap[f32_idx], na[f32_idx], nb[f32_idx], "det", "float32", 3)),
        "duct_gauss002": (run_duct, (Ap, na, nb, "gauss0.02", "float64", 7)),
        "duct_gauss01": (run_duct, (Ap, na, nb, "gauss0.1", "float64", 8)),
        "rm_full": (run_rm_full, (Ap, na, nb, 4)),
        "rm_bandit": (run_bandit, (Ap, na, nb, "rm", 5)),
        "exp3": (run_bandit, (Ap, na, nb, "exp3", 6)),
    }
    results = {}
    todo = {}
    for k, (f, a) in jobs.items():
        p = CACHE / f"{k}.npz"
        if p.exists():
            results[k] = dict(np.load(p))
        else:
            todo[k] = (f, a)
    if todo:
        with Pool(2) as pool:
            hs = {k: pool.apply_async(f, a) for k, (f, a) in todo.items()}
            for k, h in hs.items():
                results[k] = h.get(); np.savez(CACHE / f"{k}.npz", **results[k])
                print(f"  finished {k}  ({time.time() - t0:.0f}s)")

    an = {}
    an["DUCT det (visits)"] = (analyse(games, sol, results["duct_det"], "visits"), np.arange(len(games)))
    an["DUCT Bernoulli (visits)"] = (analyse(games, sol, results["duct_bern"], "visits"), np.arange(len(games)))
    an["DUCT det f32 (visits)"] = (analyse([games[i] for i in f32_idx], [sol[i] for i in f32_idx],
                                           results["duct_det_f32"], "visits"), f32_idx)
    an["DUCT gauss sd=0.02 (visits)"] = (analyse(games, sol, results["duct_gauss002"], "visits"), np.arange(len(games)))
    an["DUCT gauss sd=0.1 (visits)"] = (analyse(games, sol, results["duct_gauss01"], "visits"), np.arange(len(games)))
    an["RM full-info (avg sigma)"] = (analyse(games, sol, results["rm_full"], "avg"), np.arange(len(games)))
    an["RM bandit (visits)"] = (analyse(games, sol, results["rm_bandit"], "visits_c"), np.arange(len(games)))
    an["Exp3 (visits)"] = (analyse(games, sol, results["exp3"], "visits_c"), np.arange(len(games)))

    # identity on every DUCT checkpoint
    idmax = max(abs(e["gap"] - e["reg1"] - e["reg2"]) for rows, _ in [an["DUCT det (visits)"], an["DUCT Bernoulli (visits)"]]
                for tr in rows for e in tr)
    print(f"\n  identity on all DUCT checkpoints: max |gap - (Reg1+Reg2)/N| = {idmax:.2e}")

    fams = ["MP", "SYM2", "RPS", "RPSpert", "rand3", "rand5", "rand10"]
    print("\n=== Claim 2: NashGap of (x, y) vs N.  cells: median / p90 / frac>0.05 / frac>0.01 (count)")
    for meth, (rows, idx) in an.items():
        print(f"\n-- {meth}")
        ck = [e["N"] for e in rows[0]]
        print("  family   " + "".join(f"{'N=' + format(c, '.0e'):>30s}" for c in ck))
        for f in fams:
            sel = [i for i, g in enumerate(idx) if fam[g] == f]
            if not sel:
                continue
            line = f"  {f:8s} "
            for c in range(len(ck)):
                v = np.array([rows[i][c]["gap"] for i in sel])
                line += f"{np.median(v):8.4f}/{pct(v, 90):.4f}/{(v > .05).mean():.2f}/{(v > .01).mean():.2f}".rjust(30)
            print(line + f"  ({len(sel)})")

    # stratify random families by pure saddle vs mixed, DUCT det and Bernoulli at final N
    print("\n=== DUCT at final N, random families split by pure-saddle vs mixed equilibrium")
    for meth in ["DUCT det (visits)", "DUCT gauss sd=0.02 (visits)", "DUCT gauss sd=0.1 (visits)",
                 "DUCT Bernoulli (visits)", "RM bandit (visits)", "Exp3 (visits)"]:
        rows, idx = an[meth]
        for f in ["rand3", "rand5", "rand10"]:
            for lab, mask in [("saddle", True), ("mixed", False)]:
                sel = [i for i, g in enumerate(idx) if fam[g] == f and pure_eq[g] == mask]
                if not sel:
                    continue
                v = np.array([rows[i][-1]["gap"] for i in sel])
                print(f"  {meth:24s} {f:6s} {lab:6s} n={len(sel):3d}  median={np.median(v):.4f}  p90={pct(v, 90):.4f}"
                      f"  max={v.max():.4f}  >0.05:{(v > .05).mean():.2f}  >0.01:{(v > .01).mean():.2f}")

    # MP / RPS individual trajectories (DUCT det + Bern), plus per-side regret
    print("\n=== DUCT on MP / RPS: gap trajectory (det | Bernoulli) and per-side regrets/N at final N (det)")
    for g, d in enumerate(games):
        if fam[g] not in ("MP", "SYM2", "RPS", "RPSpert"):
            continue
        rd = an["DUCT det (visits)"][0][g]; rb = an["DUCT Bernoulli (visits)"][0][g]
        print(f"  {d['name']:16s} det: " + " ".join(f"{e['gap']:.4f}" for e in rd)
              + " | bern: " + " ".join(f"{e['gap']:.4f}" for e in rb)
              + f" | Reg1/N={rd[-1]['reg1']:+.4f} Reg2/N={rd[-1]['reg2']:+.4f}")
    f32rows, _ = an["DUCT det f32 (visits)"]
    print("  f32 replicate, final gap (MP, SYM2, RPS x7, RPSpert x7):",
          " ".join(f"{f32rows[i][-1]['gap']:.4f}" for i in range(16)))

    # per-side regret sign: linear negative regret while gap small?
    rows = an["DUCT det (visits)"][0]
    negs = [(games[g]["name"], rows[g][-1]["reg1"], rows[g][-1]["reg2"], rows[g][-1]["gap"]) for g in range(len(games))
            if min(rows[g][-1]["reg1"], rows[g][-1]["reg2"]) < -0.01]
    print(f"\n  DUCT det games with a per-side regret/N < -0.01 at final N: {len(negs)}"
          + ("; e.g. " + ", ".join(f"{n}(R1={a:+.3f},R2={b:+.3f},gap={c:.4f})" for n, a, b, c in negs[:4]) if negs else ""))

    # final-move rules
    print("\n=== Side-one exploitability val - min_b x^T A e_b at final N (median / p90 / max)")
    for meth in ["DUCT det (visits)", "DUCT Bernoulli (visits)"]:
        rows, idx = an[meth]
        for f in fams:
            sel = [i for i, g in enumerate(idx) if fam[g] == f]
            out = []
            for key in ["expl1", "expl_top", "expl_fp"]:
                v = np.array([rows[i][-1][key] for i in sel])
                out.append(f"{key}={np.median(v):.4f}/{pct(v, 90):.4f}/{v.max():.4f}")
            print(f"  {meth:24s} {f:7s} " + "  ".join(out))
    print("  reference floors from exact x*: ")
    for f in fams:
        sel = [g for g in range(len(games)) if fam[g] == f]
        fl = np.array([sol[g][0] - (fp_rule(np.clip(sol[g][1], 0, None)) @ games[g]["A"]).min() for g in sel])
        pg = np.array([sol[g][0] - games[g]["A"].min(1).max() for g in sel])
        print(f"    {f:7s} 0.75-rule(x*): median={np.median(fl):.4f} p90={pct(fl, 90):.4f} max={fl.max():.4f}"
              f" frac>0.01={(fl > .01).mean():.2f} | pure maximin gap g(A): median={np.median(pg):.4f}"
              f" p90={pct(pg, 90):.4f}")

    # worst instances
    worst = {}
    for meth in ["DUCT det (visits)", "DUCT Bernoulli (visits)"]:
        rows, idx = an[meth]
        for f in fams:
            sel = sorted([i for i, g in enumerate(idx) if fam[g] == f], key=lambda i: -rows[i][-1]["gap"])[:3]
            for i in sel:
                g = idx[i]; val, xs, ys = sol[g]; N = rows[i][-1]["N"]
                key = "duct_det" if "det" in meth else "duct_bern"
                worst.setdefault(meth, []).append(dict(
                    family=f, name=games[g]["name"], A=games[g]["A"].round(12).tolist(), val=val,
                    x_star=xs.round(6).tolist(), y_star=ys.round(6).tolist(), pure_saddle=bool(pure_eq[g]),
                    visit_shares_A=(results[key]["cntA"][-1, g, :na[g]] / N).round(6).tolist(),
                    visit_shares_B=(results[key]["cntB"][-1, g, :nb[g]] / N).round(6).tolist(),
                    trajectory=rows[i]))
    print("\n=== Worst DUCT instances per family at final N (gap)")
    for meth, lst in worst.items():
        print(f"  {meth}: " + ", ".join(f"{w['name']}={w['trajectory'][-1]['gap']:.4f}" for w in lst))

    long = long_runs(games, sol, an)
    worst["long_runs"] = long
    if not QUICK:
        (HERE / "lk4_duct_worst.json").write_text(json.dumps(worst, indent=1))
    print(f"\ntotal {time.time() - t0:.0f}s")


def _long_job(A, n_long, ck, mode, seed):
    res = duct_single(A, n_long, ck, mode=mode, seed=seed)
    traj = []
    for Nn, vA, vB, sm in res:
        x = np.array(vA) / Nn; y = np.array(vB) / Nn
        traj.append(dict(N=Nn, gap=float(gap_side(A, x, y)), reg1=float((A @ y).max() - sm / Nn),
                         reg2=float(sm / Nn - (x @ A).min()), x=x.round(6).tolist(), y=y.round(6).tolist()))
    return traj


def long_runs(games, sol, an):
    """Worst instances re-run single-game with the line-by-line replica (pure Python): det to 1e7 for the
    worst RPSpert / rand3 / rand5 instance, Bernoulli to 1e7 for the worst RPS and rand3 Bernoulli instance."""
    fam = [d["family"] for d in games]
    def worst_of(meth, f):
        rows, _ = an[meth]
        sel = [g for g in range(len(games)) if fam[g] == f]
        return max(sel, key=lambda g: rows[g][-1]["gap"])
    picks = [(worst_of("DUCT det (visits)", f), "det") for f in ["RPSpert", "rand3", "rand5"]]
    picks += [(worst_of("DUCT Bernoulli (visits)", f), "bern") for f in ["RPS", "rand3"]]
    n_long = 100_000 if QUICK else 10_000_000
    ck = {10 ** k for k in range(3, 8)} | {3 * 10 ** k for k in range(3, 7)}
    print(f"\n=== Long single-game DUCT runs (float64, line-by-line replica) to N={n_long:.0e}: NashGap")
    lc = CACHE / "long_runs.json"
    if lc.exists():
        out = json.loads(lc.read_text())
        for o in out:
            tr = o["trajectory"]
            print(f"  {o['name']:20s} {o['mode']:4s} " + " ".join(f"{e['N']:.0e}:{e['gap']:.4f}" for e in tr)
                  + f" | final x={tr[-1]['x']} y={tr[-1]['y']} x*={np.round(o['x_star'], 4).tolist()}"
                  f" Reg1/N={tr[-1]['reg1']:+.4f} Reg2/N={tr[-1]['reg2']:+.4f}  (cached)")
        return out
    with Pool(2) as pool:
        hs = [pool.apply_async(_long_job, (games[g]["A"], n_long, ck, mode, 17 + k))
              for k, (g, mode) in enumerate(picks)]
        out = []
        for (g, mode), h in zip(picks, hs):
            traj = h.get()
            print(f"  {games[g]['name']:20s} {mode:4s} " + " ".join(f"{e['N']:.0e}:{e['gap']:.4f}" for e in traj)
                  + f" | final x={traj[-1]['x']} y={traj[-1]['y']} x*={sol[g][1].round(4).tolist()}"
                  f" Reg1/N={traj[-1]['reg1']:+.4f} Reg2/N={traj[-1]['reg2']:+.4f}")
            out.append(dict(name=games[g]["name"], mode=mode, A=games[g]["A"].tolist(), val=sol[g][0],
                            x_star=sol[g][1].tolist(), y_star=sol[g][2].tolist(), trajectory=traj))
    lc.write_text(json.dumps(out))
    return out


if __name__ == "__main__":
    main()
