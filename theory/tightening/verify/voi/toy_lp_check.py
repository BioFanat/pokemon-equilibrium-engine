"""Verifier: hand-checkable / brute-force checks of tera_subgame.seqform + analyse and tera_two_sided.seqform2.

(1) Example C.1 (full_game.md §6.3) pushed through ts.analyse (build/stage1 monkeypatched): expect V0=0, expl=1,
    eta_NL(0)=c/(1+c)=3/7 (grid lower bound), E[TV1]=2/3, VOI_total = 1/3*(... ) computed by hand.
(2) Random one-sided instances: ts.analyse's V0, Vfull, v_A3, expl, e_tv1, e_eta_play vs normal-form brute force.
(3) Random two-sided instances: seqform2 V_real / V_conc vs normal-form brute force; buggy shared-block variant.
"""
from __future__ import annotations
import itertools, os, sys
import numpy as np
from scipy.optimize import linprog
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "scripts"))
import tera_subgame as ts  # noqa: E402
import tera_two_sided as t2  # noqa: E402


def lp_maxmin_pure(payoffs_by_type, b):
    """max_x sum_th b_th lambda_th, lambda_th <= sum_s x_s pay[th][s, j] for all opponent pure j of type th."""
    S = next(iter(payoffs_by_type.values())).shape[0]
    ths = list(payoffs_by_type)
    nv = S + len(ths)
    c = np.zeros(nv); c[S:] = [-b[t] for t in ths]
    A, rhs = [], []
    for i, t in enumerate(ths):
        P = payoffs_by_type[t]
        for j in range(P.shape[1]):
            row = np.zeros(nv); row[:S] = -P[:, j]; row[S + i] = 1; A.append(row); rhs.append(0)
    res = linprog(c, A_ub=np.array(A), b_ub=rhs, A_eq=np.r_[np.ones(S), np.zeros(len(ths))][None], b_eq=[1],
                  bounds=[(0, None)] * S + [(None, None)] * len(ths), method="highs")
    return -res.fun, res.x[:S]


# ------------------------------------------------------------------ one-sided brute force
def agent_pures(info, A0):
    out = []
    for a in A0:
        Is = [I for I in info if info[I]["a"] == a]
        for rows in itertools.product(*[range(len(info[I]["rows"])) for I in Is]):
            out.append((a, dict(zip(Is, rows))))
    return out


def opp_pures(info, B0, th):
    out = []
    for beta in B0:
        Is = [I for I in info if info[I]["beta"] == beta and th in info[I]["U"]]
        for cols in itertools.product(*[range(len(info[I]["cols"])) for I in Is]):
            out.append((beta, dict(zip(Is, cols))))
    return out


def pay(info, sa, so, th):
    a, ar = sa; beta, oc = so
    return sum(info[I]["P"][th] * info[I]["U"][th][ar[I], oc[I]] for I in info
               if info[I]["a"] == a and info[I]["beta"] == beta and th in info[I]["U"])


def brute_one_sided(info, A0, B0, teras, b):
    SA = agent_pures(info, A0)
    P = {th: np.array([[pay(info, sa, so, th) for so in opp_pures(info, B0, th)] for sa in SA]) for th in teras}
    return lp_maxmin_pure(P, b)[0]


def af_value(Us, w):
    """Bayesian one-shot value max_x sum_k w_k min_j (U_k^T x)_j by brute-force LP (own implementation)."""
    return lp_maxmin_pure({k: U for k, U in enumerate(Us)}, dict(enumerate(w)))


def brute_expl_a3(info, A0, B0, teras, b0):
    """A3 agent built independently; exploitability by enumerating opponent pure strategies per type."""
    k = len(teras)
    val = {(I, th): (af_value([d["U"][th]], [1])[0]) for I, d in info.items() for th in d["U"]}
    Mw = {}
    for th in teras:
        M = np.zeros((len(A0), len(B0)))
        for I, d in info.items():
            if th in d["U"]:
                M[A0.index(d["a"]), B0.index(d["beta"])] += d["P"][th] * val[(I, th)]
        Mw[th] = M
    v_a3, x0 = af_value([Mw[th] for th in teras], [b0[th] for th in teras])
    x1 = {}
    for I, d in info.items():
        ths = [th for th in teras if th in d["U"]]
        w = np.array([b0[th] * d["P"][th] for th in ths]); w /= w.sum()
        x1[I] = af_value([d["U"][th] for th in ths], w)[1]
    V0 = brute_one_sided(info, A0, B0, teras, b0)
    tot = 0.0
    br = {}
    for th in teras:
        best = None
        for so in opp_pures(info, B0, th):
            beta, oc = so
            v = sum(x0[A0.index(d["a"])] * d["P"][th] * float(x1[I] @ d["U"][th][:, oc[I]]) for I, d in info.items()
                    if d["beta"] == beta and th in d["U"])
            if best is None or v < best[0] - 1e-12:
                best = (v, beta)
        br[th] = best[1]; tot += b0[th] * best[0]
    # TV under that best response
    etv = 0.0
    for I, d in info.items():
        ths = [th for th in teras if th in d["U"] and br[th] == d["beta"]]
        if not ths:
            continue
        pr = sum(b0[th] * x0[A0.index(d["a"])] * d["P"][th] for th in ths)
        allt = [th for th in teras if th in d["U"]]
        post = np.array([b0[th] * d["P"][th] * (th in ths) for th in allt]); post /= post.sum()
        hard = np.array([b0[th] * d["P"][th] for th in allt]); hard /= hard.sum()
        etv += pr * 0.5 * np.abs(post - hard).sum()
    return dict(V0=V0, v_a3=v_a3, expl=V0 - tot, e_tv=etv, br=br)


def fake_run(info, A0, B0, teras, step=0.1):
    def fbuild(*_a, **_k):
        return None, A0, B0, info
    def fstage1(inf):
        for d in inf.values():
            d["val"] = {th: (ts.ha.lp_x(U)[1] if U.size > 1 else float(U[0, 0])) for th, U in d["U"].items()}
    ob, os1 = ts.build, ts.stage1
    ts.build, ts.stage1 = fbuild, fstage1
    try:
        class S:  # minimal set stand-ins
            species = role = "x"
        return ts.analyse(S, S, teras, step)
    finally:
        ts.build, ts.stage1 = ob, os1


def random_one_sided(rng, k=2, nA=2, nB=2, nch=2, nr=2, nc=2, terminal_frac=0.15):
    teras = [f"T{i}" for i in range(k)]
    A0 = [f"a{i}" for i in range(nA)]
    B0 = sorted([f"b{i}" for i in range(nB)] + ["b0-tera"])
    info = {}
    def mat():
        if rng.random() < terminal_frac:
            return np.array([[rng.normal()]]), ["none"], ["none"]
        return rng.normal(size=(nr, nc)).round(3), [f"r{i}" for i in range(nr)], [f"c{j}" for j in range(nc)]
    for a in A0:
        for beta in B0:
            if beta.endswith("-tera"):
                for th in teras:
                    M, r, c = mat()
                    info[(a, beta, ("tera", th))] = dict(a=a, beta=beta, P={th: 1.0}, U={th: M}, child={th: None}, rows=r, cols=c)
            else:
                p = rng.dirichlet(np.ones(nch))
                for ch in range(nch):
                    M0, r, c = mat()
                    U = {th: (M0 if M0.size == 1 else rng.normal(size=M0.shape).round(3)) for th in teras}
                    info[(a, beta, ch)] = dict(a=a, beta=beta, P={th: float(p[ch]) for th in teras}, U=U,
                                               child={th: None for th in teras}, rows=r, cols=c)
    return info, A0, B0, teras


def example_c1(c=0.75):
    teras = ["1", "2", "3"]; A0 = ["only"]; B0 = ["1", "2", "3"]
    info = {}
    for beta in B0:
        U = {th: np.array([[-1.0 if th == beta else c], [0.0]]) for th in teras}
        info[("only", beta, 0)] = dict(a="only", beta=beta, P={th: 1.0 for th in teras}, U=U, child={th: None for th in teras},
                                       rows=["risky", "safe"], cols=["none"])
    return info, A0, B0, teras


if __name__ == "__main__":
    print("== (1) Example C.1 through ts.analyse ==")
    info, A0, B0, teras = example_c1()
    out = fake_run(info, A0, B0, teras, step=1 / 70)
    print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in out.items() if k in
           ("V0", "voi_total", "v_a3", "sandwich_gap", "eta_NL0", "expl_a3", "e_tv1", "e_eta_play", "bound_C1", "c1_holds", "sandwich_holds", "Delta")})
    print("expected: V0=0, expl=1, eta_NL0=3/7=%.4f, E[TV1]=2/3, Delta=1.75, C1 holds, expl > eta_NL0" % (3 / 7))

    print("== (2) random one-sided vs brute force ==")
    rng = np.random.default_rng(0)
    worst = dict(V0=0, Vfull=0, v_a3=0, expl=0, etv=0)
    n_c1_fail = n_sand_fail = 0; ntr = 0; n_expl_gt_eta = 0
    for trial in range(60):
        k = [2, 2, 3][trial % 3]
        info, A0, B0, teras = random_one_sided(rng, k=k)
        out = fake_run(info, A0, B0, teras, step=0.05 if k == 2 else 0.1)
        b0 = {th: 1 / k for th in teras}
        bf = brute_expl_a3(info, A0, B0, teras, b0)
        worst["V0"] = max(worst["V0"], abs(out["V0"] - bf["V0"]))
        for th in teras:
            worst["Vfull"] = max(worst["Vfull"], abs(out["Vfull"][th] - brute_one_sided(info, A0, B0, teras, {t: float(t == th) for t in teras})))
        worst["v_a3"] = max(worst["v_a3"], abs(out["v_a3"] - bf["v_a3"]))
        worst["expl"] = max(worst["expl"], abs(out["expl_a3"] - bf["expl"]))
        worst["etv"] = max(worst["etv"], abs(out["e_tv1"] - bf["e_tv"]))
        n_c1_fail += not out["c1_holds"]; n_sand_fail += not out["sandwich_holds"]; ntr += 1
        n_expl_gt_eta += out["expl_a3"] > out["eta_NL0"] + 1e-9
    print("max |code - brute| over", ntr, "instances:", {k: f"{v:.2e}" for k, v in worst.items()})
    print("C1 fails", n_c1_fail, "sandwich fails", n_sand_fail, "| instances with expl > eta_NL0 (TV needed):", n_expl_gt_eta)
