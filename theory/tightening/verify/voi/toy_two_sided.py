"""Verifier: seqform2 (tera_two_sided.py) vs a normal-form brute force on random two-sided instances,
plus the buggy shared-stage-0-block variant described in voi.md §2.2.
Agent of own type phi chooses a pure strategy (a, row at each of its info sets (phi, I)); the opponent of type theta
knows theta, not phi; it picks (beta, col at each (theta, I)). Brute force: x_phi over phi's pure strategies,
max sum_th b lambda_th s.t. lambda_th <= sum_phi q_phi x_phi . pay(., y_th) for every pure y_th."""
from __future__ import annotations
import itertools, os, sys
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import lil_matrix
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "scripts"))
import tera_two_sided as t2  # noqa: E402


def random_two_sided(rng, nth=2, nph=2, nr=2, nc=2, nch=2):
    th_ = [f"T{i}" for i in range(nth)]; ph_ = [f"P{i}" for i in range(nph)]
    A0 = ["a0", "a0-tera", "a1"]; B0 = ["b0", "b0-tera", "b1"]
    info = {}
    jts = [(t, p) for t in th_ for p in ph_]
    for a in A0:
        for beta in B0:
            # public key reveals phi if a is tera, theta if beta is tera
            groups = {}
            for (t, p) in jts:
                key = (p if a.endswith("tera") else None, t if beta.endswith("tera") else None)
                groups.setdefault(key, []).append((t, p))
            for key, members in groups.items():
                pch = rng.dirichlet(np.ones(nch))
                for ch in range(nch):
                    I = (a, beta, key, ch)
                    info[I] = dict(a=a, beta=beta, P={jt: float(pch[ch]) for jt in members},
                                   U={jt: rng.normal(size=(nr, nc)).round(3) for jt in members},
                                   rows=[f"r{i}" for i in range(nr)], cols=[f"c{j}" for j in range(nc)])
    return info, A0, B0, th_, ph_


def brute(info, A0, B0, th_, ph_, b, q):
    def agent_pures(ph):
        out = []
        for a in A0:
            Is = [I for I in info if info[I]["a"] == a and any((t, ph) in info[I]["U"] for t in th_)]
            for rows in itertools.product(*[range(len(info[I]["rows"])) for I in Is]):
                out.append((a, dict(zip(Is, rows))))
        return out

    def opp_pures(th):
        out = []
        for beta in B0:
            Is = [I for I in info if info[I]["beta"] == beta and any((th, p) in info[I]["U"] for p in ph_)]
            for cols in itertools.product(*[range(len(info[I]["cols"])) for I in Is]):
                out.append((beta, dict(zip(Is, cols))))
        return out

    SA = {ph: agent_pures(ph) for ph in ph_}
    SO = {th: opp_pures(th) for th in th_}
    off = {}; n = 0
    for ph in ph_:
        off[ph] = n; n += len(SA[ph])
    nv = n + len(th_)
    c = np.zeros(nv); c[n:] = [-b[t] for t in th_]
    A, rhs = [], []
    for i, th in enumerate(th_):
        for (beta, oc) in SO[th]:
            row = np.zeros(nv); row[n + i] = 1
            for ph in ph_:
                if q[ph] == 0:
                    continue
                for s, (a, ar) in enumerate(SA[ph]):
                    v = sum(info[I]["P"][(th, ph)] * info[I]["U"][(th, ph)][ar[I], oc[I]] for I in info
                            if info[I]["a"] == a and info[I]["beta"] == beta and (th, ph) in info[I]["U"])
                    row[off[ph] + s] -= q[ph] * v
            A.append(row); rhs.append(0)
    Aeq = []
    for ph in ph_:
        r = np.zeros(nv); r[off[ph]:off[ph] + len(SA[ph])] = 1; Aeq.append(r)
    res = linprog(c, A_ub=np.array(A), b_ub=rhs, A_eq=np.array(Aeq), b_eq=np.ones(len(ph_)),
                  bounds=[(0, None)] * n + [(None, None)] * len(th_), method="highs")
    return -res.fun


def seqform2_shared(info, A0, B0, thetas, phis, b, q):
    """The pre-fix bug as described: ONE stage-0 block shared by all own types (idx_r keyed by a only)."""
    src = t2.seqform2
    keys = list(info); nA = len(A0)
    idx_r = {(ph, a): i for ph in phis for i, a in enumerate(A0)}
    off = nA
    idx_rI = {}
    for ph in phis:
        for I in keys:
            if any((th, ph) in info[I]["U"] for th in thetas):
                idx_rI[(ph, I)] = off; off += len(info[I]["rows"])
    n_nonneg = off
    idx_lam = {th: off + i for i, th in enumerate(thetas)}; off += len(thetas)
    idx_mu = {}
    for th in thetas:
        for I in keys:
            if any((th, ph) in info[I]["U"] for ph in phis):
                idx_mu[(th, I)] = off; off += 1
    nv = off
    c = np.zeros(nv)
    for th in thetas:
        c[idx_lam[th]] = -b[th]
    Aeq, beq = [], []
    row = np.zeros(nv); row[:nA] = 1; Aeq.append(row); beq.append(1.0)
    for (ph, I), j in idx_rI.items():
        row = np.zeros(nv); row[j:j + len(info[I]["rows"])] = 1; row[idx_r[(ph, info[I]["a"])]] = -1
        Aeq.append(row); beq.append(0.0)
    n_ub = len(thetas) * len(B0) + sum(len(info[I]["cols"]) for (th, I) in idx_mu)
    Aub = lil_matrix((n_ub, nv)); bub = np.zeros(n_ub); r = 0
    for th in thetas:
        for beta in B0:
            Aub[r, idx_lam[th]] = 1
            for I in keys:
                if info[I]["beta"] == beta and (th, I) in idx_mu:
                    Aub[r, idx_mu[(th, I)]] = -1
            r += 1
    for (th, I), j in idx_mu.items():
        d = info[I]
        for jb in range(len(d["cols"])):
            Aub[r, j] = 1
            for ph in phis:
                if (th, ph) in d["U"] and q[ph] > 0:
                    P, U = d["P"][(th, ph)], d["U"][(th, ph)]
                    for ia in range(len(d["rows"])):
                        Aub[r, idx_rI[(ph, I)] + ia] += -q[ph] * P * U[ia, jb]
            r += 1
    res = linprog(c, A_ub=Aub.tocsr(), b_ub=bub, A_eq=np.array(Aeq), b_eq=beq,
                  bounds=[(0, None)] * n_nonneg + [(None, None)] * (nv - n_nonneg), method="highs")
    return -res.fun


if __name__ == "__main__":
    rng = np.random.default_rng(1)
    worst = 0.0; worst_conc = 0.0; neg_conceal = 0; neg_conceal_bug = 0; bug_lower = 0; N = 25
    for trial in range(N):
        info, A0, B0, th_, ph_ = random_two_sided(rng, nth=2, nph=2 if trial % 2 else 3, nr=2, nc=2, nch=1)
        bu = {t: 1 / len(th_) for t in th_}; qu = {p: 1 / len(ph_) for p in ph_}
        vr = t2.seqform2(info, A0, B0, th_, ph_, bu, qu)
        vb = brute(info, A0, B0, th_, ph_, bu, qu)
        worst = max(worst, abs(vr - vb))
        vc = {}
        for ph in ph_:
            qq = {p: float(p == ph) for p in ph_}
            vc[ph] = t2.seqform2(info, A0, B0, th_, ph_, bu, qq)
            worst_conc = max(worst_conc, abs(vc[ph] - brute(info, A0, B0, th_, ph_, bu, qq)))
        conceal = vr - sum(qu[p] * vc[p] for p in ph_)
        neg_conceal += conceal < -1e-9
        vbug = seqform2_shared(info, A0, B0, th_, ph_, bu, qu)
        vcbug = sum(qu[p] * seqform2_shared(info, A0, B0, th_, ph_, bu, {pp: float(pp == p) for pp in ph_}) for p in ph_)
        bug_lower += vbug < vr - 1e-9
        neg_conceal_bug += (vbug - vcbug) < -1e-9
    print(f"{N} instances: max|seqform2 - brute| V_real {worst:.2e}, V_conc {worst_conc:.2e}; conceal<0: {neg_conceal}")
    print(f"shared-block (bug) variant: V_real below correct in {bug_lower}/{N}; conceal<0 in {neg_conceal_bug}/{N}")
