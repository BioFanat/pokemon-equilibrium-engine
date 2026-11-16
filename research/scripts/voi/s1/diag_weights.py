"""Diagnostic: how far apart are the A1 (live) and A2 (frozen) node weights inside multi-world groups, and how often do
the group strategies differ? Real decisions (seeds 501, 502). Run: OURBOT_DET_LEAF=1 python diag_weights.py"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "tests"))
from _common import decisions, np, ha
import info_planner as ip
from collections import defaultdict
tv, xdiff, vdiff, n = [], 0, [], 0
for seed in (501, 502):
    for state, side, view, legal, worlds in decisions(seed, 3, every=5, K=4):
        ex = ip.expand(worlds, side, legal, set(view.reveals), key_report=False)
        groups = defaultdict(list)
        for i, nd in enumerate(ex.nodes): groups[nd.key].append(i)
        for g, idx in groups.items():
            ks = [ex.nodes[i].world for i in idx]
            if len(set(ks)) < 2: continue
            n += 1
            p = np.array([ex.nodes[i].p for i in idx]); pk = defaultdict(float)
            for k, pi in zip(ks, p): pk[k] += pi
            pi1 = np.array([.25 * pi for pi in p]); pi1 /= pi1.sum()
            pi2 = np.array([.25 * pi / pk[k] for k, pi in zip(ks, p)]); pi2 /= pi2.sum()
            tv.append(0.5 * np.abs(pi1 - pi2).sum())
            rows = []
            for i in idx: rows += [r for r in ex.nodes[i].rows if r not in rows]
            Ms = [ha.align(ex.nodes[i].M, ex.nodes[i].rows, rows) for i in idx]
            x1 = ha.agent_form_lp(Ms, pi1)[0]; x2 = ha.agent_form_lp(Ms, pi2)[0]
            v1 = sum(w * (M.T @ x1).min() for w, M in zip(pi1, Ms)); v2 = sum(w * (M.T @ x2).min() for w, M in zip(pi1, Ms))
            xdiff += 0.5 * np.abs(x1 - x2).sum() > 1e-6; vdiff.append(v1 - v2)
tv = np.array(tv); vd = np.array(vdiff)
print(f"multi-world groups {n}: TV(pi1,pi2) mean {tv.mean():.3f} median {np.median(tv):.3f} frac>0.01 {np.mean(tv>0.01):.2f} max {tv.max():.3f}")
print(f"x1 != x2 in {xdiff}/{n}; V1-V2: frac>1e-9 {np.mean(vd>1e-9):.3f}, max {vd.max():.4f}, min {vd.min():.2e}")
