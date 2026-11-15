"""Checks for nash.full_matrix (the unpruned depth-2 payoff matrix). Run: .venv/bin/python test_matrix.py"""
from __future__ import annotations

import random

import numpy as np
import poke_engine as pe

import nash
import pkbattle as pb
from test_hidden import mirror


def positions(n: int, seed: int, max_moves: int = 12):
    rng = random.Random(seed)
    out = []
    while len(out) < n:
        st = pb.random_state(rng)
        for _ in range(rng.randrange(0, max_moves)):
            o1, o2 = pb.options(st)
            st = pb.step(st, rng.choice(o1), rng.choice(o2), rng)
            if not pb.alive(st.side_one) or not pb.alive(st.side_two):
                break
        if pb.alive(st.side_one) and pb.alive(st.side_two):
            out.append(st)
    return out


def raw_id(st, ms):
    r = pe.id(st, ms)
    M = np.array([np.nan if v is None else v for v in r.matrix], float).reshape(len(r.s1), len(r.s2))
    return M, [pb._canon(a) for a in r.s1], [pb._canon(b) for b in r.s2], r.depth_searched


def test_shape_and_finite(ps):
    for st in ps:
        M, s1, s2 = nash.full_matrix(st)
        assert (s1, s2) == pb.options(st) and M.shape == (len(s1), len(s2))
        assert np.isfinite(M).all()
    print(f"shape/labels/finite: ok on {len(ps)} positions")


def test_matches_engine_unpruned_cells(ps):
    """Where pe.id reached exactly depth 2, its non-pruned cells are the same depth-2 values up to
    damage-roll branching (Python generate_instructions branches on damage, the search does not)."""
    errs, checked, nan_frac = [], 0, []
    for st in ps:
        M, s1, s2 = nash.full_matrix(st, leaf="maximin")     # the engine's own leaf rule
        for ms in (2, 5, 10):
            E, e1, e2, d = raw_id(st, ms)
            if d == 2:
                break
        if d != 2:
            continue
        E = E[np.ix_([e1.index(a) for a in s1], [e2.index(b) for b in s2])]   # align labels
        known = ~np.isnan(E)
        nan_frac.append(1 - known.mean())
        rng_ = np.ptp(M) or 1.0
        errs.append(np.abs(M - E)[known].max() / rng_)
        checked += 1
    errs = np.array(errs)
    print(f"vs engine depth-2 unpruned cells: {checked} positions, pruned share median {np.median(nan_frac):.2f}, "
          f"max cell error / payoff range median {np.median(errs):.3f}, max {errs.max():.3f}")
    assert checked >= 5 and np.median(errs) < 0.05


def test_legacy_is_the_old_bug(ps):
    st = ps[0]
    E, *_ = raw_id(st, 50)
    L, *_ = nash.engine_matrix(st, 50, legacy=True)
    assert np.isnan(E).any() and (L == 0).sum() >= np.isnan(E).sum()
    print("legacy=True reproduces NaN->0: ok")


def test_mirror_asymmetry(ps, leaf):
    """Swapping sides should give N = -M^T. Side-one-maximin leaves break this badly; midpoint
    leaves mostly restore it (residual: chance branching / move-order asymmetries in the engine)."""
    gaps = []
    for st in ps:
        M, s1, s2 = nash.full_matrix(st, leaf)
        N, m1, m2 = nash.full_matrix(mirror(st), leaf)
        N = N[np.ix_([m1.index(b) for b in s2], [m2.index(a) for a in s1])]
        gaps.append(np.abs(N + M.T).max() / (np.ptp(M) or 1.0))
    gaps = np.array(gaps)
    print(f"mirror antisymmetry gap, {leaf} leaves (max cell / range): median {np.median(gaps):.3f}, max {gaps.max():.3f}")
    return gaps


def test_solvers_and_policies(ps):
    gaps = []
    for st in ps:
        M, *_ = nash.full_matrix(st)
        x, y = nash.rm_plus(M, 1000)
        gaps.append(nash.exploitability(M, x, y) / (np.ptp(M) or 1.0))
    assert max(gaps) < 1e-2, max(gaps)
    rng = random.Random(0)
    for pol in (nash.nash_policy(pick="argmax"), nash.nash_policy(), nash.safest_policy()):
        for st in ps[:3]:
            o1, o2 = pb.options(st)
            assert pol(st, o1, 1, rng) in o1 and pol(st, o2, 2, rng) in o2
    print(f"RM+ on full matrix: Nash gap max {max(gaps):.1e}; policies return legal options")


def test_whole_games():
    """Every turn of a few full games (covers forced switches, near-terminal children)."""
    rng, turns = random.Random(7), 0
    for _ in range(3):
        st = pb.random_state(rng)
        while pb.alive(st.side_one) and pb.alive(st.side_two) and turns < 600:
            M, s1, s2 = nash.full_matrix(st)
            assert np.isfinite(M).all()
            st = pb.step(st, rng.choice(s1), rng.choice(s2), rng)
            turns += 1
    print(f"whole games: {turns} turns, all matrices finite")


if __name__ == "__main__":
    ps = positions(20, seed=11)
    test_shape_and_finite(ps)
    test_matches_engine_unpruned_cells(ps)
    test_legacy_is_the_old_bug(ps)
    g_max = test_mirror_asymmetry(ps[:10], "maximin")
    g_mid = test_mirror_asymmetry(ps[:10], "mid")
    assert np.median(g_mid) < 0.5 * np.median(g_max)
    test_solvers_and_policies(ps[:10])
    import time; t = time.perf_counter()
    for st in ps: nash.full_matrix(st)
    print(f"full_matrix (mid leaves): {1e3 * (time.perf_counter() - t) / len(ps):.0f} ms per position (mean)")
    test_whole_games()
    print("ALL MATRIX TESTS PASSED")
