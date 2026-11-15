"""Checks for features.py, value_net.py and nash.vnet_matrix / vnet_policy.
Run (after value_net.py has written results/value/vnet.npz): .venv/bin/python test_value.py"""
from __future__ import annotations

import random

import numpy as np
import poke_engine as pe

import features
import nash
import pkbattle as pb
import value_net
from test_matrix import positions


def test_encode(ps):
    for st in ps:
        x, xm = features.encode_pair(st)
        assert x.shape == xm.shape == (features.DIM,) and x.dtype == np.float32
        assert np.array_equal(xm, x[features.MIRROR_PERM])      # mirror is a column permutation
    assert features.encode_many(ps).shape == (len(ps), features.DIM)
    print(f"encode: shape ({features.DIM},) and mirror permutation ok on {len(ps)} positions")


def test_antisymmetry(ps):
    v = value_net.value_many(ps)
    vm = value_net.value_many([nash._mirror(st) for st in ps])
    err = np.abs(v + vm).max()
    assert err < 1e-5, err
    print(f"antisymmetry: max |V(s) + V(mirror s)| = {err:.1e}")


def test_matrix_and_policy(ps):
    rng = random.Random(0)
    pol = nash.vnet_policy(1000, "argmax"), nash.vnet_policy(1000, "sample")
    for st in ps:
        M, s1, s2 = nash.vnet_matrix(st)
        assert (s1, s2) == pb.options(st) and M.shape == (len(s1), len(s2))
        assert np.isfinite(M).all() and np.abs(M).max() <= 1 + 1e-5   # engine branch % are float32
        for p in pol:
            assert p(st, s1, 1, rng) in s1 and p(st, s2, 2, rng) in s2
    print(f"vnet_matrix shape/finite and vnet_policy legality: ok on {len(ps)} positions")


def near_terminal():
    """Garchomp vs a lone Toxapex at 1 HP: every branch of (earthquake, recover) KOs side two."""
    st = pe.State(side_one=pe.Side(pokemon=[pb.make_pokemon(n) for n in ("garchomp", "heatran")]),
                  side_two=pe.Side(pokemon=[pb.make_pokemon("toxapex", hp_frac=0.001)]))
    s1, s2 = pb.options(st)
    kids = [st.apply_instructions(br) for br in pe.generate_instructions(st, "earthquake", "recover")]
    assert all(pb.alive(k.side_two) == 0 and pb.alive(k.side_one) > 0 for k in kids)
    M, _, _ = nash.vnet_matrix(st)
    return M[s1.index("earthquake"), s2.index("recover")], M


def test_terminal():
    v, M = near_terminal()
    assert abs(v - 1.0) < 1e-5, v                 # branch percentages are float32, sum to 1 +- ~2e-8
    assert np.isfinite(M).all() and np.abs(M).max() <= 1 + 1e-5   # engine branch % are float32
    print("terminal: a cell whose every branch KOs side two's last mon is exactly +1")


if __name__ == "__main__":
    ps = positions(20, seed=3)
    test_encode(ps)
    test_antisymmetry(ps)
    test_matrix_and_policy(ps)
    test_terminal()
    print("all value tests passed")
