"""Plain-assert tests for the hidden-information harness.  python test_hidden.py"""
from __future__ import annotations

import math
import random

import numpy as np
import poke_engine as pe

import hidden as hd
import hidden_agents as ha
import nash
import pkbattle as pb
import randbats as rb


STATS5 = ("attack", "defense", "special_attack", "special_defense", "speed")


def game_with_sets(rng):
    t1, t2 = rb.sample_team(rng), rb.sample_team(rng)
    side = lambda t: pe.Side(pokemon=[rb.to_pokemon(s) for s in t])
    return pe.State(side_one=side(t1), side_two=side(t2)), t1, t2


def test_collapse_reproduces_truth():
    """Belief collapsed to the true sets -> determinize == true state. Byte-identical with EXACT_HP;
    with Showdown-percent HP, equal except opponent HP within 1% of maxhp."""
    rng = random.Random(0)
    n_exact = n_pct = n_lock = n_statchg = 0
    for g in range(12):
        st, t1, t2 = game_with_sets(rng)
        tr = {1: hd.Tracker(1, st), 2: hd.Tracker(2, st)}
        for turn in range(40):
            for s, opp_sets in ((1, t2), (2, t1)):
                view = hd.make_view(st, tr[s])
                if view.opp_pending_move:
                    continue                      # deferred move is sampled, not copied
                fb = hd.FixedBelief(view, dict(enumerate(opp_sets)))
                hd.EXACT_HP = True
                d = hd.determinize(st, hd.make_view(st, tr[s]), fb, rng)
                hd.EXACT_HP = False
                # choice-lock `disabled` flags are derived from the hidden item (see hidden.py);
                # compare with them normalised and count the positions where they differ
                norm = lambda x: x.replace(";true;", ";false;")
                opp_t = st.side_two if s == 1 else st.side_one
                if any(tuple(getattr(p, f) for f in STATS5) != tuple(getattr(rb.to_pokemon(q), f) for f in STATS5)
                       for p, q in zip(opp_t.pokemon, opp_sets)):
                    n_statchg += 1                # stats changed in play (Power Split / Transform / forme): not public-derivable
                    continue
                assert norm(d.to_string()) == norm(st.to_string()), (g, turn, s)
                n_lock += d.to_string() != st.to_string()
                n_exact += 1
                d2 = hd.determinize(st, view, fb, rng)
                opp_t = st.side_two if s == 1 else st.side_one
                opp_d = d2.side_two if s == 1 else d2.side_one
                for a, b in zip(opp_t.pokemon, opp_d.pokemon):
                    assert abs(a.hp - b.hp) <= max(1, math.ceil(a.maxhp / 100)) and (a.hp > 0) == (b.hp > 0)
                n_pct += 1
            o1, o2 = pb.options(st)
            m1, m2 = rng.choice(o1), rng.choice(o2)
            br = pe.generate_instructions(st, m1, m2)
            pick = rng.choices(br, weights=[b.percentage for b in br])[0]
            new = st.apply_instructions(pick)
            for s in (1, 2):
                tr[s].update(st, m1, m2, pick, new)
            st = new
            if not pb.alive(st.side_one) or not pb.alive(st.side_two):
                break
    print(f"collapse: {n_exact} positions identical (exact HP; {n_lock} differ only in choice-lock flags), {n_pct} within 1% HP, {n_statchg} skipped (in-battle stat change) (percent view)")


def test_belief_weights():
    rng = random.Random(1)
    for sp in rng.sample(sorted(rb.sets()), 40):
        e = rb.sets()[sp]
        for _ in range(3):
            s = rb.sample_set(sp, rng)
            rev = frozenset(rng.sample([rb.to_id(m) for m in s.moves], rng.randrange(0, len(s.moves) + 1)))
            post = hd.role_posterior(sp, rev, s.tera.lower() if rng.random() < .5 else None)
            assert abs(sum(w for _, w in post) - 1) < 1e-12
            assert dict(post)[s.role] > 0                      # truth always consistent
            for r, w in post:
                pool = {rb.to_id(m) for m in e["roles"][r]["moves"]}
                if not rev <= pool:
                    assert w == 0
    # hand-computed case: a species with >= 2 roles, reveal a move in exactly one role's pool
    for sp, e in sorted(rb.sets().items()):
        roles = sorted(e["roles"])
        if len(roles) != 2:
            continue
        p0 = {rb.to_id(m) for m in e["roles"][roles[0]]["moves"]}
        p1 = {rb.to_id(m) for m in e["roles"][roles[1]]["moves"]}
        common = sorted(p0 & p1)
        if not common or len(p0) <= 4 or len(p1) <= 4:
            continue
        m = common[0]
        n0, n1 = len(p0), len(p1)
        w0, w1 = math.comb(n0 - 1, 3) / math.comb(n0, 4), math.comb(n1 - 1, 3) / math.comb(n1, 4)  # = 4/n
        post = dict(hd.role_posterior(sp, frozenset([m]), None))
        assert abs(post[roles[0]] - w0 / (w0 + w1)) < 1e-12 and abs(post[roles[0]] - n1 / (n0 + n1)) < 1e-12
        only0 = sorted(p0 - p1)
        if only0:
            assert dict(hd.role_posterior(sp, frozenset([only0[0]]), None))[roles[1]] == 0
        print(f"belief: hand case {sp}: P({roles[0]} | {m}) = {post[roles[0]]:.4f} = {n1}/({n0}+{n1})")
        break


def test_agent_form_lp():
    rng = np.random.default_rng(0)
    for _ in range(50):
        m = rng.integers(2, 5)
        Ms = [rng.uniform(0, 1, (m, rng.integers(2, 5))) for _ in range(2)]
        b = rng.dirichlet([1, 1])
        x, v = ha.agent_form_lp(Ms, b)
        assert abs(ha.f_b(Ms, b, x) - v) < 1e-7
        # brute force over a fine simplex grid
        best = -1e9
        N = 60 if m <= 3 else 24
        import itertools
        for c in itertools.product(range(N + 1), repeat=m - 1):
            if sum(c) <= N:
                xx = np.array(list(c) + [N - sum(c)]) / N
                best = max(best, ha.f_b(Ms, b, xx))
        assert best <= v + 1e-7 and v - best < 0.05, (v, best)
        # K = 1 equals nash.lp (compare values and guaranteed payoffs, not x)
        x1, v1 = ha.agent_form_lp([Ms[0]], [1.0])
        _, _, vlp = nash.lp(Ms[0])
        assert abs(v1 - vlp) < 1e-7 and abs((Ms[0].T @ x1).min() - vlp) < 1e-7
    # toy: blending loses, agent form picks the safe action
    M1 = np.array([[1.0, 1.0], [0.0, 0.0], [0.6, 0.6]]); M2 = np.array([[0.0, 0.0], [1.0, 1.0], [0.6, 0.6]])
    x, v = ha.agent_form_lp([M1, M2], [.5, .5])
    assert abs(v - 0.6) < 1e-9 and x[2] > 0.999
    print("agent-form LP: matches brute force; K=1 == nash.lp")


def mirror(st):
    return pe.State(side_one=st.side_two, side_two=st.side_one,
                    **{f: getattr(st, f) for f in hd.GF})


def test_perspective():
    """Side-1 agent in S and side-2 agent in mirror(S) see the same own-perspective matrix and,
    with the same rng seed, sample the same determinizations -> identical strategies."""
    # pure transform check on a synthetic matrix
    M = np.arange(6.0).reshape(2, 3)
    A, r, c = ha.own_matrix(M, ["a", "b"], ["x", "y", "z"], 2)
    assert r == ["x", "y", "z"] and c == ["a", "b"] and np.array_equal(A, -M.T)
    # engine level: own matrix of side 2 in mirror(S) == own matrix of side 1 in S, up to the
    # engine's maximin pruning (NaN cells, filled with 0 by nash.engine_matrix) and the row/column
    # order; compare cells computed in both, and the full pipeline on a fully-symmetric fake.
    rng = random.Random(3)
    st, t1, t2 = game_with_sets(rng)
    mi = mirror(st)
    tr1, tr2 = hd.Tracker(1, st), hd.Tracker(2, mi)
    v1, v2 = hd.make_view(st, tr1), hd.make_view(mi, tr2)
    for seed in range(3):
        d1 = hd.determinize(st, v1, hd.Belief(v1), random.Random(seed))
        d2 = hd.determinize(mi, v2, hd.Belief(v2), random.Random(seed))
        assert mirror(d1).to_string() == d2.to_string()
    legal1 = pb.options(st)[0]
    legal2 = pb.options(mi)[1]
    assert sorted(legal1) == sorted(legal2)
    real = nash.engine_matrix
    try:                                          # deterministic, exactly antisymmetric fake engine
        def fake(state, ms):
            o1, o2 = sorted(pb.options(state)[0]), sorted(pb.options(state)[1])
            h = lambda a, b: (hash((a, b)) % 1000) / 10.0
            M = np.array([[h(a, b) - h(b, a) for b in o2] for a in o1])
            return M, o1, o2
        nash.engine_matrix = fake
        for spec in ("oracle:0", "mostlikely:0", "pimc:3:0", "agentform:3:0"):
            p1 = ha.Agent(spec).strategy(st, v1, hd.Belief(v1), legal1, 1, random.Random(5))
            p2 = ha.Agent(spec).strategy(mi, v2, hd.Belief(v2), legal1, 2, random.Random(5))
            assert np.allclose(p1, p2, atol=1e-7), spec
    finally:
        nash.engine_matrix = real
    print("perspective: side-1 and mirrored side-2 strategies identical (4 agent kinds)")


def test_games():
    import hidden_experiments as hx
    rng = random.Random(11)
    res = []
    for g in range(20):
        w, turns, _ = hx.play_hidden(ha.Agent("agentform:2:20"), ha.Agent("mostlikely:20"), rng)
        res.append((w, turns))
    print("20 games agentform:2:20 vs mostlikely:20 ok:", res)


if __name__ == "__main__":
    test_belief_weights()
    test_agent_form_lp()
    test_perspective()
    test_collapse_reproduces_truth()
    test_games()
    print("ALL TESTS PASSED")
