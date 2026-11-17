"""No-regret MCTS variants: sampled native shares, Hedge over MCTS/LP experts, and SM-MCTS with regret matching."""
from contextlib import ExitStack
import json
import os
from pathlib import Path
import random
import sys
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

import numpy as np
import poke_engine as pe

ROOT = Path(__file__).resolve().parents[2]
FP = Path(os.environ.get("FOULPLAY_DIR", ROOT / ".local-play/foul-play"))
sys.path[:0] = [str(ROOT / "engine"), str(ROOT / "engine/showdown"), str(FP)]
import hidden_agents as ha
import pkbattle as pb
import noregret_methods as nr
import smmcts_rm


def result(**visits):
    return NS(side_one=[NS(move_choice=a, visits=n, total_score=n*.5) for a, n in visits.items()],
              total_visits=sum(visits.values()))


def state():
    p = pe.Pokemon(id='gurdurr', hp=300, maxhp=300, types=('fighting', 'typeless'),
                   terastallized=True, moves=[pe.Move(id='drainpunch'), pe.Move(id='knockoff')])
    q = pe.Pokemon(id='sableye', hp=300, maxhp=300, types=('ghost', 'dark'),
                   terastallized=True, moves=[pe.Move(id='splash')])
    return pe.State(side_one=pe.Side(pokemon=[p]), side_two=pe.Side(pokemon=[q]))


def battle(states, tag='noregret-test'):
    return NS(team_preview=False, turn=1, battle_tag=tag, _ourlines=[], opponent=NS(name='p2'),
              mode=NS(prepare_battles=lambda b, k: [(s, 1/len(states)) for s in states]))


class StubRNG:
    def __init__(self, index=0):
        self.index, self.weights = index, []

    def choices(self, population, weights):
        self.weights.append(list(weights))
        return [list(population)[self.index]]


ENV = dict(OURBOT_BELIEF_SOURCE='foulplay_baseline', OURBOT_WORLD_SAMPLER='baseline', OURBOT_MCTS_MS='100')


class SampledNativeTests(unittest.TestCase):
    def test_samples_from_shares_and_logs_them_as_pfull(self):
        import run_ourbot as ro
        legal = pb.options(state())[0]
        visits = {legal[0]: 30, legal[1]: 70}
        rng = StubRNG(1)
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, ENV), \
                patch.object(pe, 'monte_carlo_tree_search', return_value=result(**visits)) as search, \
                patch.object(ro, 'battle_to_poke_engine_state', lambda s: s), \
                patch.object(ro, 'request_options', lambda b: None), \
                patch.object(ro, 'raw_action', lambda label, b: label), patch.object(ro, 'RNG', rng):
            log = Path(tmp)/'d.jsonl'
            move = ro.make_move_fn('mcts_native_sampled', 4, log=str(log))(battle([state()]*4))
            self.assertEqual(move, legal[1])
            self.assertEqual([c.kwargs['duration_ms'] for c in search.call_args_list], [25]*4)
            rec = json.loads(log.read_text())
        np.testing.assert_allclose(rng.weights[0], [.3, .7])
        np.testing.assert_allclose(rec['pfull'], [.3, .7])
        self.assertEqual(rec['mcts']['policy'], 'weighted_root_visit_sample_v1')
        self.assertEqual(rec['method'], 'mcts_native_sampled')
        self.assertEqual(rec['mix'], {legal[1]: .7, legal[0]: .3})
        draws = [nr.sample([.3, .7], random.Random(s)) for s in range(4000)]
        self.assertAlmostEqual(np.mean(draws), .7, delta=.03)


class HedgeMCTSTests(unittest.TestCase):
    def setUp(self):
        self.legal = pb.options(state())[0]
        # Unaligned matrix: rows in reverse legal order, so alignment must be by label.
        self.aligned = np.array([[100., 0.], [0., 50.]])
        self.M, self.s1, self.s2 = self.aligned[::-1].copy(), self.legal[::-1], ['attack', 'wait']

    def run_decision(self, stack, tmp, state_path, tag='hedge-test'):
        import run_ourbot as ro
        import nash
        visits = {self.legal[0]: 20, self.legal[1]: 80}
        stack.enter_context(patch.dict(os.environ, dict(ENV, OURBOT_LOG_STATES='1')))
        stack.enter_context(patch.object(pe, 'monte_carlo_tree_search', return_value=result(**visits)))
        stack.enter_context(patch.object(nash, 'full_matrix', lambda st, side_one_options=None: (self.M, self.s1, self.s2)))
        stack.enter_context(patch.object(ro, 'RNG', StubRNG(0)))
        log = Path(tmp)/'d.jsonl'
        fn = nr.make_move_fn('hedge_mcts', 2, str(log), lambda s: s, lambda b: None, lambda a, b: a,
                             state_path=state_path)
        fn(battle([state(), state()], tag))
        return fn, log

    def test_experts_mixture_feedback_and_checkpoint(self):
        import run_ourbot as ro
        with tempfile.TemporaryDirectory() as tmp, ExitStack() as stack:
            path = str(Path(tmp)/'learning/hedge_mcts_state.json')
            _, log = self.run_decision(stack, tmp, path)
            rec = json.loads(log.read_text().splitlines()[0])
            x = ha.agent_form_lp([self.aligned, self.aligned], [.5, .5])[0]
            Q = np.stack([[0, 1], [.2, .8], x])
            np.testing.assert_allclose(rec['Q'], Q.round(4), atol=1e-4)
            self.assertEqual(list(rec['weights']), nr.HEDGE_MCTS_EXPERTS)
            np.testing.assert_allclose(rec['pfull'], Q.mean(0), atol=1e-4)
            self.assertEqual(rec['experts']['mcts_top'], self.legal[1])
            self.assertEqual(rec['mcts']['policy'], 'hedge_mcts_v1')
            self.assertEqual(rec['search']['matrices'], [self.M.tolist()]*2)
            self.assertEqual(rec['search']['rows'], [self.s1]*2)
            self.assertIsNone(rec['feedback'])
            self.assertEqual(json.load(open(path))['t'], 0)
            ro._FINISHERS.pop('hedge-test')(['|move|p2a: Foo|Attack|p1a: Bar', '|win|Winner'])
            fb = json.loads(log.read_text().splitlines()[-1])
            self.assertTrue(fb['final'])
            self.assertFalse(fb['feedback']['skipped'])
            self.assertEqual(fb['feedback']['b'], 'attack')
            saved = json.load(open(path))
            self.assertEqual((saved['t'], saved['semantics'], saved['experts']),
                             (1, 'hedge_mcts_v1', nr.HEDGE_MCTS_EXPERTS))
            expected = np.clip((Q @ self.aligned[:, 0] - self.aligned.min()) / 400, 0, 1)
            np.testing.assert_allclose(saved['cum'], expected)
            L, meta = nr.load_learner(path)
            self.assertEqual((L.t, meta['games'], meta['updates']), (1, 2, 1))

    def test_unseen_opponent_action_is_skipped(self):
        import run_ourbot as ro
        with tempfile.TemporaryDirectory() as tmp, ExitStack() as stack:
            path = str(Path(tmp)/'s.json')
            _, log = self.run_decision(stack, tmp, path, tag='hedge-skip')
            ro._FINISHERS.pop('hedge-skip')(['|move|p2a: Foo|Surf|p1a: Bar'])
            fb = json.loads(log.read_text().splitlines()[-1])['feedback']
            self.assertEqual((fb['skipped'], fb['reason']), (True, 'not_in_worlds'))
            self.assertEqual(json.load(open(path))['t'], 0)

    def test_reload_rejects_other_experts_or_semantics(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = str(Path(tmp)/'s.json')
            L, meta = nr.load_learner(path)
            nr.save_learner(path, L, meta)
            for key, value in (('semantics', 'other'), ('experts', ['top', 'sampled', 'safest']),
                               ('method', 'hedge')):
                d = json.load(open(path))
                d[key] = value
                bad = str(Path(tmp)/f'bad-{key}.json')
                json.dump(d, open(bad, 'w'))
                with self.assertRaises(ValueError):
                    nr.load_learner(bad)

    def test_in_memory_learner_without_state_path(self):
        with tempfile.TemporaryDirectory() as tmp, ExitStack() as stack:
            _, log = self.run_decision(stack, tmp, None, tag='hedge-mem')
            self.assertEqual(sorted(os.listdir(tmp)), ['d.jsonl'])
            import run_ourbot as ro
            ro._FINISHERS.pop('hedge-mem')(['|move|p2a: Foo|Wait|p1a: Bar'])
            self.assertFalse(json.loads(log.read_text().splitlines()[-1])['feedback']['skipped'])


class RegretMatchingNodeTests(unittest.TestCase):
    def test_matching_pennies_average_is_uniform(self):
        x, y = smmcts_rm.solve_matrix(np.array([[1., -1.], [-1., 1.]]), 20000, np.random.default_rng(0))
        np.testing.assert_allclose(x, [.5, .5], atol=.05)
        np.testing.assert_allclose(y, [.5, .5], atol=.05)

    def test_pure_saddle_concentrates(self):
        x, y = smmcts_rm.solve_matrix(np.array([[3., 5.], [1., 0.]]), 5000, np.random.default_rng(0))
        self.assertGreater(x[0], .9)
        self.assertGreater(y[0], .9)


class SMMCTSRMRuntimeTests(unittest.TestCase):
    def test_real_engine_search_returns_distribution_over_legal(self):
        st = pb.random_state(random.Random(3))
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, dict(ENV, OURBOT_MCTS_MS='50')):
            log = Path(tmp)/'d.jsonl'
            fn = nr.make_move_fn('smmcts_rm', 1, str(log), lambda s: s, lambda b: None, lambda a, b: a)
            move = fn(battle([st]))
            rec = json.loads(log.read_text())
        self.assertIn(move, rec['legal'])
        self.assertEqual(rec['legal'], pb.options(st)[0])
        self.assertAlmostEqual(sum(rec['pfull']), 1)
        self.assertTrue(all(p >= 0 for p in rec['pfull']))
        self.assertEqual(rec['mcts']['policy'], 'smmcts_rm_avg_sample_v1')
        self.assertGreater(rec['mcts']['searches'][0]['iterations'], 0)
        self.assertFalse(rec['mcts']['fallback'])


if __name__ == '__main__':
    unittest.main()
