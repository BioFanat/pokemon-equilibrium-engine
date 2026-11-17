"""Native search policy: weighted votes, legality, budgets, and real engine smoke."""
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

import numpy as np
import poke_engine as pe

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT/'engine'), str(ROOT/'engine/showdown'), str(ROOT/'.local-play/foul-play')]
from native_mcts import aggregate_visits, make_move_fn
from run_public_evaluation import plan


def result(**visits):
    return NS(side_one=[NS(move_choice=a, visits=n, total_score=n*.5) for a,n in visits.items()],
              total_visits=sum(visits.values()))


class AggregateTests(unittest.TestCase):
    def test_world_weights_are_not_replaced_by_number_of_iterations(self):
        p, _, fallback = aggregate_visits([result(tackle=90,growl=10), result(tackle=1,growl=9)], [1,1], ['tackle','growl'])
        np.testing.assert_allclose(p,[.5,.5]); self.assertFalse(fallback)
        p,_,_ = aggregate_visits([result(tackle=90,growl=10), result(tackle=1,growl=9)], [3,1], ['tackle','growl'])
        np.testing.assert_allclose(p,[.7,.3])

    def test_server_illegal_action_cannot_win_and_discarded_mass_is_visible(self):
        p, worlds, fallback = aggregate_visits([result(forbidden=990,tackle=10)], [1], ['tackle','growl'])
        np.testing.assert_array_equal(p,[1,0]); self.assertFalse(fallback)
        self.assertAlmostEqual(worlds[0]['discarded_visit_fraction'],.99)

    def test_canonical_switch_names_and_missing_world_actions(self):
        p,_,_ = aggregate_visits([result(**{'switch heatran':90,'tackle':10}),result(tackle=10)], [1,1], ['heatran','tackle'])
        np.testing.assert_allclose(p,[.45,.55])

    def test_zero_legal_visits_has_explicit_legal_fallback(self):
        p, worlds, fallback = aggregate_visits([result(forbidden=10)], [1], ['growl','tackle'])
        np.testing.assert_array_equal(p,[1,0]); self.assertTrue(fallback)
        self.assertTrue(worlds[0]['no_legal_visits'])

    def test_world_and_root_order_invariance(self):
        a,b=result(tackle=7,growl=3),result(growl=8,tackle=2)
        p,_,_=aggregate_visits([a,b],[.3,.7],['tackle','growl'])
        q,_,_=aggregate_visits([b,a],[.7,.3],['tackle','growl'])
        np.testing.assert_allclose(p,q)


class RuntimeTests(unittest.TestCase):
    def state(self):
        p=pe.Pokemon(id='gurdurr',hp=300,maxhp=300,types=('fighting','typeless'),
                     terastallized=True,moves=[pe.Move(id='drainpunch'),pe.Move(id='knockoff')])
        q=pe.Pokemon(id='sableye',hp=300,maxhp=300,types=('ghost','dark'),
                     terastallized=True,moves=[pe.Move(id='splash')])
        return pe.State(side_one=pe.Side(pokemon=[p]),side_two=pe.Side(pokemon=[q]))

    def battle(self, states):
        return NS(team_preview=False,turn=1,battle_tag='native-test',
                  mode=NS(prepare_battles=lambda battle,k:[(s,1/len(states)) for s in states]))

    def test_budget_split_argmax_and_snapshot_do_not_build_payoff_matrices(self):
        import run_ourbot as ro
        import nash
        states=[self.state() for _ in range(4)]
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ,OURBOT_MCTS_MS='1001',OURBOT_LOG_STATES='1',OURBOT_BELIEF_SOURCE='foulplay_baseline'), \
             patch.object(pe,'monte_carlo_tree_search',return_value=result(drainpunch=10,knockoff=90)) as search, \
             patch.object(nash,'full_matrix',side_effect=AssertionError('MCTS must bypass the matrix')), \
             patch.object(ro,'battle_to_poke_engine_state',lambda s:s), \
             patch.object(ro,'request_options',lambda b:['knockoff','drainpunch']), \
             patch.object(ro,'raw_action',lambda label,b:label):
            log=Path(tmp)/'decisions.jsonl'
            move=ro.make_move_fn('mcts_native',4,log=str(log))(self.battle(states))
            self.assertEqual(move,'knockoff')
            self.assertEqual([c.kwargs['duration_ms'] for c in search.call_args_list],[251,250,250,250])
            self.assertTrue(all(c.kwargs['threads']==1 for c in search.call_args_list))
            record=json.loads(log.read_text())
            self.assertEqual(record['mcts']['policy'],'weighted_root_visit_argmax_v1')
            self.assertEqual(len(record['search']['states']),4)
            self.assertEqual(record['search']['matrices'],[])
            self.assertEqual(max(record['pfull']),1)
            self.assertEqual(record['mix'],{'knockoff':1})

    def test_forced_server_action_skips_native_search(self):
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ,OURBOT_BELIEF_SOURCE='foulplay_baseline'), \
             patch.object(pe,'monte_carlo_tree_search',side_effect=AssertionError('forced action')):
            log=Path(tmp)/'decisions.jsonl'
            fn=make_move_fn(1,str(log),lambda s:s,lambda b:['knockoff'],lambda a,b:a)
            self.assertEqual(fn(self.battle([self.state()])),'knockoff')
            self.assertTrue(json.loads(log.read_text())['mcts']['forced'])

    def test_real_native_search_returns_legal_move_and_root_visits(self):
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ,OURBOT_MCTS_MS='25',OURBOT_LOG_STATES='1',OURBOT_BELIEF_SOURCE='foulplay_baseline'):
            log=Path(tmp)/'decisions.jsonl'
            fn=make_move_fn(1,str(log),lambda s:s,lambda b:None,lambda a,b:a)
            self.assertIn(fn(self.battle([self.state()])),['knockoff','drainpunch'])
            r=json.loads(log.read_text()); self.assertFalse(r['mcts']['fallback'])
            self.assertGreater(r['mcts']['worlds'][0]['total_visits'],0)

    def test_evaluation_budget_is_pinned_when_resuming(self):
        args=NS(games=50,method='mcts_native',concurrency=5,mcts_ms=1000,out=None)
        _,cfg,_,remaining=plan(args)
        self.assertEqual((cfg['k'],cfg['mcts_ms'],remaining),(4,1000,50))
        with tempfile.TemporaryDirectory() as tmp:
            args.out=Path(tmp); (args.out/'evaluation.json').write_text(json.dumps(cfg))
            args.mcts_ms=2000
            with self.assertRaisesRegex(ValueError,'different mcts_ms'): plan(args)


if __name__=='__main__': unittest.main()
