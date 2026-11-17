"""Coverage and probability-mass regressions for the MCTS sampler experiment."""
from collections import Counter
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

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'engine'),str(ROOT/'engine/showdown'),str(ROOT/'.local-play/foul-play')]
from stratified_worlds import allocate_strata,prepare_worlds
from run_public_evaluation import plan


class AllocationTests(unittest.TestCase):
    def test_all_material_classes_covered_with_exact_original_probability_mass(self):
        for seed in range(50):
            slots,mode=allocate_strata([672,322,370],4,random.Random(seed))
            self.assertEqual(mode,'exact_stratum_mass');self.assertEqual(len(slots),4)
            mass=Counter()
            for i,w in slots:mass[i]+=w
            for i,n in enumerate([672,322,370]): self.assertAlmostEqual(mass[i],n/1364)
            self.assertEqual(Counter(i for i,w in slots),{0:2,1:1,2:1})

    def test_rare_stratum_is_not_reweighted_to_uniform(self):
        slots,_=allocate_strata([999,1],4,random.Random(2))
        self.assertAlmostEqual(sum(w for i,w in slots if i==1),.001)
        self.assertEqual(sum(i==1 for i,w in slots),1)
        self.assertAlmostEqual(sum(w for i,w in slots),1)

    def test_overflow_remains_unbiased_across_randomized_draws(self):
        masses=[50,20,10,8,5,4,3]; observed=Counter(); rng=random.Random(21)
        for _ in range(5000):
            slots,mode=allocate_strata(masses,4,rng)
            self.assertEqual(mode,'systematic_overflow')
            for i,w in slots:observed[i]+=w/5000
        for i,n in enumerate(masses):self.assertAlmostEqual(observed[i],n/100,delta=.012)

    def test_zero_mass_and_one_class(self):
        slots,_=allocate_strata([0,5,0],4,random.Random(1))
        self.assertEqual(slots,[(1,.25)]*4)
        for bad in ([],[0,0],[1,-1],[float('inf')],[float('nan')]):
            with self.assertRaises(ValueError):allocate_strata(bad,4)

    def test_config_sampler_is_frozen_and_old_baseline_configs_still_resume(self):
        with tempfile.TemporaryDirectory() as tmp:
            args=NS(games=50,method='mcts_native',out=Path(tmp),concurrency=5,mcts_ms=1000,sampler='baseline')
            _,cfg,_,_=plan(args);cfg.pop('sampler')
            (args.out/'evaluation.json').write_text(json.dumps(cfg))
            self.assertEqual(plan(args)[1]['sampler'],'baseline')
            args.sampler='item_stratified'
            with self.assertRaisesRegex(ValueError,'different sampler'):plan(args)
            args.out=None;args.method='hedge_bayes_mw'
            with self.assertRaisesRegex(ValueError,'requires mcts_native'):plan(args)


class RealSamplerTests(unittest.TestCase):
    def setUp(self):
        from fp.config import FoulPlayConfig
        self.config=patch.object(FoulPlayConfig,'pokemon_format','gen9randombattle')
        self.config.start();self.addCleanup(self.config.stop)

    def battle(self):
        from fp.battle.state import Battle,Pokemon
        from fp.modes.random_battle import RandomBattleMode
        b=Battle('stratified-offline');b.turn=1
        b.pokemon_format='gen9randombattle';b.generation='gen9'
        b.mode=RandomBattleMode();b.mode.datasets.initialize(b.format_spec)
        b.user.active=Pokemon('gyarados',79);b.user.active.ability='moxie';b.user.active.item='heavydutyboots'
        b.user.active.add_move('earthquake');b.user.active.add_move('dragondance')
        b.opponent.active=Pokemon('dragalge',88);b.opponent.active.add_move('dracometeor')
        b.opponent.active.moves[0].current_pp=3
        b.opponent.active.hp=130
        return b

    def test_actual_catalogue_covers_specs_and_preserves_current_state(self):
        b=self.battle();hp=b.opponent.active.hp
        before=(b.opponent.active.item,b.opponent.active.ability)
        worlds,meta=prepare_worlds(b,4,random.Random(5))
        self.assertEqual(meta['mode'],'exact_stratum_mass')
        self.assertAlmostEqual(meta['represented_prior_mass'],1)
        expected={s['item']:s['prior_mass'] for s in meta['strata']}
        actual=Counter()
        for world,w in worlds:
            p=world.opponent.active; actual[p.item]+=w
            self.assertEqual(p.hp,hp)
            self.assertEqual(next(m for m in p.moves if m.name=='dracometeor').current_pp,3)
            self.assertEqual(len(world.opponent.reserve),5)
            self.assertEqual(world.user.active.hp,b.user.active.hp)
        self.assertIn('choicespecs',actual)
        for item,mass in expected.items(): self.assertAlmostEqual(actual[item],mass)
        self.assertEqual((b.opponent.active.item,b.opponent.active.ability),before)
        self.assertEqual(b.opponent.reserve,[])

    def test_known_item_and_ability_are_not_overwritten_or_falsely_diversified(self):
        b=self.battle();b.opponent.active.item='choicespecs';b.opponent.active.ability='adaptability'
        worlds,meta=prepare_worlds(b,4,random.Random(3))
        self.assertEqual(len(meta['strata']),1)
        self.assertTrue(all(w.opponent.active.item=='choicespecs' and w.opponent.active.ability=='adaptability' for w,p in worlds))
        self.assertTrue(all(abs(p-.25)<1e-12 for w,p in worlds))

    def test_empty_active_support_uses_explicit_baseline_fallback(self):
        b=self.battle();sentinel=[(b,1)]
        with patch('fp.search.random_battles.get_all_remaining_sets_for_revealed_pkmn',return_value={'dragalge':[]}), \
             patch.object(b.mode,'prepare_battles',return_value=sentinel):
            worlds,meta=prepare_worlds(b,4)
        self.assertEqual(worlds,sentinel)
        self.assertEqual(meta['fallback'],'no_positive_active_candidates')

    def test_native_policy_consumes_strata_weights_and_logs_them(self):
        import native_mcts
        import poke_engine as pe
        from battle_adapter import battle_to_poke_engine_state,request_options,raw_action
        b=self.battle()
        r=NS(side_one=[NS(move_choice='earthquake',visits=90,total_score=50),NS(move_choice='dragondance',visits=10,total_score=4)],total_visits=100)
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ,OURBOT_WORLD_SAMPLER='item_stratified',OURBOT_MCTS_MS='1000',OURBOT_LOG_STATES='1',OURBOT_BELIEF_SOURCE='foulplay_baseline'), \
             patch.object(pe,'monte_carlo_tree_search',return_value=r):
            p=Path(tmp)/'decisions.jsonl'
            fn=native_mcts.make_move_fn(4,str(p),battle_to_poke_engine_state,request_options,raw_action)
            self.assertEqual(fn(b),'earthquake')
            record=json.loads(p.read_text())
            self.assertEqual(record['sampler']['name'],'item_stratified')
            self.assertEqual(record['mcts']['budget_ms'],1000)
            self.assertEqual(len(record['search']['states']),4)
            self.assertAlmostEqual(sum(record['search']['weights']),1)
            self.assertGreater(len(set(record['search']['weights'])),1)


if __name__=='__main__':unittest.main()
