"""Lightweight adapter tests for the queued independent planner experiments."""
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

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'engine'),str(ROOT/'engine/showdown')]
from rescaled_credit import rescale_utilities,conditional_credit
import s1_live


class RescaledCreditTests(unittest.TestCase):
    def test_large_losses_do_not_all_receive_identical_maximal_reward(self):
        u=np.array([-8104.,-8104.,-8108.,-5180.])
        r=rescale_utilities(u)
        self.assertEqual(r.argmin(),2);self.assertEqual(r.argmax(),3)
        self.assertEqual(r.min(),0);self.assertEqual(r.max(),1)
        self.assertGreater(r[0],r[2]);self.assertLess(r[0],.01)

    def test_shared_offset_does_not_change_credit(self):
        u=np.array([20.,10.,0.,5.])
        np.testing.assert_allclose(rescale_utilities(u),rescale_utilities(u-10000))
        np.testing.assert_allclose(rescale_utilities(u),[.05,.025,0,.0125])

    def test_true_ties_stay_tied_and_do_not_create_weight_changes(self):
        np.testing.assert_array_equal(rescale_utilities([5,5,5,5]),[0,0,0,0])

    def test_conditional_expectation_precedes_shared_rescaling(self):
        matrices=[np.array([[1000.],[0.]]),np.array([[0.],[300.]])]
        condition=dict(weights=np.array([.2,.8]),indices={0:0,1:0})
        r,u=conditional_credit(matrices,np.eye(2),condition)
        np.testing.assert_allclose(u,[200,240]);np.testing.assert_allclose(r,[0,.1])


class S1AdapterTests(unittest.TestCase):
    def state(self):
        p=pe.Pokemon(id='gurdurr',hp=300,maxhp=300,terastallized=True,
                     moves=[pe.Move(id='drainpunch'),pe.Move(id='knockoff')])
        q=pe.Pokemon(id='sableye',hp=300,maxhp=300,terastallized=True,moves=[pe.Move(id='splash')])
        return pe.State(side_one=pe.Side(pokemon=[p]),side_two=pe.Side(pokemon=[q]))

    def test_only_previously_revealed_slots_are_passed_to_group_keys(self):
        states=[self.state(),self.state()]
        battle=NS(team_preview=False,turn=5,battle_tag='s1-test',opponent=NS(reserve=[NS()]),
                  mode=NS(prepare_battles=lambda b,k:list(zip(states,[.7,.3]))))
        result=dict(A1_rep=dict(x=np.array([0.,1.]),v=3.,R=[np.array([[1.],[3.]])]*2),
                    ex=NS(cols=[['splash']]*2,nodes=[1,2],secs_wall=.01,secs_cpu=.01,stats=dict(d1_calls=2)),
                    groups=[dict(worlds=[0,1])],lp_secs={})
        with tempfile.TemporaryDirectory() as tmp,patch.dict(os.environ,OURBOT_WORLD_SAMPLER='baseline',OURBOT_LOG_STATES='1'), \
             patch.object(s1_live.ip.nash,'DET_LEAF',True), \
             patch.object(s1_live.pb,'options',return_value=(['drainpunch','knockoff'],['splash'])), \
             patch.object(s1_live.ip,'plan',return_value=result) as planner:
            log=Path(tmp)/'decisions.jsonl';fn=s1_live.make_move_fn(2,str(log),lambda s:s,lambda b:None,lambda a,b:a,random.Random(3))
            self.assertEqual(fn(battle),'knockoff')
            self.assertEqual(planner.call_args.args[3],{0,1})
            self.assertFalse(planner.call_args.kwargs['want_a0l'])
            record=json.loads(log.read_text());self.assertEqual(record['s1']['arm'],'A1_rep')
            self.assertEqual(record['search']['weights'],[.7,.3])
            self.assertEqual(record['search']['matrix_kind'],'S1_A1_rep')


if __name__=='__main__':unittest.main()
