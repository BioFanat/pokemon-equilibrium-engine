"""Live adapter for the existing experimental S1 A1_rep matrix planner.

Uses baseline sampling; this is a planner alternative, not native MCTS. The
opponent's revealed slots are captured BEFORE filling unknown team slots.
"""
from copy import deepcopy
import json
import os
import time

import numpy as np
import info_planner as ip
import pkbattle as pb


def make_move_fn(K,log,converter,request_options,raw_action,rng):
    if not ip.nash.DET_LEAF:
        raise ValueError('s1_agentform requires OURBOT_DET_LEAF=1 before importing the engine')
    if os.environ.get('OURBOT_WORLD_SAMPLER','baseline')!='baseline':
        raise ValueError('S1 activation test requires the unchanged baseline sampler')
    index=0
    def move_fn(battle):
        nonlocal index
        started=time.perf_counter()
        battle=deepcopy(battle)
        if battle.team_preview:
            battle.user.active=battle.user.reserve.pop(0)
            battle.opponent.active=battle.opponent.reserve.pop(0)
        # Pinned Foul Play converter orders active at 0, original reserves next,
        # then the newly sampled unknown reserves. Never reveal the latter.
        revealed=set(range(1+len(battle.opponent.reserve)))
        worlds=list(battle.mode.prepare_battles(battle,K))
        states,weights=zip(*[(converter(b),w) for b,w in worlds])
        weights=np.asarray(weights,dtype=float);weights/=weights.sum()
        allowed=request_options(battle);own,_=pb.options(states[0])
        legal=[a for a in own if allowed is None or a in allowed]
        if not legal:raise ValueError('No S1 engine options match the server legal actions')
        matrices=[]
        rows=[pb.options(s)[0] for s in states];cols=[pb.options(s)[1] for s in states]
        if len(legal)==1:
            x=np.ones(1);stats=dict(arm='A1_rep',forced=True,revealed_slots=sorted(revealed))
        else:
            result=ip.plan(states,1,legal,revealed,w=weights,cache={},want_a0l=False,key_report=False)
            arm=result['A1_rep'];x=np.clip(arm['x'],0,None);x/=x.sum()
            matrices=[M.tolist() for M in arm['R']];rows=[list(legal) for _ in states];cols=result['ex'].cols
            stats=dict(arm='A1_rep',forced=False,revealed_slots=sorted(revealed),value=float(arm['v']),
                nodes=len(result['ex'].nodes),groups=len(result['groups']),
                multi_world_groups=sum(len(g['worlds'])>1 for g in result['groups']),
                expansion_wall_secs=result['ex'].secs_wall,expansion_cpu_secs=result['ex'].secs_cpu,
                d1_calls=result['ex'].stats['d1_calls'],group_cpu_secs=result['lp_secs'])
        chosen=rng.choices(range(len(legal)),weights=x)[0]
        action=raw_action(legal[chosen],battle)
        record=dict(method='s1_agentform',turn=battle.turn,battle_tag=battle.battle_tag,decision_index=index,
            worlds=len(states),legal=legal,pfull=x.tolist(),move=action,
            mix={a:float(p) for a,p in zip(legal,x) if p>=.01},
            secs=round(time.perf_counter()-started,4),s1=stats)
        if os.environ.get('OURBOT_LOG_STATES')=='1':
            record['search']=dict(states=[s.to_string() for s in states],weights=weights.tolist(),
                legal=legal,rows=rows,cols=cols,matrices=matrices,matrix_kind='S1_A1_rep')
        with open(log,'a') as f:f.write(json.dumps(record)+'\n')
        index+=1
        return action
    return move_fn
