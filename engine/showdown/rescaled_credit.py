"""Experimental Hedge reward transform; not used by native MCTS.

Normalize action-conditional expected expert utilities, not each world's clipped
reward. The floor avoids magnifying tiny differences. This changes the surrogate
objective; no fixed-scale raw-utility regret guarantee is claimed for this rule.
"""
import numpy as np

SEMANTICS='per_world_security_conditional_expert_range_floor400_v1'


def rescale_utilities(utilities,minimum_scale=400.0):
    u=np.asarray(utilities,dtype=float)
    if u.ndim!=1 or not len(u) or not np.all(np.isfinite(u)) or minimum_scale<=0:
        raise ValueError('Expected finite expert utilities and a positive scale floor')
    return (u-u.min())/max(minimum_scale,float(np.ptp(u)))


def conditional_credit(matrices,Q,conditioning,minimum_scale=400.0):
    u=sum(conditioning['weights'][k]*(Q @ matrices[k][:,j])
          for k,j in conditioning['indices'].items())
    return rescale_utilities(u,minimum_scale),u
