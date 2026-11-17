"""Budgeted native MCTS over sampled worlds, without the matrix/Hedge layer.

Each world contributes its prior weight times its normalized legal root visits.
Choose the largest aggregate share (stable first-label tie break). The native
API does not accept a root action mask: server legality is enforced on returned
root visits without altering future-state mechanics. Discarded mass is logged.
"""
from copy import deepcopy
import json
import os
import time

import numpy as np
import poke_engine as pe
import pkbattle as pb


def aggregate_visits(results, weights, legal):
    """Equal/weighted worlds, not pooled iteration counts; missing labels get zero."""
    weights = np.asarray(weights, dtype=float)
    if len(results) != len(weights) or np.any(weights < 0) or not np.all(np.isfinite(weights)) or weights.sum() <= 0:
        raise ValueError("MCTS requires finite nonnegative world weights with positive total")
    weights = weights / weights.sum()
    positions = {label: i for i, label in enumerate(legal)}
    shares = np.zeros(len(legal))
    worlds = []
    for result, weight in zip(results, weights):
        root, votes = [], np.zeros(len(legal))
        all_visits = 0
        for option in result.side_one:
            label = pb._canon(option.move_choice)
            visits = int(option.visits)
            if visits < 0:
                raise ValueError("Native MCTS returned negative visits")
            all_visits += visits
            root.append(dict(action=label, visits=visits, total_score=float(option.total_score)))
            if label in positions:
                votes[positions[label]] += visits
        retained = float(votes.sum())
        if retained:
            shares += weight * votes / retained
        worlds.append(dict(weight=float(weight), total_visits=int(result.total_visits), root=root,
                           legal_visits=int(retained),
                           discarded_visit_fraction=1-retained/all_visits if all_visits else 0,
                           no_legal_visits=not bool(retained)))
    if not legal:
        raise ValueError("No legal native MCTS actions")
    fallback = bool(shares.sum() <= 0)
    if fallback:
        shares[0] = 1  # Legal, explicitly logged fallback; never emit an invalid choice.
    else:
        shares /= shares.sum()
    return shares, worlds, fallback


def make_move_fn(K, log, converter, request_options, raw_action):
    budget_ms = int(os.environ.get("OURBOT_MCTS_MS", "1000"))
    if K < 1 or not K <= budget_ms <= 10000:
        raise ValueError("Native MCTS budget must be between K and 10000 milliseconds")
    if os.environ.get("OURBOT_BELIEF_SOURCE", "foulplay_baseline") != "foulplay_baseline":
        raise ValueError("mcts_native currently uses the foulplay_baseline sampler")
    sampler = os.environ.get('OURBOT_WORLD_SAMPLER','baseline')
    if sampler not in {'baseline','item_stratified'}:
        raise ValueError('Unknown MCTS world sampler: '+sampler)
    decision_index = 0

    def move_fn(battle):
        nonlocal decision_index
        started = time.perf_counter()
        battle = deepcopy(battle)
        if battle.team_preview:
            battle.user.active = battle.user.reserve.pop(0)
            battle.opponent.active = battle.opponent.reserve.pop(0)
        if sampler == 'item_stratified':
            from stratified_worlds import prepare_worlds
            sampled, sampling = prepare_worlds(battle,K)
        else:
            sampled = list(battle.mode.prepare_battles(battle, K))
            sampling = dict(name='baseline')
        states, weights = zip(*[(converter(b), weight) for b, weight in sampled])
        allowed = request_options(battle)
        first_options, _ = pb.options(states[0])
        legal = [a for a in first_options if allowed is None or a in allowed]
        if not legal:
            raise ValueError("No engine options match the server's legal actions")
        forced = len(legal) == 1
        worlds, searches, results = [], [], []
        if forced:
            policy = np.ones(1)
            fallback = False
        else:
            # Split the fixed budget exactly; differing native iteration counts
            # must not change prior world weights.
            durations = [budget_ms//len(states) + (i < budget_ms % len(states)) for i in range(len(states))]
            for state, duration in zip(states, durations):
                t0 = time.perf_counter()
                results.append(pe.monte_carlo_tree_search(state, duration_ms=duration, threads=1))
                searches.append(dict(budget_ms=duration, elapsed_secs=time.perf_counter()-t0))
            policy, worlds, fallback = aggregate_visits(results, weights, legal)
        chosen = int(policy.argmax())
        action = raw_action(legal[chosen], battle)
        played = np.eye(len(legal))[chosen]
        record = dict(method="mcts_native", turn=battle.turn, battle_tag=battle.battle_tag,
                      decision_index=decision_index, worlds=len(states), legal=legal,
                      move=action, mix={legal[chosen]:1.0}, pfull=played.tolist(),
                      secs=round(time.perf_counter()-started, 4),
                      mcts=dict(policy="weighted_root_visit_argmax_v1", budget_ms=budget_ms,
                                threads=1, forced=forced, fallback=fallback,
                                visit_policy=policy.tolist(), searches=searches, worlds=worlds))
        record['sampler'] = sampling
        if os.environ.get("OURBOT_LOG_STATES") == "1":
            record["search"] = dict(states=[s.to_string() for s in states], weights=list(weights),
                                    legal=legal, rows=[pb.options(s)[0] for s in states],
                                    cols=[pb.options(s)[1] for s in states], matrices=[])
        with open(log, "a") as f:
            f.write(json.dumps(record)+"\n")
        decision_index += 1
        return action
    return move_fn
