"""Coverage of active-opponent item/ability hypotheses with probability weights.

For <= K positive-mass strata, cover each stratum and give every sample its
stratum's probability divided by its quota. For more strata than slots, use
randomized systematic resampling (equal-weight, unbiased across draws). This
changes finite-ensemble coverage, not the catalogue or its inferred set masses.
"""
from collections import defaultdict
from copy import deepcopy
import math
import random


def allocate_strata(masses, K, rng=random):
    if K < 1 or not masses or any(not math.isfinite(x) or x < 0 for x in masses) or not math.isfinite(sum(masses)) or sum(masses) <= 0:
        raise ValueError("Strata need positive finite total mass and at least one slot")
    total = sum(masses)
    probabilities = [x/total for x in masses]
    positive = [i for i,p in enumerate(probabilities) if p>0]
    if len(positive) <= K:
        quotas = {i:1 for i in positive}
        while sum(quotas.values()) < K:
            i = max(positive,key=lambda i:K*probabilities[i]-quotas[i])
            quotas[i] += 1
        slots = [(i,probabilities[i]/quotas[i]) for i in positive for _ in range(quotas[i])]
        mode = 'exact_stratum_mass'
    else:
        offset = rng.random()/K
        slots, i, cumulative = [], 0, probabilities[0]
        for j in range(K):
            point = offset+j/K
            while point >= cumulative and i < len(probabilities)-1:
                i += 1; cumulative += probabilities[i]
            slots.append((i,1/K))
        mode = 'systematic_overflow'
    rng.shuffle(slots)
    return slots, mode


def prepare_worlds(battle, K, rng=random):
    from fp import constants
    from fp.search.helpers import populate_pkmn_from_set
    from fp.search.random_battles import (get_all_remaining_sets_for_revealed_pkmn,
                                         populate_randombattle_unrevealed_pkmn)
    remaining = get_all_remaining_sets_for_revealed_pkmn(deepcopy(battle))
    active = battle.opponent.active
    candidates = [c for c in remaining[active.name] if c.pkmn_set.count>0]
    if not candidates:
        return list(battle.mode.prepare_battles(battle,K)),dict(name='item_stratified',
            fallback='no_positive_active_candidates',active=active.name)
    groups = defaultdict(list)
    for candidate in candidates:
        s = candidate.pkmn_set
        item = s.item if active.item == constants.UNKNOWN_ITEM else active.item
        ability = active.ability or s.ability
        groups[(str(item),str(ability))].append(candidate)
    keys = sorted(groups)
    masses = [sum(c.pkmn_set.count for c in groups[key]) for key in keys]
    slots, mode = allocate_strata(masses,K,rng)
    sampled, counts = [], defaultdict(int)
    for i,weight in slots:
        world = deepcopy(battle)
        population = groups[keys[i]]
        candidate = rng.choices(population,weights=[c.pkmn_set.count for c in population])[0]
        populate_pkmn_from_set(world.opponent.active,candidate)
        counts[i] += 1
        # Preserve the existing baseline distribution for revealed reserves and
        # the upstream generator/constraints for unrevealed teammates.
        for p in filter(lambda p:p.is_alive(),world.opponent.reserve):
            population = remaining[p.name]
            if population:
                candidate = rng.choices(population,weights=[c.pkmn_set.count for c in population])[0]
                populate_pkmn_from_set(p,candidate)
        populate_randombattle_unrevealed_pkmn(world)
        world.opponent.lock_moves()
        sampled.append((world,weight))
    total = sum(masses)
    metadata = dict(name='item_stratified',version=1,mode=mode,active=active.name,
        candidate_count=len(candidates),represented_prior_mass=sum(masses[i]/total for i in counts),
        strata=[dict(item=key[0],ability=key[1],prior_mass=masses[i]/total,samples=counts[i])
                for i,key in enumerate(keys)])
    return sampled,metadata
