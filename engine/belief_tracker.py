"""Persistent mechanics-conditioned posterior. Pure Python; no private state.

All-candidate updates are transactional. Any unsupported candidate skips the
whole factor, rather than favoring candidates understood by our model.
"""
from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field

from belief_catalog import CatalogEntry, default_catalog, to_id
from belief_events import DamageEvidence, Reveal, Skip, SpeedEvidence
from belief_likelihoods import NoDamageOracle, Unsupported, filter_hp, incoming_likelihood, speed_likelihood


class PriorSupportError(ValueError):
    """A direct reveal contradicts the configured prior; never silently ignore it."""


@dataclass
class EntityBelief:
    entries: tuple[CatalogEntry, ...]
    logweights: list[float]
    moves: set[str] = field(default_factory=set)
    item: str | None = None
    ability: str | None = None
    tera: str | None = None
    hp: dict[int, dict[int, float]] = field(default_factory=dict)
    degraded: bool = False

    def probabilities(self):
        m = max(self.logweights)
        weights = [math.exp(w - m) for w in self.logweights]
        total = math.fsum(weights)
        return [w / total for w in weights]

    def support(self):
        return [i for i, w in enumerate(self.logweights) if math.isfinite(w)]


class BeliefSession:
    def __init__(self, catalog=None, damage_oracle=None, mechanics=True):
        self.catalog = catalog or default_catalog()
        self.damage_oracle = damage_oracle or NoDamageOracle()
        self.mechanics = mechanics
        self.entities = {}
        self.applied = set()
        self.diagnostics = Counter()

    def _entity(self, entity, species=None):
        if entity not in self.entities:
            if species is None:
                raise KeyError(entity)
            entries = self.catalog.entries(species)
            self.entities[entity] = EntityBelief(entries, [math.log(e.prior) for e in entries])
        return self.entities[entity]

    def _apply_weights(self, belief, likelihoods, hard=False):
        out = [w + math.log(p) if p > 0 else -math.inf for w, p in zip(belief.logweights, likelihoods)]
        if not any(math.isfinite(w) for w in out):
            if hard:
                belief.degraded = True
                raise PriorSupportError("direct reveal outside remaining prior support")
            self.diagnostics["model_mismatch"] += 1
            return False
        m = max(out)
        z = m + math.log(math.fsum(math.exp(w - m) for w in out))
        belief.logweights = [w - z for w in out]
        return True

    def observe(self, event):
        if event.event_id in self.applied:
            self.diagnostics["duplicate"] += 1
            return False
        if isinstance(event, Reveal):
            belief = self._entity(event.entity, event.species)
            value = to_id(event.value) if event.value is not None else None
            if event.attribute == "species":
                if to_id(belief.entries[0].candidate.species) != to_id(event.species):
                    belief.degraded = True
                    raise PriorSupportError("ambiguous/replaced species identity")
            elif event.attribute in {"move", "item", "ability", "tera"}:
                if value is None:
                    raise ValueError("reveal needs a value")
                if event.original:
                    likelihoods = []
                    for entry in belief.entries:
                        c = entry.candidate
                        likelihoods.append(float(value in c.moves if event.attribute == "move"
                                                 else value == getattr(c, event.attribute)))
                    self._apply_weights(belief, likelihoods, hard=True)
                if event.attribute == "move":
                    belief.moves.add(value)
                else:
                    setattr(belief, event.attribute, value)
            else:
                raise ValueError(f"unknown reveal attribute: {event.attribute}")
            self.applied.add(event.event_id)
            self.diagnostics["reveals"] += 1
            return True
        self.applied.add(event.event_id)
        if isinstance(event, Skip):
            self.diagnostics["skip:" + event.reason] += 1
            return False
        if not self.mechanics:
            self.diagnostics["skip:mechanics_disabled"] += 1
            return False
        belief = self._entity(event.entity)
        if belief.degraded:
            self.diagnostics["skip:degraded_entity"] += 1
            return False
        likelihoods = [0.0] * len(belief.entries)
        pending_hp = {}
        item = event.item_at_event if event.attributes_at_event else belief.item
        ability = event.ability_at_event if event.attributes_at_event else belief.ability
        try:
            if isinstance(event, DamageEvidence) and hasattr(self.damage_oracle, "prepare"):
                self.damage_oracle.prepare([belief.entries[i].candidate for i in belief.support()], event, item, ability)
            for i in belief.support():
                c = belief.entries[i].candidate
                if isinstance(event, SpeedEvidence):
                    p = speed_likelihood(c, event, self.catalog, item, ability)
                elif isinstance(event, DamageEvidence):
                    if not event.supported:
                        raise Unsupported(event.reason)
                    pmf = self.damage_oracle.distribution(c, event, item, ability)
                    if event.incoming:
                        p = incoming_likelihood(pmf, event.before, event.after)
                    else:
                        maximum = self.catalog.stats(c)["hp"]
                        prior = belief.hp.get(i)
                        if prior is None:
                            compatible = [h for h in range(maximum + 1) if event.before.compatible(h, maximum)]
                            if not compatible:
                                p = 0.0
                                likelihoods[i] = p
                                continue
                            # First contact is an explicitly documented uniform HP assumption.
                            prior = {h: 1 / len(compatible) for h in compatible}
                        elif not all(event.before.compatible(h, maximum) for h in prior):
                            raise Unsupported("hp_history_gap")
                        p, filtered = filter_hp(prior, pmf, event.after, maximum)
                        pending_hp[i] = filtered
                else:
                    raise TypeError(type(event))
                if not math.isfinite(p) or not 0 <= p <= 1 + 1e-9:
                    raise ValueError("invalid mechanics likelihood")
                likelihoods[i] = min(p, 1.0)
        except Unsupported as exc:
            self.diagnostics["skip:" + str(exc)] += 1
            return False
        if not self._apply_weights(belief, likelihoods):
            return False
        belief.hp.update(pending_hp)
        self.diagnostics["speed" if isinstance(event, SpeedEvidence) else "damage"] += 1
        return True

    def invalidate_hp(self, entity):
        if entity in self.entities:
            self.entities[entity].hp.clear()
            self.diagnostics["hp_invalidated"] += 1

    def sample(self, entity, rng):
        b = self._entity(entity)
        if b.degraded:
            raise PriorSupportError("cannot sample an out-of-support entity")
        return rng.choices([e.candidate for e in b.entries], weights=b.probabilities())[0]

    def map_set(self, entity):
        b = self._entity(entity)
        if b.degraded:
            raise PriorSupportError("cannot sample an out-of-support entity")
        return b.entries[max(range(len(b.entries)), key=lambda i: b.logweights[i])].candidate

    def marginal(self, entity, attribute):
        b = self._entity(entity)
        out = Counter()
        for entry, p in zip(b.entries, b.probabilities()):
            vals = entry.candidate.moves if attribute == "move" else [getattr(entry.candidate, attribute)]
            for value in vals:
                out[value] += p
        return dict(out)

    def summary(self):
        out = {}
        for name, b in self.entities.items():
            p = b.probabilities()
            out[name] = dict(support=len(b.support()), entropy=-math.fsum(x * math.log(x) for x in p if x),
                             degraded=b.degraded, item=self.marginal(name, "item"),
                             ability=self.marginal(name, "ability"))
        return dict(entities=out, diagnostics=dict(self.diagnostics))
