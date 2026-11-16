"""Independent belief provider for the real-Showdown agent.

Foul Play supplies only transport and our own exact side. Opponent hypotheses
come from our catalog, not from its already-conditioned prepare_battles output.
Unimplemented public-state reconstruction uses an explicit, logged fallback.
"""
from __future__ import annotations

import random

from belief_catalog import to_id
from belief_engine import PokeEngineDamageOracle
from belief_tracker import BeliefSession
from showdown.belief_observer import ProtocolBeliefObserver, own_snapshot


class MechanicsWorldProvider:
    def __init__(self, source="mechanics_v2", shadow=True, rng=None, session=None):
        if source not in {"reveals_v2", "mechanics_v2"}:
            raise ValueError(source)
        self.source, self.shadow = source, shadow
        self.rng = rng or random.Random()
        self.session = session or BeliefSession(damage_oracle=PokeEngineDamageOracle(), mechanics=source == "mechanics_v2")
        self.observer = ProtocolBeliefObserver(self.session)
        self.history = []
        self.failure = None
        self.own_side = None

    def ingest(self, lines):
        lines = list(lines)
        if lines[:len(self.history)] != self.history:
            self.failure = "protocol_history_changed"
            return
        self.observer.feed(lines[len(self.history):])
        self.history = lines

    def update(self, battle, converter):
        # Foul Play uses p1/p2 as battler names in this pinned harness.
        side = getattr(battle.user, "name", None)
        if side not in {"p1", "p2"}:
            opponent = getattr(battle.opponent, "name", None)
            side = {"p1": "p2", "p2": "p1"}.get(opponent)
        if side is None:
            self.failure = "observer_seat_unknown"
            return
        self.own_side = self.observer.own_side = side
        try:
            self.ingest(getattr(battle, "_ourlines", []))
            state = converter(battle)
            for snapshot in own_snapshot(state):
                self.observer.set_own(snapshot)
        except (ValueError, AttributeError, KeyError, RuntimeError) as exc:
            self.failure = "own_state_conversion:" + type(exc).__name__

    def finish(self, lines):
        try:
            self.ingest(self.history + list(lines))
            self.observer.finish()
        except (ValueError, AttributeError, KeyError, RuntimeError) as exc:
            self.failure = "final_observation:" + type(exc).__name__

    def sample_states(self, battle, converter, K):
        if self.shadow:
            self.failure = "shadow_only"
            return None
        observer = self.observer
        if self.failure not in {None, "shadow_only"}:
            return None
        if not self.own_side or observer.field_unsafe or observer.suspended:
            self.failure = "unsupported_public_state"
            return None
        opp_side = "p2" if self.own_side == "p1" else "p1"
        entities = [e for e in observer.entities if e.startswith(opp_side + ":")]
        if not entities or len(entities) > 6:
            self.failure = "missing_opponent_identity"
            return None
        if any(observer.entities[e]["unsafe"] or observer.entities[e]["boosted"] for e in entities):
            self.failure = "unsupported_transformed_or_boosted_state"
            return None
        import poke_engine as pe
        import hidden as hd
        import randbats as rb
        own = converter(battle).side_one       # side_two is deliberately discarded
        if own.force_switch or any(observer.entities[e]["status"] in {"slp", "frz", "tox"} for e in entities):
            self.failure = "unsupported_pending_or_status_state"
            return None
        active = observer.active.get(opp_side)
        result = []
        for _ in range(K):
            mons, taken = [], set()
            for entity in entities:
                candidate = self.session.sample(entity, self.rng)
                mon = rb.to_pokemon(candidate.as_pokeset())
                b = self.session.entities[entity]
                current = observer.entities[entity]
                hp_obs = observer.hp[entity]
                index = next(i for i, entry in enumerate(b.entries) if entry.candidate == candidate)
                dist = b.hp.get(index)
                if dist:
                    hp = self.rng.choices(list(dist), weights=list(dist.values()))[0]
                else:
                    compatible = [h for h in range(mon.maxhp + 1) if hp_obs.compatible(h, mon.maxhp)]
                    if not compatible:
                        self.failure = "hp_display_outside_support"
                        return None
                    hp = self.rng.choice(compatible)
                item = candidate.item if b.item is None else b.item
                ability = candidate.ability if b.ability is None else b.ability
                # Reconstruct choice-lock from the last publicly executed move,
                # never Foul Play's inferred disabled flags.
                last = current.get("last_move") if entity == active else None
                moves = [pe.Move(id=m, disabled=bool(item in hd.CHOICE_ITEMS and last and m != last))
                         for m in candidate.moves]
                mon = hd.copy_pokemon(mon, hp=hp, item=item, ability=ability, base_ability=ability,
                                      status=current["status"], moves=moves,
                                      terastallized=bool(current["tera"]),
                                      tera_type=current["tera"] or mon.tera_type)
                mons.append(mon)
                taken.add(hd.base_of(candidate.species))
            if len(mons) < 6:
                for candidate in rb.sample_team(self.rng, n=6 - len(mons), exclude=taken):
                    mons.append(rb.to_pokemon(candidate))
            opponent = pe.Side(pokemon=mons, active_index=entities.index(active),
                               force_switch=observer.entities[active]["fainted"])
            # Non-default fields currently cause fallback above; no inferred
            # opponent timers, volatile state or hidden bench is copied.
            result.append((pe.State(side_one=hd.copy_side(own), side_two=opponent), 1 / K))
        self.failure = None
        return result

    def summary(self):
        return dict(source=self.source, shadow=self.shadow, fallback=self.failure, **self.session.summary())
