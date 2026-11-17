"""Streaming player-visible Showdown observations for the mechanics posterior.

This adapter never accepts opponent commands, simulator states or inferred
Foul Play attributes. Exact own snapshots are explicitly player-private input.
Complex effects are skipped; the diagnostic coverage is part of the output.
"""
from __future__ import annotations

import copy
import json

from belief_catalog import to_id
from belief_engine import ORDINARY_MOVES
from belief_events import DamageEvidence, HPObservation, Reveal, Skip, SpeedEvidence
from belief_likelihoods import Unsupported, effective_speed
from belief_tracker import PriorSupportError


def hp_observation(text, exact=False):
    token = text.split()[0]
    if token == "0":
        # Denominator must come from preceding own snapshot if exact.
        return HPObservation(0, 100, False)
    numerator, denominator = token.split("/")
    return HPObservation(int(numerator), int(denominator), exact)


class ProtocolBeliefObserver:
    def __init__(self, session, own_side=None):
        self.session = session
        self.own_side = own_side
        self.entities = {}        # public species identity, not true bench indices
        self.active = {}
        self.hp = {}
        self.own = {}             # exact snapshots of our own Pokémon only
        self.turn = 0
        self.serial = 0
        self.block = None
        self.order = []
        self.turn_context = None
        self.field_unsafe = False
        self.turn_unsafe = False
        self.suspended = set()

    def _id(self, suffix):
        return f"protocol:{self.serial}:{suffix}"

    def _observe(self, event):
        try:
            self.session.observe(event)
        except (PriorSupportError, KeyError):
            self.suspended.add(event.entity)
            self.session.diagnostics["skip:prior_support"] += 1

    def set_own(self, snapshot):
        """Only call after ingesting the preceding history with its old snapshots."""
        self.own[to_id(snapshot["species"])] = copy.deepcopy(snapshot)
        if self.turn_context is not None and not self.order:
            active, entities, _ = self.turn_context
            self.turn_context = active, entities, copy.deepcopy(self.own)

    def _entity(self, side, species):
        return side + ":" + to_id(species)

    def feed(self, lines):
        for line in lines:
            self.serial += 1
            parts = line.split("|")
            if len(parts) < 2:
                continue
            kind = parts[1]
            if kind == "split":
                raise ValueError("observer requires player-visible lines, not raw simulator splits")
            if kind == "request":
                request = json.loads(parts[2])
                if "side" in request:
                    seat = request["side"]["id"]
                    if self.own_side is None:
                        self.own_side = seat
                    elif seat != self.own_side:
                        raise ValueError("opponent-private request passed to observer")
                continue
            if self.own_side is None:
                # No inference until the observing seat is explicitly established.
                continue
            if kind in {"move", "switch", "drag", "turn", "win", "tie"}:
                self._flush_damage()
            if kind in {"turn", "win", "tie"}:
                self._flush_speed()
                self.order = []
                self.turn_unsafe = False
                if kind == "turn":
                    self.turn = int(parts[2])
                    self.turn_context = copy.deepcopy((self.active, self.entities, self.own))
                continue
            if kind in {"switch", "drag"}:
                side = parts[2][:2]
                species = parts[3].split(",")[0]
                entity = self._entity(side, species)
                self.active[side] = entity
                self.entities.setdefault(entity, dict(species=species, moves=set(), stage=0, status="none",
                                                     item=None, ability=None, tera=None, unsafe=False,
                                                     fainted=False, boosted=False))
                state = self.entities[entity]
                state.update(stage=0, boosted=False, last_move=None)
                self.hp[entity] = hp_observation(parts[4], side == self.own_side)
                state["status"] = to_id(parts[4].split()[1]) if len(parts[4].split()) > 1 else "none"
                if side != self.own_side:
                    self._observe(Reveal(self._id("species"), entity, species))
                self.turn_unsafe = True
                continue
            if kind == "move":
                side = parts[2][:2]
                entity = self.active.get(side)
                if entity is None:
                    continue
                move = to_id(parts[3])
                called = any(p.startswith("[from]") for p in parts[4:])
                if side == self.own_side and (called or move not in ORDINARY_MOVES):
                    self.turn_unsafe = True
                state = self.entities[entity]
                if side != self.own_side and not called:
                    self._observe(Reveal(self._id("move"), entity, state["species"], "move", move))
                    state["moves"].add(move)
                    state["last_move"] = move
                own_entity = self.active.get(self.own_side)
                own_species = to_id(self.entities.get(own_entity, {}).get("species", ""))
                own = copy.deepcopy(self.own.get(own_species))
                self.order.append((entity, move, called))
                self.block = dict(entity=entity, move=move, called=called, damage=[], unsafe=False,
                                  own=own, pre=copy.deepcopy(self.hp), state=copy.deepcopy(state))
                continue
            if len(parts) < 3:
                continue
            if kind in {"-weather", "-fieldstart", "-fieldend", "-sidestart", "-sideend"}:
                self.field_unsafe = True
                if self.block:
                    self.block["unsafe"] = True
                continue
            side = parts[2][:2]
            entity = self.active.get(side)
            if entity is None:
                continue
            state = self.entities[entity]
            opponent = side != self.own_side
            if kind in {"detailschange", "-formechange", "replace", "-transform", "-endability"}:
                state["unsafe"] = True
                self.suspended.add(entity)
            elif kind == "-terastallize":
                state["tera"] = to_id(parts[3])
                if opponent:
                    self._observe(Reveal(self._id("tera"), entity, state["species"], "tera", parts[3]))
            elif kind in {"-item", "-enditem", "-ability"}:
                attribute = "ability" if kind == "-ability" else "item"
                value = parts[3]
                # Transfers/replacements are dynamic only. Explicit [from] item/ability
                # activations remain ambiguous and conservatively avoid original filtering.
                original = (not any(p.startswith("[from]") for p in parts[4:])
                            and state.get(attribute + "_original", True))
                if opponent:
                    self._observe(Reveal(self._id(attribute), entity, state["species"], attribute, value, original))
                    if kind == "-enditem":
                        self._observe(Reveal(self._id("removed"), entity, state["species"], "item", "none", False))
                state[attribute] = "none" if kind == "-enditem" else to_id(value)
                if not original or kind == "-enditem":
                    state[attribute + "_original"] = False
                if self.block:
                    self.block["unsafe"] = True
            elif kind in {"-damage", "-heal", "-sethp"}:
                before = self.hp.get(entity)
                text = parts[3]
                if text.split()[0] == "0" and before and before.exact:
                    after = HPObservation(0, before.denominator, True)
                else:
                    after = hp_observation(text, side == self.own_side)
                self.hp[entity] = after
                own = self.own.get(to_id(state["species"])) if not opponent else None
                if own:
                    own["hp"] = after.value
                direct = kind == "-damage" and not any(p.startswith("[from]") for p in parts[4:])
                if self.block and direct:
                    self.block["damage"].append((entity, before, after))
                elif opponent:
                    self.session.invalidate_hp(entity)
            elif kind in {"-boost", "-unboost", "-setboost", "-clearboost", "-clearallboost"}:
                state["boosted"] = True
                self.turn_unsafe = True
                if self.block:
                    self.block["unsafe"] = True
                if len(parts) > 4 and parts[3] == "spe":
                    amount = int(parts[4])
                    state["stage"] = amount if kind == "-setboost" else state["stage"] + (-amount if kind == "-unboost" else amount)
            elif kind in {"-status", "-curestatus"}:
                state["status"] = to_id(parts[3]) if kind == "-status" else "none"
                if not opponent:
                    own = self.own.get(to_id(state["species"]))
                    if own:
                        own["status"] = state["status"]
            elif kind == "faint":
                state["fainted"] = True
            elif kind in {"cant", "-crit", "-hitcount", "-miss", "-immune", "-fail", "-activate", "-start", "-end",
                           "-singleturn", "-singlemove", "-prepare", "-mustrecharge"}:
                if self.block:
                    self.block["unsafe"] = True
                if kind in {"-start", "-end", "-activate", "-singleturn", "-singlemove", "-prepare", "-mustrecharge"}:
                    state["unsafe"] = True

    def _flush_damage(self):
        block, self.block = self.block, None
        if block is None or block["entity"].startswith(self.own_side + ":"):
            return
        entity = block["entity"]
        if entity in self.suspended:
            return
        reason = None
        if block["called"] or block["unsafe"] or block["state"]["unsafe"]:
            reason = "complex_move_block"
        elif self.field_unsafe or self.turn_unsafe or block["state"]["boosted"] or block["state"]["tera"]:
            reason = "unsupported_field_boost_or_tera"
        elif block["move"] not in ORDINARY_MOVES:
            reason = "damage_move_not_supported"
        elif len(block["damage"]) != 1 or not block["own"]:
            reason = "ambiguous_damage_or_missing_own_context"
        if reason:
            self._observe(Skip(self._id("damage_skip"), entity, reason))
            return
        target, before, after = block["damage"][0]
        own = block["own"]
        own_public = self.entities.get(self.active.get(self.own_side), {})
        if not target.startswith(self.own_side + ":") or not before or not before.exact:
            self._observe(Skip(self._id("damage_skip"), entity, "incoming_only"))
            return
        if own_public.get("boosted") or own_public.get("tera") or own_public.get("unsafe"):
            self._observe(Skip(self._id("damage_skip"), entity, "unsupported_own_context"))
            return
        own["hp"] = before.value
        h = block["pre"].get(entity)
        context = (("own", json.dumps(own, sort_keys=True)), ("engine_context_supported", True),
                   ("opponent_status", block["state"]["status"]),
                   ("opponent_hp", "full" if h and h.value == h.denominator else "unknown"))
        self._observe(DamageEvidence(self._id("damage"), entity, block["move"], before, after,
                                     context=context, supported=True, attributes_at_event=True,
                                     item_at_event=block["state"]["item"], ability_at_event=block["state"]["ability"]))

    def _flush_speed(self):
        if len(self.order) != 2 or self.turn_context is None or self.turn_unsafe or self.field_unsafe:
            return
        active, entities, own_snapshots = self.turn_context
        (first, move1, called1), (second, move2, called2) = self.order
        if called1 or called2 or {first, second} != set(active.values()):
            return
        if move1 not in ORDINARY_MOVES or move2 not in ORDINARY_MOVES:
            return
        own_entity = active.get(self.own_side)
        opponent = second if first == own_entity else first
        if opponent in self.suspended:
            return
        own = own_snapshots.get(to_id(entities[own_entity]["species"]))
        state = entities[opponent]
        if not own or state["unsafe"] or state["boosted"] or state["tera"]:
            return
        try:
            speed = effective_speed(own["speed"], own["ability"], own["item"],
                                    stage=entities[own_entity]["stage"],
                                    paralyzed=entities[own_entity]["status"] == "par")
        except Unsupported as exc:
            self._observe(Skip(self._id("speed_skip"), opponent, str(exc)))
            return
        self._observe(SpeedEvidence(self._id("speed"), opponent, speed, first == opponent,
                                   opponent_stage=state["stage"], paralyzed=state["status"] == "par",
                                   supported=True, attributes_at_event=True,
                                   item_at_event=state["item"], ability_at_event=state["ability"]))

    def finish(self):
        self._flush_damage()
        self._flush_speed()
        self.order = []


def own_snapshot(state):
    """Allowlisted player-private projection; never reads state.side_two."""
    side = state.side_one
    out = []
    for p in side.pokemon:
        out.append(dict(species=p.id.lower(), level=p.level, hp=p.hp, maxhp=p.maxhp,
                        types=[str(t).lower() for t in p.types], attack=p.attack, defense=p.defense,
                        special_attack=p.special_attack, special_defense=p.special_defense, speed=p.speed,
                        ability=p.ability.lower(), item=p.item.lower(), status=p.status.lower(),
                        moves=[m.id.lower() for m in p.moves], weight_kg=p.weight_kg, tera_type=p.tera_type.lower()))
    return out
