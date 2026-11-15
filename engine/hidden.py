"""Hidden-information model for Random Battles on poke-engine.

Option strings (pkbattle._canon of pe.id output): `<moveid>`, `<moveid>-tera`, `<pokemonid>`
(switch), `none`.

Reveal rules (per observing side, about the opponent; everything keyed by opponent SLOT index,
since active_index / last_used_move / future_sight refer to slots and ids can forme-change):
  * species: a slot is revealed when it is the opponent's active mon before or after a step, or
    appears in a `Switch SideX: Pa -> Pb` instruction of the applied chance branch.
  * move: revealed when the opponent's chosen option names a move of its pre-step active mon.
    Approximation: the engine emits no "move used" instruction, so a move chosen but not executed
    (full paralysis, flinch, sleep, fainting first) still counts as revealed. Exception: a move
    deferred by a U-turn-style switch (SetSide*MoveSecondSwitchOutMove instruction) is revealed
    only when it is actually executed next step.
  * tera type: revealed when the mon is terastallized (public flag).
  * item and ability stay hidden, except that an item being consumed / removed is public
    (the determinized item is then "none"); item and ability (TODO: item/ability inference, behavioural belief updates).
HP is public as a percentage (Showdown rule: ceil(100*hp/maxhp), 99 if not full); set
EXACT_HP = True to expose the exact fraction instead.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field, replace
from functools import lru_cache

import poke_engine as pe

import randbats as rb

EXACT_HP = False
CHOICE_ITEMS = {"choiceband", "choicespecs", "choicescarf"}
PF = ["id", "level", "types", "base_types", "hp", "maxhp", "ability", "base_ability", "item", "nature",
      "evs", "attack", "defense", "special_attack", "special_defense", "speed", "status", "rest_turns",
      "sleep_turns", "weight_kg", "mega_evolved", "terastallized", "tera_type"]
SF = ["side_conditions", "active_index", "baton_passing", "shed_tailing", "volatile_status_durations",
      "wish", "future_sight", "force_switch", "force_trapped", "slow_uturn_move", "volatile_statuses",
      "substitute_health", "attack_boost", "defense_boost", "special_attack_boost",
      "special_defense_boost", "speed_boost", "accuracy_boost", "evasion_boost", "last_used_move",
      "switch_out_move_second_saved_move"]
GF = ["weather", "weather_turns_remaining", "terrain", "terrain_turns_remaining", "trick_room",
      "trick_room_turns_remaining", "team_preview"]
PUBLIC_PF = ["id", "types", "base_types", "status", "rest_turns", "sleep_turns", "weight_kg",
             "mega_evolved", "terastallized"]


# ---------------------------------------------------------------- copying (object route)

def copy_pokemon(p, **over) -> pe.Pokemon:
    kw = {f: getattr(p, f) for f in PF}
    kw["moves"] = [pe.Move(id=m.id, pp=m.pp, disabled=m.disabled) for m in p.moves]
    kw.update(over)
    return pe.Pokemon(**kw)


def copy_side(s, pokemon=None, **over) -> pe.Side:
    kw = {f: getattr(s, f) for f in SF}
    kw.update(over)
    return pe.Side(pokemon=pokemon if pokemon is not None else [copy_pokemon(p) for p in s.pokemon], **kw)


def make_state(side_one, side_two, fields: dict) -> pe.State:
    return pe.State(side_one=side_one, side_two=side_two, **fields)


# ---------------------------------------------------------------- species lookup

@lru_cache(maxsize=None)
def _id_map() -> dict:
    return {rb.to_id(k): k for k in rb.sets()}


def species_key(engine_id: str) -> str:
    m, engine_id = _id_map(), engine_id.lower()
    if engine_id in m:
        return m[engine_id]
    pre = [k for k in m if engine_id.startswith(k)]          # forme change, e.g. palafinhero
    if not pre:
        raise KeyError(engine_id)
    return m[max(pre, key=len)]


def base_of(species: str) -> str:
    return rb.to_id(species.split("-")[0])


# ---------------------------------------------------------------- reveal tracking

@dataclass
class SlotReveal:
    species: str
    moves: dict = field(default_factory=dict)   # move slot index -> engine move id
    tera: str | None = None


def _sides(state, side):
    return (state.side_one, state.side_two) if side == 1 else (state.side_two, state.side_one)


class Tracker:
    """What `side` has seen of the opponent so far."""

    def __init__(self, side: int, state: pe.State):
        self.side = side
        self.slots: dict[int, SlotReveal] = {}
        self.last_move: str | None = None      # opponent active's last executed move since switch-in
        self._see(state)

    def _reveal(self, opp, i):
        if i not in self.slots:
            self.slots[i] = SlotReveal(species_key(opp.pokemon[i].id))

    def _see(self, state):
        _, opp = _sides(state, self.side)
        self._reveal(opp, int(opp.active_index))
        for i, sr in self.slots.items():
            p = opp.pokemon[i]
            if p.terastallized:
                sr.tera = p.tera_type

    def update(self, pre, m1, m2, branch, post):
        _, opp = _sides(pre, self.side)
        tag = "SideTwo" if self.side == 1 else "SideOne"
        choice = m2 if self.side == 1 else m1
        reprs = [repr(x) for x in branch.instruction_list]
        oi = int(opp.active_index)
        mv = choice[:-5] if choice.endswith("-tera") else choice
        ids = [m.id.lower() for m in opp.pokemon[oi].moves]
        deferred = any(r.startswith(f"Set{tag}MoveSecondSwitchOutMove") for r in reprs)
        switched = [int(r.split("-> P")[1]) for r in reprs if r.startswith(f"Switch {tag}:")]
        used = mv in ids and not deferred
        if used:
            self.slots[oi].moves[ids.index(mv)] = mv
        if switched:
            self.last_move = None
        elif used:
            self.last_move = mv
        for j in switched:
            self._reveal(opp, j)
        self._see(post)


# ---------------------------------------------------------------- view

def hp_public(hp: int, maxhp: int) -> float:
    if EXACT_HP or hp <= 0:
        return hp / maxhp
    pct = math.ceil(100 * hp / maxhp)
    return (99 if pct == 100 and hp < maxhp else pct) / 100


@dataclass
class View:
    side: int
    own: object                  # own pe.Side (exact)
    opp_fields: dict             # opponent side-level fields (public)
    opp_public: dict             # slot -> public per-mon info
    reveals: dict                # slot -> SlotReveal
    unrevealed: list             # opponent slots not yet seen
    field: dict                  # weather / terrain / trick room
    opp_pending_move: bool       # opponent's move is deferred while we pick a U-turn switch-in
    opp_last_move: str | None


def make_view(state: pe.State, tr: Tracker) -> View:
    own, opp = _sides(state, tr.side)
    pub = {}
    for i in tr.slots:
        p = opp.pokemon[i]
        d = {f: getattr(p, f) for f in PUBLIC_PF}
        d["hp_frac"] = hp_public(p.hp, p.maxhp)
        d["item_gone"] = p.item.lower() == "none"     # consumed / knocked off is announced publicly
        pub[i] = d
    fields = {f: getattr(opp, f) for f in SF}
    pending = own.force_switch and fields["switch_out_move_second_saved_move"] != "NONE"
    return View(tr.side, own, fields, pub, {i: SlotReveal(s.species, dict(s.moves), s.tera) for i, s in tr.slots.items()},
                [i for i in range(6) if i not in tr.slots], {f: getattr(state, f) for f in GF},
                pending, tr.last_move)


# ---------------------------------------------------------------- belief

def _role_lists(species, role_name):
    e = rb.sets()[species]
    role = e["roles"][role_name]
    return (role["moves"], role.get("abilities") or e["abilities"],
            role.get("items") or e.get("items") or ["None"], role["teraTypes"])


@lru_cache(maxsize=None)
def role_posterior(species: str, moves: frozenset, tera: str | None) -> tuple:
    """Exact posterior over roles under sample_set's prior given hard evidence:
    w_r ∝ (1/R) · C(n-k, 4-k)/C(n, 4) [revealed ⊆ pool] · count(tera)/|teraTypes|.
    Within a role, consistent sets are distributed as the prior restricted to the evidence
    (uniform completions of the move subset, ability/item/tera by list frequency)."""
    roles = sorted(rb.sets()[species]["roles"])
    w = []
    for r in roles:
        pool, _, _, teras = _role_lists(species, r)
        ids = [rb.to_id(m) for m in pool]
        n, k = len(ids), len(moves)
        if not moves <= set(ids):
            w.append(0.0); continue
        p = math.comb(n - k, min(4, n) - k) / math.comb(n, min(4, n)) if n > 4 else 1.0
        if tera is not None:
            p *= sum(t.lower() == tera for t in teras) / len(teras)
        w.append(p / len(roles))
    s = sum(w)
    if s == 0:                                   # evidence outside every pool: fall back to prior
        return tuple((r, 1 / len(roles)) for r in roles)
    return tuple((r, x / s) for r, x in zip(roles, w))


def make_set(species, role_name, moves, ability, item, tera) -> rb.PokeSet:
    e = rb.sets()[species]
    role = e["roles"][role_name]
    ev = {**e.get("evs", {}), **role.get("evs", {})}
    iv = {**e.get("ivs", {}), **role.get("ivs", {})}
    return rb.PokeSet(species, role_name, e["level"], ability, item, tera, tuple(moves),
                      tuple(ev.get(s, 84) for s in rb.STATS), tuple(iv.get(s, 31) for s in rb.STATS))


class Belief:
    def __init__(self, view: View):
        self.view = view
        self.post = {i: role_posterior(sr.species, frozenset(sr.moves.values()), sr.tera)
                     for i, sr in view.reveals.items()}

    def _evidence(self, slot):
        sr = self.view.reveals[slot]
        return sr, set(sr.moves.values())

    def sample(self, slot: int, rng: random.Random) -> rb.PokeSet:
        sr, rev = self._evidence(slot)
        roles, ws = zip(*self.post[slot])
        r = rng.choices(roles, weights=ws)[0]
        pool, abil, items, teras = _role_lists(sr.species, r)
        known = [m for m in pool if rb.to_id(m) in rev]
        rest = [m for m in pool if rb.to_id(m) not in rev]
        moves = known + rng.sample(rest, max(0, min(4, len(pool)) - len(known)))
        tera = next((t for t in teras if t.lower() == sr.tera), sr.tera) if sr.tera else rng.choice(teras)
        return make_set(sr.species, r, moves, rng.choice(abil), rng.choice(items), tera)

    def map_set(self, slot: int) -> rb.PokeSet:
        """Single most probable full set. Within a role all move completions are equiprobable, so
        the score is w_r / C(n-k, 4-k) · max P(ability) · max P(item) · max P(tera | evidence).
        Ties: first role in sorted order; moves = revealed + first unrevealed in pool order;
        ability/item/tera = most frequent, first in list order."""
        sr, rev = self._evidence(slot)
        best = None
        for r, w in self.post[slot]:
            if w <= 0:
                continue
            pool, abil, items, teras = _role_lists(sr.species, r)
            n, k = len(pool), len(rev)
            pmax = lambda L: max(L.count(x) for x in L) / len(L)
            sc = w / math.comb(n - k, max(0, min(4, n) - k)) * pmax(abil) * pmax(items) * (1 if sr.tera else pmax(teras))
            if best is None or sc > best[0] + 1e-15:
                best = (sc, r)
        r = best[1]
        pool, abil, items, teras = _role_lists(sr.species, r)
        known = [m for m in pool if rb.to_id(m) in rev]
        rest = [m for m in pool if rb.to_id(m) not in rev]
        moves = known + rest[:max(0, min(4, len(pool)) - len(known))]
        mode = lambda L: max(L, key=L.count)
        tera = next((t for t in teras if t.lower() == sr.tera), sr.tera) if sr.tera else mode(teras)
        return make_set(sr.species, r, moves, mode(abil), mode(items), tera)

    def sample_unrevealed(self, rng: random.Random) -> dict:
        """Species uniform over keys whose base is not already on the opponent's team (only the
        opponent's revealed species are excluded; teams are independent), then the set prior."""
        taken = {base_of(sr.species) for sr in self.view.reveals.values()}
        out, keys = {}, sorted(rb.sets())
        for i in self.view.unrevealed:
            sp = rng.choice([k for k in keys if base_of(k) not in taken])
            taken.add(base_of(sp))
            out[i] = rb.sample_set(sp, rng)
        return out


class FixedBelief(Belief):
    """Belief collapsed onto given sets (tests)."""

    def __init__(self, view, sets_by_slot):
        super().__init__(view)
        self.fixed = sets_by_slot

    def sample(self, slot, rng):
        return self.fixed[slot]

    map_set = lambda self, slot: self.fixed[slot]

    def sample_unrevealed(self, rng):
        return {i: self.fixed[i] for i in self.view.unrevealed}


# ---------------------------------------------------------------- determinization

def _revealed_mon(s: rb.PokeSet, pub: dict, sr: SlotReveal, active: bool, last_move) -> pe.Pokemon:
    base = rb.to_pokemon(s)
    sampled = [rb.to_id(m) for m in s.moves if rb.to_id(m) not in sr.moves.values()]
    n = max(len(s.moves), max(sr.moves, default=-1) + 1)
    ids = [sr.moves.get(j) for j in range(n)]
    it = iter(sampled)
    ids = [m if m is not None else next(it, "none") for m in ids]
    item = "none" if pub["item_gone"] else base.item
    lock = active and item in CHOICE_ITEMS and last_move in ids
    moves = [pe.Move(id=m, disabled=bool(lock and m != last_move)) for m in ids]
    if pub["hp_frac"] <= 0:
        hp = 0
    else:
        hp = min(base.maxhp, max(1, round(pub["hp_frac"] * base.maxhp)))
    over = {f: pub[f] for f in PUBLIC_PF}
    cur = pub["id"].lower()
    if cur != rb.to_id(s.species) and cur in rb.dex():   # public forme change: that forme's stats
        alt = rb.to_pokemon(replace(s, species=cur))
        over.update({f: getattr(alt, f) for f in ("attack", "defense", "special_attack", "special_defense", "speed")})
    return copy_pokemon(base, hp=hp, moves=moves, item=item, **over)


def determinize(true_state, view: View, belief: Belief, rng: random.Random, map_: bool = False) -> pe.State:
    """Full state consistent with `view`. Reads only `view` (true_state is accepted for the agent
    signature). Own side copied exactly; revealed opponent mons keep public info with set-dependent
    fields from a belief sample (or MAP set); unrevealed slots get sampled species+sets at full HP."""
    own = copy_side(view.own)
    opp_act = int(view.opp_fields["active_index"])
    unrev = belief.sample_unrevealed(rng)
    mons = []
    for i in range(6):
        if i in view.reveals:
            s = belief.map_set(i) if map_ else belief.sample(i, rng)
            mons.append(_revealed_mon(s, view.opp_public[i], view.reveals[i], i == opp_act, view.opp_last_move))
        else:
            mons.append(rb.to_pokemon(unrev[i]))
    fields = dict(view.opp_fields)
    if view.opp_pending_move:                     # deferred opponent move: unknown, sample from its set
        cand = [m.id for m in mons[opp_act].moves if m.id not in ("none", "")]
        if cand:
            fields["switch_out_move_second_saved_move"] = rng.choice(cand).upper()
    opp = pe.Side(pokemon=mons, **fields)
    return make_state(own, opp, view.field) if view.side == 1 else make_state(opp, own, view.field)
