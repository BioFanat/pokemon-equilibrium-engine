"""Gen 9 Random Battle teams for poke-engine, from Showdown's published set data.

Data (see data/SOURCES.md): per species a level and a list of roles; each role lists the
possible abilities, items, Tera types and a move pool (often > 4 moves), plus optional
EV/IV overrides (default 84 EVs / 31 IVs, neutral nature).

`sample_set` draws role -> 4 moves -> ability -> item -> Tera type uniformly. Showdown's real
generator adds rules on top (guaranteed STAB, at most one setup move, team-level limits such as
one weather setter); this is an approximation of its distribution, not a reimplementation.
The same role structure is the hypothesis space for a Bayesian belief over hidden sets.

Engine coverage was checked against poke-engine 0.0.48's enums: all 509 species, every move
and ability are known; 3 items (Light Clay, Scope Lens, Leppa Berry; 7 of 873 roles) are not and
silently become UNKNOWNITEM (no effect).
"""
from __future__ import annotations

import json
import random
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import poke_engine as pe

DATA = Path(__file__).parent / "data"
STATS = ("hp", "atk", "def", "spa", "spd", "spe")


def to_id(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", s.lower())


@lru_cache(maxsize=None)
def sets() -> dict:
    return json.loads((DATA / "gen9randombattle.json").read_text())


@lru_cache(maxsize=None)
def dex() -> dict:
    return json.loads((DATA / "dex_gen9randbats.json").read_text())


@dataclass(frozen=True)
class PokeSet:
    species: str          # Showdown display name, e.g. "Ogerpon-Wellspring"
    role: str
    level: int
    ability: str
    item: str
    tera: str
    moves: tuple[str, ...]
    evs: tuple[int, ...]  # hp atk def spa spd spe
    ivs: tuple[int, ...]


def sample_set(species: str, rng: random.Random) -> PokeSet:
    e = sets()[species]
    role_name = rng.choice(sorted(e["roles"]))
    role = e["roles"][role_name]
    pool = role["moves"]
    moves = tuple(rng.sample(pool, min(4, len(pool))))
    abilities = role.get("abilities") or e["abilities"]
    items = role.get("items") or e.get("items") or ["None"]
    ev = {**e.get("evs", {}), **role.get("evs", {})}
    iv = {**e.get("ivs", {}), **role.get("ivs", {})}
    return PokeSet(species, role_name, e["level"], rng.choice(abilities), rng.choice(items),
                   rng.choice(role["teraTypes"]), moves,
                   tuple(ev.get(s, 84) for s in STATS), tuple(iv.get(s, 31) for s in STATS))


def _stat(base: int, level: int, ev: int, iv: int, hp: bool) -> int:
    core = (2 * base + iv + ev // 4) * level // 100          # neutral nature
    return core + level + 10 if hp else core + 5


def to_pokemon(s: PokeSet, hp_frac: float = 1.0) -> pe.Pokemon:
    d = dex()[to_id(s.species)]
    b = d["baseStats"]
    st = [_stat(b[k], s.level, ev, iv, k == "hp") for k, ev, iv in zip(STATS, s.evs, s.ivs)]
    if to_id(s.species) == "shedinja":
        st[0] = 1
    types = tuple(t.lower() for t in d["types"]) + ("typeless",) * (2 - len(d["types"]))
    ability = to_id(s.ability)
    return pe.Pokemon(
        id=to_id(s.species), level=s.level, types=types, base_types=types,
        hp=max(1, int(st[0] * hp_frac)), maxhp=st[0], ability=ability, base_ability=ability,
        item=to_id(s.item), tera_type=s.tera.lower(), evs=s.evs,
        attack=st[1], defense=st[2], special_attack=st[3], special_defense=st[4], speed=st[5],
        weight_kg=float(d["weightkg"]), moves=[pe.Move(id=to_id(m)) for m in s.moves],
    )


def sample_team(rng: random.Random, n: int = 6, exclude: set[str] = frozenset()) -> list[PokeSet]:
    """n distinct species (species clause on the base name, e.g. one Ogerpon forme)."""
    names, bases = [], set(exclude)
    for sp in rng.sample(sorted(sets()), len(sets())):
        base = to_id(sp.split("-")[0])
        if base not in bases:
            names.append(sp); bases.add(base)
        if len(names) == n:
            break
    return [sample_set(sp, rng) for sp in names]


def random_state(rng: random.Random) -> pe.State:
    t1 = sample_team(rng)
    t2 = sample_team(rng)
    side = lambda t: pe.Side(pokemon=[to_pokemon(s) for s in t])
    return pe.State(side_one=side(t1), side_two=side(t2))
