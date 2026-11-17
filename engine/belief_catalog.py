"""Exact, dependency-free catalog for randbats.sample_set's approximate prior.

Role names are provenance, not part of a mechanics hypothesis: duplicate paths
are summed. Importing this module never imports poke_engine.
"""
from __future__ import annotations

import itertools
import json
import math
import re
from collections import Counter
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

DATA = Path(__file__).resolve().parent / "data"
STATS = ("hp", "atk", "def", "spa", "spd", "spe")


def to_id(value):
    return re.sub(r"[^a-z0-9]", "", str(value).lower())


@dataclass(frozen=True)
class Candidate:
    species: str
    level: int
    moves: tuple[str, ...]
    ability: str
    item: str
    tera: str
    evs: tuple[int, ...]
    ivs: tuple[int, ...]

    def as_pokeset(self):
        # Lazy: catalog/update/replay tests need no Rust engine.
        from randbats import PokeSet
        return PokeSet(self.species, "catalog", self.level, self.ability,
                       self.item, self.tera, self.moves, self.evs, self.ivs)


@dataclass(frozen=True)
class CatalogEntry:
    candidate: Candidate
    prior: float


def enumerate_species(species, entry):
    mass = Counter()
    roles = entry["roles"]
    for name in sorted(roles):
        role = roles[name]
        pool = role["moves"]
        subsets = list(itertools.combinations(pool, min(4, len(pool))))
        if not subsets:
            raise ValueError(f"empty move pool: {species}/{name}")
        ability = Counter(map(to_id, role.get("abilities") or entry["abilities"]))
        item = Counter(map(to_id, role.get("items") or entry.get("items") or ["None"]))
        tera = Counter(map(to_id, role["teraTypes"]))
        ev = {**entry.get("evs", {}), **role.get("evs", {})}
        iv = {**entry.get("ivs", {}), **role.get("ivs", {})}
        denom = len(roles) * len(subsets) * sum(ability.values()) * sum(item.values()) * sum(tera.values())
        for moves, a, i, t in itertools.product(subsets, ability, item, tera):
            c = Candidate(species, entry["level"], tuple(sorted(map(to_id, moves))), a, i, t,
                          tuple(ev.get(s, 84) for s in STATS), tuple(iv.get(s, 31) for s in STATS))
            mass[c] += ability[a] * item[i] * tera[t] / denom
    total = math.fsum(mass.values())
    if not math.isclose(total, 1.0, abs_tol=1e-12):
        raise ValueError(f"invalid catalog prior for {species}: {total}")
    return tuple(CatalogEntry(c, p / total) for c, p in mass.items())


class Catalog:
    def __init__(self, data=None, dex=None):
        self.data = data if data is not None else json.loads((DATA / "gen9randombattle.json").read_text())
        self.dex = dex if dex is not None else json.loads((DATA / "dex_gen9randbats.json").read_text())
        self.names = {to_id(k): k for k in self.data}
        self._entries = {}

    def entries(self, species):
        name = self.names.get(to_id(species))
        if name is None:
            raise KeyError(f"species outside catalog: {species}")
        if name not in self._entries:
            self._entries[name] = enumerate_species(name, self.data[name])
        return self._entries[name]

    def stats(self, candidate):
        base = self.dex[to_id(candidate.species)]["baseStats"]
        out = {}
        for s, ev, iv in zip(STATS, candidate.evs, candidate.ivs):
            core = (2 * base[s] + iv + ev // 4) * candidate.level // 100
            out[s] = core + (candidate.level + 10 if s == "hp" else 5)
        if to_id(candidate.species) == "shedinja":
            out["hp"] = 1
        return out


@lru_cache(maxsize=1)
def default_catalog():
    return Catalog()
