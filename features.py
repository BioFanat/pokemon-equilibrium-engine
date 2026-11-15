"""Species-agnostic, full-information position encoding for the value net.

encode(state) = [side-one block | side-two block | field]. No feature in a side block depends on
the other side and the field is shared, so swapping sides is a column permutation:
encode(nash._mirror(s)) == encode(s)[MIRROR_PERM]. Training and inference use that identity.

Per mon (6 slots): hp/maxhp, alive, active, status one-hot, type one-hot (base types, plus the
Tera type once terastallized), atk/def/spa/spd/spe / 400, level / 100, terastallized, and from
data/moves_gen9.json: per-type max base power / 100 over its moves (18) and the counts of
physical / special / status moves / 4. Per side: boosts / 6, side conditions, force_switch.
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import numpy as np

STATUS = ["none", "burn", "freeze", "paralyze", "poison", "sleep", "toxic"]
WEATHER = ["none", "sun", "rain", "sand", "hail", "snow", "harshsun", "heavyrain"]
TERRAIN = ["none", "grassyterrain", "electricterrain", "mistyterrain", "psychicterrain"]
TYPES = ["normal", "fire", "water", "electric", "grass", "ice", "fighting", "poison", "ground",
         "flying", "psychic", "bug", "rock", "ghost", "dragon", "dark", "steel", "fairy"]
CATS = ["physical", "special", "status"]
TI = {t: i for i, t in enumerate(TYPES)}

MON_F = 3 + len(STATUS) + len(TYPES) + 5 + 1 + 1 + len(TYPES) + len(CATS)
SIDE_F = 6 * MON_F + 7 + 9
FIELD_F = len(WEATHER) + len(TERRAIN) + 1
DIM = 2 * SIDE_F + FIELD_F
MIRROR_PERM = np.r_[np.arange(SIDE_F, 2 * SIDE_F), np.arange(SIDE_F), np.arange(2 * SIDE_F, DIM)]


@lru_cache(maxsize=None)
def moves_table() -> dict:
    return json.loads((Path(__file__).parent / "data" / "moves_gen9.json").read_text())


def _idx(lst, v) -> int:
    v = str(v).lower()
    return lst.index(v) if v in lst else 0


def _active(side) -> int:
    return int(str(side.active_index)[-1])


def _mon(out: np.ndarray, p, active: bool) -> None:
    out[0] = p.hp / p.maxhp if p.maxhp else 0.0
    out[1] = p.hp > 0
    out[2] = active
    out[3 + _idx(STATUS, p.status)] = 1
    o = 3 + len(STATUS)
    types = list(p.base_types) + ([p.tera_type] if p.terastallized else [])
    for t in types:                   # engine returns upper case after a step ('GRASS', 'TYPELESS')
        t = str(t).lower()
        if t in TI:
            out[o + TI[t]] = 1
    o += len(TYPES)
    out[o:o + 5] = np.array([p.attack, p.defense, p.special_attack, p.special_defense, p.speed]) / 400.0
    out[o + 5] = p.level / 100.0
    out[o + 6] = p.terastallized
    o += 7
    mt = moves_table()
    for m in p.moves:
        d = mt.get(m.id.lower())      # engine ids are upper case ('EARTHQUAKE'); 'NONE' is absent
        if d is None:
            continue
        if d["category"] != "status" and d["type"] in TI:
            out[o + TI[d["type"]]] = max(out[o + TI[d["type"]]], d["basePower"] / 100.0)
        out[o + len(TYPES) + CATS.index(d["category"])] += 0.25


def enc_side(side) -> np.ndarray:
    act = _active(side)
    v = np.zeros(SIDE_F, np.float32)
    for i, p in enumerate(side.pokemon[:6]):
        _mon(v[i * MON_F:(i + 1) * MON_F], p, i == act)
    o, sc = 6 * MON_F, side.side_conditions
    v[o:o + 7] = np.array([side.attack_boost, side.defense_boost, side.special_attack_boost,
                           side.special_defense_boost, side.speed_boost, side.accuracy_boost,
                           side.evasion_boost]) / 6.0
    v[o + 7:] = [sc.stealth_rock, sc.spikes / 3, sc.toxic_spikes / 2, sc.sticky_web, sc.reflect > 0,
                 sc.light_screen > 0, sc.aurora_veil > 0, sc.tailwind > 0, side.force_switch]
    return v


def encode(state) -> np.ndarray:
    """Side-one perspective feature vector of length DIM."""
    field = np.zeros(FIELD_F, np.float32)
    field[_idx(WEATHER, state.weather)] = 1
    field[len(WEATHER) + _idx(TERRAIN, state.terrain)] = 1
    field[-1] = state.trick_room
    return np.concatenate([enc_side(state.side_one), enc_side(state.side_two), field])


def encode_pair(state) -> tuple[np.ndarray, np.ndarray]:
    """(x, x_mirror) with x_mirror = encode of the side-swapped position."""
    import nash
    return encode(state), encode(nash._mirror(state))


def encode_many(states) -> np.ndarray:
    return np.stack([encode(s) for s in states]) if states else np.zeros((0, DIM), np.float32)
