"""Minimal full-battle runner on poke-engine (Gen 9 + Terastallization build).

Plays complete 6v6 singles battles: each turn both sides pick an option, poke-engine
enumerates every chance outcome with its probability, we sample one and apply it.
Teams: Showdown Gen 9 Random Battle sets by default (randbats.py); TEAMS='hand' gives the 12 hand-built sets.
"""
from __future__ import annotations

import random
import time

import poke_engine as pe

# name: (types, base stats hp/atk/def/spa/spd/spe, ability, item, tera, moves, weight_kg)
SETS = {
    "garchomp": (("dragon", "ground"), (108, 130, 95, 80, 85, 102), "roughskin", "lifeorb", "steel",
                 ["earthquake", "dragonclaw", "swordsdance", "stoneedge"], 95.0),
    "rotomwash": (("electric", "water"), (50, 65, 107, 105, 107, 86), "levitate", "leftovers", "water",
                  ["hydropump", "voltswitch", "willowisp", "painsplit"], 0.3),
    "greattusk": (("ground", "fighting"), (115, 131, 131, 53, 53, 87), "protosynthesis", "boosterenergy", "ground",
                  ["headlongrush", "closecombat", "icespinner", "rapidspin"], 320.0),
    "gholdengo": (("steel", "ghost"), (87, 60, 95, 133, 91, 84), "goodasgold", "choicescarf", "steel",
                  ["makeitrain", "shadowball", "focusblast", "trick"], 30.0),
    "kingambit": (("dark", "steel"), (100, 135, 120, 60, 85, 50), "supremeoverlord", "leftovers", "dark",
                  ["kowtowcleave", "suckerpunch", "ironhead", "swordsdance"], 120.0),
    "dragonite": (("dragon", "flying"), (91, 134, 95, 100, 100, 80), "multiscale", "heavydutyboots", "normal",
                  ["extremespeed", "dragondance", "earthquake", "roost"], 210.0),
    "corviknight": (("flying", "steel"), (98, 87, 105, 53, 85, 67), "pressure", "leftovers", "dragon",
                    ["bravebird", "uturn", "roost", "defog"], 75.0),
    "toxapex": (("poison", "water"), (50, 63, 152, 53, 142, 35), "regenerator", "blacksludge", "fairy",
                ["toxic", "recover", "haze", "surf"], 14.5),
    "ironvaliant": (("fairy", "fighting"), (74, 130, 90, 120, 60, 116), "quarkdrive", "boosterenergy", "fairy",
                    ["moonblast", "closecombat", "knockoff", "swordsdance"], 35.0),
    "volcarona": (("bug", "fire"), (85, 60, 65, 135, 105, 100), "flamebody", "heavydutyboots", "grass",
                  ["quiverdance", "fierydance", "gigadrain", "morningsun"], 46.0),
    "clefable": (("fairy", "typeless"), (95, 70, 73, 95, 90, 60), "magicguard", "lifeorb", "steel",
                 ["moonblast", "flamethrower", "calmmind", "moonlight"], 40.0),
    "heatran": (("fire", "steel"), (91, 90, 106, 130, 106, 77), "flashfire", "leftovers", "grass",
                ["magmastorm", "earthpower", "stealthrock", "taunt"], 430.0),
}


def _stat(base: int, level: int, hp: bool = False) -> int:
    core = (2 * base + 31 + 84 // 4) * level // 100   # randbats: 31 IVs, 84 EVs, neutral nature
    return core + level + 10 if hp else core + 5


def make_pokemon(name: str, level: int = 82, hp_frac: float = 1.0) -> pe.Pokemon:
    types, b, ability, item, tera, moves, wt = SETS[name]
    hp = _stat(b[0], level, hp=True)
    return pe.Pokemon(
        id=name, level=level, types=types, base_types=types, hp=max(1, int(hp * hp_frac)), maxhp=hp,
        ability=ability, base_ability=ability, item=item, tera_type=tera,
        evs=(84,) * 6, attack=_stat(b[1], level), defense=_stat(b[2], level),
        special_attack=_stat(b[3], level), special_defense=_stat(b[4], level),
        speed=_stat(b[5], level), weight_kg=wt, moves=[pe.Move(id=m) for m in moves],
    )


TEAMS = "randbats"   # "randbats" (Showdown Gen 9 Random Battle sets) or "hand" (the 12 sets above)


def random_state(rng: random.Random, teams: str | None = None) -> pe.State:
    if (teams or TEAMS) == "randbats":
        import randbats
        return randbats.random_state(rng)
    names = rng.sample(sorted(SETS), 12)
    side = lambda ns: pe.Side(pokemon=[make_pokemon(n) for n in ns])
    return pe.State(side_one=side(names[:6]), side_two=side(names[6:]))


def options(state: pe.State) -> tuple[list[str], list[str]]:
    """Legal options for both sides, read off a zero-time expectiminimax root (~0.1 ms;
    a 1-iteration MCTS gives the same list but costs ~1.6 ms of thread overhead)."""
    r = pe.id(state, 0)
    return [_canon(x) for x in r.s1], [_canon(x) for x in r.s2]


def _canon(choice: str) -> str:
    """MCTS display strings ('switch volcarona', 'No Move') -> generate_instructions input."""
    if choice.startswith("switch "):
        return choice[len("switch "):]
    return "none" if choice == "No Move" else choice


def alive(side: pe.Side) -> int:
    return sum(p.hp > 0 for p in side.pokemon)


def step(state: pe.State, m1: str, m2: str, rng: random.Random) -> pe.State:
    branches = pe.generate_instructions(state, m1, m2)
    pick = rng.choices(branches, weights=[b.percentage for b in branches])[0]
    return state.apply_instructions(pick)


def play(policy1, policy2, rng: random.Random, max_turns: int = 300, log=None,
         teams: str | None = None) -> tuple[int, int]:
    """Returns (winner: 1/2/0 for draw, turns)."""
    state = random_state(rng, teams)
    for turn in range(1, max_turns + 1):
        o1, o2 = options(state)
        m1, m2 = policy1(state, o1, 1, rng), policy2(state, o2, 2, rng)
        if log:
            log(turn, m1, m2)
        state = step(state, m1, m2, rng)
        a1, a2 = alive(state.side_one), alive(state.side_two)
        if a1 == 0 or a2 == 0:
            return (0 if a1 == a2 else (1 if a1 else 2)), turn
    return 0, max_turns


def random_policy(state, opts, side, rng):
    return rng.choice(opts)


def mcts_policy(iterations: int):
    def pol(state, opts, side, rng):
        if len(opts) == 1:
            return opts[0]
        r = pe.mcts(state, 0, iterations, 1)
        res = r.s1 if side == 1 else r.s2
        return _canon(max(res, key=lambda x: x.visits).move_choice)
    return pol


if __name__ == "__main__":
    rng = random.Random(0)
    print("--- one logged battle, MCTS(500) vs random ---")
    w, t = play(mcts_policy(500), random_policy, rng, log=lambda t, a, b: t <= 6 and print(f"turn {t}: {a:22s} | {b}"))
    print(f"winner: side {w} after {t} turns")

    n, t0, turns = 200, time.perf_counter(), 0
    for _ in range(n):
        turns += play(random_policy, random_policy, rng)[1]
    dt = time.perf_counter() - t0
    print(f"random vs random: {n} battles, {turns} turns in {dt:.2f}s -> {n/dt:.0f} battles/s, {turns/dt:.0f} turns/s (1 core, incl. Python)")

    n, t0, wins = 40, time.perf_counter(), 0
    for _ in range(n):
        wins += play(mcts_policy(500), random_policy, rng)[0] == 1
    print(f"MCTS(500 it) vs random: {wins}/{n} wins, {time.perf_counter()-t0:.1f}s")
