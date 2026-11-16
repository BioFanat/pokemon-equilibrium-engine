"""Conservative bridge to poke-engine's documented damage-roll CLI.

POKE_ENGINE_BIN must name a Gen 9/Tera build of poke-engine 0.0.48. The CLI
contract is checked for every result: two explicit 16-roll lists, not extrema.
No dependency is downloaded and no approximate PMF is synthesized from maxima.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
from dataclasses import replace
from functools import lru_cache

from belief_likelihoods import DamagePMF, Unsupported

# A deliberately small, fixed-power, single-hit support surface. This is also
# the equal-priority whitelist: unsupported move effects must not be guessed.
ORDINARY_MOVES = frozenset("""
thunderbolt icebeam flamethrower earthquake shadowball psychic surf darkpulse
moonblast dragonpulse energyball airslash dazzlinggleam sludgebomb sludgewave
flashcannon bugbuzz powergem dragonclaw ironhead waterfall crunch leafblade
playrough xscissor rockslide stoneedge hydropump fireblast focusblast
""".split())
HP_ABILITIES = frozenset("blaze torrent overgrow swarm defeatist multiscale shadowshield berserk emergencyexit wimpout".split())
UNSUPPORTED_ITEMS = frozenset("lightclay scopelens leppaberry unknownitem".split())


def parse_rolls(output, incoming):
    lines = re.findall(r"Damage Rolls:\s*([0-9, ]+)", output)
    if len(lines) != 2:
        raise Unsupported("engine_damage_output_contract")
    all_rolls = [[int(x.strip()) for x in line.split(",") if x.strip()] for line in lines]
    rolls = all_rolls[1 if incoming else 0]
    if len(rolls) != 16 or rolls != sorted(rolls):
        raise Unsupported("engine_requires_16_explicit_rolls")
    return DamagePMF.from_uniform_rolls(rolls)


class PokeEngineDamageOracle:
    def __init__(self, executable=None, timeout=2.0, max_groups=64):
        if timeout <= 0 or max_groups < 1:
            raise ValueError("invalid damage oracle budget")
        self.executable = executable or os.environ.get("POKE_ENGINE_BIN")
        self.timeout = timeout
        self.max_groups = max_groups

    @staticmethod
    def signature(candidate, evidence, current_item, current_ability):
        return replace(candidate, moves=(evidence.move,), tera="normal",
                       item=candidate.item if current_item is None else current_item,
                       ability=candidate.ability if current_ability is None else current_ability)

    def prepare(self, candidates, evidence, current_item, current_ability):
        if len({self.signature(c, evidence, current_item, current_ability) for c in candidates}) > self.max_groups:
            raise Unsupported("damage_work_budget")

    def distribution(self, candidate, evidence, current_item=None, current_ability=None):
        if not self.executable:
            raise Unsupported("set_POKE_ENGINE_BIN_for_damage")
        if not evidence.incoming:
            raise Unsupported("outgoing_engine_context_not_implemented")
        if evidence.move not in ORDINARY_MOVES:
            raise Unsupported("damage_move_not_supported")
        item = candidate.item if current_item is None else current_item
        ability = candidate.ability if current_ability is None else current_ability
        if item in UNSUPPORTED_ITEMS:
            raise Unsupported("unsupported_engine_item")
        context = dict(evidence.context)
        if not context.get("engine_context_supported"):
            raise Unsupported("unsupported_damage_context")
        if ability in HP_ABILITIES and context.get("opponent_hp") != "full":
            raise Unsupported("unknown_hp_dependent_ability")
        signature = self.signature(candidate, evidence, current_item, current_ability)
        key_event = replace(evidence, event_id="", entity="", after=evidence.before)
        try:
            return self._cached(signature, key_event, item, ability)
        except (ValueError, TypeError, AttributeError, KeyError, IndexError) as exc:
            raise Unsupported("engine_state_contract:" + type(exc).__name__) from exc

    @lru_cache(maxsize=2048)
    def _cached(self, candidate, evidence, item, ability):
        try:
            import poke_engine as pe
            import hidden as hd
            import randbats as rb
        except ImportError as exc:
            raise Unsupported("poke_engine_unavailable") from exc
        ctx = dict(evidence.context)
        own = json.loads(ctx["own"])
        opponent = rb.to_pokemon(candidate.as_pokeset())
        opponent = hd.copy_pokemon(opponent, item=item, ability=ability, base_ability=ability,
                                   status=ctx.get("opponent_status", "none"))
        defender = pe.Pokemon(
            id=own["species"], level=own["level"], hp=own["hp"], maxhp=own["maxhp"],
            types=tuple(own["types"]), base_types=tuple(own["types"]),
            attack=own["attack"], defense=own["defense"], special_attack=own["special_attack"],
            special_defense=own["special_defense"], speed=own["speed"],
            ability=own["ability"], base_ability=own["ability"], item=own["item"],
            status=own["status"], moves=[pe.Move(id=m) for m in own["moves"]],
            weight_kg=own["weight_kg"], tera_type=own["tera_type"],
        )
        # Irrelevant bench slots are fainted copies, not guesses about hidden teammates.
        def side(p):
            return pe.Side(pokemon=[p] + [hd.copy_pokemon(p, hp=0) for _ in range(5)])
        state = pe.State(side_one=side(defender), side_two=side(opponent))
        own_move = own["moves"][0]
        try:
            result = subprocess.run([self.executable, "calculate-damage", "--state", state.to_string(),
                                     "-o", own_move, "-t", evidence.move],
                                    capture_output=True, text=True, timeout=self.timeout, check=True)
        except (OSError, subprocess.SubprocessError) as exc:
            raise Unsupported("engine_damage_command_failed") from exc
        return parse_rolls(result.stdout, evidence.incoming)
