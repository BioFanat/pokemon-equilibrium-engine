"""Mechanics likelihoods, deliberately separate from strategic payoff evaluation.

Damage oracles return an actual PMF (or raise Unsupported). Min/max ranges and
the engine's heuristic payoff values are never treated as probabilities.
"""
from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass

from belief_catalog import to_id


class Unsupported(Exception):
    pass


@dataclass(frozen=True)
class DamagePMF:
    outcomes: tuple[tuple[int, float], ...]

    def __post_init__(self):
        if not self.outcomes or any(d < 0 or not isinstance(d, int) or p < 0 or not math.isfinite(p)
                                    for d, p in self.outcomes):
            raise ValueError("invalid damage PMF")
        if not math.isclose(math.fsum(p for _, p in self.outcomes), 1.0, abs_tol=1e-9):
            raise ValueError("damage probabilities must sum to one")

    @classmethod
    def from_uniform_rolls(cls, rolls):
        rolls = list(rolls)
        if not rolls:
            raise Unsupported("empty_damage_rolls")
        counts = Counter(rolls)
        return cls(tuple((d, n / len(rolls)) for d, n in sorted(counts.items())))


# Only abilities whose speed behavior is irrelevant are allowed here. An unknown
# ability must skip the entire factor; assigning it likelihood zero would bias
# inference toward the subset the implementation happens to understand.
SPEED_NEUTRAL_ABILITIES = frozenset("""
levitate roughskin ironbarbs pressure intimidate static flamebody flashfire
waterabsorb voltabsorb lightningrod stormdrain sturdy naturalcure regenerator
magicguard unaware overcoat infiltrator clearbody innerfocus soundproof
technician skilllink compoundeyes sheerforce guts moxie defiant competitive
contrary thickfat hugepower purepower adaptability sniper solidrock filter
multiscale shadowshield goodasgold supremeoverlord moldbreaker torrent blaze
overgrow swarm poisonheal immun​ity limber vitalspirit insomnia own​tempo
serenegrace superluck noguard rockhead recoilproof keeneye hypercutter
""".replace("\u200b", "").split())
SPEED_NEUTRAL_ITEMS = frozenset("""
none leftovers blacksludge lifeorb choiceband choicespecs assaultvest
heavydutyboots focussash rockyhelmet airballoon expertbelt whiteherb
powerherb mentalherb clearamulet covertcloak lightclay scopelens leppaberry
sitrusberry lumberry chopleberry shucaberry colburberry weaknesspolicy
flameorb toxicorb eviolite mysticwater blackglasses charcoal magnet
""".split())


def effective_speed(base, ability, item, *, stage=0, paralyzed=False, tailwind=False):
    ability, item = to_id(ability), to_id(item)
    if ability not in SPEED_NEUTRAL_ABILITIES:
        raise Unsupported(f"speed_ability:{ability}")
    if item not in SPEED_NEUTRAL_ITEMS | {"choicescarf"}:
        raise Unsupported(f"speed_item:{item}")
    if not -6 <= stage <= 6 or base <= 0:
        raise ValueError("invalid Speed input")
    if sum((item == "choicescarf", tailwind, paralyzed)) > 1:
        raise Unsupported("combined_speed_modifiers_not_validated")
    speed = base * max(2, 2 + stage) // max(2, 2 - stage)
    # Gen 9 modifier order for the supported ordinary case. Exclude other
    # abilities/items rather than approximating their interaction or priority.
    if item == "choicescarf":
        speed = speed * 3 // 2
    if tailwind:
        speed *= 2
    if paralyzed:
        speed //= 2
    return max(1, speed)


def speed_likelihood(candidate, evidence, catalog, current_item=None, current_ability=None):
    if not evidence.supported:
        raise Unsupported(evidence.reason)
    speed = effective_speed(catalog.stats(candidate)["spe"],
                            current_ability or candidate.ability,
                            candidate.item if current_item is None else current_item,
                            stage=evidence.opponent_stage, paralyzed=evidence.paralyzed,
                            tailwind=evidence.tailwind)
    if speed == evidence.own_speed:
        return 0.5
    first = speed < evidence.own_speed if evidence.trick_room else speed > evidence.own_speed
    return float(first == evidence.opponent_first)


def incoming_likelihood(pmf, before, after):
    if not before.exact or not after.exact or before.denominator != after.denominator:
        raise Unsupported("incoming_requires_exact_own_hp")
    # Censor overkill instead of treating the HP lost as uncapped damage.
    return math.fsum(p for d, p in pmf.outcomes if max(0, before.value - d) == after.value)


def filter_hp(prior, pmf, observed, maxhp):
    """Unnormalized forward filter; returns (evidence likelihood, conditional HP)."""
    posterior = Counter()
    for hp, p_hp in prior.items():
        for damage, p_damage in pmf.outcomes:
            h = max(0, hp - damage)
            if observed.compatible(h, maxhp):
                posterior[h] += p_hp * p_damage
    mass = math.fsum(posterior.values())
    return mass, {h: p / mass for h, p in posterior.items()} if mass else {}


class NoDamageOracle:
    def distribution(self, candidate, evidence, current_item=None, current_ability=None):
        raise Unsupported("damage_oracle_unavailable")
