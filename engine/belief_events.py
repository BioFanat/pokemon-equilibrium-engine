"""Observer-visible evidence only. No simulator objects or opponent commands."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Reveal:
    event_id: str
    entity: str
    species: str
    attribute: str = "species"
    value: str | None = None
    original: bool = True


@dataclass(frozen=True)
class HPObservation:
    value: int
    denominator: int
    exact: bool = False

    def __post_init__(self):
        if self.denominator <= 0 or not 0 <= self.value <= self.denominator:
            raise ValueError("invalid HP observation")

    def compatible(self, hp, maxhp):
        if self.exact:
            return maxhp == self.denominator and hp == self.value
        # Explicit percentage-display model, validated by the observer adapter's fixtures.
        # Keep the non-full -> 99 edge used by this project's internal observer.
        if hp == 0:
            return self.value == 0
        pct = (hp * self.denominator + maxhp - 1) // maxhp
        if pct == self.denominator and hp < maxhp:
            pct -= 1
        return pct == self.value


@dataclass(frozen=True)
class SpeedEvidence:
    event_id: str
    entity: str
    own_speed: int                 # known effective Speed at action selection
    opponent_first: bool
    opponent_stage: int = 0
    paralyzed: bool = False
    tailwind: bool = False
    trick_room: bool = False
    # Adapter certifies equal, known priority and no ambiguous order effects.
    supported: bool = False
    reason: str = "uncertified_order_context"
    attributes_at_event: bool = False
    item_at_event: str | None = None
    ability_at_event: str | None = None


@dataclass(frozen=True)
class DamageEvidence:
    event_id: str
    entity: str
    move: str
    before: HPObservation
    after: HPObservation
    incoming: bool = True
    # Immutable public inputs consumed by a mechanics oracle, not a full state.
    context: tuple = ()
    supported: bool = False
    reason: str = "uncertified_damage_context"
    attributes_at_event: bool = False
    item_at_event: str | None = None
    ability_at_event: str | None = None


@dataclass(frozen=True)
class Skip:
    event_id: str
    entity: str
    reason: str
