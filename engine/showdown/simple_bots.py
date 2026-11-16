"""Cheap baseline bots for the gauntlet (no poke-engine search, no matrices), dispatched from headless.play_game.

random: uniform over the legal options (moves, their -tera variants when tera is available, switches; on a forced
switch, switches only). maxpower: the max-damage baseline (poke-env's MaxBasePowerPlayer plus type effectiveness and
STAB): the legal move with the highest basePower x STAB (1.5) x effectiveness against the opponent's active; status
moves score 0; if no move scores > 0, or on a forced switch / team preview, the alive reserve whose best known move
scores highest against the opponent's active (else a random legal option). Never teras; ties go to the first option.

Legal options come from foul-play's battle copy, which async_pick_move has already updated from the request JSON
(active moves with disabled / pp, can_terastallize, trapped, reserve hp). Return values use foul-play's decision
format (fp.modes.base.format_decision): a move name, "<move>-tera", or "switch <reserve name>".
"""
import random

from fp.battle.helpers import type_effectiveness_modifier
from fp.data import all_move_json

METHODS = ("random", "maxpower")
STAB = 1.5


def legal_moves(battle):
    """Names of the active Pokemon's selectable moves (not disabled; pp left when the request says so)."""
    active = battle.user.active
    req_moves = (battle.request_json or {}).get("active", [{}])[0].get("moves", [])
    # the request is the ground truth: foul-play re-enables Gigaton Hammer / Blood Moon after use (fp/battle/state.py),
    # but the simulator still rejects them, so read disabled flags from the request (index-aligned with active.moves)
    bad = {i for i, m in enumerate(req_moves) if m.get("disabled") or m.get("maybeDisabled")}
    moves = [m for i, m in enumerate(active.moves) if not m.disabled and i not in bad]
    with_pp = [m for m in moves if m.current_pp > 0]
    return [m.name for m in (with_pp or moves)]


def legal_switches(battle):
    """Reserve names the user can switch to (fainted ones only while reviving)."""
    user = battle.user
    if not battle.team_preview and not battle.force_switch and user.trapped:
        return []
    reviving = bool(getattr(user.active, "reviving", False)) and battle.force_switch
    return [f"switch {p.name}" for p in user.reserve if (p.hp <= 0) == reviving]


def legal_options(battle, tera=True):
    """(moves, switches) in foul-play's decision format; forced switch / team preview: no moves."""
    switches = legal_switches(battle)
    if battle.team_preview or battle.force_switch:
        return [], switches
    moves = legal_moves(battle)
    if tera and battle.user.active.can_terastallize:
        moves = moves + [f"{m}-tera" for m in moves]
    return moves, switches


def current_types(pkmn):
    """Types after terastallization (stellar keeps the original types)."""
    if pkmn.terastallized and pkmn.tera_type and pkmn.tera_type != "stellar":
        return [pkmn.tera_type]
    return list(pkmn.types)


def move_score(move_name, attacker, defender_types):
    """basePower x STAB x type effectiveness; 0 for status moves and unknown moves."""
    mv = all_move_json.get(move_name)
    if not mv or mv.get("category") == "status" or not mv.get("basePower"):
        return 0.0
    stab_types = set(attacker.types) | set(current_types(attacker))
    stab = STAB if mv["type"] in stab_types else 1.0
    return mv["basePower"] * stab * type_effectiveness_modifier(mv["type"], defender_types)


def best_reserve(battle, switches):
    """The switch whose Pokemon's best known move scores highest against the opponent's active, or None."""
    opp = battle.opponent.active
    if opp is None or not opp.name:                         # team preview: dummy opponent
        return None
    opp_types = current_types(opp)
    by_name = {p.name: p for p in battle.user.reserve}
    best, best_s = None, 0.0
    for sw in switches:
        p = by_name[sw[len("switch "):]]
        s = max((move_score(m.name, p, opp_types) for m in p.moves), default=0.0)
        if s > best_s:
            best, best_s = sw, s
    return best


def make_move_fn(method, seed=0, game=0, role="A"):
    """fn(battle) -> foul-play decision string; a per-player RNG keeps the two players' draws independent."""
    assert method in METHODS, method
    rng = random.Random(f"{method}-{seed}-{game}-{role}")

    def random_move(battle):
        moves, switches = legal_options(battle, tera=True)
        return rng.choice(moves + switches)

    def maxpower_move(battle):
        moves, switches = legal_options(battle, tera=False)
        opp = battle.opponent.active
        if moves and opp is not None and opp.name:
            opp_types = current_types(opp)
            scores = [move_score(m, battle.user.active, opp_types) for m in moves]
            top = max(scores)
            if top > 0:
                return moves[scores.index(top)]
        return best_reserve(battle, switches) or rng.choice(moves + switches)

    return random_move if method == "random" else maxpower_move
