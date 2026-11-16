"""Revival Blessing (and forced-recharge) guard for foul-play's battle loop (all methods, stock included).

After Revival Blessing, Showdown sends a forceSwitch request whose active Pokemon has "reviving": true; the player must
pick a FAINTED reserve. foul-play (8f541f8) stores pkmn.reviving but never uses it and poke-engine has no notion of it,
so a search either returns 'No Move' with 0 visits (ZeroDivisionError in fp/search/main.py
select_move_from_mcts_results; AttributeError in format_decision for our solvers) or picks a living reserve, which
Showdown rejects ("You have to pass to a fainted Pokemon" -> invalid_choice_stall). Diagnosed 2026-11-16 by replaying
G1_stockat25_stockat400-s4-g1 deterministically; 33 past games across experiments ended this way.

guard(fn) answers such requests with the first fainted reserve (a fixed legal rule; no method evaluates revival
targets) and calls fn otherwise. install_keep_last_move() keeps battle.user.last_selected_move at the move actually
used (revivalblessing): the pick happens mid-turn, and fp.battle.inference.check_speed_ranges would otherwise look up
'switch <name>' as a move (KeyError)."""


def is_revival(battle):
    req = battle.request_json or {}
    mons = (req.get("side") or {}).get("pokemon") or []
    return bool(req.get("forceSwitch")) and any(p.get("reviving") for p in mons)


def revival_choice(battle):
    if not is_revival(battle):
        return None
    fainted = [p for p in battle.user.reserve if p.hp <= 0]
    return f"switch {fainted[0].name}" if fainted else None


def recharge_choice(battle):
    """A recharge turn (after Hyper Beam / Giga Impact) offers only the move 'recharge'. When the user is also
    trapped (e.g. by Arena Trap) poke-engine sees no option, returns 'No Move', and format_decision fails
    (AttributeError ... can_z). 'recharge' is the only legal choice, so answer it directly."""
    req = battle.request_json or {}
    act = req.get("active") or []
    if req.get("forceSwitch") or len(act) != 1:
        return None
    moves = act[0].get("moves") or []
    if len(moves) == 1 and moves[0].get("id") == "recharge":
        return "recharge"
    return None


def guard(fn):
    def guarded(battle):
        pick = revival_choice(battle)
        if pick is None:
            pick = recharge_choice(battle)
        return pick if pick is not None else fn(battle)
    return guarded


def install_keep_last_move():
    import fp.run_battle as rb
    if getattr(rb.async_pick_move, "_revival_guard", False):
        return
    orig = rb.async_pick_move

    async def pick(battle):
        prev, rev = battle.user.last_selected_move, is_revival(battle)
        out = await orig(battle)
        if rev:
            battle.user.last_selected_move = prev
        return out
    pick._revival_guard = True
    rb.async_pick_move = pick
