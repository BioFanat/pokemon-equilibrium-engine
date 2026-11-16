"""Gate (d): no information leak — A1's grouping keys use only public observations.

For children of real decisions (every node of every world), mutate the hidden-world attributes of the opponent in the
CHILD state and recompute obs_key; it must not change:
  * active mon: item (unless already gone), ability, base_ability, every move except the one just used (ids and slot
    order), tera_type (unless terastallized), the five non-HP stats, evs; exact HP within the same hp_public bucket;
  * every opponent slot not revealed in the decider's view (and not the post-turn active): replaced by another
    species' sampled set;
  * side field switch_out_move_second_saved_move (hidden.py copies it unmasked; report §6) and
    volatile_status_durations are not read.
Sanity (the key is not vacuous): changing the opponent's public hp bucket, or our own active's HP, changes the key.
Also: groups never mix our root actions or the opponent's root labels (the key contains a and beta).
Run: OURBOT_DET_LEAF=1 python test_d_leak.py
"""
import random

from _common import check, decisions, finish, np, pe, hd
import info_planner as ip
import randbats as rb

rng = random.Random(7)


def opp_side(st, side):
    return st.side_two if side == 1 else st.side_one


def with_opp(st, side, new):
    a, b = (st.side_one, new) if side == 1 else (new, st.side_two)
    return pe.State(side_one=a, side_two=b, **{f: getattr(st, f) for f in hd.GF})


def mutate_hidden(st, side, b, revealed):
    sd = opp_side(st, side)
    a = int(sd.active_index)
    mons = [hd.copy_pokemon(p) for p in sd.pokemon]
    p = sd.pokemon[a]
    used = (b[:-5] if b.endswith("-tera") else b)
    others = [m for m in ("toxic", "protect", "surf", "earthquake", "recover", "uturn") ]
    new_moves = []
    for m in p.moves:
        if m.id.lower() == used:
            new_moves.append(pe.Move(id=m.id, pp=m.pp, disabled=m.disabled))
        else:
            cand = [o for o in others if o != used and o not in [x.id.lower() for x in new_moves]]
            new_moves.append(pe.Move(id=rng.choice(cand), disabled=m.disabled))
    rng.shuffle(new_moves)
    over = dict(moves=new_moves, ability="pressure", base_ability="pressure",
                attack=p.attack + 7, defense=p.defense + 5, special_attack=p.special_attack + 3,
                special_defense=p.special_defense + 2, speed=p.speed + 11, evs=(0, 252, 0, 0, 4, 252))
    if p.item.lower() != "none":
        over["item"] = "choicescarf" if p.item.lower() != "choicescarf" else "leftovers"
    if not p.terastallized:
        over["tera_type"] = "fairy" if p.tera_type.lower() != "fairy" else "ghost"
    if 0 < p.hp < p.maxhp:                       # exact HP, same public bucket
        pub = hd.hp_public(p.hp, p.maxhp)
        for hp in (p.hp - 1, p.hp + 1):
            if 0 < hp < p.maxhp and hd.hp_public(hp, p.maxhp) == pub:
                over["hp"] = hp
                break
    mons[a] = hd.copy_pokemon(p, **over)
    for i in range(6):
        if i != a and i not in revealed:
            sp = rng.choice(sorted(rb.sets()))
            mons[i] = rb.to_pokemon(rb.sample_set(sp, rng))
            mons[i] = hd.copy_pokemon(mons[i], hp=sd.pokemon[i].hp if sd.pokemon[i].hp <= 0 else mons[i].maxhp)
    side_new = hd.copy_side(sd, pokemon=mons, switch_out_move_second_saved_move="EARTHQUAKE")
    return with_opp(st, side, side_new)


def mutate_public(st, side):
    sd = opp_side(st, side)
    a = int(sd.active_index)
    p = sd.pokemon[a]
    mons = [hd.copy_pokemon(x) for x in sd.pokemon]
    mons[a] = hd.copy_pokemon(p, hp=max(1, p.hp // 3) if p.hp > 3 else p.maxhp)
    return with_opp(st, side, hd.copy_side(sd, pokemon=mons))


pos = decisions(401, 4, every=4, K=4) + decisions(402, 4, every=4, K=4)
n = same = pub_changed = own_changed = 0
mixed = 0
for state, side, view, legal, worlds in pos:
    rev = set(view.reveals)
    for w in worlds:
        s1, s2 = pe.id(w, 0).s1, pe.id(w, 0).s2
        import pkbattle as pb
        o1, o2 = pb.options(w)
        own, opp = (o1, o2) if side == 1 else (o2, o1)
        for a in own[::2]:
            for b in opp[::2]:
                for br in (pe.generate_instructions(w, a, b) if side == 1 else pe.generate_instructions(w, b, a))[:3]:
                    c = w.apply_instructions(br)
                    k0 = ip.obs_key(c, side, a, b, rev)
                    if opp_side(c, side).pokemon[int(opp_side(c, side).active_index)].hp <= 0:
                        continue
                    n += 1
                    same += ip.obs_key(mutate_hidden(c, side, b, rev), side, a, b, rev) == k0
                    pub_changed += ip.obs_key(mutate_public(c, side), side, a, b, rev) != k0
                    ownsd, _ = hd._sides(c, side)
                    oa = int(ownsd.active_index)
                    mons = [hd.copy_pokemon(x) for x in ownsd.pokemon]
                    mons[oa] = hd.copy_pokemon(ownsd.pokemon[oa], hp=max(1, ownsd.pokemon[oa].hp - 1))
                    c2 = (pe.State(side_one=hd.copy_side(ownsd, pokemon=mons), side_two=c.side_two, **{f: getattr(c, f) for f in hd.GF})
                          if side == 1 else
                          pe.State(side_one=c.side_one, side_two=hd.copy_side(ownsd, pokemon=mons), **{f: getattr(c, f) for f in hd.GF}))
                    own_changed += ip.obs_key(c2, side, a, b, rev) != k0 or ownsd.pokemon[oa].hp <= 1
    res = ip.plan(worlds, side, legal, rev, want_a0l=False, key_report=False)
    g = {}
    for nd in res["ex"].nodes:
        g.setdefault(nd.key, set()).add((nd.a, nd.b))
    mixed += sum(len(v) > 1 for v in g.values())
check("(d) hidden-attribute mutations leave the key unchanged", same == n, f"({same}/{n} children)")
check("(d) sanity: public opponent HP change changes the key", pub_changed == n, f"({pub_changed}/{n})")
check("(d) sanity: our own HP change changes the key", own_changed == n, f"({own_changed}/{n})")
check("(d) no group mixes root actions or opponent labels", mixed == 0, f"({len(pos)} decisions)")
finish()
