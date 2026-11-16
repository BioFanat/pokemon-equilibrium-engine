"""Build a fixed, serialized set of child states (same recipe as theory/tightening/scripts/leaf_nondet.py)
plus their mirrors, so every probe compares the exact same engine inputs.
Usage: engine/.venv/bin/python build_states.py <seed> <n_trials> <kids_per_trial> <out.jsonl>"""
import json, random, sys
sys.path.insert(0, "/Users/biofanat/Documents/classes/f26/cs535/final/engine")
import poke_engine as pe, randbats as rb, pkbattle as pb, nash

seed, ntr, kpt, out = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
rng = random.Random(seed)
rows = []
for trial in range(ntr):
    if trial % 2 == 0:
        t = rb.sample_team(rng, n=2)
        st = pe.State(side_one=pe.Side(pokemon=[rb.to_pokemon(t[0])]), side_two=pe.Side(pokemon=[rb.to_pokemon(t[1])])); kind = "1v1"
    else:
        st = rb.random_state(rng); kind = "6v6"
        for _ in range(rng.randint(0, 6)):
            o1, o2 = pb.options(st); st = pb.step(st, rng.choice(o1), rng.choice(o2), rng)
            if pb.alive(st.side_one) == 0 or pb.alive(st.side_two) == 0: break
    s1, s2 = pb.options(st)
    kids = [st.apply_instructions(br) for a in s1 for b in s2 for br in pe.generate_instructions(st, a, b)]
    rng.shuffle(kids)
    for c in kids[:kpt]:
        rows.append(dict(kind=kind, trial=trial, root=st.to_string(), child=c.to_string(), mirror=nash._mirror(c).to_string()))
with open(out, "w") as f:
    for r in rows: f.write(json.dumps(r) + "\n")
print(len(rows), "children written to", out)
