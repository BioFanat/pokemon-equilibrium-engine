"""Extend the B12 replay for the 'sampled' target only (its exploitability runs entirely through the drift
channel) to more logged W1 pairs, so the along-path drift has a usable SE.
Wraps scratchpad/bridges/b12/replay_b12.py unchanged (imported as a module).
  python replay_more.py K0 K1 out.jsonl     # replays W1 pairs K0..K1-1 (both seats) for wb2:sample:4:0.8
"""
import json, sys
B12 = "/private/tmp/claude-501/-Users-biofanat-Documents-classes-f26-cs535/2bbe670b-2f97-4d9a-bfad-e1139e0352c6/scratchpad/bridges/b12"
sys.path.insert(0, B12)
import replay_b12 as rb

k0, k1, out = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
A = "wb2:sample:4:0.8"
B, tname, _ = rb.TARGETS[A]
logged = {}
for line in open(f"{rb.ENGINE}/results/experiments/W1.jsonl"):
    r = json.loads(line)
    logged[(r["A"], r["pair"], r["a_side"])] = r
with open(out, "a") as fo:
    for k in range(k0, k1):
        for a_side in (1, 2):
            L = logged[(A, k, a_side)]
            assert L["B"] == B
            rec, turns = rb.play_logged(A, B, tname, L["seed"], a_side)
            rec.update(pair=k, logged_winner=L["winner"], logged_turns=L["turns"], logged_score=L["score_A"],
                       tname=tname, per_turn=turns)
            fo.write(json.dumps(rec) + "\n"); fo.flush()
            print(f"pair {k:3d} side {a_side}: winner {rec['winner']} (logged {L['winner']}), turns {rec['turns']} "
                  f"(logged {L['turns']}), {rec['wall_s']:.1f}s", flush=True)
