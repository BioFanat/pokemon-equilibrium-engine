import json, glob, sys, collections
for tag in sys.argv[1:]:
    tot = collections.Counter(); games = collections.Counter(); seen=set(); mono=0
    for f in sorted(glob.glob(f"/Users/biofanat/Documents/classes/f26/cs535/final/engine/showdown/results/{tag}/replays_headless/*.decisions.jsonl")):
        game = f.rsplit('/',1)[1].split('.')[0]; seen.add(game)
        prev=None
        for l in open(f):
            if '"leaf_stats"' not in l: continue
            s = json.loads(l)["leaf_stats"]
            if prev and s["calls"] < prev: mono+=1
            prev = s["calls"]
            for k in ("calls","non_d1","retries","fallback"): tot[k] += s.get(k,0)
            games[game] += s.get("non_d1",0)
    g1 = sum(1 for g in seen if games[g]>0)
    print(tag, "games", len(seen), dict(tot), "games with >=1 non_d1:", g1, "rate %.4f%% 1 in %.0f"%(100*tot["non_d1"]/tot["calls"], tot["calls"]/tot["non_d1"]), "calls decreases", mono)
