import json, glob, collections
tag="LSn"
for role in ("A","B"):
    tot=collections.Counter(); games=collections.Counter()
    for f in sorted(glob.glob(f"/Users/biofanat/Documents/classes/f26/cs535/final/engine/showdown/results/{tag}/replays_headless/*.{role}.decisions.jsonl")):
        game=f.rsplit('/',1)[1].split('.')[0]
        games[game]+=0
        for l in open(f):
            if '"leaf_stats"' not in l: continue
            s=json.loads(l)["leaf_stats"]
            for k in ("calls","non_d1"): tot[k]+=s[k]
            games[game]+=s["non_d1"]
    print(role, dict(tot), "games>=1:", sum(v>0 for v in games.values()), "of", len(games), "rate %.4f%%"%(100*tot['non_d1']/tot['calls']))
