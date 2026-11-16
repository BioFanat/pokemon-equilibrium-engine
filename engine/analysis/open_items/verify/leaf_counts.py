import json, glob, sys, collections
for tag in sys.argv[1:]:
    tot = collections.Counter(); games_with = 0; ngames = 0; per_role = collections.Counter()
    for f in sorted(glob.glob(f"/Users/biofanat/Documents/classes/f26/cs535/final/engine/showdown/results/{tag}/replays_headless/*.decisions.jsonl")):
        last = None
        for l in open(f):
            if '"leaf_stats"' in l:
                r = json.loads(l); last = r["leaf_stats"]
        ngames += 1
        if last is None: continue
        for k in ("calls","non_d1","retries","fallback"): tot[k] += last.get(k,0)
        role = f.split('.')[-3]; per_role[role] += last.get("non_d1",0)
        games_with += last.get("non_d1",0) > 0
    print(tag, "files", ngames, dict(tot), "files with >=1 non_d1:", games_with, "by role", dict(per_role),
          "rate %.5f%% 1 in %.0f" % (100*tot["non_d1"]/tot["calls"], tot["calls"]/max(tot["non_d1"],1)))
