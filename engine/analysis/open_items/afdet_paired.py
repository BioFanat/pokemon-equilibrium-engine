"""AFdet (deterministic leaf) vs AF (time-budgeted leaf): seed-paired score difference and det-leaf firing stats.
Run from engine/showdown/results:  python ../../analysis/open_items/afdet_paired.py"""
import json, glob, collections

def per_seed(tag):
    g = collections.defaultdict(list)
    for f in glob.glob(f"{tag}/replays_headless/{tag}-s*-g*.json"):
        d = json.load(open(f))
        g[d["seed"]].append(None if d.get("error") else
                            {"A": 1.0, "B": 0.0}.get(d["winner_role"], 0.5))
    return {s: sum(v) / 2 for s, v in g.items() if len(v) == 2 and None not in v}

a, b = per_seed("AFdet"), per_seed("AF")
common = sorted(set(a) & set(b)); d = [a[i] - b[i] for i in common]
m = sum(d) / len(d); se = (sum((x - m) ** 2 for x in d) / (len(d) - 1)) ** .5 / len(d) ** .5
cl = collections.defaultdict(list)
for i, x in zip(common, d): cl[(i - 1) % 10].append(x)
cm = [sum(v) / len(v) for v in cl.values()]; mm = sum(cm) / len(cm)
sec = (sum((x - mm) ** 2 for x in cm) / (len(cm) - 1)) ** .5 / len(cm) ** .5
same = sum(json.load(open(f"AFdet/replays_headless/AFdet-s{i}-g{g}.json"))["team_seeds"] ==
           json.load(open(f"AF/replays_headless/AF-s{i}-g{g}.json"))["team_seeds"] for i in common for g in (1, 2))
tot = collections.Counter()
for f in glob.glob("AFdet/replays_headless/*.A.decisions.jsonl"):
    for l in open(f):
        ls = json.loads(l).get("leaf_stats")
        if ls: tot.update({k: ls[k] for k in ("calls", "non_d1", "retries", "fallback")})
out = dict(AFdet=sum(a.values()) / len(a), n_AFdet=len(a), AF=sum(b.values()) / len(b), n_AF=len(b), n_paired=len(common),
           diff=m, ci=[m - 1.96 * se, m + 1.96 * se], ci_stream_t9=[mm - 2.262 * sec, mm + 2.262 * sec],
           same_team_games=same, leaf=dict(tot))
print(json.dumps(out, indent=1))
