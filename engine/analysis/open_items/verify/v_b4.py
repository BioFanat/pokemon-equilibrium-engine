"""B4: played vs expert labels in E5 hedge decisions."""
import collections
from vlib import decs, games
c = collections.Counter(); n = 0; dis = 0; dis_c = collections.Counter(); keys = collections.Counter()
g = games("E5")
for (s, k) in g:
    rs = decs("E5", s, k, "A")
    if rs is None: continue
    for r in rs:
        if r.get("final") or "experts" not in r: continue
        keys[tuple(sorted(r.keys()))] += 1
        mv = r["move"]; mv = mv[7:] if mv.startswith("switch ") else mv
        e = r["experts"]; n += 1
        for x in ("top", "sampled", "safest", "safex"): c[x] += mv == e[x]
        if e["top"] != e["safest"]:
            dis += 1; dis_c["top"] += mv == e["top"]; dis_c["safest"] += mv == e["safest"]
            dis_c["neither"] += (mv != e["top"] and mv != e["safest"]); dis_c["safex"] += mv == e["safex"]
            dis_c["top_eq_sampled"] += e["top"] == e["sampled"]
print("decisions", n, {x: round(v / n, 4) for x, v in c.items()})
print("disagree", dis, round(dis / n, 4), {x: round(v / dis, 4) for x, v in dis_c.items()})
print("record key sets:", list(keys.items())[:3])
