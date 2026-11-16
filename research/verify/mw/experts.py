"""Exploit-expert behaviour in MWs vs BHs: hedge weights (per decision and final state), expert argmax agreement,
matched turn-1 states (same teams/battle seed) to compare the safex expert directly, posterior MAP per stream,
and fields that would reveal LP fallbacks / degenerate play."""
import glob, json, os, math
import numpy as np
from collections import Counter, defaultdict

SB = os.path.expanduser("~/sandbox/ourbot")
D = {"MWs": f"{SB}/mw_MWs", "BHs": f"{SB}/mw_BHs"}
E = ["top", "sampled", "safest", "safex"]


def recs(t, s, g):
    out = []
    for l in open(f"{D[t]}/replays_headless/{t}-s{s}-g{g}.A.decisions.jsonl"):
        if l.strip():
            out.append(json.loads(l))
    return out


R = {(t, s, g): recs(t, s, g) for t in D for s in range(1, 201) for g in (1, 2)}

for t in D:
    print(f"== {t}")
    for k in range(10):
        h = json.load(open(f"{D[t]}/hedge_state/{t}-stream{k}-A.json"))
        b = json.load(open(f"{D[t]}/hedge_state/{t}-stream{k}-A.bayes.json"))
        post = b["posterior"]
        print(f"  stream {k}: final w={dict(zip(E, [round(x,3) for x in h['weights']]))} cum-max={np.round(np.array(h['cum'])-max(h['cum']),1).tolist()} "
              f"t={h['t']} MAP={max(post, key=post.get)} ({post[max(post, key=post.get)]:.2f})")
    dec = [r for (tt, s, g), rs in R.items() if tt == t for r in rs if "weights" in r and isinstance(r["weights"], dict)]
    W = np.array([[r["weights"][e] for e in E] for r in dec])
    print(f"  decisions with hedge mix: {len(dec)}; mean weights {dict(zip(E, W.mean(0).round(3)))}")
    # by position in stream: early (first 5 seeds of stream) vs late
    agree = Counter()
    for r in dec:
        ex = r["experts"]
        agree["safex==top"] += ex["safex"] == ex["top"]
        agree["safex==safest"] += ex["safex"] == ex["safest"]
        agree["safex==sampled"] += ex["safex"] == ex["sampled"]
        agree["move==safex"] += r.get("move", "").endswith(ex["safex"]) or ex["safex"] in r.get("move", "")
        agree["all4 same"] += len(set(ex.values())) == 1
    print("  argmax agreement:", {k: round(v / len(dec), 3) for k, v in agree.items()})
    mx = np.array([max(r["p"].values()) for r in dec])
    print(f"  played p: mean max-prob={mx.mean():.3f}; share of decisions with max-prob>=0.99: {np.mean(mx>=0.99):.3f}")
    allr = [r for (tt, s, g), rs in R.items() if tt == t for r in rs]
    keys = Counter(k for r in allr for k in r)
    print("  record keys:", dict(keys))
    forced = sum(1 for r in allr if "mix" in r and len(r["mix"]) == 1 and "weights" not in r)
    print(f"  single-legal decisions: {forced}")
    post_maps = Counter()
    for r in dec:
        p = r["bayes"]["posterior"]
        post_maps[max(p, key=p.get)] += 1
    print("  decision-time posterior MAP shares:", {k: round(v / len(dec), 3) for k, v in post_maps.items()})

# matched turn-1 states: first decision record of each (seed, game) in both arms
same_af, same_safex, n, tv, sx_mass = 0, 0, 0, [], []
diff_detail = Counter()
for s in range(1, 201):
    for g in (1, 2):
        a = next((r for r in R[("MWs", s, g)] if "experts" in r), None)
        b = next((r for r in R[("BHs", s, g)] if "experts" in r), None)
        if a is None or b is None or a["turn"] != b["turn"]:
            diff_detail["no matched first decision"] += 1
            continue
        n += 1
        ea, eb = a["experts"], b["experts"]
        af = all(ea[e] == eb[e] for e in ("top", "sampled", "safest"))
        same_af += af
        if af:
            same_safex += ea["safex"] == eb["safex"]
diff_detail = dict(diff_detail)
print(f"\nmatched first decisions: n={n}, top/sampled/safest argmax identical in {same_af}; "
      f"of those safex argmax identical in {same_safex} ({same_safex/max(same_af,1):.3f}); {diff_detail}")
# stream-first game (no carried state): s=1..10, g=1, turn-1 record: weights should be uniform & posterior uniform in both
for s in range(1, 11):
    a = next(r for r in R[("MWs", s, 1)] if "experts" in r)
    b = next(r for r in R[("BHs", s, 1)] if "experts" in r)
    print(f"  s{s} g1 t{a['turn']}: MW experts={a['experts']} p={a['p']} | BH experts={b['experts']} p={b['p']}")
