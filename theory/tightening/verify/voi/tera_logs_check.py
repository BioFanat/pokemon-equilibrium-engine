"""Verifier: independent tera counting from Showdown headless replays. For each game and side: did any input_log
command of that side contain the token 'terastallize' (as a separate word)? Also cross-checks the output log
('-terastallize|' protocol lines) when available, per run, for the 150-game shuffled subset and for all games."""
import json, os, random, re, sys, collections
RES = "/Users/biofanat/Documents/classes/f26/cs535/final/engine/showdown/results"
for limit in (150, None):
    tot = collections.Counter(); ter = collections.Counter(); proto_mismatch = 0; nproto = 0
    for run in ["AF", "AF16", "E2", "BH"]:
        d = os.path.join(RES, run, "replays_headless")
        files = sorted(f for f in os.listdir(d) if f.endswith(".json"))
        random.Random(0).shuffle(files)
        for f in files[:limit] if limit else files:
            g = json.load(open(os.path.join(d, f)))
            if g.get("error") or not g.get("turns"):
                continue
            for role in ("A", "B"):
                p = g["sides"][role]; m = f"{run}:{g['methods'][role]}"
                tot[m] += 1
                hit = any(re.match(rf">{p} (move|switch) ", l) and re.search(r"\bterastallize\b", l) for l in g["input_log"])
                ter[m] += hit
    print(f"--- limit={limit}")
    ours = [m for m in tot if not m.endswith(":stock")]; stock = [m for m in tot if m.endswith(":stock")]
    for m in sorted(tot):
        print(f"{m:>20} {tot[m]:5d} {ter[m] / tot[m]:.3f}")
    print("pooled ours %.3f (n=%d)  stock %.3f (n=%d)" % (sum(ter[m] for m in ours) / sum(tot[m] for m in ours), sum(tot[m] for m in ours),
                                                         sum(ter[m] for m in stock) / sum(tot[m] for m in stock), sum(tot[m] for m in stock)))
