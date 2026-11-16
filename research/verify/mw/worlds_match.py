"""Are the bot's sampled worlds at the first decision identical across arms (same seed/game)? Compare bayes.labels
(union of the sampled worlds' opponent options) and shapes. Also BH (Modal) stream layout."""
import json, os
SB = os.path.expanduser("~/sandbox/ourbot")
def first(t, s, g):
    for l in open(f"{SB}/mw_{t}/replays_headless/{t}-s{s}-g{g}.A.decisions.jsonl"):
        r = json.loads(l)
        if "bayes" in r and "labels" in r["bayes"]:
            return r
same_lab = same_shape = same_turn = n = 0
for s in range(1, 201):
    for g in (1, 2):
        a, b = first("MWs", s, g), first("BHs", s, g)
        n += 1; same_turn += a["turn"] == b["turn"]
        same_lab += a["bayes"]["labels"] == b["bayes"]["labels"]
        same_shape += a["abar_shape"] == b["abar_shape"]
print(f"first hedge decision: n={n} same turn {same_turn}, identical sampled-world label union {same_lab}, same Abar shape {same_shape}")
BH = "/Users/biofanat/Documents/classes/f26/cs535/final/engine/showdown/results/BH/replays_headless"
ok = sum(json.load(open(f"{BH}/BH-s{s}-g1.json"))["stream"] == (s - 1) % 10 for s in range(1, 401))
print(f"BH stream == (seed-1) % 10 for {ok}/400 seeds (so seeds 1-200 sit at stream positions 0-19, as in MWs/BHs)")
