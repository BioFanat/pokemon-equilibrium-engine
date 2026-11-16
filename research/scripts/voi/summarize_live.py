"""Summarise out/two_sided_live.jsonl against the pre-registered predictions P1-P5 (two_sided_live.py docstring)."""
from __future__ import annotations

import collections
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "out", "two_sided_live.jsonl")
games = [json.loads(l) for l in open(path)]
D0 = [r for g in games for r in g["decisions"]]
D = [r for r in D0 if r["payoff_range"] > 0]
print(f"dropped {len(D0) - len(D)} decisions with payoff_range == 0")
print(f"games {len(games)}  turns {[g['turns'] for g in games]}  sampled decisions {len(D)}  "
      f"wall {sum(g['wall_s'] for g in games) / 60:.1f} min  secs/decision median {np.median([r['secs'] for r in D]):.1f}")
rng = np.array([r["payoff_range"] for r in D])
ls = D[-1]["leaf_stats"]
print(f"leaf stats (last record of last game): {ls}")


def q(a, name, unit=""):
    a = np.asarray(a, float)
    if a.size == 0:
        print(f"  {name}: (none)"); return
    print(f"  {name:38s} n={a.size:3d} mean {a.mean():.4f} median {np.median(a):.4f} p90 {np.percentile(a, 90):.4f} "
          f"max {a.max():.4f}  zero(<1e-6) {(np.abs(a) < 1e-6).mean():.2f}  >1% {(a > 0.01).mean():.2f}  >5% {(a > 0.05).mean():.2f}{unit}")


print("\n(1) DECISION REGIME (K=4 full-set worlds, our true team)  [P1: disagree 40-60%, af_outside < 15%]")
print(f"  per-world argmax disagree: {np.mean([r['worlds_disagree'] for r in D]):.2f}   AF argmax outside world argmaxes: "
      f"{np.mean([r['af_outside'] for r in D]):.2f}")
q([r["eta_hat_set"] / r["payoff_range"] for r in D], "eta_hat_set / range")
q([max(r["regret_af_in_world"]) / r["payoff_range"] for r in D], "max_k regret(x_AF in world k) / range")
q([np.mean(r["regret_af_in_world"]) / r["payoff_range"] for r in D], "mean_k regret(x_AF in world k) / range")

print("\n(3) ATTRIBUTE SPLIT on the active opponent's MAP set  [P2: moves > item > tera; tera median 0]")
for a in ("moves", "item", "ability", "tera"):
    rs = [(r, r["attr"][a]) for r in D if a in r["attr"]]
    if not rs:
        print(f"  {a}: none"); continue
    print(f"  {a}: n={len(rs)}  argmax disagree {np.mean([x['disagree'] for _, x in rs]):.2f}  "
          f"n_worlds median {np.median([x['n'] for _, x in rs]):.0f}")
    q([x["eta_hat"] / r["payoff_range"] for r, x in rs], f"    eta_hat_{a} / range")
    q([max(x["regret_af"]) / r["payoff_range"] for r, x in rs], f"    max regret(AF act) / range")
    q([(max(x["values"]) - min(x["values"])) / r["payoff_range"] for r, x in rs], f"    spread of world values / range")

for cond in ("spk", "pub"):
    rs = [r for r in D if "conc_value" in r.get(cond, {})]
    sk = sum(1 for r in D if r.get(cond, {}).get("skipped"))
    print(f"\n(2) CONCEALMENT condition {cond} (n={len(rs)}, skipped {sk})  [P3: conc median 0, >1% in <10%; P4: argmax change <15%, tera <3%]")
    q([r[cond]["conc_value"] / r["payoff_range"] for r in rs], "ex-ante concealment value / range")
    q([r[cond]["regret_af_vs_honest"] / r["payoff_range"] for r in rs], "regret of x_AF vs honest opp / range")
    q([r[cond]["regret_true_vs_omni"] / r["payoff_range"] for r in rs], "regret of x_true vs omniscient / range")
    q([r[cond]["tv"] for r in rs], "TV(x_AF, x_true)")
    ch = [r for r in rs if r[cond]["argmax_change"]]
    print(f"  argmax change: {len(ch)}/{len(rs)} = {len(ch) / max(1, len(rs)):.2f}")
    cls = collections.Counter((r[cond]["from_class"], r[cond]["to_class"]) for r in ch)
    print(f"  change classes (from -> to): {dict(cls)}")
    tera = [r for r in ch if "tera" in (r[cond]["from_class"], r[cond]["to_class"])]
    unrev = [r for r in ch if "move_unrevealed" in (r[cond]["from_class"], r[cond]["to_class"])]
    print(f"  tera-involving changes {len(tera)}/{len(rs)} = {len(tera) / max(1, len(rs)):.3f};  "
          f"unrevealed-move-involving {len(unrev)}/{len(rs)} = {len(unrev) / max(1, len(rs)):.3f}")
    print(f"  our tera already used at decision: {np.mean([r['our_tera_used'] for r in rs]):.2f}; "
          f"revealed moves of our active (mean) {np.mean([len(r['our_revealed_moves']) for r in rs]):.2f}")

both = [r for r in D if "conc_value" in r.get("spk", {}) and "conc_value" in r.get("pub", {})]
if both:
    a = np.array([r["pub"]["conc_value"] for r in both]); b = np.array([r["spk"]["conc_value"] for r in both])
    print(f"\n[P5] pub vs spk concealment value: mean {a.mean():.3f} vs {b.mean():.3f}; pub > spk in {(a > b + 1e-9).mean():.2f} of decisions; "
          f"median ratio pub/spk over decisions with spk>1e-6: "
          f"{np.median([x / y for x, y in zip(a, b) if y > 1e-6]) if any(b > 1e-6) else float('nan'):.2f}")

print("\nby game stage (turn buckets of 10): conc(spk)/range mean, eta_set/range mean, n")
by = collections.defaultdict(list)
for r in D:
    if "conc_value" in r.get("spk", {}):
        by[min(r["turn"] // 10, 3)].append((r["spk"]["conc_value"] / r["payoff_range"], r["eta_hat_set"] / r["payoff_range"]))
for k, v in sorted(by.items()):
    v = np.array(v); print(f"  turns {k * 10}-{k * 10 + 9}: conc {v[:, 0].mean():.4f}  eta_set {v[:, 1].mean():.4f}  n={len(v)}")
print("\nrevealed opponent moves (active) by turn bucket:",
      {k: round(float(np.mean([len(r['opp_revealed_moves']) for r in D if min(r['turn'] // 10, 3) == k])), 2) for k in sorted(by)})

print("\nconcealment (spk) vs how much of our active set the opponent has seen (revealed moves 0-4), and vs our tera used")
byr = collections.defaultdict(list)
for r in D:
    if "conc_value" in r.get("spk", {}):
        byr[len(r["our_revealed_moves"])].append(r["spk"]["conc_value"] / r["payoff_range"])
for k, v in sorted(byr.items()):
    print(f"  revealed {k}: conc mean {np.mean(v):.4f} median {np.median(v):.4f} n={len(v)}")
for flag in (False, True):
    v = [r["spk"]["conc_value"] / r["payoff_range"] for r in D if "conc_value" in r.get("spk", {}) and r["our_tera_used"] == flag]
    if v:
        print(f"  our tera used={flag}: conc mean {np.mean(v):.4f} median {np.median(v):.4f} n={len(v)}")
# robustness of 'argmax change': count only changes where the AF argmax has regret > 1% of range vs the honest opponent
for cond in ("spk", "pub"):
    rs = [r for r in D if "conc_value" in r.get(cond, {})]
    strong = [r for r in rs if r[cond]["argmax_change"] and r[cond]["regret_af_vs_honest"] > 0.01 * r["payoff_range"]]
    print(f"  {cond}: argmax changes with regret(x_AF vs honest) > 1% range: {len(strong)}/{len(rs)} = {len(strong) / max(1, len(rs)):.2f}")
