"""E4: the logged node-level time-average Nash gap (Lecture 7 §2.2 / L4 identity: gap = (Reg1+Reg2)/N) at every
bandit root decision; by tag, two-sided decisions only; plus per-game Theorem-A sums and the RM+ side's gap."""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import game_files, records  # noqa: E402

out = {}
for tag in ("E4e", "E4u", "E4emv", "E4umv"):
    for role in ("A", "B"):
        gaps, per_game, shapes, rules = [], [], [], {}
        for (seed, g), path in sorted(game_files(tag, role).items()):
            gsum = 0.0
            for rec in records(path):
                if "gap" not in rec or rec.get("final"):
                    continue
                sh = rec.get("abar_shape") or rec.get("shape")
                if not sh or min(sh) < 2:
                    continue
                gaps.append(float(rec["gap"])); gsum += float(rec["gap"]); shapes.append(sh)
                rules[rec.get("played_rule", "n/a")] = rules.get(rec.get("played_rule", "n/a"), 0) + 1
            per_game.append(gsum)
        gaps = np.array(gaps); shapes = np.array(shapes)
        if len(gaps) == 0:
            continue
        bycol = {}
        for lo, hi in ((2, 8), (8, 14), (14, 27), (27, 99)):
            m = (shapes[:, 1] >= lo) & (shapes[:, 1] < hi)
            if m.any():
                bycol[f"n in [{lo},{hi})"] = (float(np.median(gaps[m])), int(m.sum()))
        d = dict(decisions=len(gaps), median=float(np.median(gaps)), mean=float(gaps.mean()), p90=float(np.percentile(gaps, 90)),
                 frac_gt_005=float((gaps > 0.05).mean()), per_game_sum_mean=float(np.mean(per_game)),
                 rows_median=float(np.median(shapes[:, 0])), cols_median=float(np.median(shapes[:, 1])), rules=rules, by_cols=bycol)
        out[f"{tag}-{role}"] = d
        print(f"{tag}-{role}: {d['decisions']} two-sided decisions; gap median {d['median']:.4f} mean {d['mean']:.4f} p90 {d['p90']:.3f}; "
              f">0.05: {d['frac_gt_005']:.2f}; Thm-A sum/game {d['per_game_sum_mean']:.2f}; shape med {d['rows_median']:.0f}x{d['cols_median']:.0f}; rules {rules}")
        print("   by column count:", {k: (round(v[0], 4), v[1]) for k, v in bycol.items()})
json.dump(out, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "out_e4_gap.json"), "w"), indent=1)
