# Average vs last iterate on in-engine stage matrices

Source of the paper's "Average or last iterate" numbers (final5.tex, Regret and no-regret dynamics).

- `build_data.py` builds the stage matrices: 1,477 positions with at least two options per side, from 20 in-engine
  games of sampled play against MCTS with 2,000 iterations, rebuilt by deterministic replay. Its output `data.pkl`
  (9.9 MB) is not committed; rerun the script to regenerate it, then point `D` in the scripts at it.
- `b3_bandit_vs_full.py` samples 120 of these matrices (normalised to range 1) and runs simultaneous Hedge and
  optimistic Hedge (OMWU), step 0.1, 1,000 iterations, plus RM+ (alternating, linear averaging). Output: `b3.out`.
- `b3_followup.py` adds the checks reported in the paper.

Results (rechecked 2026-11-17):

| Quantity | Hedge | OMWU | RM+ |
|---|---|---|---|
| Average strategies, median duality gap | 0.032 | 0.032 | 0.0000 (p90 0.0001) |
| Last iterate, median gap | 0.0087 | 0.0085 | — |
| Last iterate, 90th percentile | 0.092 | 0.086 | — |

- 70 of 120 matrices (58%) have a pure saddle point, where the last iterate settles.
- On the other 50: last-iterate median 0.049 (Hedge) / 0.044 (OMWU) vs average 0.040 / 0.039.
- Hedge's average satisfies the no-regret-dynamics bound with the fixed-step Hedge regret on every matrix
  (at most 0.67 of the bound). The gap equals the sum of average regrets exactly for simultaneous updates.
