"""Gate 0 (prerequisite of every other gate): info_planner.full_d1 returns the engine's own depth-1 matrix, unpruned.

Checks on children of real positions (random self-play, seeds 101..104):
  * no NaN; every cell the stock pe.id(child, 0) did compute is reproduced (|diff| <= 1e-3, f32 rounding), counted
    separately for re-computed move / '-tera' / switch rows (each re-computed row has >= 1 stock cell: its first);
  * pure maximin of the full matrix == nash._child_value(child) (the link to today's leaf);
  * mirror check: the mirrored child's stock cells vs -F^T (reported, not asserted: the engine is not exactly
    side-symmetric);
  * leaf change: |val(F) - nash._leaf(child, 'mid')| / ptp(F) (reported).
Run: OURBOT_DET_LEAF=1 python test_full_d1.py
"""
from _common import check, decisions, finish, np, pe, nash, pb  # noqa: F401
import info_planner as ip

kids = []
for seed in (101, 102, 103, 104):
    for state, s, view, legal, worlds in decisions(seed, 4, every=4, K=1):
        w = worlds[0]
        s1, s2 = pb.options(w)
        for a in s1[::3]:
            for b in s2[::3]:
                br = pe.generate_instructions(w, a, b)
                kids.append(w.apply_instructions(br[0]))
print(f"{len(kids)} children")
cmp = {"move": [0, 0], "tera": [0, 0], "switch": [0, 0], "row0/full": [0, 0]}
mm_bad = anti_n = anti_bad = 0
leafdiff = []
for c in kids:
    M, rows, cols = ip._d1(c)
    F, r2, c2 = ip.full_d1(c)
    assert r2 == rows and c2 == cols and not np.isnan(F).any()
    own_ids = {p.id.lower() for p in c.side_one.pokemon}
    for i, lab in enumerate(rows):
        k = ~np.isnan(M[i])
        kind = "row0/full" if k.all() else ("tera" if lab.endswith("-tera") else ("switch" if lab in own_ids else "move"))
        cmp[kind][0] += int(k.sum())
        cmp[kind][1] += int((np.abs(F[i, k] - M[i, k]) > 1e-3).sum())
    if abs(F.min(1).max() - nash._child_value(c)) > 1e-3:
        mm_bad += 1
    M2, m1, m2 = ip._d1(nash._mirror(c))
    for j, b in enumerate(m1):
        for i, a in enumerate(m2):
            if not np.isnan(M2[j, i]):
                anti_n += 1
                anti_bad += abs(M2[j, i] + F[rows.index(a), cols.index(b)]) > 1e-3
    leafdiff.append(abs(ip.value_of(F)[1] - nash._leaf(c, "mid")) / (np.ptp(F) or 1.0))
for kind, (n, bad) in cmp.items():
    check(f"stock cells reproduced, {kind} rows", bad == 0 and (n > 0 or kind == "tera"), f"({n} cells, {bad} mismatches)")
check("tera rows were exercised", cmp["tera"][0] > 0, f"({cmp['tera'][0]} cells)")
check("pure maximin(F) == nash._child_value", mm_bad == 0, f"({len(kids)} children, {mm_bad} bad)")
print(f"    mirror: {anti_bad}/{anti_n} stock mirror cells differ from -F^T (engine side asymmetry, not asserted)")
ld = np.array(leafdiff)
print(f"    leaf change |val(F) - mid leaf| / ptp(F): mean {ld.mean():.4f} median {np.median(ld):.4f} "
      f"p90 {np.quantile(ld, .9):.4f} max {ld.max():.4f}; zero (pure saddle, both sides) in {np.mean(ld < 1e-9):.2f}")
print(f"    full_d1 stats {ip.FULL_D1_STATS}")
finish()
