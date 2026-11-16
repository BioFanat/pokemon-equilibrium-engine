Leaf non-determinism probe (2026-11-16). Python: engine/.venv/bin/python, run under `nice -n 10`. Each load level =
N busy `while time.time()<deadline` processes (start_new_session, killpg in finally). poke-engine 0.0.48 source:
~/.cache/uv/sdists-v9/pypi/poke-engine/0.0.48/wF7abHZGrk3nbZD69_lLJ/src (src/search.rs, poke-engine-py/src/lib.rs).

States: 480 children (+480 mirrors) from leaf_nondet.py's recipe, seed 0 (12 roots alternating 1v1 / 6v6, 40 kids each).

probe_load.py 20 0,4,8,12,14,20,28  -> out/probe_load.json  main run: depth/matrix/value per call, terminal formula,
                                                           bit-exact reproduction, retry wrapper (E)
probe_curve.py all 10 3 0,8,...,28  -> out/curve_all.json   load curve, 3 trials/level; 6v6 deeper results reproduced 210/210
taskpolicy -b ... probe_curve.py ecore 10 2 0,1,2,3,4,6,8 -> out/curve_ecore.json  (4 E-cores only; small-box proxy)
probe_wrap.py / probe_wrap1.py      -> out/wrap*.json       i8 depth wrap; wrapped depth-1 returns: maximin identical 14/14
probe_d1hunt.py 12,13,16 4 10       -> out/d1hunt.json      0 depth-1 mismatches in 113,905 depth-1 returns under load
probe_matrix.py 0,12,13,20 5        -> out/matrix.json      full_matrix impact vs deterministic reference (retry wrapper)
build_states.py                     -> states_s0.jsonl      serialized copy (note: to_string/from_string is not
                                                            byte-stable on the first trip; probes use in-memory states)

Result: depth 1 is never truncated. Under load the ID worker thread sometimes finishes depth >= 2 before the main
thread (sleep(0) = no syscall) stops it, and that complete deeper result is returned with depth_searched >= 2
(i8, wraps for terminal / small trees; terminal value = evaluate + 100*(d-1)*battle_is_over).
Fix: retry pe.id(c, 0) while depth_searched != 1 (deterministic maximin, same cost).

Counting caveat (code review 2026-11-16): LEAF_STATS is process-wide and not thread-local. In ours-vs-ours runs (e.g. LSn) the two bots search concurrently in foul-play worker threads, so per-decision deltas mix both bots, the other bot's tail after side A's last decision is missed (~1-2% of raw counts), and += is not atomic. Totals are approximate there; the non-depth-1 rate is unaffected. AFdet/LSs (vs stock, which never calls nash) are exact.
