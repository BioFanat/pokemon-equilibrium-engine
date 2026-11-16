# Synthesis analyses

Scripts behind `theory/synthesis.md` and the current report's provenance rows (Bayes regret = ln 5, meta-game time
averages, E4 node gaps, weight replay, VOI toy). Each `out_*` file is the recorded output of the script with the same stem.

They read the per-decision logs under `engine/showdown/results/<TAG>/replays_headless/`, which are kept out of git
(size); regenerate them with the Modal runs (`showdown/launch_modal_batch.sh`, `showdown/launch_open_items.sh`) or point
`SHOWDOWN_RESULTS` at a copy. Run from `engine/` with `.venv/bin/python analysis/synthesis/<script>.py`.

| script | output | what |
|---|---|---|
| `bh_regret.py` | `out_bh_regret.json` | style posterior's log-loss regret per BH stream (headline panel D) |
| `bayes_identity.py` | `out_bayes_identity.txt` | prequential identity R_T(s) = ln K + ln P_T(s) check |
| `hedge_dyn.py` | `out_hedge_dyn.{txt,json}` | external / dynamic regret, last iterate vs time average of carried Hedge |
| `weight_replay.py` | `out_weight_replay.txt` | logged vs replayed Hedge weights; update-count integrity |
| `e4_gap.py` | `out_e4_gap.{txt,json}` | logged Nash gap at bandit roots (Lecture 7 identity, Prop. L4) |
| `checks2.py` | `out_checks2.txt` | BH prediction metrics (log loss, accuracy) |
| `voi_toy.py` | `out_voi_toy.txt` | value-of-information toy (eta_NL) |
