# Style-switch analysis (research/prereg_style_switch.md)

### SW2 (bayes, lambda=1.0, root /Users/biofanat/sandbox/ourbot/sw/replays_headless)

| stream | T | t0 | P_t0(u) | val err | clean | tau_map | tau_90 | tau_99 | tau_leave | tau_hat | tau_leave_hat | X | X_logged | FA | P90_b6 | id err rec | meta |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 396 | 87 | 1.41e-11 | 0.000127 | y | 108 | 109 | 123 | 109 | 70.6 | 34.9 | 24.98 | 24.98 | no | 196 | 0.000303 | 396 |
| 1 | 379 | 112 | 2.13e-09 | 0.000060 | y | 59 | 60 | 70 | 4 | 66.2 | -2.8 | 19.97 | 19.97 | no | 172 | 0.000174 | 379 |
| 2 | 308 | 82 | 6.17e-15 | 0.000203 | y | 95 | 99 | 107 | 1 | 111.0 | -14.7 | 32.72 | 32.72 | no | 181 | 0.000292 | 308 |
| 3 | 445 | 110 | 5.71e-11 | 0.000109 | y | 107 | 114 | 125 | 49 | 59.7 | 59.7 | 23.59 | 23.59 | no | 224 | 0.000324 | 445 |
| 4 | 380 | 116 | 1.09e-21 | 0.000071 | y | 127 | 133 | 134 | 41 | 158.2 | 17.1 | 48.27 | 48.27 | no | 249 | 0.000418 | 380 |
| 5 | 355 | 87 | 1.15e-12 | 0.000139 | y | 92 | 94 | 96 | 94 | 69.6 | 69.6 | 27.49 | 27.49 | no | 181 | 0.000345 | 355 |
| 6 | 326 | 84 | 4.14e-14 | 0.000078 | y | 124 | 128 | 130 | 99 | 78.0 | 78.0 | 30.82 | 30.82 | no | 212 | 0.000394 | 326 |
| 7 | 392 | 105 | 6.70e-02 | 0.000086 | y | 12 | 17 | 26 | 1 | 9.0 | -152.2 | 2.70 | 2.70 | yes@30 | 122 | 0.000161 | 392 |
| 8 | 411 | 158 | 5.72e-19 | 0.000096 | y | >=253 | >=253 | >=253 | 1 | 143.4 | -76.0 | 42.01 | 42.01 | no | 405 | 0.000385 | 411 |
| 9 | 340 | 126 | 2.70e-19 | 0.000212 | y | 167 | 173 | 176 | 87 | 108.2 | 108.2 | 42.76 | 42.76 | no | 299 | 0.000604 | 340 |

- tau_map: median 107.50 [77.00, 137.70] (n=10, censored 1)
- tau_90: median 111.50 [84.50, 150.50] (n=10, censored 1)
- tau_99: median 124.00 [83.00, 153.00] (n=10, censored 1)
- tau_leave: median 45.00 [1.00, 94.00] (n=10)
- tau_hat: median 74.31 [64.65, 110.98] (n=10)
- tau_leave_hat: median 26.00 [-39.44, 69.59] (n=10)
- X: median 29.15 [22.46, 42.01] (n=10)
- X_logged: median 29.15 [23.59, 42.01] (n=10)
- tau_map_over_t0: median 1.13 [0.81, 1.36] (n=10, censored 1)
- tau_map_over_tau_hat: median 1.43 [0.89, 1.65] (n=10, censored 1)
- tau90_minus_tau_map: median 4.00 [1.00, 6.00] (n=9, median bounds under censoring 4.00..4.50)
- Spearman rho(tau_hat, tau_map) = 0.770 (p=0.00922, n=10)
- clean 10/10; false-alarm streams 1; flips <= 30 updates 1; max val err 0.000212 (phase 1 0.000109, post 0.000212); max id err rec 0.000604, logged 0.004572; max |X - identity| 0.000000 (logged q-hat 0.000409)
- score pre: 20 seeds, mean 0.925; vs_RQsafe: +0.000 [-0.142, +0.142] (n=20)
- score post: 80 seeds, mean 1.000; vs_RQunif: +0.006 [-0.006, +0.019] (n=78)

### SW4 (bayes, lambda=1.0, root /Users/biofanat/sandbox/ourbot/sw/replays_headless)

| stream | T | t0 | P_t0(u) | val err | clean | tau_map | tau_90 | tau_99 | tau_leave | tau_hat | tau_leave_hat | X | X_logged | FA | P90_b6 | id err rec | meta |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 599 | 189 | 1.05e-20 | 0.000203 | y | 268 | 275 | 276 | 48 | 134.0 | 66.0 | 46.00 | 46.00 | no | 464 | 0.000586 | 599 |
| 1 | 643 | 225 | 1.10e-39 | 0.000147 | y | 247 | 251 | 258 | 82 | 245.7 | 173.8 | 89.71 | 89.71 | no | 476 | 0.000507 | 643 |
| 2 | 674 | 177 | 1.16e-18 | 0.000111 | y | 121 | 130 | 143 | 71 | 137.6 | 5.2 | 41.30 | 41.30 | no | 307 | 0.000220 | 674 |
| 3 | 635 | 177 | 6.65e-25 | 0.000115 | y | 206 | 239 | 242 | 1 | 190.0 | -48.6 | 55.67 | 55.67 | no | 416 | 0.000303 | 635 |
| 4 | 638 | 199 | 7.71e-27 | 0.000054 | y | 213 | 223 | 227 | 1 | 205.2 | -74.8 | 60.13 | 60.13 | no | 422 | 0.000170 | 638 |
| 5 | 716 | 246 | 5.54e-43 | 0.000186 | y | 334 | 388 | 392 | 1 | 332.1 | -211.7 | 97.30 | 97.30 | no | 634 | 0.000632 | 716 |
| 6 | 666 | 219 | 3.15e-37 | 0.000083 | y | 237 | 240 | 256 | 90 | 274.7 | 34.6 | 84.05 | 84.05 | no | 459 | 0.000461 | 666 |
| 7 | 631 | 154 | 1.24e-08 | 0.000097 | y | 65 | 71 | 77 | 71 | 47.0 | 42.2 | 18.21 | 18.21 | no | 225 | 0.000293 | 631 |
| 8 | 626 | 203 | 4.35e-32 | 0.000118 | y | 150 | 153 | 158 | 77 | 191.2 | 158.6 | 72.21 | 72.21 | no | 356 | 0.000584 | 626 |
| 9 | 688 | 252 | 4.03e-46 | 0.000234 | y | 365 | 368 | 370 | 205 | 349.9 | 18.1 | 104.53 | 104.53 | no | 620 | 0.000476 | 688 |

- tau_map: median 225.00 [150.00, 273.50] (n=10)
- tau_90: median 239.50 [153.00, 321.50] (n=10)
- tau_99: median 249.00 [158.00, 314.00] (n=10)
- tau_leave: median 71.00 [1.00, 82.00] (n=10)
- tau_hat: median 198.23 [137.56, 288.87] (n=10)
- tau_leave_hat: median 26.35 [-48.62, 100.41] (n=10)
- X: median 66.17 [46.00, 90.67] (n=10)
- X_logged: median 66.17 [46.00, 90.68] (n=10)
- tau_map_over_t0: median 1.09 [0.74, 1.36] (n=10)
- tau_map_over_tau_hat: median 1.02 [0.88, 1.21] (n=10)
- tau90_minus_tau_map: median 6.50 [3.00, 20.00] (n=10)
- Spearman rho(tau_hat, tau_map) = 0.721 (p=0.0186, n=10)
- clean 10/10; false-alarm streams 0; flips <= 30 updates 0; max val err 0.000234 (phase 1 0.000097, post 0.000234); max id err rec 0.000632, logged 0.004934; max |X - identity| 0.000000 (logged q-hat 0.000671)
- score pre: 40 seeds, mean 0.850; vs_RQsafe: +0.000 [-0.099, +0.099] (n=40)
- score post: 140 seeds, mean 0.996; vs_RQunif: +0.009 [-0.008, +0.026] (n=58)

### SW4p (ptd, lambda=0.98, root /Users/biofanat/sandbox/ourbot/sw_p/replays_headless)

| stream | T | t0 | P_t0(u) | val err | clean | tau_map | tau_90 | tau_99 | tau_leave | tau_hat | tau_leave_hat | X | X_logged | FA | P90_b6 | id err rec | meta |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 333 | 176 | 6.81e-07 | 0.000091 | y | 26 | 29 | 33 | 28 | 32.9 | -8.6 | 10.72 | 10.72 | no | 205 | - | 333 |
| 1 | 392 | 201 | 1.09e-06 | 0.000092 | y | 37 | 59 | 66 | 31 | 30.6 | 8.3 | 10.45 | 10.45 | no | 260 | - | 392 |
| 2 | 373 | 197 | 3.87e-10 | 0.000078 | y | 114 | 115 | 116 | 77 | 42.2 | 12.7 | 4.48 | 4.48 | no | 312 | - | 373 |
| 3 | 359 | 256 | 3.00e-06 | 0.000078 | y | 37 | 41 | 43 | 1 | 30.8 | -44.5 | 6.40 | 6.40 | no | 297 | - | 359 |
| 4 | 303 | 189 | 2.83e-03 | 0.000075 | y | 25 | 33 | 36 | 1 | 16.6 | -36.8 | 4.72 | 4.72 | no | 222 | - | 303 |
| 5 | 306 | 195 | 3.11e-08 | 0.000102 | y | 44 | 59 | >=111 | 20 | 31.6 | 39.7 | 10.24 | 10.24 | no | 254 | - | 306 |
| 6 | 274 | 173 | 5.54e-08 | 0.000086 | y | 20 | 23 | 48 | 23 | 30.3 | 42.3 | 13.28 | 13.28 | no | 196 | - | 274 |
| 7 | 433 | 282 | 1.49e-11 | 0.000090 | y | 80 | 85 | 88 | 76 | 40.4 | 63.1 | 10.55 | 10.55 | no | 367 | - | 433 |
| 8 | 401 | 276 | 7.07e-04 | 0.000082 | y | 38 | 71 | 74 | 13 | 16.4 | 7.7 | 6.21 | 6.21 | no | 347 | - | 401 |
| 9 | 291 | 194 | 6.39e-09 | 0.000082 | y | 46 | 48 | 56 | 44 | 35.7 | 26.8 | 14.26 | 14.26 | no | 242 | - | 291 |

- tau_map: median 37.50 [26.00, 62.00] (n=10)
- tau_90: median 53.50 [33.00, 72.00] (n=10)
- tau_99: median 61.00 [42.00, 92.50] (n=10, censored 1)
- tau_leave: median 25.50 [10.50, 53.50] (n=10)
- tau_hat: median 31.24 [23.71, 36.93] (n=10)
- tau_leave_hat: median 10.45 [-14.29, 39.71] (n=10)
- X: median 10.34 [5.56, 11.92] (n=10)
- X_logged: median 10.34 [6.21, 11.87] (n=10)
- tau_map_over_t0: median 0.17 [0.14, 0.26] (n=10)
- tau_map_over_tau_hat: median 1.34 [1.02, 1.98] (n=10)
- tau90_minus_tau_map: median 4.50 [2.50, 15.00] (n=10)
- Spearman rho(tau_hat, tau_map) = 0.717 (p=0.0195, n=10)
- clean 10/10; false-alarm streams 0; flips <= 30 updates 3; max val err 0.000102 (phase 1 0.000092, post 0.000102); max id err rec -, logged -; max |X - identity| - (logged q-hat -)
- score pre: 40 seeds, mean 0.900; vs_RQsafe: +0.050 [-0.054, +0.154] (n=40)
- score post: 40 seeds, mean 1.000; vs_RQunif: +0.013 [-0.012, +0.038] (n=39)

### Predictions

| P | status | detail |
|---|---|---|
| P1 | PASS | {"SW2": {"median_tau_map": {"median": 107.5, "lo": 77.0, "hi": 137.6999999999989, "n": 10, "n_censored": 1, "median_bounds": [107.5, 107.5]}, "median_ratio_t0": {"median": 1.126682085786375, "lo": 0.8108066502463054, "hi": 1.3587848932676518, "n": 10, "n_censored": 1, "median_bounds": [1.126682085786375, 1.126682085786375]}, "refuted_ratio": false, "early_flips_le30": 1, "refuted_early": false, "interval": [70, 170], "median_in_interval": true, "censored_counting_against": 1, "censored_inconclusive": 0}, "SW4": {"median_tau_map": {"median": 225.0, "lo": 150.0, "hi": 273.5, "n": 10, "n_censored |
| P2 | PASS | {"ratio_point": 2.0930232558139537, "ratio_bounds": [2.0930232558139537, 2.0930232558139537], "in_interval_1p5_2p7": true, "refuted_lt_1p25": false} |
| P3 | PASS | {"n_streams": 20, "arms": ["SW2", "SW4"], "median_ratio_bounds": [1.0637074527568808, 1.0637074527568808], "n_tau_hat_zero": 0, "spearman": 0.8571428571428572, "ratio_in_0p7_1p4": true, "rho_ge_0p5": true, "refuted_ratio": false, "refuted_rho": false} |
| P4 | PASS | {"SW2": {"tau90_minus_tau_map": {"median": 4.0, "lo": 1.0, "hi": 6.0, "n": 9, "n_unknown": 1, "median_bounds": [4.0, 4.5]}, "in_0_25": true, "tau_leave": {"median": 45.0, "lo": 1.0, "hi": 94.0, "n": 10, "n_censored": 0, "median_bounds": [45.0, 45.0]}, "tau_leave_hat": {"median": 26.003266121183124, "lo": -39.43505832994916, "hi": 69.58830082981193, "n": 10, "n_censored": 0, "median_bounds": [26.003266121183124, 26.003266121183124]}}, "SW4": {"tau90_minus_tau_map": {"median": 6.5, "lo": 3.0, "hi": 20.0, "n": 10, "n_unknown": 0, "median_bounds": [6.5, 6.5]}, "in_0_25": true, "tau_leave": {"media NOTE: no refutation clause in the prereg; the [0, 25] interval is used as the criterion |
| P5 | PASS | {"SW2": {"X": {"median": 29.15171615006101, "lo": 22.4571801494192, "hi": 42.0051162709065, "n": 10, "n_censored": 0, "median_bounds": [29.15171615006101, 29.15171615006101]}, "interval": [20, 60], "in_interval": true, "refuted_lt_10": false}, "SW4": {"X": {"median": 66.1702562614871, "lo": 46.00312403484362, "hi": 90.6748453696199, "n": 10, "n_censored": 0, "median_bounds": [66.1702562614871, 66.1702562614871]}, "interval": [45, 110], "in_interval": true, "refuted_lt_10": false}} |
| P6 | PASS | {"false_alarm_streams": {"SW2": 1, "SW4": 0, "SW4p": 0}, "SW4p_median_tau_map": {"median": 37.5, "lo": 26.0, "hi": 62.0, "n": 10, "n_censored": 0, "median_bounds": [37.5, 37.5]}, "SW4p_in_15_70": true, "SW4p_ratio_tau_hat": {"median": 1.3392574865545541, "lo": 1.0246368907226884, "hi": 1.9798523090816131, "n": 10, "n_censored": 0, "median_bounds": [1.3392574865545541, 1.3392574865545541]}, "ratio_SW4p_SW4_bounds": [0.16666666666666666, 0.16666666666666666], "ratio_le_0p4": true} |
| P7 | PASS | {"SW2_post_vs_RQunif": {"mean_diff": 0.00641025641025641, "se": 0.006410256410256409, "lo": -0.006153846153846152, "hi": 0.018974358974358972, "n_pairs": 78, "in_interval": true, "bound": 0.05}, "SW2_pre_vs_RQsafe": {"mean_diff": 0.0, "se": 0.07254762501100116, "lo": -0.14219334502156228, "hi": 0.14219334502156228, "n_pairs": 20, "in_interval": true, "bound": 0.1}, "SW4_post_vs_RQunif": {"mean_diff": 0.008620689655172414, "se": 0.008620689655172412, "lo": -0.008275862068965512, "hi": 0.02551724137931034, "n_pairs": 58, "in_interval": true, "bound": 0.05}, "SW4_pre_vs_RQsafe": {"mean_diff": 0.0 NOTE: descriptive, no refutation in the prereg; FAIL here only means outside the stated interval |
| P8 | PASS | {"SW2": {"max_id_err_rec": 0.0006042697513066742, "max_id_err_logged": 0.004572199579769176, "n_streams": 10, "violated": false}, "SW4": {"max_id_err_rec": 0.0006319316296694577, "max_id_err_logged": 0.004934231386933163, "n_streams": 10, "violated": false}} |
