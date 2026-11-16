# Gauntlet G1: Bradley-Terry ratings (Elo scale, agentform = 0)

| bot | rating | 95% CI | CPU-s / decision | games used | error games |
|---|---|---|---|---|---|
| stock@1000 | 192 | 152 to 233 | 2.4321 | 400 | 0 |
| stock@100 | 160 | 119 to 200 | 0.3234 | 400 | 0 |
| stock@400 | 150 | 110 to 192 | 0.9711 | 400 | 0 |
| stock@25 | 135 | 94 to 179 | 0.1429 | 400 | 0 |
| top | 43 | -4 to 92 | 0.3873 | 320 | 0 |
| agentform | 0 | 0 to 0 | 0.4047 | 680 | 0 |
| pimc | -41 | -90 to 5 | 0.384 | 360 | 0 |
| maxpower | -286 | -367 to -220 | 0.0 | 320 | 0 |
| random | -911 | -1495 to -752 | 0.0 | 320 | 0 |

| A | B | score of A | 95% CI | pairs | error games |
|---|---|---|---|---|---|
| agentform | stock@25 | 0.3167 | 0.242 to 0.392 | 60 | 0 |
| agentform | stock@100 | 0.3167 | 0.242 to 0.400 | 60 | 0 |
| agentform | stock@400 | 0.275 | 0.200 to 0.342 | 60 | 0 |
| agentform | stock@1000 | 0.225 | 0.158 to 0.292 | 60 | 0 |
| agentform | pimc | 0.5375 | 0.425 to 0.650 | 40 | 0 |
| agentform | top | 0.55 | 0.400 to 0.675 | 20 | 0 |
| agentform | maxpower | 0.8 | 0.675 to 0.925 | 20 | 0 |
| agentform | random | 1.0 | 1.000 to 1.000 | 20 | 0 |
| stock@25 | stock@100 | 0.475 | 0.325 to 0.625 | 20 | 0 |
| stock@25 | stock@400 | 0.45 | 0.300 to 0.601 | 20 | 0 |
| stock@25 | stock@1000 | 0.475 | 0.325 to 0.625 | 20 | 0 |
| stock@25 | pimc | 0.725 | 0.575 to 0.875 | 20 | 0 |
| stock@25 | top | 0.6 | 0.450 to 0.750 | 20 | 0 |
| stock@25 | maxpower | 0.925 | 0.849 to 1.000 | 20 | 0 |
| stock@25 | random | 1.0 | 1.000 to 1.000 | 20 | 0 |
| stock@100 | stock@400 | 0.625 | 0.500 to 0.750 | 20 | 0 |
| stock@100 | stock@1000 | 0.475 | 0.350 to 0.600 | 20 | 0 |
| stock@100 | pimc | 0.775 | 0.675 to 0.875 | 20 | 0 |
| stock@100 | top | 0.625 | 0.475 to 0.775 | 20 | 0 |
| stock@100 | maxpower | 0.925 | 0.850 to 1.000 | 20 | 0 |
| stock@100 | random | 1.0 | 1.000 to 1.000 | 20 | 0 |
| stock@400 | stock@1000 | 0.425 | 0.300 to 0.550 | 20 | 0 |
| stock@400 | pimc | 0.825 | 0.725 to 0.925 | 20 | 0 |
| stock@400 | top | 0.575 | 0.425 to 0.725 | 20 | 0 |
| stock@400 | maxpower | 0.95 | 0.875 to 1.000 | 20 | 0 |
| stock@400 | random | 1.0 | 1.000 to 1.000 | 20 | 0 |
| stock@1000 | pimc | 0.775 | 0.649 to 0.900 | 20 | 0 |
| stock@1000 | top | 0.75 | 0.600 to 0.900 | 20 | 0 |
| stock@1000 | maxpower | 0.9 | 0.825 to 0.975 | 20 | 0 |
| stock@1000 | random | 1.0 | 1.000 to 1.000 | 20 | 0 |
| pimc | top | 0.4 | 0.225 to 0.575 | 20 | 0 |
| pimc | maxpower | 0.8 | 0.650 to 0.925 | 20 | 0 |
| pimc | random | 1.0 | 1.000 to 1.000 | 20 | 0 |
| top | maxpower | 0.9 | 0.825 to 0.975 | 20 | 0 |
| top | random | 1.0 | 1.000 to 1.000 | 20 | 0 |
| maxpower | random | 0.95 | 0.875 to 1.000 | 20 | 0 |

| bot | foul-play equivalent budget (ms per state) |
|---|---|
| agentform | out of range: rating 0 vs stock@25..1000 ratings 135..192 |
| pimc | out of range: rating -41 vs stock@25..1000 ratings 135..192 |
| top | out of range: rating 43 vs stock@25..1000 ratings 135..192 |
| maxpower | out of range: rating -286 vs stock@25..1000 ratings 135..192 |
| random | out of range: rating -911 vs stock@25..1000 ratings 135..192 |

Meta-game equilibrium (symmetric zero-sum, P = score - 0.5; value -0.0000, exploitability 0.0e+00, check passed)

| bot | sigma | bootstrap mean | 95% CI | P(sigma > 0.01) | Nash-averaged payoff |
|---|---|---|---|---|---|
| stock@1000 | 1.000 | 0.466 | 0.000 to 1.000 | 0.623 | +0.0000 |
| stock@25 | 0.000 | 0.147 | 0.000 to 1.000 | 0.301 | -0.0250 |
| stock@100 | 0.000 | 0.348 | 0.000 to 1.000 | 0.510 | -0.0250 |
| stock@400 | 0.000 | 0.037 | 0.000 to 0.400 | 0.133 | -0.0750 |
| top | 0.000 | 0.002 | 0.000 to 0.000 | 0.014 | -0.2500 |
| agentform | 0.000 | 0.000 | 0.000 to 0.000 | 0.000 | -0.2750 |
| pimc | 0.000 | 0.000 | 0.000 to 0.000 | 0.000 | -0.2750 |
| maxpower | 0.000 | 0.000 | 0.000 to 0.000 | 0.000 | -0.4000 |
| random | 0.000 | 0.000 | 0.000 to 0.000 | 0.000 | -0.5000 |

Bootstrap: 1000 resamples of seeds within each pairing; L2 prior 0.01 on natural log-strengths.
