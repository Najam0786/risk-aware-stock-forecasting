# Sealed test results: MSFT (2024-01-02 to 2026-09-18)

Pre-registration tag `preregistered-v2`; run at commit `1d3055e3da` on 2026-09-19T19:55:16+00:00. 681 one-day-ahead forecasts, expanding window, refit every 21 days, configuration frozen in `config/preregistered_extension.json`.

## Risk task: volatility (QLIKE lower is better; coverage target 95%)
```
             qlike  mae_vol  rmse_var  coverage  kupiec_p  christoffersen_ind_p  cond_coverage_p  dm_vs_rolling_stat  dm_vs_rolling_p
rolling_21 -7.0845   0.0097    0.0011    0.9207    0.0012                0.0426           0.0007                 NaN              NaN
ewma_094   -7.1878   0.0097    0.0011    0.9295    0.0205                0.1071           0.0187              2.3465           0.0190
garch_t    -7.2282   0.0098    0.0011    0.9457    0.6088                0.4073           0.6223              2.9612           0.0031
```

QLIKE by year:
```
target_date    2024    2025    2026
rolling_21  -7.4973 -7.3864 -6.0815
ewma_094    -7.6122 -7.4297 -6.2525
garch_t     -7.6431 -7.4909 -6.2773
```

Joint interval calibration (ARIMA mean + GARCH-t sigma):
```
     coverage  kupiec_p  christoffersen_ind_p  cond_coverage_p
50%    0.5140    0.4665                0.5771           0.6567
80%    0.7944    0.7168                0.0654           0.1714
95%    0.9471    0.7340                0.4499           0.7095
```

## Mean task: next-day return
```
                          rmse      mae  directional_acc  dm_vs_naive_stat  dm_vs_naive_p
naive_zero             0.01673      NaN              NaN               NaN            NaN
arima_100              0.01688  0.01144          0.52276          -1.97577        0.04818
always_up_directional      NaN      NaN          0.52423               NaN            NaN
```

## Strategies vs buy-and-hold MSFT (5 bps costs)
```
                                                          cumulative_return  annual_return  sharpe  max_drawdown  hit_ratio  exposure  trades
buy_and_hold_MSFT                                                    0.3405         0.1145  0.5373       -0.3450     0.5242    1.0000     1.0
market_reference_buy_and_hold_SPY                                    0.6541         0.2047  1.2716       -0.1876     0.5639    1.0000     1.0
A_score_rule (primary: next-close execution)                         0.1638         0.0578  0.3474       -0.3512     0.5134    0.8752    14.0
A_score_rule (shift-1 convention: trade at signal close)             0.1404         0.0498  0.3175       -0.3834     0.5134    0.8752    14.0
B_vol_filter (primary: next-close execution)                         0.0431         0.0157  0.1775       -0.2100     0.5203    0.6153    46.0
B_vol_filter (shift-1 convention: trade at signal close)             0.1361         0.0484  0.3779       -0.1881     0.5251    0.6153    46.0
```

Bootstrap (stationary, 2000 draws) of the Sharpe difference vs buy-and-hold:
```
              sharpe_diff  diff_ci_low  diff_ci_high  prob_strategy_not_better  strategy_sharpe_ci_low  strategy_sharpe_ci_high  benchmark_sharpe_ci_low  benchmark_sharpe_ci_high
A_score_rule      -0.1899      -0.5984        0.2115                    0.8215                 -0.7447                   1.4720                  -0.6126                     1.693
B_vol_filter      -0.3598      -1.5425        0.8112                    0.7110                 -1.0251                   1.3788                  -0.6126                     1.693
```

Sharpe by transaction cost (primary execution):
```
                    0 bps   5 bps  10 bps  20 bps
buy_and_hold_MSFT  0.5380  0.5373  0.5366  0.5351
A_score_rule       0.3577  0.3474  0.3370  0.3162
B_vol_filter       0.2302  0.1775  0.1248  0.0195
```

By calendar year (primary execution):
```
                             strategy  year  days  cumulative_return  sharpe  max_drawdown  exposure
0                   buy_and_hold_MSFT  2024   252             0.1287  0.7067       -0.1549    1.0000
1                   buy_and_hold_MSFT  2025   250             0.1558  0.7212       -0.2056    1.0000
2                   buy_and_hold_MSFT  2026   179             0.0275  0.2836       -0.2672    1.0000
3   market_reference_buy_and_hold_SPY  2024   252             0.2482  1.8273       -0.0841    1.0000
4   market_reference_buy_and_hold_SPY  2025   250             0.1772  0.9401       -0.1876    1.0000
5   market_reference_buy_and_hold_SPY  2026   179             0.1257  1.3240       -0.0888    1.0000
6                        A_score_rule  2024   252             0.1453  0.7813       -0.1549    0.9921
7                        A_score_rule  2025   250             0.0580  0.3598       -0.2064    0.8440
8                        A_score_rule  2026   179            -0.0395 -0.0072       -0.2704    0.7542
9                        B_vol_filter  2024   252             0.0338  0.2715       -0.1603    0.8492
10                       B_vol_filter  2025   250             0.0132  0.1623       -0.1548    0.6800
11                       B_vol_filter  2026   179            -0.0041  0.0205       -0.1122    0.1955
```

## Pre-registered acceptance (Sharpe >= buy-and-hold and smaller drawdown)
```
{
  "A_score_rule": false,
  "B_vol_filter": false
}
```
