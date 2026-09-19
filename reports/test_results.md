# Sealed test results (2024-01-02 to 2026-09-18)

Pre-registration tag `preregistered-v1`; run at commit `3d52e81f49` on 2026-09-19T18:22:13+00:00. 681 one-day-ahead forecasts, expanding window, refit every 21 days, configuration frozen in `config/preregistered.json`.

## Risk task: volatility (QLIKE lower is better; coverage target 95%)
```
             qlike  mae_vol  rmse_var  coverage  kupiec_p  christoffersen_ind_p  cond_coverage_p  dm_vs_rolling_stat  dm_vs_rolling_p
rolling_21 -8.3559   0.0054    0.0004    0.9295    0.0205                0.0185           0.0043                 NaN              NaN
ewma_094   -8.4352   0.0055    0.0004    0.9383    0.1768                0.1512           0.1434              1.5296           0.1261
garch_t    -8.5017   0.0054    0.0004    0.9515    0.8528                0.5937           0.8526              2.5613           0.0104
```

QLIKE by year:
```
target_date    2024    2025    2026
rolling_21  -8.4692 -8.1483 -8.4863
ewma_094    -8.6322 -8.1629 -8.5382
garch_t     -8.6591 -8.3113 -8.5459
```

Joint interval calibration (ARIMA mean + GARCH-t sigma):
```
     coverage  kupiec_p  christoffersen_ind_p  cond_coverage_p
50%    0.4949    0.7885                0.3173           0.5851
80%    0.7871    0.4029                0.2535           0.3673
95%    0.9530    0.7159                0.6467           0.8426
```

## Mean task: next-day return
```
                          rmse      mae  directional_acc  dm_vs_naive_stat  dm_vs_naive_p
naive_zero             0.00980      NaN              NaN               NaN            NaN
arima_200              0.00975  0.00656          0.52717           0.61162        0.54079
always_up_directional      NaN      NaN          0.56388               NaN            NaN
```

## Strategies vs buy-and-hold SPY (5 bps costs)
```
                                                          cumulative_return  annual_return  sharpe  max_drawdown  hit_ratio  exposure  trades
buy_and_hold_SPY                                                     0.6541         0.2047  1.2716       -0.1876     0.5639    1.0000     1.0
A_score_rule (primary: next-close execution)                         0.2649         0.0909  0.7309       -0.2483     0.5552    0.8120   189.0
A_score_rule (shift-1 convention: trade at signal close)             0.3194         0.1080  0.7836       -0.1632     0.5497    0.8120   190.0
B_vol_filter (primary: next-close execution)                         0.0272         0.0100  0.1719       -0.0826     0.5235    0.4376    73.0
B_vol_filter (shift-1 convention: trade at signal close)            -0.0542        -0.0204 -0.2491       -0.1691     0.5101    0.4376    74.0
```

Bootstrap (stationary, 2000 draws) of the Sharpe difference vs buy-and-hold:
```
              sharpe_diff  diff_ci_low  diff_ci_high  prob_strategy_not_better  strategy_sharpe_ci_low  strategy_sharpe_ci_high  benchmark_sharpe_ci_low  benchmark_sharpe_ci_high
A_score_rule      -0.5407      -1.1951        0.0604                    0.9655                 -0.5611                   2.0654                   0.2373                    2.4638
B_vol_filter      -1.0997      -2.3284        0.0303                    0.9700                 -0.7684                   1.1909                   0.2373                    2.4638
```

Sharpe by transaction cost (primary execution):
```
                   0 bps   5 bps  10 bps  20 bps
buy_and_hold_SPY  1.2728  1.2716  1.2703  1.2678
A_score_rule      0.9993  0.7309  0.4627 -0.0709
B_vol_filter      0.3560  0.1719 -0.0115 -0.3741
```

By calendar year (primary execution):
```
           strategy  year  days  cumulative_return  sharpe  max_drawdown  exposure
0  buy_and_hold_SPY  2024   252             0.2482  1.8273       -0.0841    1.0000
1  buy_and_hold_SPY  2025   250             0.1772  0.9401       -0.1876    1.0000
2  buy_and_hold_SPY  2026   179             0.1257  1.3240       -0.0888    1.0000
3      A_score_rule  2024   252             0.1577  1.3836       -0.0627    0.7937
4      A_score_rule  2025   250            -0.0551 -0.2938       -0.2246    0.7920
5      A_score_rule  2026   179             0.1564  1.7400       -0.0620    0.8659
6      B_vol_filter  2024   252             0.0498  0.6611       -0.0401    0.5317
7      B_vol_filter  2025   250             0.0101  0.1890       -0.0375    0.3720
8      B_vol_filter  2026   179            -0.0313 -0.5314       -0.0826    0.3966
```

## Pre-registered acceptance (Sharpe >= buy-and-hold and smaller drawdown)
```
{
  "A_score_rule": false,
  "B_vol_filter": false
}
```
