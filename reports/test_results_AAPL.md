# Sealed test results: AAPL (2024-01-02 to 2026-09-18)

Pre-registration tag `preregistered-v2`; run at commit `1d3055e3da` on 2026-09-19T19:55:08+00:00. 681 one-day-ahead forecasts, expanding window, refit every 21 days, configuration frozen in `config/preregistered_extension.json`.

## Risk task: volatility (QLIKE lower is better; coverage target 95%)
```
             qlike  mae_vol  rmse_var  coverage  kupiec_p  christoffersen_ind_p  cond_coverage_p  dm_vs_rolling_stat  dm_vs_rolling_p
rolling_21 -7.0563   0.0104     0.001    0.9192    0.0007                0.0842           0.0007                 NaN              NaN
ewma_094   -7.1508   0.0103     0.001    0.9251    0.0054                0.8901           0.0205              2.0188           0.0435
garch_t    -7.2104   0.0105     0.001    0.9413    0.3081                0.8335           0.5819              2.6495           0.0081
```

QLIKE by year:
```
target_date    2024    2025    2026
rolling_21  -7.2792 -6.9425 -6.9015
ewma_094    -7.4067 -6.9955 -7.0075
garch_t     -7.4394 -7.0727 -7.0804
```

Joint interval calibration (ARIMA mean + GARCH-t sigma):
```
     coverage  kupiec_p  christoffersen_ind_p  cond_coverage_p
50%    0.5433    0.0237                0.0121           0.0033
80%    0.8150    0.3238                0.4747           0.4760
95%    0.9369    0.1299                0.8249           0.3099
```

## Mean task: next-day return
```
                          rmse      mae  directional_acc  dm_vs_naive_stat  dm_vs_naive_p
naive_zero             0.01739      NaN              NaN               NaN            NaN
arima_000              0.01737  0.01175          0.54038           0.51251        0.60829
always_up_directional      NaN      NaN          0.54038               NaN            NaN
```

## Strategies vs buy-and-hold AAPL (5 bps costs)
```
                                                          cumulative_return  annual_return  sharpe  max_drawdown  hit_ratio  exposure  trades
buy_and_hold_AAPL                                                    0.7662         0.2343  0.8974       -0.3336     0.5404    1.0000     1.0
market_reference_buy_and_hold_SPY                                    0.6541         0.2047  1.2716       -0.1876     0.5639    1.0000     1.0
A_score_rule (primary: next-close execution)                         0.8318         0.2511  0.9483       -0.3336     0.5412    0.9985     1.0
A_score_rule (shift-1 convention: trade at signal close)             0.7662         0.2343  0.8974       -0.3336     0.5404    1.0000     1.0
B_vol_filter (primary: next-close execution)                        -0.0079        -0.0029  0.0701       -0.2046     0.5120    0.4905    74.0
B_vol_filter (shift-1 convention: trade at signal close)            -0.0918        -0.0350 -0.1279       -0.2014     0.5120    0.4905    74.0
```

Bootstrap (stationary, 2000 draws) of the Sharpe difference vs buy-and-hold:
```
              sharpe_diff  diff_ci_low  diff_ci_high  prob_strategy_not_better  strategy_sharpe_ci_low  strategy_sharpe_ci_high  benchmark_sharpe_ci_low  benchmark_sharpe_ci_high
A_score_rule       0.0510      -0.0007        0.1748                    0.3540                 -0.1492                   2.1086                  -0.2096                      2.06
B_vol_filter      -0.8273      -1.7785        0.0998                    0.9575                 -1.0941                   1.2553                  -0.2096                      2.06
```

Sharpe by transaction cost (primary execution):
```
                    0 bps   5 bps  10 bps  20 bps
buy_and_hold_AAPL  0.8981  0.8974  0.8966  0.8951
A_score_rule       0.9490  0.9483  0.9477  0.9463
B_vol_filter       0.1485  0.0701 -0.0082 -0.1649
```

By calendar year (primary execution):
```
                             strategy  year  days  cumulative_return  sharpe  max_drawdown  exposure
0                   buy_and_hold_AAPL  2024   252             0.3064  1.2910       -0.1535    1.0000
1                   buy_and_hold_AAPL  2025   250             0.0905  0.4286       -0.3107    1.0000
2                   buy_and_hold_AAPL  2026   179             0.2398  1.2567       -0.1271    1.0000
3   market_reference_buy_and_hold_SPY  2024   252             0.2482  1.8273       -0.0841    1.0000
4   market_reference_buy_and_hold_SPY  2025   250             0.1772  0.9401       -0.1876    1.0000
5   market_reference_buy_and_hold_SPY  2026   179             0.1257  1.3240       -0.0888    1.0000
6                        A_score_rule  2024   252             0.3549  1.4689       -0.1535    0.9960
7                        A_score_rule  2025   250             0.0905  0.4286       -0.3107    1.0000
8                        A_score_rule  2026   179             0.2398  1.2567       -0.1271    1.0000
9                        B_vol_filter  2024   252             0.1458  0.8048       -0.1307    0.6389
10                       B_vol_filter  2025   250            -0.0301 -0.1099       -0.1773    0.4400
11                       B_vol_filter  2026   179            -0.1073 -0.8715       -0.1290    0.3520
```

## Pre-registered acceptance (Sharpe >= buy-and-hold and smaller drawdown)
```
{
  "A_score_rule": true,
  "B_vol_filter": false
}
```
