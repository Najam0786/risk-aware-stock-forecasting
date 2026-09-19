# Sealed test results: JPM (2024-01-02 to 2026-09-18)

Pre-registration tag `preregistered-v2`; run at commit `1d3055e3da` on 2026-09-19T19:56:16+00:00. 681 one-day-ahead forecasts, expanding window, refit every 21 days, configuration frozen in `config/preregistered_extension.json`.

## Risk task: volatility (QLIKE lower is better; coverage target 95%)
```
             qlike  mae_vol  rmse_var  coverage  kupiec_p  christoffersen_ind_p  cond_coverage_p  dm_vs_rolling_stat  dm_vs_rolling_p
rolling_21 -7.2756   0.0089    0.0007    0.9295    0.0205                0.3741           0.0460                 NaN              NaN
ewma_094   -7.3529   0.0088    0.0007    0.9413    0.3081                0.2957           0.3443              1.8011           0.0717
garch_t    -7.3933   0.0089    0.0007    0.9486    0.8679                0.4952           0.7816              3.1079           0.0019
```

QLIKE by year:
```
target_date    2024    2025    2026
rolling_21  -7.0762 -7.3458 -7.4584
ewma_094    -7.2077 -7.3897 -7.5060
garch_t     -7.2762 -7.4459 -7.4847
```

Joint interval calibration (ARIMA mean + GARCH-t sigma):
```
     coverage  kupiec_p  christoffersen_ind_p  cond_coverage_p
50%    0.5066    0.7302                0.9424           0.9398
80%    0.8150    0.3238                0.4747           0.4760
95%    0.9471    0.7340                0.4499           0.7095
```

## Mean task: next-day return
```
                          rmse      mae  directional_acc  dm_vs_naive_stat  dm_vs_naive_p
naive_zero             0.01497      NaN              NaN               NaN            NaN
arima_202              0.01502  0.01051          0.51542          -0.58379        0.55936
always_up_directional      NaN      NaN          0.56681               NaN            NaN
```

## Strategies vs buy-and-hold JPM (5 bps costs)
```
                                                          cumulative_return  annual_return  sharpe  max_drawdown  hit_ratio  exposure  trades
buy_and_hold_JPM                                                     1.1788         0.3340  1.3337       -0.2442     0.5668    1.0000     1.0
market_reference_buy_and_hold_SPY                                    0.6541         0.2047  1.2716       -0.1876     0.5639    1.0000     1.0
A_score_rule (primary: next-close execution)                         0.6709         0.2092  1.0116       -0.1963     0.5725    0.8003   149.0
A_score_rule (shift-1 convention: trade at signal close)             0.9281         0.2750  1.2158       -0.2454     0.5678    0.8018   149.0
B_vol_filter (primary: next-close execution)                         0.3060         0.1038  0.7009       -0.0985     0.5633    0.4875    53.0
B_vol_filter (shift-1 convention: trade at signal close)             0.5516         0.1765  1.1038       -0.1031     0.5856    0.4890    53.0
```

Bootstrap (stationary, 2000 draws) of the Sharpe difference vs buy-and-hold:
```
              sharpe_diff  diff_ci_low  diff_ci_high  prob_strategy_not_better  strategy_sharpe_ci_low  strategy_sharpe_ci_high  benchmark_sharpe_ci_low  benchmark_sharpe_ci_high
A_score_rule      -0.3221      -0.8296        0.1767                    0.8945                 -0.0010                   2.1672                   0.3019                    2.4712
B_vol_filter      -0.6328      -1.6275        0.3510                    0.9055                 -0.2777                   1.6653                   0.3019                    2.4712
```

Sharpe by transaction cost (primary execution):
```
                   0 bps   5 bps  10 bps  20 bps
buy_and_hold_JPM  1.3344  1.3337  1.3329  1.3315
A_score_rule      1.1434  1.0116  0.8797  0.6157
B_vol_filter      0.7623  0.7009  0.6393  0.5157
```

By calendar year (primary execution):
```
                             strategy  year  days  cumulative_return  sharpe  max_drawdown  exposure
0                    buy_and_hold_JPM  2024   252             0.4422  1.6756       -0.1013    1.0000
1                    buy_and_hold_JPM  2025   250             0.3727  1.4102       -0.2442    1.0000
2                    buy_and_hold_JPM  2026   179             0.1006  0.7116       -0.1547    1.0000
3   market_reference_buy_and_hold_SPY  2024   252             0.2482  1.8273       -0.0841    1.0000
4   market_reference_buy_and_hold_SPY  2025   250             0.1772  0.9401       -0.1876    1.0000
5   market_reference_buy_and_hold_SPY  2026   179             0.1257  1.3240       -0.0888    1.0000
6                        A_score_rule  2024   252             0.1951  0.9350       -0.0897    0.7183
7                        A_score_rule  2025   250             0.2804  1.2718       -0.1963    0.8480
8                        A_score_rule  2026   179             0.0920  0.7302       -0.1417    0.8492
9                        B_vol_filter  2024   252             0.1685  0.8863       -0.0884    0.5675
10                       B_vol_filter  2025   250             0.1579  1.0869       -0.0926    0.5360
11                       B_vol_filter  2026   179            -0.0347 -0.4068       -0.0929    0.3073
```

## Pre-registered acceptance (Sharpe >= buy-and-hold and smaller drawdown)
```
{
  "A_score_rule": false,
  "B_vol_filter": false
}
```
