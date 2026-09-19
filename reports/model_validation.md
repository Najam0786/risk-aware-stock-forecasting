# Model validation (train 2010-2021, validation 2022-2023; test sealed)

Walk-forward, expanding window, refit every 21 days. 501 one-day-ahead forecasts.

## Risk task: volatility forecasts (QLIKE lower is better; coverage target 95%)
```
              qlike  mae_vol  rmse_var  coverage  kupiec_p  christoffersen_ind_p  cond_coverage_p  dm_vs_rolling_stat  dm_vs_rolling_p
rolling_21  -7.9479   0.0066    0.0003    0.9361    0.1711                0.3905           0.2711                 NaN              NaN
ewma_094    -7.9738   0.0066    0.0003    0.9421    0.4291                0.0587           0.1225              1.7302           0.0836
garch_t     -7.9462   0.0068    0.0003    0.9521    0.8284                0.1197           0.2911             -0.0836           0.9334
gjr_garch_t -7.9439   0.0066    0.0003    0.9421    0.4291                0.5491           0.6113             -0.1199           0.9046
```

QLIKE by year:
```
date           2022    2023
rolling_21  -7.3102 -8.5881
ewma_094    -7.3400 -8.6101
garch_t     -7.3008 -8.5942
gjr_garch_t -7.2666 -8.6238
```

## Mean task: ARMA order selection on train (BIC)
```
   p  q        aic        bic
0  2  0  8858.9942  8883.0183
1  0  2  8859.4385  8883.4627
2  1  1  8863.8828  8887.9069
3  1  2  8859.4132  8889.4434
4  2  1  8860.9287  8890.9588
5  1  0  8874.0734  8892.0915
6  2  2  8860.3324  8896.3686
7  0  1  8881.5535  8899.5716
8  0  0  8925.7208  8937.7328
```

## Exogenous set selection on train (BIC)
```
                      exog        aic        bic
0                     none  8858.9942  8883.0183
1               vix_change  8857.4286  8887.4587
2  vix_change+dgs10_change  8857.6630  8893.6992
```

## Mean task: next-day return forecasts on validation
```
                                   rmse      mae  directional_acc  dm_vs_naive_stat  dm_vs_naive_p
naive_zero                      0.01229  0.00924              NaN               NaN            NaN
always_up                           NaN      NaN          0.49701               NaN            NaN
arima                           0.01247  0.00947          0.50299          -2.31278        0.02073
arimax_vix_change               0.01246  0.00943          0.48303          -2.50756        0.01216
arimax_vix_change+dgs10_change  0.01247  0.00945          0.49900          -2.41172        0.01588
```

RMSE by year:
```
date                                2022      2023
naive_zero                      0.015272  0.008266
arima                           0.015524  0.008334
arimax_vix_change               0.015522  0.008314
arimax_vix_change+dgs10_change  0.015543  0.008301
```

```
{
  "train_end": "2021-12-31",
  "validation_end": "2023-12-31",
  "arma_order": [
    2,
    0,
    0
  ],
  "exog_selected_by_bic_on_train": "none",
  "best_vol_model_on_validation": "garch_t"
}
```
