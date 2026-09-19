# EDA summary (train + validation only)

Data: `gold_market_daily`, 2010-02-03 to 2023-12-29. Sealed test period (2024+) excluded. Primary asset: SPY.

## Q1 Stationarity (ADF: H0 unit root; KPSS: H0 stationary)
```
ticker     series  adf_p  kpss_p
  AAPL  log_price 0.8535    0.01
  AAPL log_return 0.0000    0.10
   JPM  log_price 0.9049    0.01
   JPM log_return 0.0000    0.10
  MSFT  log_price 0.9866    0.01
  MSFT log_return 0.0000    0.10
   SPY  log_price 0.8787    0.01
   SPY log_return 0.0000    0.10
```

## Q2/Q3 Ljung-Box p-values, lag 10 (low p = autocorrelation)
```
ticker  returns_p_lag10  squared_p_lag10  abs_p_lag10
  AAPL              0.0              0.0          0.0
   JPM              0.0              0.0          0.0
  MSFT              0.0              0.0          0.0
   SPY              0.0              0.0          0.0
```

## Q4 Distribution
```
ticker    n  mean_bp  std_pct    skew  excess_kurtosis  jarque_bera_p  min_pct  max_pct
  AAPL 3501   9.9409   1.7740 -0.2524           5.5031            0.0 -13.7708  11.3157
   JPM 3501   5.1279   1.7625 -0.0889           9.5400            0.0 -16.2106  16.5620
  MSFT 3501   8.1623   1.6338 -0.1856           8.0724            0.0 -15.9454  13.2929
   SPY 3501   4.9213   1.0951 -0.7196          11.5537            0.0 -11.5887   8.6731
```

Ten largest |moves| for SPY (return in %):
```
      date  log_return  vix_close
2020-03-16    -11.5887      82.69
2020-03-12    -10.0569      75.47
2020-03-24      8.6731      61.67
2020-03-13      8.2028      57.83
2020-03-09     -8.1313      54.46
2011-08-08     -6.7340      48.00
2020-04-06      6.5007      45.24
2020-06-11     -5.9377      40.79
2020-03-26      5.6749      61.00
2022-11-10      5.3497      23.53
```

## Q5 Exogenous variables (SPY, feature on day t vs target on day t+1)
```
     feature              target  pearson_r      p
  vix_change     next_day_return     0.1109 0.0000
  vix_change next_day_abs_return     0.0461 0.0064
dgs10_change     next_day_return     0.0020 0.9078
dgs10_change next_day_abs_return    -0.0792 0.0000
      t10y2y     next_day_return     0.0009 0.9563
      t10y2y next_day_abs_return    -0.0745 0.0000
 roll_vol_21     next_day_return     0.0181 0.2854
 roll_vol_21 next_day_abs_return     0.4552 0.0000
```

## Q6 Regimes
```
        regime  days  ann_return_pct  ann_vol_pct  worst_day_pct  best_day_pct
2010-2019 calm  2495         12.8290      14.7007        -6.7340        4.9290
    2020 COVID   253         16.7656      33.6477       -11.5887        8.6731
 2021 recovery   252         25.2537      12.9940        -2.4744        2.3951
     2022 bear   251        -20.1391      24.2592        -4.4456        5.3497
 2023 recovery   250         23.4366      13.0650        -2.0265        2.2673
```

## Q7 Leverage effect (SPY)
```
corr_r_t_vs_abs_r_t1: -0.11921
corr_r_t_vs_sq_r_t1: -0.08760
mean_abs_next_after_down_pct: 0.82948
mean_abs_next_after_up_pct: 0.64346
mannwhitney_p: 0.00000
```

## Q8 Weekday effects (SPY)
```
     mean_bp  std_pct    n
Mon   2.3899   1.1769  653
Tue   8.1066   1.0438  718
Wed   5.6719   1.0610  719
Thu   3.9864   1.1467  708
Fri   4.1932   1.0496  703
```

Kruskal-Wallis p (returns): 0.9657 | (absolute returns): 0.5955

## Extremes clustering (top 1% |return| days, share by year)
```
date
2020    0.472
2011    0.194
2022    0.139
```
