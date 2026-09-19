# Extension assets: model validation (train 2010-2021, validation 2022-2023; test sealed)
## AAPL
ARMA order by BIC on training data: (0, 0, 0)
Volatility (QLIKE lower is better; coverage target 95%):
```
             qlike  mae_vol  rmse_var  coverage  kupiec_p  christoffersen_ind_p  cond_coverage_p  dm_vs_rolling_stat  dm_vs_rolling_p
rolling_21 -7.0757   0.0097    0.0006    0.9401    0.3244                0.8759           0.6079                 NaN              NaN
ewma_094   -7.1227   0.0097    0.0006    0.9501    0.9918                0.8079           0.9708              2.2129           0.0269
garch_t    -7.1241   0.0100    0.0006    0.9541    0.6702                0.9525           0.9117              1.1658           0.2437
```

Next-day return:
```
                          rmse     mae  directional_acc  dm_vs_naive_stat  dm_vs_naive_p
naive_zero             0.01829     NaN              NaN               NaN            NaN
arima                  0.01831  0.0136          0.51497          -0.51184        0.60876
always_up_directional      NaN     NaN          0.51497               NaN            NaN
```

ARMA order selection (BIC):
```
   p  q         aic         bic
0  0  0  11919.3108  11931.3228
1  1  0  11913.5160  11931.5341
2  0  1  11913.7586  11931.7767
3  1  1  11914.0066  11938.0308
4  2  0  11914.7463  11938.7704
5  0  2  11914.9921  11939.0162
6  1  2  11916.0002  11946.0304
7  2  1  11916.0010  11946.0311
8  2  2  11918.0062  11954.0424
```

## MSFT
ARMA order by BIC on training data: (1, 0, 0)
Volatility (QLIKE lower is better; coverage target 95%):
```
             qlike  mae_vol  rmse_var  coverage  kupiec_p  christoffersen_ind_p  cond_coverage_p  dm_vs_rolling_stat  dm_vs_rolling_p
rolling_21 -6.9113   0.0108    0.0007    0.9341    0.1194                0.8960           0.2949                 NaN              NaN
ewma_094   -6.9295   0.0107    0.0007    0.9381    0.2388                0.4397           0.3707              1.2021           0.2293
garch_t    -6.9008   0.0106    0.0007    0.9321    0.0811                0.3035           0.1286             -0.5442           0.5863
```

Next-day return:
```
                          rmse      mae  directional_acc  dm_vs_naive_stat  dm_vs_naive_p
naive_zero             0.01930      NaN              NaN               NaN            NaN
arima                  0.01944  0.01483          0.51098          -1.26439        0.20609
always_up_directional      NaN      NaN          0.50699               NaN            NaN
```

ARMA order selection (BIC):
```
   p  q         aic         bic
0  1  0  11201.0000  11219.0181
1  0  1  11203.4110  11221.4291
2  0  2  11202.1310  11226.1551
3  2  0  11202.4344  11226.4585
4  1  1  11202.5420  11226.5661
5  2  1  11197.9989  11228.0291
6  1  2  11204.1168  11234.1470
7  2  2  11198.7436  11234.7798
8  0  0  11253.5683  11265.5804
```

## JPM
ARMA order by BIC on training data: (2, 0, 2)
Volatility (QLIKE lower is better; coverage target 95%):
```
             qlike  mae_vol  rmse_var  coverage  kupiec_p  christoffersen_ind_p  cond_coverage_p  dm_vs_rolling_stat  dm_vs_rolling_p
rolling_21 -7.2944   0.0091    0.0005    0.9142    0.0008                0.6830           0.0033                 NaN              NaN
ewma_094   -7.3178   0.0090    0.0005    0.9421    0.4291                0.7998           0.7083              1.0946           0.2737
garch_t    -7.3186   0.0092    0.0005    0.9501    0.9918                0.8079           0.9708              0.6342           0.5259
```

Next-day return:
```
                          rmse      mae  directional_acc  dm_vs_naive_stat  dm_vs_naive_p
naive_zero             0.01617      NaN              NaN               NaN            NaN
arima                  0.01638  0.01205          0.51497          -1.67245        0.09444
always_up_directional      NaN      NaN          0.52096               NaN            NaN
```

ARMA order selection (BIC):
```
   p  q         aic         bic
0  2  2  11912.8436  11948.8798
1  2  0  11938.6095  11962.6336
2  0  2  11940.3855  11964.4096
3  1  1  11941.2114  11965.2355
4  1  0  11949.1973  11967.2154
5  1  2  11939.1882  11969.2184
6  2  1  11940.2007  11970.2308
7  0  1  11954.5509  11972.5690
8  0  0  11991.7704  12003.7825
```
