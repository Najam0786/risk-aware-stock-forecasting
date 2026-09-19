# Extension assets: threshold calibration (train 2010-2021, validation 2022-2023; test sealed)
## AAPL
Buy-and-hold on validation: Sharpe 0.304, max drawdown -0.309, cumulative 0.096
Feasible: max drawdown no worse than buy-and-hold and at least 10 trades.
Strategy A (score rule):
```
    theta v_max_label   v_max  cumulative_return  annual_return  sharpe  max_drawdown  hit_ratio  exposure  trades  sharpe_same_close  buy_signals
0    0.05        none     inf             0.0693         0.0343  0.2608       -0.3095      0.514     0.998       1             0.3036          367
1    0.10        none     inf             0.0000         0.0000     NaN        0.0000        NaN     0.000       0                NaN            0
2    0.15        none     inf             0.0000         0.0000     NaN        0.0000        NaN     0.000       0                NaN            0
3    0.20        none     inf             0.0000         0.0000     NaN        0.0000        NaN     0.000       0                NaN            0
4    0.05         P75  0.0188             0.0693         0.0343  0.2608       -0.3095      0.514     0.998       1             0.3036          327
5    0.10         P75  0.0188             0.0000         0.0000     NaN        0.0000        NaN     0.000       0                NaN            0
6    0.15         P75  0.0188             0.0000         0.0000     NaN        0.0000        NaN     0.000       0                NaN            0
7    0.20         P75  0.0188             0.0000         0.0000     NaN        0.0000        NaN     0.000       0                NaN            0
8    0.05         P90  0.0236             0.0693         0.0343  0.2608       -0.3095      0.514     0.998       1             0.3036          367
9    0.10         P90  0.0236             0.0000         0.0000     NaN        0.0000        NaN     0.000       0                NaN            0
10   0.15         P90  0.0236             0.0000         0.0000     NaN        0.0000        NaN     0.000       0                NaN            0
11   0.20         P90  0.0236             0.0000         0.0000     NaN        0.0000        NaN     0.000       0                NaN            0
```

Selected: theta=0.05, v_max=none, feasible=False
Strategy B (volatility filter):
```
  v_max_label   v_max  cumulative_return  annual_return  sharpe  max_drawdown  hit_ratio  exposure  trades  sharpe_same_close
0         P50  0.0156             0.2689         0.1272  0.9796       -0.1780     0.5260    0.3453      25             1.2552
1         P60  0.0166             0.2955         0.1391  0.9019       -0.1652     0.5391    0.4591      35             1.3873
2         P70  0.0180             0.1174         0.0574  0.4003       -0.2437     0.5017    0.5888      41             1.0900
3         P75  0.0188             0.0206         0.0103  0.1509       -0.3038     0.4985    0.6527      33             0.5624
4         P80  0.0200            -0.0838        -0.0431 -0.1085       -0.3571     0.5028    0.7146      23             0.4365
5         P90  0.0236             0.1107         0.0542  0.3360       -0.2894     0.5164    0.8543      27             0.2431
6         P95  0.0268             0.2173         0.1040  0.5005       -0.2637     0.5161    0.9321      15             0.3986
```

Selected: v_max=P50, feasible=True

## MSFT
Buy-and-hold on validation: Sharpe 0.365, max drawdown -0.359, cumulative 0.138
Feasible: max drawdown no worse than buy-and-hold and at least 10 trades.
Strategy A (score rule):
```
    theta v_max_label   v_max  cumulative_return  annual_return  sharpe  max_drawdown  hit_ratio  exposure  trades  sharpe_same_close  buy_signals
0    0.05        none     inf             0.2848         0.1344  0.6276       -0.2537     0.5183    0.7086     157             0.9330          257
1    0.10        none     inf             0.3600         0.1673  0.7298       -0.3628     0.5216    0.7385      77             0.7054          159
2    0.15        none     inf             0.5236         0.2359  0.9081       -0.2390     0.5299    0.7685      39             0.6458           95
3    0.20        none     inf             0.1661         0.0804  0.4123       -0.3295     0.5217    0.8263      11             0.1515           46
4    0.05         P75  0.0163            -0.0688        -0.0352 -0.1391       -0.2540     0.5090    0.3333      53             0.1343           93
5    0.10         P75  0.0163             0.1193         0.0583  0.4144       -0.2625     0.5562    0.3553      33            -0.0637           50
6    0.15         P75  0.0163             0.1878         0.0904  0.5591       -0.2164     0.5489    0.3673      19            -0.0471           27
7    0.20         P75  0.0163             0.1663         0.0805  0.5897       -0.1299     0.5775    0.3733       5             0.5188            9
8    0.05         P90  0.0203            -0.0689        -0.0353 -0.0608       -0.3126     0.5148    0.5389     115             0.3571          183
9    0.10         P90  0.0203             0.1233         0.0602  0.3742       -0.3305     0.5208    0.5749      49             0.2354          105
10   0.15         P90  0.0203             0.3259         0.1524  0.7263       -0.2272     0.5359    0.6108      27             0.4051           60
11   0.20         P90  0.0203            -0.0293        -0.0148  0.0731       -0.3517     0.5269    0.6667       9            -0.1747           24
```

Selected: theta=0.15, v_max=none, feasible=True
Strategy B (volatility filter):
```
  v_max_label   v_max  cumulative_return  annual_return  sharpe  max_drawdown  hit_ratio  exposure  trades  sharpe_same_close
0         P50  0.0135            -0.0522        -0.0266 -0.3705       -0.0778     0.5532    0.0938      19             0.3429
1         P60  0.0144             0.0050         0.0025  0.0770       -0.1038     0.5432    0.1617      29            -0.0133
2         P70  0.0156             0.0890         0.0438  0.3630       -0.1679     0.5396    0.2774      41            -0.0768
3         P75  0.0163            -0.0165        -0.0083  0.0264       -0.2688     0.5176    0.3393      45            -0.2779
4         P80  0.0172             0.0002         0.0001  0.0933       -0.2940     0.5280    0.4271      45            -0.2018
5         P90  0.0203            -0.1312        -0.0683 -0.1714       -0.4740     0.5072    0.6966      55            -0.0163
6         P95  0.0241             0.1286         0.0627  0.3577       -0.3460     0.5091    0.8782      31             0.2574
```

Selected: v_max=P70, feasible=True

## JPM
Buy-and-hold on validation: Sharpe 0.385, max drawdown -0.379, cumulative 0.141
Feasible: max drawdown no worse than buy-and-hold and at least 10 trades.
Strategy A (score rule):
```
    theta v_max_label   v_max  cumulative_return  annual_return  sharpe  max_drawdown  hit_ratio  exposure  trades  sharpe_same_close  buy_signals
0    0.05        none     inf            -0.1604        -0.0842 -0.2980       -0.3196     0.4970    0.6547     247            -0.1932          223
1    0.10        none     inf             0.1570         0.0761  0.4368       -0.2771     0.5347    0.7764     135             0.0959          146
2    0.15        none     inf            -0.0419        -0.0213  0.0207       -0.3288     0.5233    0.8124      63            -0.0077           87
3    0.20        none     inf            -0.1191        -0.0618 -0.1648       -0.3403     0.5054    0.7345      19             0.0467           34
4    0.05         P75  0.0172            -0.2464        -0.1326 -0.8038       -0.3665     0.5000    0.4631     165            -0.1423          149
5    0.10         P75  0.0172            -0.0162        -0.0082  0.0364       -0.2610     0.5531    0.5449      95             0.0681           89
6    0.15         P75  0.0172             0.0054         0.0027  0.1080       -0.3341     0.5455    0.6148      49             0.1163           56
7    0.20         P75  0.0172             0.0005         0.0003  0.1012       -0.3043     0.5273    0.6208      13             0.1781           20
8    0.05         P90  0.0235            -0.2711        -0.1471 -0.6774       -0.3873     0.4854    0.6168     235            -0.5369          210
9    0.10         P90  0.0235             0.0034         0.0017  0.1165       -0.3373     0.5286    0.7325     131            -0.1941          137
10   0.15         P90  0.0235            -0.0844        -0.0434 -0.0892       -0.3821     0.5249    0.7605      59            -0.0229           82
11   0.20         P90  0.0235            -0.1046        -0.0541 -0.1293       -0.3403     0.5068    0.7325      17             0.0548           31
```

Selected: theta=0.1, v_max=none, feasible=True
Strategy B (volatility filter):
```
  v_max_label   v_max  cumulative_return  annual_return  sharpe  max_drawdown  hit_ratio  exposure  trades  sharpe_same_close
0         P50  0.0140             0.1715         0.0829  0.6764       -0.1473     0.5729    0.3832      19             1.0714
1         P60  0.0150             0.1690         0.0817  0.6219       -0.2287     0.5671    0.4611      35             0.6525
2         P70  0.0163             0.0793         0.0391  0.3101       -0.2790     0.5548    0.5649      33             0.3033
3         P75  0.0172             0.0817         0.0403  0.3109       -0.2834     0.5517    0.6367      33             0.3918
4         P80  0.0184            -0.0870        -0.0448 -0.1209       -0.4096     0.5299    0.7345      35             0.1237
5         P90  0.0235            -0.0134        -0.0067  0.0927       -0.3810     0.5195    0.9222      21             0.2144
6         P95  0.0308             0.1170         0.0572  0.3446       -0.3793     0.5200    0.9980       1             0.3854
```

Selected: v_max=P50, feasible=True
