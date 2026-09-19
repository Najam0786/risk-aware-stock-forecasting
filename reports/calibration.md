# Validation calibration (2022-2023, primary execution t+1, 5 bps costs)

Buy-and-hold on validation: Sharpe 0.178, max drawdown -0.245, cumulative 0.032
Feasibility: max drawdown no worse than buy-and-hold and at least 10 trades.

## Strategy A: score rule (theta_buy = theta_sell = theta)
```
    theta v_max_label   v_max  cumulative_return  annual_return  sharpe  max_drawdown  hit_ratio  exposure  trades  sharpe_same_close  buy_signals
0    0.05        none     inf             0.1174         0.0574  0.4338       -0.1344     0.5055    0.7265     197             0.1457          251
1    0.10        none     inf            -0.0079        -0.0040  0.0597       -0.2132     0.5000    0.8064     113            -0.0531          174
2    0.15        none     inf             0.0005         0.0002  0.0918       -0.2727     0.5181    0.8283      39            -0.2819           96
3    0.20        none     inf             0.0638         0.0316  0.2599       -0.2485     0.5100    0.9002       5             0.1567           44
4    0.05         P75  0.0109            -0.0113        -0.0057 -0.0205       -0.1248     0.5248    0.4032      73            -0.4153          128
5    0.10         P75  0.0109            -0.1091        -0.0565 -0.5694       -0.2269     0.5179    0.4471      37            -0.6382           82
6    0.15         P75  0.0109            -0.0302        -0.0153 -0.0286       -0.2416     0.5307    0.6168      13            -0.3429           36
7    0.20         P75  0.0109             0.0071         0.0036  0.1102       -0.2387     0.5098    0.8144       5            -0.0152           18
8    0.05         P90  0.0149             0.0633         0.0314  0.3080       -0.1488     0.5151    0.5968     147             0.0178          198
9    0.10         P90  0.0149            -0.0323        -0.0164 -0.0550       -0.2127     0.5154    0.6467      79            -0.3726          131
10   0.15         P90  0.0149            -0.0206        -0.0104  0.0210       -0.2849     0.5197    0.7605      31            -0.3363           70
11   0.20         P90  0.0149             0.0638         0.0316  0.2599       -0.2485     0.5100    0.9002       5             0.1567           32
```

Selected: theta=0.05, v_max=none, feasible=True

## Strategy B: volatility filter
```
  v_max_label   v_max  cumulative_return  annual_return  sharpe  max_drawdown  hit_ratio  exposure  trades  sharpe_same_close
0         P50  0.0077             0.0203         0.0102  0.2382       -0.0743     0.5281    0.1776      35            -0.0178
1         P60  0.0087             0.0190         0.0095  0.1915       -0.1010     0.5468    0.2774      25             0.5562
2         P70  0.0101             0.0198         0.0099  0.1641       -0.1144     0.5279    0.3932      49             0.3725
3         P75  0.0109             0.0072         0.0036  0.0849       -0.1558     0.5238    0.4611      45             0.2415
4         P80  0.0118            -0.0466        -0.0237 -0.1581       -0.2128     0.5035    0.5669      45             0.2947
5         P90  0.0149            -0.0419        -0.0213 -0.0677       -0.2639     0.5065    0.7645      41             0.1282
6         P95  0.0183             0.0805         0.0397  0.3064       -0.2478     0.5078    0.9002      25             0.4887
```

Selected: v_max=P50, feasible=True
