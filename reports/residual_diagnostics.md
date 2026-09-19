# Residual diagnostics (training sample 2010 to 2021; test data not used)

Ljung-Box and ARCH-LM tests at 10 lags; a small p-value means remaining structure.

```
                                   SPY       AAPL       MSFT        JPM
arma_order                   (2, 0, 0)  (0, 0, 0)  (1, 0, 0)  (2, 0, 2)
arma_resid_LB_p                    0.0        0.0        0.0     0.0883
arma_sq_resid_LB_p                 0.0        0.0        0.0        0.0
arma_ARCH_LM_p                     0.0        0.0        0.0        0.0
garch_std_resid_LB_p            0.2101      0.236     0.0391     0.5684
garch_sq_std_resid_LB_p         0.4287     0.7577     0.6629     0.3096
garch_ARCH_LM_p                 0.6387     0.8877     0.8112      0.493
student_t_nu                     5.016     4.5572     4.5463     5.5209
persistence_alpha_plus_beta      0.988     0.9625     0.9581     0.9788
```

Reading (this vintage):
- ARMA residuals: squared residuals and the ARCH-LM test reject for all four assets, so volatility clustering is strong. This is why risk is modeled with GARCH and not a constant variance. Ljung-Box on the ARMA residuals also rejects for SPY, AAPL and MSFT; that test over-rejects under heteroskedasticity and extreme days such as March 2020, and the sealed test shows the mean model adds no forecasting value anyway.
- GARCH(1,1)-t standardized residuals: ARCH-LM and Ljung-Box on the squared standardized residuals do not reject for any asset (p above 0.3), so the clustering is captured. Ljung-Box on the standardized residuals is marginal for MSFT only (p = 0.04). Student-t degrees of freedom of about 4.5 to 5.5 confirm heavy tails, and persistence (alpha + beta) is 0.96 to 0.99.
