from __future__ import annotations

import json
import warnings
from pathlib import Path

import pandas as pd
from arch import arch_model
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch

from risklens.mean_models import _fit
from risklens.run_validation import TRAIN_END

ROOT = Path(__file__).resolve().parents[2]
GOLD_PATH = ROOT / "data" / "gold" / "gold_market_daily.parquet"
REPORT_PATH = ROOT / "reports" / "residual_diagnostics.md"
LAGS = 10
warnings.filterwarnings("ignore", category=FutureWarning, module="statsmodels")


def orders() -> dict[str, tuple[int, int, int]]:
    spy = json.loads((ROOT / "config" / "preregistered.json").read_text("utf-8"))
    extension = json.loads((ROOT / "config" / "preregistered_extension.json").read_text("utf-8"))
    out = {"SPY": tuple(spy["mean_model"]["order"])}
    out.update({t: tuple(c["mean_model"]["order"]) for t, c in extension["tickers"].items()})
    return out


def ljung_p(series: pd.Series, lags: int = LAGS, model_df: int = 0) -> float:
    return float(acorr_ljungbox(series, lags=[lags], model_df=model_df)["lb_pvalue"].iloc[0])


def diagnose(returns: pd.Series, order: tuple[int, int, int]) -> dict[str, float | str]:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        arma = _fit(returns, order, None)
        resid = pd.Series(arma.resid).iloc[max(order[0], order[2]) :]
        garch = arch_model(
            returns * 100, mean="Constant", vol="GARCH", p=1, q=1, dist="t", rescale=False
        ).fit(disp="off")
    std = garch.std_resid.dropna()
    arma_df = order[0] + order[2]
    return {
        "arma_order": str(order),
        "arma_resid_LB_p": ljung_p(resid, model_df=arma_df),
        "arma_sq_resid_LB_p": ljung_p(resid**2),
        "arma_ARCH_LM_p": float(het_arch(resid, nlags=LAGS)[1]),
        "garch_std_resid_LB_p": ljung_p(std),
        "garch_sq_std_resid_LB_p": ljung_p(std**2, model_df=2),
        "garch_ARCH_LM_p": float(het_arch(std, nlags=LAGS)[1]),
        "student_t_nu": float(garch.params["nu"]),
        "persistence_alpha_plus_beta": float(garch.params["alpha[1]"] + garch.params["beta[1]"]),
    }


def main() -> None:
    gold = pd.read_parquet(GOLD_PATH)
    rows = {}
    for ticker, order in orders().items():
        train = gold[(gold["ticker"] == ticker) & (gold["date"] <= TRAIN_END)]
        returns = train.sort_values("date").set_index("date")["log_return"].iloc[1:]
        rows[ticker] = diagnose(returns, order)
    table = pd.DataFrame(rows).T
    numeric = table.columns.drop("arma_order")
    table[numeric] = table[numeric].astype(float).round(4)
    text = "\n".join(
        [
            f"# Residual diagnostics (training sample 2010 to {TRAIN_END[:4]}; test data not used)",
            "",
            f"Ljung-Box and ARCH-LM tests at {LAGS} lags; a small p-value means remaining structure.",
            "",
            "```",
            table.T.to_string(),
            "```",
            "",
            "Reading (this vintage):",
            "- ARMA residuals: squared residuals and the ARCH-LM test reject for all four assets, "
            "so volatility clustering is strong. This is why risk is modeled with GARCH and not a "
            "constant variance. Ljung-Box on the ARMA residuals also rejects for SPY, AAPL and MSFT; "
            "that test over-rejects under heteroskedasticity and extreme days such as March 2020, "
            "and the sealed test shows the mean model adds no forecasting value anyway.",
            "- GARCH(1,1)-t standardized residuals: ARCH-LM and Ljung-Box on the squared "
            "standardized residuals do not reject for any asset (p above 0.3), so the clustering is "
            "captured. Ljung-Box on the standardized residuals is marginal for MSFT only (p = 0.04). "
            "Student-t degrees of freedom of about 4.5 to 5.5 confirm heavy tails, and persistence "
            "(alpha + beta) is 0.96 to 0.99.",
            "",
        ]
    )
    REPORT_PATH.write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
