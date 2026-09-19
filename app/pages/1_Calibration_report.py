from __future__ import annotations

from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
REPORTS = ROOT / "reports"
EXTENSION_TICKERS = ("AAPL", "MSFT", "JPM")

st.set_page_config(page_title="RiskLens: calibration report", layout="wide")
st.title("Calibration report")
st.caption(
    "Evidence behind the dashboard: validation, the sealed test, and the exploratory analysis. "
    "Academic prototype, not financial advice."
)

tab_test, tab_other, tab_val, tab_calib, tab_eda = st.tabs(
    [
        "Sealed test (SPY)",
        "Other assets",
        "Model validation (SPY)",
        "Threshold calibration (SPY)",
        "Exploratory analysis",
    ]
)


def show(path: Path) -> None:
    st.markdown(path.read_text(encoding="utf-8"))


with tab_test:
    show(REPORTS / "test_results.md")
with tab_other:
    st.markdown(
        "AAPL, MSFT and JPM followed the same protocol as SPY, pre-registered together "
        "(tag `preregistered-v2`) before their test windows were opened."
    )
    for ticker in EXTENSION_TICKERS:
        with st.expander(f"{ticker}: sealed test results", expanded=ticker == "AAPL"):
            show(REPORTS / f"test_results_{ticker}.md")
    with st.expander("Validation tables"):
        show(REPORTS / "extension_validation.md")
    with st.expander("Threshold calibration"):
        show(REPORTS / "extension_calibration.md")
with tab_val:
    show(REPORTS / "model_findings.md")
    with st.expander("Full validation tables"):
        show(REPORTS / "model_validation.md")
with tab_calib:
    show(REPORTS / "calibration.md")
with tab_eda:
    show(REPORTS / "eda_findings.md")
    figures = sorted((REPORTS / "figures").glob("*.png"))
    for figure in figures:
        st.image(str(figure), caption=figure.stem.replace("_", " "))
