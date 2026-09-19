# Data quality report

Rows failing a validity rule are counted here, never dropped silently.

## Raw files (frozen vintage)
```
           file  rows      first       last  duplicate_dates  missing_values  sha256_matches_manifest
 prices_SPY.csv  4203 2010-01-04 2026-09-18                0               0                     True
prices_AAPL.csv  4203 2010-01-04 2026-09-18                0               0                     True
prices_MSFT.csv  4203 2010-01-04 2026-09-18                0               0                     True
 prices_JPM.csv  4203 2010-01-04 2026-09-18                0               0                     True
        vix.csv  4205 2010-01-04 2026-09-18                0               0                     True
 fred_dgs10.csv  4359 2010-01-04 2026-09-17                0             179                     True
fred_t10y2y.csv  4360 2010-01-04 2026-09-18                0             179                     True
   fred_cpi.csv   200 2010-01-01 2026-08-01                0               1                     True
```

## Calendar and alignment
- all four tickers share one trading calendar: True
- VIX dates that are not trading days (dropped by the spine): ['2026-05-25', '2026-09-07']
- DGS10 blank days (bond-market holidays, forward-filled): 179
- latest DGS10 date vs latest price date: 2026-09-17 vs 2026-09-18

## Gold layer validity
- gold rows: 16728
- duplicate (date, ticker) keys: 0
- adj_close <= 0: 0
- non-finite log returns: 0
- nulls in mandatory columns: 0
- weekend dates: 0
- zero-volume days: 0
- cpi_yoy nulls (desirable field, first year has no year-over-year): 1048

## Extreme moves kept on purpose (real events, not errors)
```
ticker       date  log_return  vix_close
  AAPL 2013-01-24      -13.19  12.690000
  AAPL 2020-03-16      -13.77  82.690002
  AAPL 2025-04-09       14.26  33.619999
   JPM 2020-03-09      -14.56  54.459999
   JPM 2020-03-13       16.56  57.830002
   JPM 2020-03-16      -16.21  82.690002
  MSFT 2020-03-13       13.29  57.830002
  MSFT 2020-03-16      -15.95  82.690002
  MSFT 2026-07-30       14.42  17.090000
   SPY 2020-03-12      -10.06  75.470001
   SPY 2020-03-16      -11.59  82.690002
   SPY 2025-04-09        9.99  33.619999
```
