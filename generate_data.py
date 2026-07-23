#!/usr/bin/env python3
"""
One-off helper used to generate the deliverable data.csv (10,000 rows of
synthetic OHLCV data) with a fixed seed so the sample dataset itself is
reproducible. Not part of the run.py pipeline / not required at run time.
"""

import numpy as np
import pandas as pd

SEED = 42
N_ROWS = 10000

rng = np.random.default_rng(SEED)

# Simulate a plausible-looking price series via a small random walk on
# log-returns, then derive open/high/low/volume around each close.
daily_returns = rng.normal(loc=0.0002, scale=0.01, size=N_ROWS)
close = 100 * np.exp(np.cumsum(daily_returns))

open_ = np.empty(N_ROWS)
open_[0] = close[0] * (1 + rng.normal(0, 0.001))
open_[1:] = close[:-1] * (1 + rng.normal(0, 0.001, size=N_ROWS - 1))

intraday_range = np.abs(rng.normal(loc=0.004, scale=0.002, size=N_ROWS))
high = np.maximum(open_, close) * (1 + intraday_range)
low = np.minimum(open_, close) * (1 - intraday_range)

volume = rng.integers(low=1000, high=1_000_000, size=N_ROWS)

timestamps = pd.date_range("2020-01-01", periods=N_ROWS, freq="min")

df = pd.DataFrame({
    "timestamp": timestamps,
    "open": open_.round(4),
    "high": high.round(4),
    "low": low.round(4),
    "close": close.round(4),
    "volume": volume,
})

df.to_csv("data.csv", index=False)
print(f"Wrote {len(df)} rows to data.csv")
