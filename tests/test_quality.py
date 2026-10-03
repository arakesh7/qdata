from datetime import datetime, timezone
import numpy as np
import pandas as pd
import pytest

from qdata.quality import QualityFailureError, validate_bars


def test_validate_clean_bars():
    ts_list = pd.date_range("2026-09-25 10:00:00", periods=5, freq="1min", tz="UTC")
    df = pd.DataFrame({
        "ts": ts_list,
        "open": [100.0, 101.0, 102.0, 103.0, 104.0],
        "high": [105.0, 106.0, 107.0, 108.0, 109.0],
        "low": [99.0, 100.0, 101.0, 102.0, 103.0],
        "close": [102.0, 103.0, 104.0, 105.0, 106.0],
        "volume": [1000.0, 1200.0, 1100.0, 1300.0, 1400.0],
        "provider": ["mock"] * 5,
        "ingested_at": [datetime.now(timezone.utc)] * 5,
    })

    cleaned_df, report = validate_bars(df, timeframe="1min")
    assert report.status == "PASSED"
    assert report.valid_bars == 5
    assert report.duplicates_dropped == 0
    assert report.ohlc_violations == 0
    assert len(cleaned_df) == 5


def test_validate_deduplication():
    ts = pd.to_datetime("2026-09-25 10:00:00", utc=True)
    t1 = datetime(2026, 9, 25, 10, 0, 0, tzinfo=timezone.utc)
    t2 = datetime(2026, 9, 25, 10, 1, 0, tzinfo=timezone.utc)
    df = pd.DataFrame({
        "ts": [ts, ts],
        "open": [100.0, 105.0],
        "high": [110.0, 115.0],
        "low": [95.0, 100.0],
        "close": [108.0, 112.0],
        "volume": [1000.0, 2000.0],
        "provider": ["mock", "mock"],
        "ingested_at": [t1, t2],  # Later ingested_at should supersede earlier
    })

    cleaned_df, report = validate_bars(df, timeframe="1min")
    assert report.duplicates_dropped == 1
    assert len(cleaned_df) == 1
    assert cleaned_df.iloc[0]["close"] == 112.0


def test_validate_ohlc_consistency_and_sanity():
    ts_list = pd.date_range("2026-09-25 10:00:00", periods=4, freq="1min", tz="UTC")
    df = pd.DataFrame({
        "ts": ts_list,
        "open": [100.0, 100.0, -10.0, 100.0],  # row 2: negative price
        "high": [110.0, 90.0, 110.0, 110.0],   # row 1: high < open (violation)
        "low": [95.0, 95.0, 95.0, 115.0],      # row 3: low > high (violation)
        "close": [105.0, 105.0, 105.0, 105.0],
        "volume": [1000.0, 1000.0, 1000.0, 1000.0],
        "provider": ["mock"] * 4,
        "ingested_at": [datetime.now(timezone.utc)] * 4,
    })

    cleaned_df, report = validate_bars(df, timeframe="1min")
    assert report.status == "WARN"
    assert report.ohlc_violations == 2
    assert report.insane_prices == 1
    # Only row 0 was completely valid
    assert len(cleaned_df) == 1


def test_validate_strict_mode_raises():
    ts_list = pd.date_range("2026-09-25 10:00:00", periods=2, freq="1min", tz="UTC")
    # All rows invalid
    df = pd.DataFrame({
        "ts": ts_list,
        "open": [-1.0, -2.0],
        "high": [10.0, 20.0],
        "low": [5.0, 10.0],
        "close": [8.0, 15.0],
        "volume": [100.0, 100.0],
        "provider": ["mock", "mock"],
        "ingested_at": [datetime.now(timezone.utc)] * 2,
    })

    with pytest.raises(QualityFailureError):
        validate_bars(df, timeframe="1min", strict=True)
