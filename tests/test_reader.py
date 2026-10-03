from datetime import datetime, timezone
from pathlib import Path
import pandas as pd
import pyarrow as pa
import pytest

from qdata import DataReader
from qdata.providers.mock import MockProvider
from qdata.sync import SyncEngine


def test_reader_initialization(tmp_data_dir: Path):
    """Verify DataReader initialization and invalid directory handling."""
    reader = DataReader(tmp_data_dir)
    assert reader.data_dir == tmp_data_dir.resolve()
    assert reader.layer == "raw"
    assert "DataReader" in repr(reader)

    # Nonexistent path must raise FileNotFoundError
    with pytest.raises(FileNotFoundError):
        DataReader(tmp_data_dir / "nonexistent_subfolder")


def test_reader_load_1d_with_datetime_index(tmp_data_dir: Path):
    """Test loading daily bars with default DatetimeIndex."""
    engine = SyncEngine(tmp_data_dir)
    fixed_time = datetime(2026, 9, 25, 0, 0, tzinfo=timezone.utc)
    mock = MockProvider(data_dir=tmp_data_dir, mock_latest_time=fixed_time)

    # Sync 30 days of daily data
    res = engine.sync_dataset(mock, symbol="RELIANCE", timeframe="1d")
    assert res["status"] == "success"

    # Read using DataReader
    reader = DataReader(tmp_data_dir)
    df = reader.load("RELIANCE", timeframe="1d")

    assert not df.empty
    assert isinstance(df.index, pd.DatetimeIndex)
    assert df.index.name == "ts"
    assert str(df.index.tz) == "UTC"
    assert df.index.is_monotonic_increasing
    assert set(["open", "high", "low", "close", "volume"]).issubset(df.columns)


def test_reader_date_filtering(tmp_data_dir: Path):
    """Test date filtering with start and end bounds."""
    engine = SyncEngine(tmp_data_dir)
    fixed_time = datetime(2026, 9, 25, 15, 30, tzinfo=timezone.utc)
    mock = MockProvider(data_dir=tmp_data_dir, mock_latest_time=fixed_time)

    engine.sync_dataset(mock, symbol="TCS", timeframe="1min")

    reader = DataReader(tmp_data_dir)
    start_filter = "2026-09-25 12:00:00"
    end_filter = "2026-09-25 13:00:00"

    df = reader.load(
        "TCS",
        timeframe="1min",
        start=start_filter,
        end=end_filter,
    )

    assert not df.empty
    assert df.index.min() >= pd.to_datetime(start_filter, utc=True)
    assert df.index.max() <= pd.to_datetime(end_filter, utc=True)


def test_reader_multi_partition_span(tmp_data_dir: Path):
    """Verify that reader correctly reads across multiple monthly partitions."""
    engine = SyncEngine(tmp_data_dir)
    # Sync spanning August and September
    aug_time = datetime(2026, 8, 30, 15, 30, tzinfo=timezone.utc)
    mock_aug = MockProvider(data_dir=tmp_data_dir, mock_latest_time=aug_time)
    engine.sync_dataset(mock_aug, symbol="INFY", timeframe="1min")

    sep_time = datetime(2026, 9, 5, 15, 30, tzinfo=timezone.utc)
    mock_sep = MockProvider(data_dir=tmp_data_dir, mock_latest_time=sep_time)
    engine.sync_dataset(mock_sep, symbol="INFY", timeframe="1min")

    reader = DataReader(tmp_data_dir)
    df = reader.load("INFY", timeframe="1min")

    assert not df.empty
    months = df.index.month.unique().tolist()
    assert 8 in months
    assert 9 in months
    assert df.index.is_monotonic_increasing


def test_reader_column_projection(tmp_data_dir: Path):
    """Verify that column projection works as expected."""
    engine = SyncEngine(tmp_data_dir)
    fixed_time = datetime(2026, 9, 25, 0, 0, tzinfo=timezone.utc)
    mock = MockProvider(data_dir=tmp_data_dir, mock_latest_time=fixed_time)

    engine.sync_dataset(mock, symbol="HDFCBANK", timeframe="1d")

    reader = DataReader(tmp_data_dir)
    cols = ["close", "volume"]
    df = reader.load("HDFCBANK", timeframe="1d", columns=cols)

    assert list(df.columns) == ["close", "volume"]
    assert df.index.name == "ts"
    assert isinstance(df.index, pd.DatetimeIndex)


def test_reader_set_index_false(tmp_data_dir: Path):
    """Verify set_index=False leaves 'ts' as a regular column."""
    engine = SyncEngine(tmp_data_dir)
    fixed_time = datetime(2026, 9, 25, 0, 0, tzinfo=timezone.utc)
    mock = MockProvider(data_dir=tmp_data_dir, mock_latest_time=fixed_time)

    engine.sync_dataset(mock, symbol="SBIN", timeframe="1d")

    reader = DataReader(tmp_data_dir)
    df = reader.load("SBIN", timeframe="1d", set_index=False)

    assert "ts" in df.columns
    assert isinstance(df.index, pd.RangeIndex)


def test_reader_as_arrow(tmp_data_dir: Path):
    """Verify as_arrow=True returns a pyarrow Table."""
    engine = SyncEngine(tmp_data_dir)
    fixed_time = datetime(2026, 9, 25, 0, 0, tzinfo=timezone.utc)
    mock = MockProvider(data_dir=tmp_data_dir, mock_latest_time=fixed_time)

    engine.sync_dataset(mock, symbol="ITC", timeframe="1d")

    reader = DataReader(tmp_data_dir)
    table = reader.load("ITC", timeframe="1d", as_arrow=True)

    assert isinstance(table, pa.Table)
    assert "ts" in table.column_names
    assert "close" in table.column_names
    assert table.num_rows > 0


def test_reader_discovery_helpers(tmp_data_dir: Path):
    """Verify list_symbols, list_timeframes, and dataset_info helpers."""
    engine = SyncEngine(tmp_data_dir)
    fixed_time = datetime(2026, 9, 25, 0, 0, tzinfo=timezone.utc)
    mock = MockProvider(data_dir=tmp_data_dir, mock_latest_time=fixed_time)

    engine.sync_dataset(mock, symbol="WIPRO", timeframe="1d")
    engine.sync_dataset(mock, symbol="WIPRO", timeframe="1min")

    reader = DataReader(tmp_data_dir)

    symbols = reader.list_symbols()
    assert "WIPRO" in symbols

    timeframes = reader.list_timeframes("WIPRO")
    assert "1d" in timeframes
    assert "1min" in timeframes

    info = reader.dataset_info("WIPRO", "1d")
    assert info is not None
    assert info["symbol"] == "WIPRO"
    assert info["row_count"] > 0


def test_reader_multi_instance_isolation(tmp_path: Path):
    """Verify two DataReader instances pointing to distinct stores do not collide."""
    dir_a = tmp_path / "store_a"
    dir_b = tmp_path / "store_b"
    dir_a.mkdir(parents=True)
    dir_b.mkdir(parents=True)

    fixed_time = datetime(2026, 9, 25, 0, 0, tzinfo=timezone.utc)
    engine_a = SyncEngine(dir_a)
    mock_a = MockProvider(data_dir=dir_a, mock_latest_time=fixed_time)
    engine_a.sync_dataset(mock_a, symbol="AAA", timeframe="1d")

    engine_b = SyncEngine(dir_b)
    mock_b = MockProvider(data_dir=dir_b, mock_latest_time=fixed_time)
    engine_b.sync_dataset(mock_b, symbol="BBB", timeframe="1d")

    reader_a = DataReader(dir_a)
    reader_b = DataReader(dir_b)

    assert "AAA" in reader_a.list_symbols()
    assert "BBB" not in reader_a.list_symbols()

    assert "BBB" in reader_b.list_symbols()
    assert "AAA" not in reader_b.list_symbols()

    df_a = reader_a.load("AAA", timeframe="1d")
    assert not df_a.empty

    df_b_empty = reader_a.load("BBB", timeframe="1d")
    assert df_b_empty.empty


def test_reader_timezone_intraday(tmp_data_dir: Path):
    """Verify timezone conversion and naive bound filtering for intraday bars."""
    engine = SyncEngine(tmp_data_dir)
    fixed_time = datetime(2026, 9, 25, 10, 0, tzinfo=timezone.utc)
    mock = MockProvider(data_dir=tmp_data_dir, mock_latest_time=fixed_time)
    engine.sync_dataset(mock, symbol="TCS", timeframe="1min")

    reader = DataReader(tmp_data_dir)

    # 1. Query with 'Asia/Kolkata'
    df_kolkata = reader.load("TCS", timeframe="1min", tz="Asia/Kolkata")
    assert not df_kolkata.empty
    assert str(df_kolkata.index.tz) == "Asia/Kolkata"

    # 2. Query with 'IST' alias (should resolve to Asia/Kolkata)
    df_ist = reader.load("TCS", timeframe="1min", tz="IST")
    assert str(df_ist.index.tz) == "Asia/Kolkata"
    assert len(df_ist) == len(df_kolkata)

    # 3. Naive filter bounds in IST should be properly translated to UTC on disk
    # 13:00 IST is 07:30 UTC
    df_filtered = reader.load(
        "TCS",
        timeframe="1min",
        start="2026-09-25 13:00:00",
        end="2026-09-25 14:00:00",
        tz="Asia/Kolkata",
    )
    assert not df_filtered.empty
    assert df_filtered.index.min() >= pd.to_datetime("2026-09-25 13:00:00", utc=False).tz_localize("Asia/Kolkata")
    assert df_filtered.index.max() <= pd.to_datetime("2026-09-25 14:00:00", utc=False).tz_localize("Asia/Kolkata")


def test_reader_timezone_daily_session_date_normalization(tmp_data_dir: Path):
    """Verify that daily bars with tz are anchored at midnight without 5:30 AM drift."""
    engine = SyncEngine(tmp_data_dir)
    fixed_time = datetime(2026, 9, 25, 0, 0, tzinfo=timezone.utc)
    mock = MockProvider(data_dir=tmp_data_dir, mock_latest_time=fixed_time)
    engine.sync_dataset(mock, symbol="RELIANCE", timeframe="1d")

    reader = DataReader(tmp_data_dir)

    # Default (tz=None) retains UTC DatetimeIndex
    df_utc = reader.load("RELIANCE", timeframe="1d")
    assert str(df_utc.index.tz) == "UTC"
    assert (df_utc.index.hour == 0).all()

    # With tz='Asia/Kolkata', session dates stay anchored at midnight
    df_ist = reader.load("RELIANCE", timeframe="1d", tz="Asia/Kolkata")
    assert str(df_ist.index.tz) == "Asia/Kolkata"
    assert (df_ist.index.hour == 0).all()  # No 5:30 AM drift!
    assert (df_ist.index.minute == 0).all()
    # Calendar dates should match exactly
    assert (df_utc.index.strftime("%Y-%m-%d") == df_ist.index.strftime("%Y-%m-%d")).all()


def test_reader_timezone_as_arrow(tmp_data_dir: Path):
    """Verify that as_arrow=True updates timestamp schema metadata zero-copy."""
    engine = SyncEngine(tmp_data_dir)
    fixed_time = datetime(2026, 9, 25, 0, 0, tzinfo=timezone.utc)
    mock = MockProvider(data_dir=tmp_data_dir, mock_latest_time=fixed_time)
    engine.sync_dataset(mock, symbol="HDFC", timeframe="1d")

    reader = DataReader(tmp_data_dir)
    table = reader.load("HDFC", timeframe="1d", as_arrow=True, tz="Asia/Kolkata")

    assert isinstance(table, pa.Table)
    ts_field = table.schema.field("ts")
    assert ts_field.type == pa.timestamp("us", tz="Asia/Kolkata")
    assert table.num_rows > 0


def test_reader_multi_symbol_dict(tmp_data_dir: Path):
    """Verify loading multiple symbols returns a dictionary of DataFrames."""
    engine = SyncEngine(tmp_data_dir)
    fixed_time = datetime(2026, 9, 25, 0, 0, tzinfo=timezone.utc)
    mock = MockProvider(data_dir=tmp_data_dir, mock_latest_time=fixed_time)

    engine.sync_dataset(mock, symbol="RELIANCE", timeframe="1d")
    engine.sync_dataset(mock, symbol="TCS", timeframe="1d")

    reader = DataReader(tmp_data_dir)
    res_dict = reader.load(["RELIANCE", "TCS"], timeframe="1d")

    assert isinstance(res_dict, dict)
    assert set(res_dict.keys()) == {"RELIANCE", "TCS"}
    assert not res_dict["RELIANCE"].empty
    assert not res_dict["TCS"].empty
    assert isinstance(res_dict["RELIANCE"].index, pd.DatetimeIndex)
    assert isinstance(res_dict["TCS"].index, pd.DatetimeIndex)


def test_reader_multi_symbol_as_arrow(tmp_data_dir: Path):
    """Verify loading multiple symbols as Arrow returns a dictionary of Tables."""
    engine = SyncEngine(tmp_data_dir)
    fixed_time = datetime(2026, 9, 25, 0, 0, tzinfo=timezone.utc)
    mock = MockProvider(data_dir=tmp_data_dir, mock_latest_time=fixed_time)

    engine.sync_dataset(mock, symbol="HDFC", timeframe="1d")
    engine.sync_dataset(mock, symbol="ICICI", timeframe="1d")

    reader = DataReader(tmp_data_dir)
    tables = reader.load(["HDFC", "ICICI"], timeframe="1d", as_arrow=True)

    assert isinstance(tables, dict)
    assert set(tables.keys()) == {"HDFC", "ICICI"}
    assert isinstance(tables["HDFC"], pa.Table)
    assert isinstance(tables["ICICI"], pa.Table)
    assert tables["HDFC"].num_rows > 0
    assert tables["ICICI"].num_rows > 0


def test_reader_multi_symbol_partial_missing(tmp_data_dir: Path):
    """Verify multi-symbol query handles nonexistent tickers gracefully."""
    engine = SyncEngine(tmp_data_dir)
    fixed_time = datetime(2026, 9, 25, 0, 0, tzinfo=timezone.utc)
    mock = MockProvider(data_dir=tmp_data_dir, mock_latest_time=fixed_time)

    engine.sync_dataset(mock, symbol="SBIN", timeframe="1d")

    reader = DataReader(tmp_data_dir)
    res = reader.load(["SBIN", "NONEXISTENT_TICKER"], timeframe="1d")

    assert isinstance(res, dict)
    assert not res["SBIN"].empty
    assert res["NONEXISTENT_TICKER"].empty
    assert isinstance(res["NONEXISTENT_TICKER"], pd.DataFrame)
