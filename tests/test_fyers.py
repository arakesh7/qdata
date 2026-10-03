from datetime import datetime, timezone, timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch
import pandas as pd
import pytest

from qdata.auth import save_cached_token, save_credential_to_file
from qdata.providers import get_provider
from qdata.providers.fyers import FyersProvider
from qdata.store import CANONICAL_COLUMNS
from qdata.sync import SyncEngine


def test_fyers_provider_registration():
    prov = get_provider("fyers")
    assert isinstance(prov, FyersProvider)
    assert prov.name == "fyers"


def test_fyers_format_symbol():
    prov = FyersProvider()
    assert prov.format_symbol("RELIANCE") == "NSE:RELIANCE-EQ"
    assert prov.format_symbol("TCS", exchange="BSE") == "BSE:TCS-EQ"
    assert prov.format_symbol("NSE:INFY-EQ") == "NSE:INFY-EQ"
    assert prov.format_symbol("MCX:CRUDEOIL-FUT") == "MCX:CRUDEOIL-FUT"


def test_fyers_map_resolution():
    prov = FyersProvider()
    assert prov.map_resolution("1min") == "1"
    assert prov.map_resolution("1m") == "1"
    assert prov.map_resolution("1d") == "D"
    assert prov.map_resolution("daily") == "D"
    assert prov.map_resolution("5min") == "5"
    assert prov.map_resolution("15min") == "15"


def test_fyers_split_into_chunks():
    prov = FyersProvider()
    start = datetime(2025, 1, 1, tzinfo=timezone.utc)
    end = datetime(2025, 8, 1, tzinfo=timezone.utc)  # ~212 days
    chunks = prov.split_into_chunks(start, end, max_days=99)

    assert len(chunks) == 3
    assert chunks[0] == ("2025-01-01", "2025-04-10")
    assert chunks[1] == ("2025-04-11", "2025-07-19")
    assert chunks[2] == ("2025-07-20", "2025-08-01")


def test_fyers_health_check_states(tmp_data_dir: Path):
    prov = FyersProvider(data_dir=tmp_data_dir)

    # Missing credentials initially
    with patch("qdata.providers.fyers.get_credential", return_value=None):
        h1 = prov.health_check()
        assert h1["status"] == "missing_credentials"
        assert h1["has_client_id"] is False

    # Configured with client_id and secret_key
    prov2 = FyersProvider(
        data_dir=tmp_data_dir,
        client_id="TEST_CLIENT_ID",
        secret_key="TEST_SECRET_KEY",
    )
    h2 = prov2.health_check()
    assert h2["status"] == "configured"
    assert h2["has_client_id"] is True
    assert h2["has_secret_key"] is True


def test_fyers_fetch_canonical_data(tmp_data_dir: Path):
    prov = FyersProvider(
        data_dir=tmp_data_dir,
        client_id="TEST_CLIENT",
        secret_key="TEST_SECRET",
    )

    # Cache token so ensure_auth passes
    save_cached_token(
        tmp_data_dir / ".tokens",
        "fyers",
        {"access_token": "mock_token", "expires_at": datetime.now(timezone.utc).timestamp() + 3600},
    )

    # Mock FyersModel
    mock_client = MagicMock()
    # Mock candle response: [epoch_sec, open, high, low, close, volume]
    base_epoch = int(datetime(2026, 9, 25, 10, 0, tzinfo=timezone.utc).timestamp())
    mock_candles = [
        [base_epoch, 2500.0, 2510.0, 2495.0, 2505.0, 1000.0],
        [base_epoch + 60, 2505.0, 2515.0, 2500.0, 2510.0, 1200.0],
    ]
    mock_client.history.return_value = {
        "s": "ok",
        "candles": mock_candles,
    }

    with patch.object(prov, "_get_client", return_value=mock_client):
        start = datetime(2026, 9, 25, 10, 0, tzinfo=timezone.utc)
        end = datetime(2026, 9, 25, 10, 1, tzinfo=timezone.utc)
        df = prov.fetch("RELIANCE", "1min", start=start, end=end)

        assert len(df) == 2
        assert list(df.columns) == CANONICAL_COLUMNS
        assert df["provider"].iloc[0] == "fyers"
        assert df["close"].iloc[0] == 2505.0
        assert df["close"].iloc[1] == 2510.0
        assert df["ts"].iloc[0].tzinfo == timezone.utc


def test_fyers_sync_engine_integration(tmp_data_dir: Path):
    """Test full SyncEngine pipeline using FyersProvider."""
    engine = SyncEngine(tmp_data_dir)
    prov = FyersProvider(
        data_dir=tmp_data_dir,
        client_id="TEST_CLIENT",
        secret_key="TEST_SECRET",
    )

    save_cached_token(
        tmp_data_dir / ".tokens",
        "fyers",
        {"access_token": "mock_token", "expires_at": datetime.now(timezone.utc).timestamp() + 3600},
    )

    mock_client = MagicMock()
    base_epoch = int(datetime(2026, 9, 25, 9, 15, tzinfo=timezone.utc).timestamp())
    mock_candles = [
        [base_epoch + (i * 60), 100.0 + i, 105.0 + i, 99.0 + i, 102.0 + i, 500.0]
        for i in range(5)
    ]
    mock_client.history.return_value = {
        "s": "ok",
        "candles": mock_candles,
    }

    with patch.object(prov, "_get_client", return_value=mock_client):
        res = engine.sync_dataset(prov, symbol="TCS", timeframe="1min")
        assert res["status"] == "success"
        assert res["synced_bars"] == 5

        # Check partition file created
        open_part = res["open_partition"]
        assert (tmp_data_dir / open_part).exists()
