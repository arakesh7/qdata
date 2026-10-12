from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch
import pandas as pd
import pytest
import requests

from qdata.catalog import CatalogManager
from qdata.providers import get_provider
from qdata.providers.upstox import UpstoxProvider, BASE_URL
from qdata.store import CANONICAL_COLUMNS, ManifestManager
from qdata.sync import SyncEngine


MOCK_CANDLES_RESPONSE = {
    "status": "success",
    "data": {
        "candles": [
            ["2024-01-05T00:00:00+05:30", 1301.45, 1309.9, 1299.0, 1305.0, 6200000, 0],
            ["2024-01-04T00:00:00+05:30", 1295.0, 1305.0, 1290.0, 1300.5, 5800000, 0],
            ["2024-01-03T00:00:00+05:30", 1290.0, 1298.0, 1285.0, 1292.0, 5100000, 0],
            ["2024-01-02T00:00:00+05:30", 1288.5, 1300.0, 1282.0, 1289.0, 4900000, 0],
            ["2024-01-01T00:00:00+05:30", 1280.0, 1290.0, 1275.0, 1285.0, 4500000, 0],
        ]
    },
}


def test_upstox_provider_registration():
    prov = get_provider("upstox")
    assert isinstance(prov, UpstoxProvider)
    assert prov.name == "upstox"


def test_upstox_format_instrument_key():
    prov = UpstoxProvider()

    # Direct instrument keys
    assert prov.format_instrument_key("NSE_EQ|INE002A01018") == "NSE_EQ|INE002A01018"
    assert prov.format_instrument_key("NSE_EQ:INE002A01018") == "NSE_EQ|INE002A01018"
    assert prov.format_instrument_key("NSE_EQ_INE002A01018") == "NSE_EQ|INE002A01018"
    assert prov.format_instrument_key("BSE_EQ_INE002A01018") == "BSE_EQ|INE002A01018"
    assert prov.format_instrument_key("BSE_INDEX|SENSEX") == "BSE_INDEX|SENSEX"

    # Index aliases
    assert prov.format_instrument_key("NIFTY") == "NSE_INDEX|Nifty 50"
    assert prov.format_instrument_key("NIFTY 50") == "NSE_INDEX|Nifty 50"
    assert prov.format_instrument_key("BANKNIFTY") == "NSE_INDEX|Nifty Bank"
    assert prov.format_instrument_key("SENSEX") == "BSE_INDEX|SENSEX"
    assert prov.format_instrument_key("BSE:SENSEX") == "BSE_INDEX|SENSEX"
    assert prov.format_instrument_key("BANKEX") == "BSE_INDEX|BANKEX"

    # Human-readable stock symbols with mock instruments table
    mock_instruments = pd.DataFrame([
        {"instrument_key": "NSE_EQ|INE002A01018", "tradingsymbol": "RELIANCE", "exchange": "NSE_EQ", "instrument_type": "EQUITY"},
        {"instrument_key": "BSE_EQ|INE002A01018", "tradingsymbol": "RELIANCE", "exchange": "BSE_EQ", "instrument_type": "EQUITY"},
        {"instrument_key": "NSE_EQ|INE467B01029", "tradingsymbol": "TCS", "exchange": "NSE_EQ", "instrument_type": "EQUITY"},
        {"instrument_key": "BSE_EQ|INE467B01029", "tradingsymbol": "TCS", "exchange": "BSE_EQ", "instrument_type": "EQUITY"},
    ])

    with patch.object(prov, "list_instruments", return_value=mock_instruments):
        # Default exchange is NSE
        assert prov.format_instrument_key("RELIANCE") == "NSE_EQ|INE002A01018"
        assert prov.format_instrument_key("TCS") == "NSE_EQ|INE467B01029"

        # Explicit BSE prefix
        assert prov.format_instrument_key("BSE:RELIANCE") == "BSE_EQ|INE002A01018"
        assert prov.format_instrument_key("BSE:TCS") == "BSE_EQ|INE467B01029"
        assert prov.format_instrument_key("RELIANCE", exchange="BSE") == "BSE_EQ|INE002A01018"


def test_upstox_date_math_and_leap_years():
    # Normal year
    d1 = date(2023, 5, 15)
    assert UpstoxProvider._add_years(d1, 1) == date(2024, 5, 15)

    # Leap day
    leap_day = date(2024, 2, 29)
    assert UpstoxProvider._add_years(leap_day, 1) == date(2025, 2, 28)

    # Date chunking
    prov = UpstoxProvider(max_window_years=1)
    chunks = list(prov._date_chunks(date(2022, 1, 1), date(2024, 6, 1)))
    assert len(chunks) == 3
    assert chunks[0] == (date(2022, 1, 1), date(2022, 12, 31))
    assert chunks[1] == (date(2023, 1, 1), date(2023, 12, 31))
    assert chunks[2] == (date(2024, 1, 1), date(2024, 6, 1))


def test_upstox_build_url():
    prov = UpstoxProvider()
    url = prov._build_url("NSE_EQ|INE002A01018", date(2024, 1, 1), date(2024, 1, 10))
    # Note: URL format is {BASE_URL}/{instrument}/day/{to_date}/{from_date}
    assert "NSE_EQ%7CINE002A01018/day/2024-01-10/2024-01-01" in url


def test_upstox_timeframe_restriction():
    prov = UpstoxProvider()
    # 1d succeeds (checked via mock)
    # Non-1d timeframes must raise ValueError
    with pytest.raises(ValueError, match="only supports daily data"):
        prov.fetch("NSE_EQ|INE002A01018", timeframe="1min")

    with pytest.raises(ValueError, match="only supports daily data"):
        prov.fetch("NSE_EQ|INE002A01018", timeframe="5min")

    with pytest.raises(ValueError, match="only supports daily data"):
        prov.latest_available("NSE_EQ|INE002A01018", timeframe="1min")


def test_upstox_unauthenticated_flow(tmp_data_dir: Path):
    prov = UpstoxProvider(data_dir=tmp_data_dir)
    # V2 requires no credentials
    token = prov._authenticate()
    assert token["access_token"] == "upstox_v2_public_access"
    assert token["api_version"] == "v2"
    assert token["provider"] == "upstox"

    # ensure_auth should succeed without error
    ensured = prov.ensure_auth()
    assert ensured["access_token"] == "upstox_v2_public_access"


def test_upstox_normalize_candles():
    prov = UpstoxProvider()
    candles = MOCK_CANDLES_RESPONSE["data"]["candles"]
    df = prov._normalize(candles, start=date(2024, 1, 2), end=date(2024, 1, 4))

    assert list(df.columns) == CANONICAL_COLUMNS
    assert len(df) == 3
    assert df["open"].dtype == "float64"
    assert df["high"].dtype == "float64"
    assert df["low"].dtype == "float64"
    assert df["close"].dtype == "float64"
    assert df["volume"].dtype == "float64"
    assert (df["provider"] == "upstox").all()
    # Ascending sort
    assert df["ts"].iloc[0] == pd.Timestamp("2024-01-02 00:00:00+00:00")
    assert df["ts"].iloc[-1] == pd.Timestamp("2024-01-04 00:00:00+00:00")


def test_upstox_request_candles_retry_on_429():
    prov = UpstoxProvider(retries=2, retry_delay=0.01)

    mock_resp_429 = MagicMock()
    mock_resp_429.status_code = 429
    mock_resp_429.headers = {"Retry-After": "0.01"}

    mock_resp_200 = MagicMock()
    mock_resp_200.status_code = 200
    mock_resp_200.json.return_value = MOCK_CANDLES_RESPONSE

    with patch.object(prov.session, "get", side_effect=[mock_resp_429, mock_resp_200]):
        candles = prov._request_candles("NSE_EQ|INE002A01018", date(2024, 1, 1), date(2024, 1, 5))
        assert len(candles) == 5


def test_upstox_request_candles_api_error():
    prov = UpstoxProvider(retries=1)

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {"status": "error", "errors": [{"message": "Invalid symbol"}]}

    with patch.object(prov.session, "get", return_value=mock_resp):
        with pytest.raises(RuntimeError, match="Upstox returned an API error"):
            prov._request_candles("NSE_EQ|INVALID", date(2024, 1, 1), date(2024, 1, 5))


def test_upstox_fetch_and_latest_available(tmp_data_dir: Path):
    prov = UpstoxProvider(data_dir=tmp_data_dir)

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = MOCK_CANDLES_RESPONSE

    with patch.object(prov.session, "get", return_value=mock_resp):
        # 1. Fetch
        start = datetime(2024, 1, 1, tzinfo=timezone.utc)
        end = datetime(2024, 1, 5, tzinfo=timezone.utc)
        df = prov.fetch("NSE_EQ|INE002A01018", timeframe="1d", start=start, end=end)
        assert len(df) == 5
        assert list(df.columns) == CANONICAL_COLUMNS

        # 2. Latest available
        latest = prov.latest_available("NSE_EQ|INE002A01018", timeframe="1d")
        assert latest is not None
        assert latest == datetime(2024, 1, 5, 0, 0, tzinfo=timezone.utc)


def test_upstox_health_check():
    prov = UpstoxProvider()

    mock_resp = MagicMock()
    mock_resp.status_code = 200

    with patch.object(prov.session, "get", return_value=mock_resp):
        h = prov.health_check()
        assert h["provider"] == "upstox"
        assert h["status"] == "ok"
        assert h["api_version"] == "v2"
        assert h["auth_required"] is False
        assert h["supported_timeframes"] == ["1d"]


def test_upstox_sync_engine_integration(tmp_data_dir: Path):
    engine = SyncEngine(tmp_data_dir)
    prov = UpstoxProvider(data_dir=tmp_data_dir)

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = MOCK_CANDLES_RESPONSE

    with patch.object(prov.session, "get", return_value=mock_resp):
        # Perform sync
        res = engine.sync_dataset(
            provider=prov,
            symbol="NSE_EQ_INE002A01018",
            timeframe="1d",
            start=datetime(2024, 1, 1, tzinfo=timezone.utc),
            end=datetime(2024, 1, 5, tzinfo=timezone.utc),
        )
        assert res["status"] == "success"
        assert res["synced_bars"] == 5

        # Check catalog
        cat = CatalogManager(tmp_data_dir)
        ds = cat.get_dataset("NSE_EQ_INE002A01018", "1d", layer="raw")
        assert ds is not None
        assert ds["provider"] == "upstox"
        assert ds["row_count"] == 5

        # Check partition file
        part_path = tmp_data_dir / "raw" / "NSE_EQ_INE002A01018" / "1d" / "2024.parquet"
        assert part_path.exists()
        part_df = pd.read_parquet(part_path)
        assert len(part_df) == 5
        assert list(part_df.columns) == CANONICAL_COLUMNS

        # Idempotency run
        res2 = engine.sync_dataset(
            provider=prov,
            symbol="NSE_EQ_INE002A01018",
            timeframe="1d",
            start=datetime(2024, 1, 1, tzinfo=timezone.utc),
            end=datetime(2024, 1, 5, tzinfo=timezone.utc),
        )
        assert res2["status"] == "up_to_date"
        assert res2["synced_bars"] == 0
