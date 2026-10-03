from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from qdata.catalog import CatalogManager
from qdata.providers import get_provider
from qdata.providers.amfi import (
    AmfiProvider,
    build_scheme_name,
    split_into_date_pairs,
)
from qdata.store import CANONICAL_COLUMNS, ManifestManager
from qdata.sync import SyncEngine


MOCK_NAVALL_CONTENT = """Scheme Code;ISIN Div Payout/ ISIN Growth;ISIN Div Reinvestment;Scheme Name;Plan;Option;Net Asset Value;Date

Open Ended Schemes(Equity Scheme - Large Cap Fund)

Axis Mutual Fund

120503;INF846K01164;-;Axis Bluechip Fund;Direct Plan;Growth Option;55.4321;29-Sep-2026
120504;INF846K01172;-;Axis Bluechip Fund;Regular Plan;Growth Option;49.8765;29-Sep-2026

HDFC Mutual Fund

100033;INF179K01BE2;-;HDFC Top 100 Fund;Direct Plan;Growth Option;987.6543;28-Sep-2026
"""

MOCK_HISTORY_JSON_CHUNK = {
    "data": {
        "mf_name": "Axis Mutual Fund",
        "scheme_name": "Axis Bluechip Fund",
        "date_range": "From 01-Jan-2024 to 05-Jan-2024",
        "nav_groups": [
            {
                "nav_name": "Axis Bluechip Fund - Direct Plan - Growth Option",
                "historical_records": [
                    {"date": "2024-01-01", "nav": 50.10, "upload_time": "2024-01-01T22:00:00.000Z"},
                    {"date": "2024-01-02", "nav": 50.50, "upload_time": "2024-01-02T22:00:00.000Z"},
                    {"date": "2024-01-03", "nav": 50.25, "upload_time": "2024-01-03T22:00:00.000Z"},
                    {"date": "2024-01-04", "nav": 51.00, "upload_time": "2024-01-04T22:00:00.000Z"},
                    {"date": "2024-01-05", "nav": 51.20, "upload_time": "2024-01-05T22:00:00.000Z"},
                ],
            }
        ],
    }
}


def test_split_into_date_pairs():
    # Multi-year span chunked into <= 1800 days
    pairs = split_into_date_pairs("2010-01-01", "2020-01-01", n_days=1800)
    assert len(pairs) >= 2
    assert pairs[0][0] == "2010-01-01"
    assert pairs[-1][1] == "2020-01-01"

    # Single day
    pairs_single = split_into_date_pairs("2024-01-01", "2024-01-01", n_days=1800)
    assert pairs_single == [("2024-01-01", "2024-01-01")]

    # Inverted date range
    pairs_inv = split_into_date_pairs("2024-01-05", "2024-01-01", n_days=1800)
    assert pairs_inv == []


def test_build_scheme_name():
    name = build_scheme_name("Axis Bluechip Fund", "Direct Plan", "Growth Option")
    assert name == "Axis Bluechip Fund - Direct Plan - Growth Option"

    # With empty / hyphen options
    name2 = build_scheme_name("Axis Children's Fund", "-", "Growth Option")
    assert name2 == "Axis Children's Fund - Growth Option"


def test_amfi_provider_registered(tmp_data_dir: Path):
    provider = get_provider("amfi", data_dir=tmp_data_dir)
    assert isinstance(provider, AmfiProvider)
    assert provider.name == "amfi"


def test_amfi_provider_auth(tmp_data_dir: Path):
    provider = AmfiProvider(data_dir=tmp_data_dir)
    token = provider.ensure_auth()
    assert token["access_token"] == "amfi_public_access"
    assert token["provider"] == "amfi"
    assert token["expires_at"] > datetime.now(timezone.utc).timestamp()


def test_amfi_provider_schemes_parsing_and_search(tmp_data_dir: Path):
    cache_path = tmp_data_dir / "catalog" / "amfi_schemes.parquet"
    provider = AmfiProvider(data_dir=tmp_data_dir, schemes_cache_file=cache_path)

    with patch.object(provider, "_fetch_raw_nav_lines", return_value=MOCK_NAVALL_CONTENT.splitlines()):
        df = provider.list_all_schemes(force_refresh=True)

    assert len(df) == 3
    assert cache_path.exists()

    # Verify columns
    for col in ["scheme_code", "isin_growth", "scheme_name", "fund_house", "nav"]:
        assert col in df.columns

    # Test get_fund_houses
    houses = provider.get_fund_houses()
    assert "Axis Mutual Fund" in houses
    assert "HDFC Mutual Fund" in houses

    # Test get_schemes_by_fund_house
    axis_schemes = provider.get_schemes_by_fund_house("Axis Mutual Fund")
    assert len(axis_schemes) == 2
    assert "Axis Bluechip Fund - Direct Plan - Growth Option" in axis_schemes["scheme_name"].values

    # Test search_schemes
    res = provider.search_schemes("120503")
    assert len(res) == 1
    assert res.iloc[0]["scheme_code"] == "120503"

    res_isin = provider.search_schemes("INF179K01BE2")
    assert len(res_isin) == 1
    assert res_isin.iloc[0]["scheme_code"] == "100033"

    # Test resolve_scheme_code
    assert provider.resolve_scheme_code("120503") == "120503"
    assert provider.resolve_scheme_code("INF846K01164") == "120503"


def test_amfi_provider_latest_available(tmp_data_dir: Path):
    cache_path = tmp_data_dir / "catalog" / "amfi_schemes.parquet"
    provider = AmfiProvider(data_dir=tmp_data_dir, schemes_cache_file=cache_path)

    with patch.object(provider, "_fetch_raw_nav_lines", return_value=MOCK_NAVALL_CONTENT.splitlines()):
        provider.refresh_schemes()

    latest_dt = provider.latest_available("120503", timeframe="1d")
    assert latest_dt is not None
    assert latest_dt.year == 2026
    assert latest_dt.month == 9
    assert latest_dt.day == 29

    # Non-1d timeframe should return None
    assert provider.latest_available("120503", timeframe="1min") is None


def test_amfi_provider_fetch_canonical_bars(tmp_data_dir: Path):
    provider = AmfiProvider(data_dir=tmp_data_dir)

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = MOCK_HISTORY_JSON_CHUNK

    with patch.object(provider.session, "get", return_value=mock_resp):
        df = provider.fetch(
            symbol="120503",
            timeframe="1d",
            start=datetime(2024, 1, 1, tzinfo=timezone.utc),
            end=datetime(2024, 1, 5, tzinfo=timezone.utc),
        )

    assert len(df) == 5
    assert list(df.columns) == CANONICAL_COLUMNS

    # Verify OHLC matches NAV and volume is 0
    first_row = df.iloc[0]
    assert first_row["open"] == 50.10
    assert first_row["high"] == 50.10
    assert first_row["low"] == 50.10
    assert first_row["close"] == 50.10
    assert first_row["volume"] == 0.0
    assert first_row["provider"] == "amfi"
    assert first_row["ts"] == pd.Timestamp("2024-01-01 00:00:00+00:00")


def test_amfi_provider_sync_engine_integration(tmp_data_dir: Path):
    """
    Test end-to-end sync using SyncEngine with AMFI provider:
    - Verifies partition files written: raw/120503/1d/2024.parquet
    - Verifies manifest hash tracking
    - Verifies catalog datasets and instruments entries
    - Verifies idempotency
    """
    engine = SyncEngine(tmp_data_dir)
    provider = AmfiProvider(data_dir=tmp_data_dir)

    # Seed scheme info
    with patch.object(provider, "_fetch_raw_nav_lines", return_value=MOCK_NAVALL_CONTENT.splitlines()):
        provider.refresh_schemes()

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = MOCK_HISTORY_JSON_CHUNK

    with patch.object(provider.session, "get", return_value=mock_resp):
        # 1. First sync
        res1 = engine.sync_dataset(
            provider=provider,
            symbol="120503",
            timeframe="1d",
            start="2024-01-01",
            end="2024-01-05",
        )

    assert res1["status"] == "success"
    assert res1["synced_bars"] == 5

    # Check partition parquet file
    part_file = tmp_data_dir / "raw" / "120503" / "1d" / "2024.parquet"
    assert part_file.exists()

    # Check catalog
    catalog = CatalogManager(tmp_data_dir)
    inst = catalog.get_instrument("120503")
    assert inst is not None
    assert inst["exchange"] == "AMFI"
    assert inst["asset_class"] == "MUTUAL_FUND"
    assert inst["provider"] == "amfi"

    ds = catalog.get_dataset("120503", "1d", layer="raw")
    assert ds is not None
    assert ds["row_count"] == 5
    assert ds["provider"] == "amfi"

    # Check manifest
    manifest = ManifestManager(tmp_data_dir)
    manifest_files = manifest.load()
    assert "raw/120503/1d/2024.parquet" in manifest_files

    # 2. Second sync (Idempotent check)
    with patch.object(provider.session, "get", return_value=mock_resp):
        res2 = engine.sync_dataset(
            provider=provider,
            symbol="120503",
            timeframe="1d",
            start="2024-01-01",
            end="2024-01-05",
        )

    assert res2["status"] == "up_to_date"
    assert res2["synced_bars"] == 0


def test_provider_dynamic_cli_app():
    from qdata.providers.base import BaseProvider
    assert BaseProvider.get_cli_app() is None

    app = AmfiProvider.get_cli_app()
    assert app is not None
    assert app.info.name == "amfi"


def test_cli_amfi_dynamic_subcommands(tmp_data_dir: Path):
    from typer.testing import CliRunner
    from qdata.cli import app

    runner = CliRunner()
    result = runner.invoke(app, ["amfi", "--help"])
    assert result.exit_code == 0
    assert "search" in result.output
    assert "fund-houses" in result.output
    assert "schemes" in result.output


def test_amfi_retry_on_network_timeout(tmp_data_dir: Path):
    """Verify that AmfiProvider retries on transient ReadTimeout and succeeds."""
    provider = AmfiProvider(data_dir=tmp_data_dir, max_retries=3, backoff_factor=0.01, rate_limit_delay=0.0)

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = MOCK_HISTORY_JSON_CHUNK

    import requests

    with patch.object(
        provider.session,
        "get",
        side_effect=[
            requests.exceptions.ReadTimeout("Connection timed out"),
            mock_resp,
        ],
    ):
        records = provider._fetch_historical_records("120503", "2024-01-01", "2024-01-05")

    assert len(records) == 5
    assert records[0]["nav"] == 50.10


def test_amfi_retry_on_server_error_503(tmp_data_dir: Path):
    """Verify that AmfiProvider retries on HTTP 503 and succeeds."""
    provider = AmfiProvider(data_dir=tmp_data_dir, max_retries=3, backoff_factor=0.01, rate_limit_delay=0.0)

    err_resp = MagicMock()
    err_resp.status_code = 503

    ok_resp = MagicMock()
    ok_resp.status_code = 200
    ok_resp.json.return_value = MOCK_HISTORY_JSON_CHUNK

    with patch.object(provider.session, "get", side_effect=[err_resp, ok_resp]):
        records = provider._fetch_historical_records("120503", "2024-01-01", "2024-01-05")

    assert len(records) == 5


def test_amfi_adaptive_chunk_subdivision(tmp_data_dir: Path):
    """Verify adaptive subdivision when a large date span fails repeatedly."""
    provider = AmfiProvider(data_dir=tmp_data_dir, max_retries=2, backoff_factor=0.01, rate_limit_delay=0.0)

    import requests

    # When querying 2024-01-01 to 2024-06-01 (~150 days), fail.
    # When sub-divided, succeed.
    def mock_get(url, params=None, timeout=None):
        if params and params.get("from_date") == "2024-01-01" and params.get("to_date") == "2024-06-01":
            raise requests.exceptions.ReadTimeout("AMFI server timeout on large query")
        resp = MagicMock()
        resp.status_code = 200
        resp.json.return_value = MOCK_HISTORY_JSON_CHUNK
        return resp

    with patch.object(provider.session, "get", side_effect=mock_get):
        records = provider._fetch_historical_records("120503", "2024-01-01", "2024-06-01")

    # Both sub-chunks return 5 records each
    assert len(records) == 10


def test_amfi_schemes_cache_fallback_on_network_error(tmp_data_dir: Path):
    """Verify refresh_schemes falls back to disk cache if network fails."""
    cache_path = tmp_data_dir / "catalog" / "amfi_schemes.parquet"
    provider = AmfiProvider(data_dir=tmp_data_dir, schemes_cache_file=cache_path, max_retries=1, backoff_factor=0.01)

    # First seed the cache
    with patch.object(provider, "_fetch_raw_nav_lines", return_value=MOCK_NAVALL_CONTENT.splitlines()):
        df_initial = provider.refresh_schemes()
    assert len(df_initial) == 3

    import requests

    # Second refresh fails due to network error -> falls back to cached parquet
    with patch.object(
        provider.session,
        "get",
        side_effect=requests.exceptions.ConnectionError("Network is down"),
    ):
        df_fallback = provider.refresh_schemes()

    assert len(df_fallback) == 3


