import json
from pathlib import Path
from typer.testing import CliRunner
import pytest

from qdata.cli import app
from qdata.store import SyncLock

runner = CliRunner()


def test_cli_init_dry_run_and_actual(tmp_data_dir: Path):
    # Dry run
    res_dry = runner.invoke(app, ["init", "--data-dir", str(tmp_data_dir), "--dry-run", "--json"])
    assert res_dry.exit_code == 0
    dry_json = json.loads(res_dry.stdout)
    assert dry_json["status"] == "dry_run"

    # Actual init
    res = runner.invoke(app, ["init", "--data-dir", str(tmp_data_dir), "--json"])
    assert res.exit_code == 0
    data = json.loads(res.stdout)
    assert data["status"] == "success"
    assert (tmp_data_dir / "catalog").exists()


def test_cli_sync_and_catalog_commands(tmp_data_dir: Path):
    # Init first
    runner.invoke(app, ["init", "--data-dir", str(tmp_data_dir)])

    # Sync AAPL
    sync_res = runner.invoke(
        app,
        ["sync", "--provider", "mock", "--symbol", "AAPL", "--timeframe", "1min", "--data-dir", str(tmp_data_dir), "--json"]
    )
    assert sync_res.exit_code == 0
    sync_data = json.loads(sync_res.stdout)
    assert len(sync_data) == 1
    assert sync_data[0]["symbol"] == "AAPL"
    assert sync_data[0]["status"] == "success"

    # Sync with explicit start and end
    sync_range_res = runner.invoke(
        app,
        ["sync", "--provider", "mock", "--symbol", "MSFT", "--timeframe", "1min", "--start", "2025-01-01", "--end", "2025-01-02", "--data-dir", str(tmp_data_dir), "--json"]
    )
    assert sync_range_res.exit_code == 0
    range_data = json.loads(sync_range_res.stdout)
    assert range_data[0]["symbol"] == "MSFT"
    assert range_data[0]["status"] == "success"

    # Catalog list
    cat_res = runner.invoke(app, ["catalog", "list", "--data-dir", str(tmp_data_dir), "--json"])
    assert cat_res.exit_code == 0
    cat_data = json.loads(cat_res.stdout)
    assert len(cat_data) >= 1
    assert cat_data[0]["symbol"] == "AAPL"

    # Catalog show
    show_res = runner.invoke(app, ["catalog", "show", "AAPL", "1min", "--data-dir", str(tmp_data_dir), "--json"])
    assert show_res.exit_code == 0
    show_data = json.loads(show_res.stdout)
    assert show_data["dataset"]["symbol"] == "AAPL"
    assert len(show_data["partitions"]) >= 1

    # Inspect
    inspect_res = runner.invoke(app, ["inspect", "AAPL", "1min", "--data-dir", str(tmp_data_dir), "--json"])
    assert inspect_res.exit_code == 0
    inspect_data = json.loads(inspect_res.stdout)
    assert inspect_data["symbol"] == "AAPL"
    assert "quality" in inspect_data

    # Verify
    verify_res = runner.invoke(app, ["verify", "--data-dir", str(tmp_data_dir), "--json"])
    assert verify_res.exit_code == 0
    verify_data = json.loads(verify_res.stdout)
    assert verify_data["status"] == "ok"
    assert verify_data["mismatches_count"] == 0

    # Rebuild adjusted
    rebuild_res = runner.invoke(
        app,
        ["rebuild", "adjusted", "--symbol", "AAPL", "--data-dir", str(tmp_data_dir), "--json"]
    )
    assert rebuild_res.exit_code == 0
    rebuild_data = json.loads(rebuild_res.stdout)
    assert rebuild_data["status"] == "success"
    assert len(rebuild_data["rebuilt_files"]) >= 1

    # Doctor
    doc_res = runner.invoke(app, ["doctor", "--data-dir", str(tmp_data_dir), "--json"])
    assert doc_res.exit_code == 0
    doc_data = json.loads(doc_res.stdout)
    assert doc_data["status"] == "PASS"


def test_cli_auth_commands(tmp_data_dir: Path):
    # Auth login mock
    login_res = runner.invoke(app, ["auth", "login", "mock", "--data-dir", str(tmp_data_dir), "--json"])
    assert login_res.exit_code == 0

    # Auth status
    stat_res = runner.invoke(app, ["auth", "status", "mock", "--data-dir", str(tmp_data_dir), "--json"])
    assert stat_res.exit_code == 0
    stat_data = json.loads(stat_res.stdout)
    assert stat_data["authenticated"] is True

    # Auth refresh
    ref_res = runner.invoke(app, ["auth", "refresh", "mock", "--data-dir", str(tmp_data_dir), "--json"])
    assert ref_res.exit_code == 0

    # Auth logout
    logout_res = runner.invoke(app, ["auth", "logout", "mock", "--data-dir", str(tmp_data_dir), "--json"])
    assert logout_res.exit_code == 0
    stat_res2 = runner.invoke(app, ["auth", "status", "mock", "--data-dir", str(tmp_data_dir), "--json"])
    stat_data2 = json.loads(stat_res2.stdout)
    assert stat_data2["authenticated"] is False


def test_cli_lock_held_exit_code_3(tmp_data_dir: Path):
    runner.invoke(app, ["init", "--data-dir", str(tmp_data_dir)])
    lock = SyncLock(tmp_data_dir / ".sync.lock")
    lock.acquire()

    try:
        res = runner.invoke(
            app,
            ["sync", "--provider", "mock", "--symbol", "AAPL", "--timeframe", "1min", "--data-dir", str(tmp_data_dir), "--json"]
        )
        assert res.exit_code == 3
        data = json.loads(res.stdout)
        assert data["error"] == "lock_held"
    finally:
        lock.release()


def test_cli_sync_comma_separated_symbols(tmp_data_dir: Path):
    runner.invoke(app, ["init", "--data-dir", str(tmp_data_dir)])
    sync_res = runner.invoke(
        app,
        ["sync", "--provider", "mock", "--symbol", "TCS,INFY", "--timeframe", "1d", "--data-dir", str(tmp_data_dir), "--json"]
    )
    assert sync_res.exit_code == 0
    sync_data = json.loads(sync_res.stdout)
    assert len(sync_data) == 2
    symbols = [r["symbol"] for r in sync_data]
    assert symbols == ["TCS", "INFY"]


def test_cli_sync_symbols_file(tmp_data_dir: Path, tmp_path: Path):
    runner.invoke(app, ["init", "--data-dir", str(tmp_data_dir)])
    csv_file = tmp_path / "tickers.csv"
    csv_file.write_text("# Top Indian Stocks\nsymbol\nRELIANCE\nHDFCBANK\n", encoding="utf-8")

    sync_res = runner.invoke(
        app,
        ["sync", "--provider", "mock", "--symbols-file", str(csv_file), "--timeframe", "1d", "--data-dir", str(tmp_data_dir), "--json"]
    )
    assert sync_res.exit_code == 0
    sync_data = json.loads(sync_res.stdout)
    assert len(sync_data) == 2
    symbols = [r["symbol"] for r in sync_data]
    assert symbols == ["RELIANCE", "HDFCBANK"]


def test_cli_symbol_parsing_methods(tmp_path: Path):
    from qdata.cli import (
        clean_symbol,
        read_symbols_from_file,
        parse_symbols_from_str,
        parse_symbols,
    )

    # 1. clean_symbol unit tests
    assert clean_symbol("  reliance  ") == "RELIANCE"
    assert clean_symbol('"TCS"') == "TCS"
    assert clean_symbol("'INFY'") == "INFY"
    assert clean_symbol("symbol") is None
    assert clean_symbol("TICKER") is None
    assert clean_symbol("   ") is None

    # 2. parse_symbols_from_str unit tests
    assert parse_symbols_from_str("AAPL, MSFT, , GOOG") == ["AAPL", "MSFT", "GOOG"]
    assert parse_symbols_from_str(["nifty", "banknifty"]) == ["NIFTY", "BANKNIFTY"]

    # 3. read_symbols_from_file unit tests
    f_path = tmp_path / "symbols_test.csv"
    f_path.write_text(
        "# Header comment\nsymbol\nRELIANCE\n  TCS  \n\"INFY\", \"HDFC\"\n\n",
        encoding="utf-8",
    )
    from_file = read_symbols_from_file(f_path)
    assert from_file == ["RELIANCE", "TCS", "INFY", "HDFC"]

    # 4. parse_symbols composition & deduplication
    combined = parse_symbols(symbol="TCS, WIPRO", symbols_file=f_path)
    # Order preserved, deduplicated: RELIANCE, TCS, INFY, HDFC, WIPRO
    assert combined == ["RELIANCE", "TCS", "INFY", "HDFC", "WIPRO"]
