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
