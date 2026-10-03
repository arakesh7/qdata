from datetime import datetime, timezone
from pathlib import Path
import pytest
import pandas as pd

from qdata.catalog import CatalogManager
from qdata.providers.mock import MockProvider
from qdata.store import (
    LockHeldError,
    ManifestManager,
    SyncLock,
    compute_sha256,
)
from qdata.sync import SyncEngine, plan_sync


def test_sync_idempotency_hash_snapshot(tmp_data_dir: Path):
    """
    THE CRITICAL TEST:
    Sync → hash snapshot → sync again → assert file hashes and catalog are completely unchanged.
    """
    engine = SyncEngine(tmp_data_dir)
    fixed_time = datetime(2026, 9, 25, 15, 30, tzinfo=timezone.utc)
    mock = MockProvider(data_dir=tmp_data_dir, mock_latest_time=fixed_time)

    # First sync
    res1 = engine.sync_dataset(mock, symbol="AAPL", timeframe="1min")
    assert res1["status"] == "success"
    assert res1["synced_bars"] > 0

    manifest = ManifestManager(tmp_data_dir)
    manifest_snap1 = manifest.load()
    assert len(manifest_snap1) > 0

    # Snapshot disk file hashes
    disk_hashes_snap1 = {}
    for rel_path in manifest_snap1.keys():
        abs_p = tmp_data_dir / rel_path
        assert abs_p.exists()
        disk_hashes_snap1[rel_path] = compute_sha256(abs_p)

    # Snapshot catalog
    catalog = CatalogManager(tmp_data_dir)
    cat_snap1 = catalog.get_dataset("AAPL", "1min", layer="raw")
    assert cat_snap1 is not None

    # SECOND SYNC (Idempotency run)
    res2 = engine.sync_dataset(mock, symbol="AAPL", timeframe="1min")
    assert res2["status"] == "up_to_date"
    assert res2["synced_bars"] == 0

    # Assert manifest is completely identical
    manifest_snap2 = manifest.load()
    assert manifest_snap1 == manifest_snap2

    # Assert disk file hashes are completely identical
    for rel_path, old_hash in disk_hashes_snap1.items():
        abs_p = tmp_data_dir / rel_path
        new_hash = compute_sha256(abs_p)
        assert new_hash == old_hash, f"Hash changed for {rel_path}!"

    # Assert catalog record is completely identical
    cat_snap2 = catalog.get_dataset("AAPL", "1min", layer="raw")
    for k in ["symbol", "timeframe", "layer", "start_ts", "end_ts", "row_count", "open_partition"]:
        assert cat_snap1[k] == cat_snap2[k], f"Catalog field {k} changed!"


def test_partition_freeze_on_month_boundary(tmp_data_dir: Path):
    """
    Situation: New month starts -> Old partition frozen forever; new file created.
    """
    engine = SyncEngine(tmp_data_dir)
    # Month 1: End of August 2026
    aug_time = datetime(2026, 8, 31, 23, 59, tzinfo=timezone.utc)
    mock = MockProvider(data_dir=tmp_data_dir, mock_latest_time=aug_time)

    res1 = engine.sync_dataset(mock, symbol="AAPL", timeframe="1min")
    assert res1["status"] == "success"
    aug_partition = "raw/AAPL/1min/2026/2026-08.parquet"
    assert res1["open_partition"] == aug_partition

    aug_file = tmp_data_dir / aug_partition
    assert aug_file.exists()
    aug_hash_initial = compute_sha256(aug_file)

    # Month 2: September 2026
    sep_time = datetime(2026, 9, 1, 10, 0, tzinfo=timezone.utc)
    mock.set_mock_latest(sep_time)

    res2 = engine.sync_dataset(mock, symbol="AAPL", timeframe="1min")
    assert res2["status"] == "success"
    sep_partition = "raw/AAPL/1min/2026/2026-09.parquet"
    assert res2["open_partition"] == sep_partition

    sep_file = tmp_data_dir / sep_partition
    assert sep_file.exists()

    # The August file MUST BE FROZEN FOREVER (hash unchanged)
    aug_hash_after = compute_sha256(aug_file)
    assert aug_hash_after == aug_hash_initial, "Frozen partition was unexpectedly modified!"

    # Catalog open_partition now points to September
    catalog = CatalogManager(tmp_data_dir)
    cat_ds = catalog.get_dataset("AAPL", "1min", layer="raw")
    assert cat_ds["open_partition"] == sep_partition


def test_sync_lock_prevents_concurrent_runs(tmp_data_dir: Path):
    """
    Test that holding data/.sync.lock raises LockHeldError.
    """
    engine = SyncEngine(tmp_data_dir)
    lock = SyncLock(tmp_data_dir / ".sync.lock")
    lock.acquire()

    try:
        # Another sync attempt while lock is held
        with pytest.raises(LockHeldError):
            engine.sync(provider_name="mock", symbols=["AAPL"], timeframe="1min")
    finally:
        lock.release()


def test_hash_mismatch_quarantine(tmp_data_dir: Path):
    """
    Situation: Hash mismatch vs manifest -> Quarantine, re-fetch, alert.
    """
    engine = SyncEngine(tmp_data_dir)
    fixed_time = datetime(2026, 9, 25, 12, 0, tzinfo=timezone.utc)
    mock = MockProvider(data_dir=tmp_data_dir, mock_latest_time=fixed_time)

    res1 = engine.sync_dataset(mock, symbol="MSFT", timeframe="1min")
    open_part = res1["open_partition"]
    open_file = tmp_data_dir / open_part

    # Tamper with file contents on disk
    with open(open_file, "ab") as f:
        f.write(b"CORRUPTED_BYTES")

    # Advance provider latest to trigger sync diff
    mock.set_mock_latest(datetime(2026, 9, 25, 15, 30, tzinfo=timezone.utc))

    # Next sync should detect hash mismatch, quarantine corrupted file, and succeed
    res2 = engine.sync_dataset(mock, symbol="MSFT", timeframe="1min")
    assert res2["status"] == "success"

    # Verify quarantine directory exists and contains the corrupted file
    quarantine_dir = tmp_data_dir / "quarantine"
    assert quarantine_dir.exists()
    quarantined_files = list(quarantine_dir.rglob("*.parquet"))
    assert len(quarantined_files) >= 1


def test_sync_daily_timeframe(tmp_data_dir: Path):
    """
    1d timeframe uses yearly partitioning: raw/<SYMBOL>/1d/<YYYY>.parquet
    """
    engine = SyncEngine(tmp_data_dir)
    fixed_time = datetime(2026, 9, 25, 0, 0, tzinfo=timezone.utc)
    mock = MockProvider(data_dir=tmp_data_dir, mock_latest_time=fixed_time)

    res = engine.sync_dataset(mock, symbol="RELIANCE", timeframe="1d")
    assert res["status"] == "success"
    expected_part = "raw/RELIANCE/1d/2026.parquet"
    assert res["open_partition"] == expected_part
    assert (tmp_data_dir / expected_part).exists()


def test_plan_sync_logic():
    latest = datetime(2026, 9, 25, 15, 30, tzinfo=timezone.utc)

    # Initial sync with explicit start and end
    p1 = plan_sync(None, None, req_start="2024-01-01", req_end="2024-12-31", latest_avail=latest)
    assert p1.fetch_start == datetime(2024, 1, 1, tzinfo=timezone.utc)
    assert p1.fetch_end == datetime(2024, 12, 31, tzinfo=timezone.utc)
    assert not p1.is_up_to_date

    # Backfill earlier than catalog start
    c_start = datetime(2025, 1, 1, tzinfo=timezone.utc)
    c_end = datetime(2026, 9, 25, 15, 30, tzinfo=timezone.utc)
    p2 = plan_sync(c_start, c_end, req_start="2024-01-01", req_end=None, latest_avail=latest)
    assert p2.fetch_start == datetime(2024, 1, 1, tzinfo=timezone.utc)
    assert not p2.is_up_to_date

    # Already covered range
    p3 = plan_sync(c_start, c_end, req_start="2025-06-01", req_end="2025-08-01", latest_avail=latest)
    assert p3.is_up_to_date


def test_sync_historical_backfill(tmp_data_dir: Path):
    """
    Test historical backfilling:
    1. First sync recent period (June 2025).
    2. Then backfill an earlier period (January 2025).
    3. Assert catalog start_ts expands, open_partition stays at June 2025.
    """
    engine = SyncEngine(tmp_data_dir)
    mock = MockProvider(data_dir=tmp_data_dir)

    # Sync June 2025
    res1 = engine.sync_dataset(
        mock,
        symbol="INFY",
        timeframe="1min",
        start="2025-06-01 10:00:00",
        end="2025-06-01 11:00:00",
    )
    assert res1["status"] == "success"
    assert res1["open_partition"] == "raw/INFY/1min/2025/2025-06.parquet"

    # Backfill January 2025
    res2 = engine.sync_dataset(
        mock,
        symbol="INFY",
        timeframe="1min",
        start="2025-01-01 10:00:00",
        end="2025-01-01 11:00:00",
    )
    assert res2["status"] == "success"
    # Backfilled file was created
    assert "raw/INFY/1min/2025/2025-01.parquet" in res2["written_partitions"]
    # Open partition should remain the June 2025 file
    assert res2["open_partition"] == "raw/INFY/1min/2025/2025-06.parquet"

    # Catalog start_ts must expand backward
    catalog = CatalogManager(tmp_data_dir)
    ds = catalog.get_dataset("INFY", "1min", layer="raw")
    assert pd.to_datetime(ds["start_ts"], utc=True) == pd.to_datetime("2025-01-01 10:00:00", utc=True)
    assert ds["open_partition"] == "raw/INFY/1min/2025/2025-06.parquet"
