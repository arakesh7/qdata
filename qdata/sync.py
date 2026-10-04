from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import pandas as pd

from qdata.auth import AuthError
from qdata.catalog import CatalogManager
from qdata.config import get_settings
from qdata.providers import get_provider
from qdata.providers.base import BaseProvider
from qdata.quality import QualityFailureError, QualityReport, validate_bars
from qdata.store import (
    CANONICAL_COLUMNS,
    LockHeldError,
    ManifestManager,
    SyncLock,
    atomic_write_parquet,
    compute_sha256,
    get_partition_relpath,
    is_monthly_partition,
    normalize_timeframe,
    quarantine_file,
    read_partition_parquet,
)


@dataclass
class SyncPlan:
    """Calculated window for data fetching and idempotency decision."""
    fetch_start: Optional[datetime]
    fetch_end: datetime
    is_up_to_date: bool = False
    message: str = ""


def plan_sync(
    cat_start: Optional[Any],
    cat_end: Optional[Any],
    req_start: Optional[Union[datetime, str]],
    req_end: Optional[Union[datetime, str]],
    latest_avail: datetime,
) -> SyncPlan:
    """
    Compute the exact fetch window based on catalog state and user parameters:
    - Initial Ingestion (no catalog data): fetch [req_start -> req_end or latest_avail].
    - Forward Incremental (no req_start): fetch (cat_end -> req_end or latest_avail].
    - Historical Backfill (req_start < cat_start): fetch [req_start -> req_end or latest_avail].
    - Already Covered Range: return is_up_to_date=True.
    """
    start_dt = pd.to_datetime(req_start, utc=True).to_pydatetime() if req_start else None
    end_dt = pd.to_datetime(req_end, utc=True).to_pydatetime() if req_end else None

    target_end = end_dt if end_dt is not None else latest_avail
    c_start = pd.to_datetime(cat_start, utc=True).to_pydatetime() if cat_start is not None else None
    c_end = pd.to_datetime(cat_end, utc=True).to_pydatetime() if cat_end is not None else None

    # Case 1: Brand new dataset (no prior catalog data)
    if c_end is None:
        return SyncPlan(fetch_start=start_dt, fetch_end=target_end)

    # Case 2: Historical backfill requested (start_dt is earlier than existing cat_start)
    if start_dt is not None and c_start is not None and start_dt < c_start:
        return SyncPlan(fetch_start=start_dt, fetch_end=target_end)

    # Case 3: Explicit bounded range that is already completely covered locally
    if start_dt is not None and c_start is not None and start_dt >= c_start and target_end <= c_end:
        return SyncPlan(fetch_start=None, fetch_end=target_end, is_up_to_date=True, message="Already up to date.")

    # Case 4: Incremental forward sync (no start specified or start is already covered)
    if c_end >= target_end:
        return SyncPlan(fetch_start=None, fetch_end=target_end, is_up_to_date=True, message="Already up to date.")

    return SyncPlan(fetch_start=c_end, fetch_end=target_end)


class PartitionSyncHandler:
    """
    Manages partition file routing, atomic writes, manifest updates,
    and integrity verification with automatic quarantine.
    """

    def __init__(self, data_dir: Path, manifest: ManifestManager):
        self.data_dir = data_dir
        self.manifest = manifest

    def read_and_verify(self, part_rel_str: str) -> pd.DataFrame:
        """Read partition parquet file and verify hash against manifest. Quarantine if tampered."""
        part_path = self.data_dir / part_rel_str
        if not part_path.exists():
            return pd.DataFrame(columns=CANONICAL_COLUMNS)

        manifest_entries = self.manifest.load()
        if part_rel_str in manifest_entries:
            curr_sha = compute_sha256(part_path)
            expected_sha = manifest_entries[part_rel_str].get("sha256")
            if curr_sha != expected_sha:
                quarantine_dest = quarantine_file(self.data_dir, part_rel_str, "hash_mismatch")
                print(f"ALERT: Hash mismatch on {part_rel_str}. Quarantined to {quarantine_dest}.")
                return pd.DataFrame(columns=CANONICAL_COLUMNS)

        return read_partition_parquet(part_path)

    def merge_and_dedupe(self, existing_df: pd.DataFrame, new_df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
        """Merge existing and new bars, deduping by timestamp (newest ingested_at wins)."""
        if existing_df.empty:
            return new_df
        combined = pd.concat([existing_df, new_df], ignore_index=True)
        merged_df, _ = validate_bars(combined, timeframe=timeframe, strict=False)
        return merged_df

    def save_partition(self, part_rel_str: str, df: pd.DataFrame) -> int:
        """Write DataFrame atomically to disk and record into manifest."""
        part_path = self.data_dir / part_rel_str
        row_count = atomic_write_parquet(df, part_path)
        self.manifest.record_file(part_rel_str, row_count)
        return row_count

    def process_group(
        self,
        part_rel_str: str,
        part_df: pd.DataFrame,
        timeframe: str,
        open_part: Optional[str],
    ) -> Optional[str]:
        """
        Process a partition group according to the 4 storage rules:
        - Open partition: verify hash, merge, dedupe, rewrite.
        - Initial sync (no open partition): merge if exists, write.
        - Closed historical partition: frozen forever, never touch.
        - New partition (rollover or backfilled file): write directly.
        """
        part_path = self.data_dir / part_rel_str

        # Case A: Active open partition
        if open_part is not None and part_rel_str == open_part:
            existing_df = self.read_and_verify(part_rel_str)
            merged_df = self.merge_and_dedupe(existing_df, part_df, timeframe)
            self.save_partition(part_rel_str, merged_df)
            return part_rel_str

        # Case B: Initial sync without open partition
        if open_part is None:
            existing_df = (
                read_partition_parquet(part_path)
                if part_path.exists()
                else pd.DataFrame(columns=CANONICAL_COLUMNS)
            )
            merged_df = self.merge_and_dedupe(existing_df, part_df, timeframe)
            self.save_partition(part_rel_str, merged_df)
            return part_rel_str

        # Case C: Historical closed partition already on disk (frozen)
        if part_path.exists():
            return None

        # Case D: Brand new partition (month rollover or historical backfilled partition)
        self.save_partition(part_rel_str, part_df)
        return part_rel_str

    def write_all(
        self,
        symbol: str,
        timeframe: str,
        clean_df: pd.DataFrame,
        open_part: Optional[str],
    ) -> Tuple[List[str], Optional[str]]:
        """
        Route bars into monthly or yearly partitions and write them.
        Returns (written_files, new_open_partition).
        """
        df_routed = clean_df.copy()
        df_routed["_part_rel"] = df_routed["ts"].apply(
            lambda t: get_partition_relpath(symbol, timeframe, t, layer="raw").as_posix()
        )

        written_files: List[str] = []
        for part_rel_str, grp in df_routed.groupby("_part_rel", sort=True):
            part_df = grp.drop(columns=["_part_rel"]).copy()
            written = self.process_group(
                part_rel_str=part_rel_str,
                part_df=part_df,
                timeframe=timeframe,
                open_part=open_part,
            )
            if written:
                written_files.append(written)

        # Resolve open_partition:
        # Initial sync: latest written partition becomes open_partition.
        # Existing open_partition: only advance if a newer partition was written.
        if open_part is None:
            new_open_partition = max(written_files) if written_files else None
        else:
            new_open_partition = max([open_part] + written_files)

        return written_files, new_open_partition


def parse_symbols(
    symbols: Optional[Union[str, List[str]]] = None,
    symbols_file: Optional[Union[str, Path]] = None,
) -> Optional[List[str]]:
    """Parse symbol inputs using qdata.cli symbol parsing utilities."""
    from qdata.cli import parse_symbols as _parse
    return _parse(symbol=symbols, symbols_file=symbols_file)


class SyncEngine:
    """
    Universal sync engine implementing the 4 core design principles:
    1. Raw data is immutable — closed partitions are frozen.
    2. Catalog is the source of truth.
    3. Sync is idempotent — running twice produces the exact same state.
    4. Partition minimally — sync touches only open or newly created partitions.
    """

    def __init__(self, data_dir: Optional[Path] = None):
        self.settings = get_settings(override_data_dir=data_dir)
        self.data_dir = self.settings.data_dir
        self.catalog = CatalogManager(self.data_dir)
        self.manifest = ManifestManager(self.data_dir)
        self.lock = SyncLock(self.settings.lock_file)
        self.partition_handler = PartitionSyncHandler(self.data_dir, self.manifest)

    def sync_dataset(
        self,
        provider: BaseProvider,
        symbol: str,
        timeframe: str,
        start: Optional[Union[datetime, str]] = None,
        end: Optional[Union[datetime, str]] = None,
        dry_run: bool = False,
        strict_quality: bool = False,
    ) -> Dict[str, Any]:
        """
        Sync a single (symbol, timeframe) dataset incrementally or over an explicit range.
        Universal across all providers (mock, fyers, upstox).
        """
        sym = symbol.upper()
        norm_tf = normalize_timeframe(timeframe)
        self.catalog.init_catalog()

        # Step 1: Read catalog state
        cat_ds = self.catalog.get_dataset(sym, norm_tf, layer="raw") or {}
        cat_start = cat_ds.get("start_ts")
        cat_end = cat_ds.get("end_ts")
        open_part = cat_ds.get("open_partition")

        # Step 2: Query provider latest available
        latest_avail = provider.latest_available(sym, norm_tf)
        if latest_avail is None:
            return {
                "symbol": sym,
                "timeframe": norm_tf,
                "status": "no_data",
                "synced_bars": 0,
                "message": "Provider has no data available.",
            }
        latest_avail = latest_avail.replace(tzinfo=timezone.utc) if latest_avail.tzinfo is None else latest_avail.astimezone(timezone.utc)

        # Step 3: Compute sync plan
        plan = plan_sync(
            cat_start=cat_start,
            cat_end=cat_end,
            req_start=start,
            req_end=end,
            latest_avail=latest_avail,
        )

        if plan.is_up_to_date:
            return {
                "symbol": sym,
                "timeframe": norm_tf,
                "status": "up_to_date",
                "synced_bars": 0,
                "start_ts": str(cat_start),
                "end_ts": str(cat_end),
                "open_partition": open_part,
                "message": plan.message or "Already up to date.",
            }

        # Step 4: Fetch diff and validate quality
        df_new = provider.fetch(sym, norm_tf, start=plan.fetch_start, end=plan.fetch_end)
        if df_new.empty:
            return {
                "symbol": sym,
                "timeframe": norm_tf,
                "status": "empty_fetch",
                "synced_bars": 0,
                "message": "Provider returned 0 bars.",
            }

        clean_df, quality_rep = validate_bars(df_new, timeframe=norm_tf, strict=strict_quality)
        if clean_df.empty:
            return {
                "symbol": sym,
                "timeframe": norm_tf,
                "status": "quality_empty",
                "synced_bars": 0,
                "quality": quality_rep.to_dict(),
                "message": "All fetched bars dropped during validation.",
            }

        # Filter strictly newer bars if doing forward sync without start
        if start is None and cat_end is not None:
            cat_end_dt = pd.to_datetime(cat_end, utc=True)
            clean_df = clean_df[clean_df["ts"] > cat_end_dt].copy()
            if clean_df.empty:
                return {
                    "symbol": sym,
                    "timeframe": norm_tf,
                    "status": "up_to_date",
                    "synced_bars": 0,
                    "start_ts": str(cat_start),
                    "end_ts": str(cat_end),
                    "open_partition": open_part,
                    "message": "Already up to date.",
                }

        # Step 5: Dry-run check
        if dry_run:
            return {
                "symbol": sym,
                "timeframe": norm_tf,
                "status": "dry_run_success",
                "synced_bars": len(clean_df),
                "open_partition": open_part,
                "quality": quality_rep.to_dict(),
                "message": f"[DRY RUN] Would sync {len(clean_df)} bars.",
            }

        # Step 6: Write partition files
        written_files, new_open = self.partition_handler.write_all(
            symbol=sym,
            timeframe=norm_tf,
            clean_df=clean_df,
            open_part=open_part,
        )

        # Step 7: Finalize catalog and manifest
        return self._finalize_sync(
            provider=provider,
            symbol=sym,
            timeframe=norm_tf,
            clean_df=clean_df,
            cat_start=cat_start,
            cat_end=cat_end,
            new_open_partition=new_open,
            quality_rep=quality_rep,
            written_files=written_files,
        )

    def _finalize_sync(
        self,
        provider: BaseProvider,
        symbol: str,
        timeframe: str,
        clean_df: pd.DataFrame,
        cat_start: Optional[Any],
        cat_end: Optional[Any],
        new_open_partition: Optional[str],
        quality_rep: QualityReport,
        written_files: List[str],
    ) -> Dict[str, Any]:
        """Update catalog records with expanded boundaries and total rows."""
        prefix = f"raw/{symbol}/{timeframe}/"
        all_manifest = self.manifest.load()
        total_rows = sum(
            meta.get("row_count", 0)
            for fpath, meta in all_manifest.items()
            if fpath.startswith(prefix)
        )

        new_start = clean_df["ts"].min()
        new_end = clean_df["ts"].max()
        if cat_start is not None:
            new_start = min(pd.to_datetime(cat_start, utc=True), new_start)
        if cat_end is not None:
            new_end = max(pd.to_datetime(cat_end, utc=True), new_end)

        # Upsert instrument if missing
        if self.catalog.get_instrument(symbol) is None:
            exchange = "AMFI" if provider.name == "amfi" else self.settings.default_exchange
            asset_class = "MUTUAL_FUND" if provider.name == "amfi" else self.settings.default_asset_class
            self.catalog.upsert_instrument(
                symbol=symbol,
                exchange=exchange,
                asset_class=asset_class,
                currency=self.settings.default_currency,
                provider=provider.name,
                active=True,
            )

        # Upsert dataset
        self.catalog.upsert_dataset(
            symbol=symbol,
            timeframe=timeframe,
            layer="raw",
            provider=provider.name,
            start_ts=new_start,
            end_ts=new_end,
            row_count=total_rows,
            open_partition=new_open_partition or "",
            quality_status=quality_rep.status,
            schema_version="1.0",
        )

        return {
            "symbol": symbol,
            "timeframe": timeframe,
            "status": "success",
            "synced_bars": len(clean_df),
            "start_ts": new_start.isoformat(),
            "end_ts": new_end.isoformat(),
            "open_partition": new_open_partition,
            "written_partitions": written_files,
            "total_rows": total_rows,
            "quality": quality_rep.to_dict(),
        }

    def sync(
        self,
        provider_name: str = "mock",
        symbols: Optional[Union[str, List[str]]] = None,
        symbols_file: Optional[Union[str, Path]] = None,
        timeframe: str = "1min",
        start: Optional[Union[datetime, str]] = None,
        end: Optional[Union[datetime, str]] = None,
        dry_run: bool = False,
        strict_quality: bool = False,
    ) -> List[Dict[str, Any]]:
        """
        Execute sync for multiple symbols under a cross-process lock.
        Raises LockHeldError if another process holds the sync lock.
        """
        target_symbols = parse_symbols(symbols=symbols, symbols_file=symbols_file)
        if not target_symbols:
            self.catalog.init_catalog()
            inst_df = self.catalog.list_instruments()
            target_symbols = inst_df["symbol"].tolist() if not inst_df.empty else ["AAPL"]

        provider = get_provider(provider_name, data_dir=self.data_dir)
        provider.connect()

        results = []
        try:
            with self.lock:
                for sym in target_symbols:
                    res = self.sync_dataset(
                        provider=provider,
                        symbol=sym,
                        timeframe=timeframe,
                        start=start,
                        end=end,
                        dry_run=dry_run,
                        strict_quality=strict_quality,
                    )
                    results.append(res)
        finally:
            provider.close()

        return results
