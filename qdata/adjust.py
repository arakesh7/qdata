from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from qdata.catalog import CatalogManager
from qdata.config import get_settings
from qdata.store import (
    ManifestManager,
    atomic_write_parquet,
    normalize_timeframe,
    read_partition_parquet,
)


def rebuild_adjusted_layer(
    symbol: str,
    timeframe: Optional[str] = None,
    data_dir: Optional[Path] = None,
    dry_run: bool = False,
) -> Dict[str, Any]:
    """
    Rebuild the derived 'adjusted' layer from raw parquet files.
    Raw data remains untouched.
    """
    settings = get_settings(override_data_dir=data_dir)
    catalog = CatalogManager(settings.data_dir)
    manifest = ManifestManager(settings.data_dir)

    sym = symbol.upper()
    raw_sym_dir = settings.raw_dir / sym
    if not raw_sym_dir.exists():
        return {
            "symbol": sym,
            "status": "error",
            "message": f"No raw data found for symbol {sym} at {raw_sym_dir}",
            "rebuilt_files": [],
        }

    # Find timeframes
    if timeframe:
        norm_tf = normalize_timeframe(timeframe)
        tf_dirs = [raw_sym_dir / norm_tf]
    else:
        tf_dirs = [p for p in raw_sym_dir.iterdir() if p.is_dir()]

    rebuilt_files: List[str] = []
    total_bars = 0

    for tf_dir in tf_dirs:
        if not tf_dir.exists():
            continue
        tf_name = tf_dir.name
        # Find all parquet files in this timeframe
        parquet_files = list(tf_dir.rglob("*.parquet"))
        parquet_files.sort()

        if not parquet_files:
            continue

        tf_total_bars = 0
        latest_adjusted_file = None
        min_ts = None
        max_ts = None

        for raw_file in parquet_files:
            rel_to_raw = raw_file.relative_to(settings.raw_dir)
            adj_rel = Path("adjusted") / rel_to_raw
            adj_target = settings.data_dir / adj_rel

            if dry_run:
                rebuilt_files.append(adj_rel.as_posix())
                continue

            # Read raw
            df = read_partition_parquet(raw_file)
            if df.empty:
                continue

            # Corporate action / split adjustment formula (default identity factor)
            # If adjustments are provided in future, factor applied here:
            # df["open"] *= factor, df["close"] *= factor, etc.
            df_adj = df.copy()

            row_count = atomic_write_parquet(df_adj, adj_target)
            manifest.record_file(adj_rel.as_posix(), row_count)

            rebuilt_files.append(adj_rel.as_posix())
            tf_total_bars += row_count
            latest_adjusted_file = adj_rel.as_posix()

            p_min_ts = df["ts"].min()
            p_max_ts = df["ts"].max()
            min_ts = min(min_ts, p_min_ts) if min_ts is not None else p_min_ts
            max_ts = max(max_ts, p_max_ts) if max_ts is not None else p_max_ts

        total_bars += tf_total_bars

        if not dry_run and latest_adjusted_file:
            catalog.upsert_dataset(
                symbol=sym,
                timeframe=tf_name,
                layer="adjusted",
                provider="derived",
                start_ts=min_ts,
                end_ts=max_ts,
                row_count=tf_total_bars,
                open_partition=latest_adjusted_file,
                quality_status="PASSED",
                schema_version="1.0",
            )

    return {
        "symbol": sym,
        "status": "dry_run" if dry_run else "success",
        "rebuilt_files": rebuilt_files,
        "total_bars": total_bars,
        "message": f"{'[DRY RUN] Would rebuild' if dry_run else 'Rebuilt'} {len(rebuilt_files)} files.",
    }
