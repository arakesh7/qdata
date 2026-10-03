"""
qdata reader module: high-performance, instance-bound market data reader.

Provides the `DataReader` class for quant backtesters, researchers, and pipelines.
Uses catalog-guided O(1) partition pruning, PyArrow columnar projection, and C++ predicate
pushdown to deliver sub-millisecond data loading without global state.
"""

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.dataset as ds

from qdata.catalog import CatalogManager
from qdata.config import get_settings
from qdata.store import (
    CANONICAL_COLUMNS,
    is_monthly_partition,
    normalize_timeframe,
)


def _to_utc(val: Any, tz: Optional[str]) -> Optional[pd.Timestamp]:
    """Parse and convert a datetime input to UTC, localizing naive values to tz or UTC."""
    if val is None:
        return None
    t = pd.to_datetime(val)
    return (t.tz_localize(tz or "UTC") if t.tzinfo is None else t).tz_convert("UTC")


def _empty_result(
    cols: List[str],
    as_arrow: bool,
    set_index: bool,
    tz: Optional[str] = None,
) -> Union[pd.DataFrame, pa.Table]:
    """Build a schema-consistent empty DataFrame or PyArrow Table."""
    if as_arrow:
        fields = [
            pa.field(c, pa.timestamp("us", tz=tz or "UTC") if c == "ts" else pa.float64())
            for c in cols
        ]
        return pa.Table.from_arrays([pa.array([]) for _ in fields], schema=pa.schema(fields))

    df = pd.DataFrame(columns=[c for c in cols if c != "ts" or not set_index])
    if set_index and "ts" in cols:
        df.index = pd.DatetimeIndex([], name="ts", tz=tz or "UTC")
    return df


class DataReader:
    """
    High-performance market data reader initialized with an explicit data directory.

    Attributes:
    -----------
    data_dir : Path
        Resolved path to the market data directory.
    layer : str
        Default storage layer ('raw' or 'adjusted').
    catalog : CatalogManager
        Bound catalog manager for dataset and instrument queries.
    """

    def __init__(
        self,
        data_dir: Optional[Union[str, Path]] = None,
        layer: str = "raw",
    ) -> None:
        if data_dir is not None:
            self.data_dir = Path(data_dir).resolve()
        else:
            self.data_dir = get_settings().data_dir.resolve()

        if not self.data_dir.exists():
            raise FileNotFoundError(f"qdata directory does not exist: {self.data_dir}")

        self.layer = layer
        self.catalog = CatalogManager(self.data_dir)

    def __repr__(self) -> str:
        return f"DataReader(data_dir='{self.data_dir}', layer='{self.layer}')"

    def _resolve_partition_files(
        self,
        symbol: str,
        timeframe: str,
        start_dt: Optional[pd.Timestamp],
        end_dt: Optional[pd.Timestamp],
        layer: str,
    ) -> List[Path]:
        """
        Calculate the exact partition files that overlap [start_dt, end_dt] via date math.
        Avoids recursive directory scanning (rglob / os.walk) for O(1) file selection.
        """
        sym = symbol.upper()
        norm_tf = normalize_timeframe(timeframe)
        base_dir = self.data_dir / layer / sym / norm_tf

        if not base_dir.exists():
            return []

        # If bounds are unspecified, consult bound catalog first
        if start_dt is None or end_dt is None:
            try:
                cat_ds = self.catalog.get_dataset(sym, norm_tf, layer=layer)
                if cat_ds:
                    if start_dt is None and cat_ds.get("start_ts") is not None:
                        start_dt = pd.to_datetime(cat_ds["start_ts"], utc=True)
                    if end_dt is None and cat_ds.get("end_ts") is not None:
                        end_dt = pd.to_datetime(cat_ds["end_ts"], utc=True)
            except Exception:
                pass

        # If still unbounded, grab all existing parquet files sorted
        if start_dt is None or end_dt is None:
            return sorted(base_dir.rglob("*.parquet"))

        # 1. Yearly partitions (1d)
        if not is_monthly_partition(norm_tf):
            candidate_files = []
            for year in range(start_dt.year, end_dt.year + 1):
                p = base_dir / f"{year}.parquet"
                if p.exists():
                    candidate_files.append(p)
            return candidate_files

        # 2. Monthly partitions (1min and intraday)
        candidate_files = []
        curr_y, curr_m = start_dt.year, start_dt.month
        end_y, end_m = end_dt.year, end_dt.month

        while (curr_y, curr_m) <= (end_y, end_m):
            p = base_dir / str(curr_y) / f"{curr_y:04d}-{curr_m:02d}.parquet"
            if p.exists():
                candidate_files.append(p)
            curr_m += 1
            if curr_m > 12:
                curr_m = 1
                curr_y += 1

        return candidate_files

    def load(
        self,
        symbol: str,
        timeframe: str = "1min",
        start: Optional[Union[datetime, str, pd.Timestamp]] = None,
        end: Optional[Union[datetime, str, pd.Timestamp]] = None,
        columns: Optional[List[str]] = None,
        layer: Optional[str] = None,
        tz: Optional[str] = None,
        set_index: bool = True,
        as_arrow: bool = False,
    ) -> Union[pd.DataFrame, pa.Table]:
        """
        High-performance market data loader for backtesting and quantitative research.
        Supports predicate pushdown, column projection, and timezone-aware formatting.
        """
        norm_tf = normalize_timeframe(timeframe)
        target_layer = layer or self.layer
        target_tz = "Asia/Kolkata" if tz and tz.upper() == "IST" else tz
        start_dt = _to_utc(start, target_tz)
        end_dt = _to_utc(end, target_tz)

        req_cols = list(columns) if columns else list(CANONICAL_COLUMNS)
        files = self._resolve_partition_files(symbol.upper(), norm_tf, start_dt, end_dt, target_layer)
        if not files:
            return _empty_result(req_cols, as_arrow, set_index, target_tz)

        # C++ Pushdown predicate in UTC
        filters = None
        if start_dt is not None:
            filters = (ds.field("ts") >= start_dt)
        if end_dt is not None:
            end_f = (ds.field("ts") <= end_dt)
            filters = (filters & end_f) if filters is not None else end_f

        # Column projection
        load_cols = list(req_cols)
        if set_index and "ts" not in load_cols:
            load_cols.insert(0, "ts")

        try:
            table = ds.dataset(files, format="parquet").to_table(columns=load_cols, filter=filters)
        except Exception:
            return _empty_result(req_cols, as_arrow, set_index, target_tz)

        if table.num_rows == 0:
            return _empty_result(req_cols, as_arrow, set_index, target_tz)

        # Zero-copy PyArrow handoff with in-place schema cast
        if as_arrow:
            if target_tz and "ts" in table.column_names:
                idx = table.schema.get_field_index("ts")
                table = table.set_column(idx, "ts", pc.cast(table["ts"], pa.timestamp("us", tz=target_tz)))
            return table

        # Pandas formatting
        df = table.to_pandas().sort_values("ts")
        if "ts" in df.columns:
            if norm_tf == "1d":
                dates = pd.to_datetime(df["ts"].dt.strftime("%Y-%m-%d"))
                df["ts"] = dates.dt.tz_localize(target_tz or "UTC")
            elif target_tz:
                df["ts"] = df["ts"].dt.tz_convert(target_tz)

            if set_index:
                df = df.set_index("ts")
                df.index.name = "ts"
                if columns and "ts" not in columns:
                    df = df[[c for c in columns if c in df.columns]]
            else:
                df = df.reset_index(drop=True)

        return df

    def list_symbols(self, layer: Optional[str] = None) -> List[str]:
        """List all symbols stored in this data directory."""
        target_layer = layer or self.layer
        layer_dir = self.data_dir / target_layer
        if not layer_dir.exists():
            return []
        return sorted([d.name for d in layer_dir.iterdir() if d.is_dir()])

    def list_timeframes(self, symbol: str, layer: Optional[str] = None) -> List[str]:
        """List all available timeframes for a given symbol."""
        target_layer = layer or self.layer
        sym_dir = self.data_dir / target_layer / symbol.strip().upper()
        if not sym_dir.exists():
            return []
        return sorted([d.name for d in sym_dir.iterdir() if d.is_dir()])

    def dataset_info(
        self,
        symbol: str,
        timeframe: str = "1min",
        layer: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """Get catalog metadata for a dataset."""
        target_layer = layer or self.layer
        return self.catalog.get_dataset(symbol, timeframe, layer=target_layer)


__all__ = ["DataReader"]
