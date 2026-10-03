import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from qdata.store import normalize_timeframe

INSTRUMENTS_COLUMNS = [
    "symbol",
    "exchange",
    "asset_class",
    "currency",
    "provider",
    "provider_symbol",
    "active",
    "listed_date",
]

INSTRUMENTS_SCHEMA = pa.schema([
    ("symbol", pa.string()),
    ("exchange", pa.string()),
    ("asset_class", pa.string()),
    ("currency", pa.string()),
    ("provider", pa.string()),
    ("provider_symbol", pa.string()),
    ("active", pa.bool_()),
    ("listed_date", pa.string()),
])

DATASETS_COLUMNS = [
    "symbol",
    "timeframe",
    "layer",
    "provider",
    "start_ts",
    "end_ts",
    "row_count",
    "open_partition",
    "quality_status",
    "schema_version",
]

DATASETS_SCHEMA = pa.schema([
    ("symbol", pa.string()),
    ("timeframe", pa.string()),
    ("layer", pa.string()),
    ("provider", pa.string()),
    ("start_ts", pa.timestamp("us", tz="UTC")),
    ("end_ts", pa.timestamp("us", tz="UTC")),
    ("row_count", pa.int64()),
    ("open_partition", pa.string()),
    ("quality_status", pa.string()),
    ("schema_version", pa.string()),
])


class CatalogManager:
    """
    Manages instruments.parquet and datasets.parquet in data/catalog/.
    Catalog is the single source of truth.
    All writes use temp file -> fsync -> atomic rename.
    """

    def __init__(self, data_dir: Path):
        self.data_dir = Path(data_dir).resolve()
        self.catalog_dir = self.data_dir / "catalog"
        self.instruments_path = self.catalog_dir / "instruments.parquet"
        self.datasets_path = self.catalog_dir / "datasets.parquet"

    def init_catalog(self) -> None:
        """Initialize catalog files if they do not exist."""
        self.catalog_dir.mkdir(parents=True, exist_ok=True)
        if not self.instruments_path.exists():
            df_inst = pd.DataFrame(columns=INSTRUMENTS_COLUMNS)
            self._write_instruments(df_inst)
        if not self.datasets_path.exists():
            df_ds = pd.DataFrame(columns=DATASETS_COLUMNS)
            self._write_datasets(df_ds)

    def _write_instruments(self, df: pd.DataFrame) -> None:
        self.catalog_dir.mkdir(parents=True, exist_ok=True)
        df_to_write = df[INSTRUMENTS_COLUMNS].copy()
        df_to_write["symbol"] = df_to_write["symbol"].astype(str)
        df_to_write["exchange"] = df_to_write["exchange"].astype(str)
        df_to_write["asset_class"] = df_to_write["asset_class"].astype(str)
        df_to_write["currency"] = df_to_write["currency"].astype(str)
        df_to_write["provider"] = df_to_write["provider"].astype(str)
        df_to_write["provider_symbol"] = df_to_write["provider_symbol"].astype(str)
        df_to_write["active"] = df_to_write["active"].astype(bool)
        df_to_write["listed_date"] = df_to_write["listed_date"].astype(str)

        table = pa.Table.from_pandas(df_to_write, schema=INSTRUMENTS_SCHEMA, preserve_index=False)
        tmp_file = self.instruments_path.with_suffix(f".tmp.{os.getpid()}.{uuid.uuid4().hex[:6]}")
        pq.write_table(table, tmp_file, compression="zstd")
        with open(tmp_file, "r+b") as f:
            os.fsync(f.fileno())
        os.replace(tmp_file, self.instruments_path)

    def _write_datasets(self, df: pd.DataFrame) -> None:
        self.catalog_dir.mkdir(parents=True, exist_ok=True)
        df_to_write = df[DATASETS_COLUMNS].copy()
        df_to_write["symbol"] = df_to_write["symbol"].astype(str)
        df_to_write["timeframe"] = df_to_write["timeframe"].astype(str)
        df_to_write["layer"] = df_to_write["layer"].astype(str)
        df_to_write["provider"] = df_to_write["provider"].astype(str)

        if not pd.api.types.is_datetime64_any_dtype(df_to_write["start_ts"]):
            df_to_write["start_ts"] = pd.to_datetime(df_to_write["start_ts"], utc=True)
        else:
            df_to_write["start_ts"] = df_to_write["start_ts"].dt.tz_convert("UTC")

        if not pd.api.types.is_datetime64_any_dtype(df_to_write["end_ts"]):
            df_to_write["end_ts"] = pd.to_datetime(df_to_write["end_ts"], utc=True)
        else:
            df_to_write["end_ts"] = df_to_write["end_ts"].dt.tz_convert("UTC")

        df_to_write["row_count"] = df_to_write["row_count"].astype("int64")
        df_to_write["open_partition"] = df_to_write["open_partition"].astype(str)
        df_to_write["quality_status"] = df_to_write["quality_status"].astype(str)
        df_to_write["schema_version"] = df_to_write["schema_version"].astype(str)

        table = pa.Table.from_pandas(df_to_write, schema=DATASETS_SCHEMA, preserve_index=False)
        tmp_file = self.datasets_path.with_suffix(f".tmp.{os.getpid()}.{uuid.uuid4().hex[:6]}")
        pq.write_table(table, tmp_file, compression="zstd")
        with open(tmp_file, "r+b") as f:
            os.fsync(f.fileno())
        os.replace(tmp_file, self.datasets_path)

    # --- Instruments operations ---

    def list_instruments(self) -> pd.DataFrame:
        if not self.instruments_path.exists():
            return pd.DataFrame(columns=INSTRUMENTS_COLUMNS)
        return pq.read_table(self.instruments_path).to_pandas()

    def get_instrument(self, symbol: str) -> Optional[Dict[str, Any]]:
        df = self.list_instruments()
        match = df[df["symbol"].str.upper() == symbol.upper()]
        if match.empty:
            return None
        return match.iloc[0].to_dict()

    def upsert_instrument(
        self,
        symbol: str,
        exchange: str = "NSE",
        asset_class: str = "EQUITY",
        currency: str = "INR",
        provider: str = "mock",
        provider_symbol: Optional[str] = None,
        active: bool = True,
        listed_date: str = "",
    ) -> None:
        sym = symbol.upper()
        df = self.list_instruments()
        row = {
            "symbol": sym,
            "exchange": exchange.upper(),
            "asset_class": asset_class.upper(),
            "currency": currency.upper(),
            "provider": provider.lower(),
            "provider_symbol": provider_symbol or sym,
            "active": active,
            "listed_date": listed_date or "",
        }
        if df.empty:
            df = pd.DataFrame([row])
        elif sym in df["symbol"].values:
            idx = df[df["symbol"] == sym].index[0]
            for k, v in row.items():
                df.at[idx, k] = v
        else:
            df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)

        self._write_instruments(df)

    # --- Datasets operations ---

    def list_datasets(
        self,
        symbol: Optional[str] = None,
        timeframe: Optional[str] = None,
        layer: Optional[str] = None,
    ) -> pd.DataFrame:
        if not self.datasets_path.exists():
            return pd.DataFrame(columns=DATASETS_COLUMNS)
        df = pq.read_table(self.datasets_path).to_pandas()
        if symbol:
            df = df[df["symbol"].str.upper() == symbol.upper()]
        if timeframe:
            norm_tf = normalize_timeframe(timeframe)
            df = df[df["timeframe"].apply(normalize_timeframe) == norm_tf]
        if layer:
            df = df[df["layer"].str.lower() == layer.lower()]
        return df.reset_index(drop=True)

    def get_dataset(
        self,
        symbol: str,
        timeframe: str,
        layer: str = "raw",
    ) -> Optional[Dict[str, Any]]:
        df = self.list_datasets(symbol=symbol, timeframe=timeframe, layer=layer)
        if df.empty:
            return None
        return df.iloc[0].to_dict()

    def upsert_dataset(
        self,
        symbol: str,
        timeframe: str,
        layer: str,
        provider: str,
        start_ts: Union[datetime, pd.Timestamp, str],
        end_ts: Union[datetime, pd.Timestamp, str],
        row_count: int,
        open_partition: str,
        quality_status: str = "OK",
        schema_version: str = "1.0",
    ) -> None:
        sym = symbol.upper()
        norm_tf = normalize_timeframe(timeframe)
        lay = layer.lower()

        # Format timestamps
        start_dt = pd.to_datetime(start_ts, utc=True)
        end_dt = pd.to_datetime(end_ts, utc=True)

        df = self.list_datasets()
        row = {
            "symbol": sym,
            "timeframe": norm_tf,
            "layer": lay,
            "provider": provider.lower(),
            "start_ts": start_dt,
            "end_ts": end_dt,
            "row_count": int(row_count),
            "open_partition": str(open_partition),
            "quality_status": quality_status,
            "schema_version": schema_version,
        }

        if df.empty:
            df = pd.DataFrame([row])
        else:
            match = df[
                (df["symbol"].str.upper() == sym)
                & (df["timeframe"].apply(normalize_timeframe) == norm_tf)
                & (df["layer"].str.lower() == lay)
            ]
            if not match.empty:
                idx = match.index[0]
                for k, v in row.items():
                    df.at[idx, k] = v
            else:
                df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)

        self._write_datasets(df)
