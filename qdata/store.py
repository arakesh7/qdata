import hashlib
import json
import os
import shutil
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from qdata.config import Settings, get_settings

CANONICAL_COLUMNS = [
    "ts",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "provider",
    "ingested_at",
]

CANONICAL_SCHEMA = pa.schema([
    ("ts", pa.timestamp("us", tz="UTC")),
    ("open", pa.float64()),
    ("high", pa.float64()),
    ("low", pa.float64()),
    ("close", pa.float64()),
    ("volume", pa.float64()),
    ("provider", pa.string()),
    ("ingested_at", pa.timestamp("us", tz="UTC")),
])


class LockHeldError(Exception):
    """Raised when the sync lock is held by another process."""
    pass


class SyncLock:
    """
    Cross-platform lock using data/.sync.lock.
    Writes PID, timestamp, and hostname.
    Detects stale locks if the process has terminated.
    """

    def __init__(self, lock_file: Path):
        self.lock_file = Path(lock_file).resolve()
        self.acquired = False

    def is_pid_running(self, pid: int) -> bool:
        """Check if a process with given PID is still active."""
        if pid <= 0:
            return False
        try:
            if os.name == "nt":
                import ctypes
                PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
                SYNCHRONIZE = 0x00100000
                handle = ctypes.windll.kernel32.OpenProcess(
                    PROCESS_QUERY_LIMITED_INFORMATION | SYNCHRONIZE, False, pid
                )
                if handle == 0:
                    return False
                ctypes.windll.kernel32.CloseHandle(handle)
                return True
            else:
                os.kill(pid, 0)
                return True
        except (OSError, ProcessLookupError):
            return False

    def acquire(self) -> None:
        self.lock_file.parent.mkdir(parents=True, exist_ok=True)

        # Check existing lock
        if self.lock_file.exists():
            try:
                with open(self.lock_file, "r", encoding="utf-8") as f:
                    lock_info = json.load(f)
                locked_pid = int(lock_info.get("pid", -1))
                if locked_pid > 0 and self.is_pid_running(locked_pid):
                    raise LockHeldError(
                        f"Sync lock held by process {locked_pid} at {lock_info.get('locked_at')} "
                        f"(lock file: {self.lock_file})"
                    )
                # Stale lock: remove it
                try:
                    self.lock_file.unlink()
                except Exception:
                    pass
            except (json.JSONDecodeError, ValueError):
                # Malformed lock file, try removing
                try:
                    self.lock_file.unlink()
                except Exception:
                    pass

        # Write lock atomically
        tmp_lock = self.lock_file.with_suffix(f".tmp.{os.getpid()}.{uuid.uuid4().hex[:6]}")
        lock_data = {
            "pid": os.getpid(),
            "locked_at": datetime.now(timezone.utc).isoformat(),
        }
        with open(tmp_lock, "w", encoding="utf-8") as f:
            json.dump(lock_data, f)
            f.flush()
            os.fsync(f.fileno())

        try:
            # Atomic rename (on Windows os.replace replaces existing, but we checked)
            os.replace(tmp_lock, self.lock_file)
            self.acquired = True
        except Exception as e:
            if tmp_lock.exists():
                tmp_lock.unlink()
            raise LockHeldError(f"Failed to acquire sync lock: {e}")

    def release(self) -> None:
        if self.acquired and self.lock_file.exists():
            try:
                # Only unlink if our PID created it
                with open(self.lock_file, "r", encoding="utf-8") as f:
                    lock_info = json.load(f)
                if lock_info.get("pid") == os.getpid():
                    self.lock_file.unlink()
            except Exception:
                try:
                    self.lock_file.unlink()
                except Exception:
                    pass
            self.acquired = False

    def __enter__(self):
        self.acquire()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()


def compute_sha256(file_path: Path) -> str:
    """Compute SHA-256 hash of a file."""
    h = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def normalize_timeframe(tf: str) -> str:
    """Normalize timeframe string (e.g. '1m' -> '1min', 'D' -> '1d')."""
    tf = tf.strip().lower()
    if tf in ("1m", "1min", "m1"):
        return "1min"
    if tf in ("1d", "d", "daily"):
        return "1d"
    return tf


def is_monthly_partition(timeframe: str) -> bool:
    """Determine whether timeframe uses monthly or yearly partitioning."""
    norm = normalize_timeframe(timeframe)
    return norm != "1d"


def get_partition_filename(timeframe: str, dt: datetime) -> str:
    """
    Format partition file name:
    1-min -> YYYY-MM.parquet
    1d    -> YYYY.parquet
    """
    if is_monthly_partition(timeframe):
        return f"{dt.strftime('%Y-%m')}.parquet"
    return f"{dt.strftime('%Y')}.parquet"


def get_partition_relpath(symbol: str, timeframe: str, dt: datetime, layer: str = "raw") -> Path:
    """
    Returns partition path relative to data_dir.
    1-min: raw/<SYMBOL>/<timeframe>/<YYYY>/<YYYY-MM>.parquet
    1d:    raw/<SYMBOL>/1d/<YYYY>.parquet
    """
    norm_tf = normalize_timeframe(timeframe)
    sym = symbol.upper()
    year = dt.strftime("%Y")
    if is_monthly_partition(norm_tf):
        return Path(layer) / sym / norm_tf / year / f"{dt.strftime('%Y-%m')}.parquet"
    return Path(layer) / sym / norm_tf / f"{year}.parquet"


def atomic_write_parquet(
    df: pd.DataFrame,
    target_path: Path,
    schema: Optional[pa.Schema] = CANONICAL_SCHEMA,
) -> int:
    """
    Writes DataFrame to Parquet via temp file -> fsync -> atomic rename.
    Enforces canonical columns and UTC timestamps.
    Returns row count.
    """
    target_path = Path(target_path).resolve()
    target_path.parent.mkdir(parents=True, exist_ok=True)

    # Ensure canonical columns exist
    for col in CANONICAL_COLUMNS:
        if col not in df.columns:
            raise ValueError(f"Missing required canonical column: {col}")

    # Order columns
    df_to_write = df[CANONICAL_COLUMNS].copy()

    # Ensure timestamp UTC
    if not pd.api.types.is_datetime64_any_dtype(df_to_write["ts"]):
        df_to_write["ts"] = pd.to_datetime(df_to_write["ts"], utc=True)
    elif df_to_write["ts"].dt.tz is None:
        df_to_write["ts"] = df_to_write["ts"].dt.tz_localize("UTC")
    else:
        df_to_write["ts"] = df_to_write["ts"].dt.tz_convert("UTC")

    if not pd.api.types.is_datetime64_any_dtype(df_to_write["ingested_at"]):
        df_to_write["ingested_at"] = pd.to_datetime(df_to_write["ingested_at"], utc=True)
    elif df_to_write["ingested_at"].dt.tz is None:
        df_to_write["ingested_at"] = df_to_write["ingested_at"].dt.tz_localize("UTC")
    else:
        df_to_write["ingested_at"] = df_to_write["ingested_at"].dt.tz_convert("UTC")

    # Sort by ts
    df_to_write = df_to_write.sort_values("ts").reset_index(drop=True)

    table = pa.Table.from_pandas(df_to_write, schema=schema, preserve_index=False)

    tmp_path = target_path.with_suffix(f".tmp.{os.getpid()}.{uuid.uuid4().hex[:8]}")
    try:
        pq.write_table(table, tmp_path, compression="zstd")
        # fsync
        with open(tmp_path, "r+b") as f:
            os.fsync(f.fileno())
        # atomic rename
        os.replace(tmp_path, target_path)
    finally:
        if tmp_path.exists():
            try:
                tmp_path.unlink()
            except Exception:
                pass

    return len(df_to_write)


def read_partition_parquet(file_path: Path) -> pd.DataFrame:
    """Read a partition parquet file into pandas DataFrame."""
    if not file_path.exists():
        return pd.DataFrame(columns=CANONICAL_COLUMNS)
    table = pq.read_table(file_path)
    return table.to_pandas()


class ManifestManager:
    """
    Manages catalog/manifest.json which stores SHA-256 + row_count per file.
    """

    def __init__(self, data_dir: Path):
        self.data_dir = Path(data_dir).resolve()
        self.manifest_file = self.data_dir / "catalog" / "manifest.json"

    def load(self) -> Dict[str, Dict[str, Any]]:
        if not self.manifest_file.exists():
            return {}
        try:
            with open(self.manifest_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            return data.get("files", {})
        except Exception:
            return {}

    def save(self, files_dict: Dict[str, Dict[str, Any]]) -> None:
        self.manifest_file.parent.mkdir(parents=True, exist_ok=True)
        tmp_file = self.manifest_file.with_suffix(".tmp")
        payload = {
            "version": "1.0",
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "files": files_dict,
        }
        with open(tmp_file, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_file, self.manifest_file)

    def record_file(self, rel_path: Union[str, Path], row_count: int) -> Dict[str, Any]:
        """Compute sha256 and record file in manifest."""
        rel_str = Path(rel_path).as_posix()
        abs_path = self.data_dir / rel_str
        if not abs_path.exists():
            raise FileNotFoundError(f"File not found: {abs_path}")

        sha = compute_sha256(abs_path)
        stat = abs_path.stat()
        entry = {
            "sha256": sha,
            "row_count": row_count,
            "size_bytes": stat.st_size,
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
        manifest_files = self.load()
        manifest_files[rel_str] = entry
        self.save(manifest_files)
        return entry

    def remove_file(self, rel_path: Union[str, Path]) -> None:
        rel_str = Path(rel_path).as_posix()
        manifest_files = self.load()
        if rel_str in manifest_files:
            del manifest_files[rel_str]
            self.save(manifest_files)

    def verify_all(self) -> List[Dict[str, Any]]:
        """
        Verify every file in manifest against actual disk state.
        Returns list of discrepancies:
        [{ "file": ..., "status": "ok" | "mismatch" | "missing", "expected_sha": ..., "actual_sha": ... }]
        """
        manifest_files = self.load()
        results = []

        for rel_str, meta in manifest_files.items():
            abs_path = self.data_dir / rel_str
            if not abs_path.exists():
                results.append({
                    "file": rel_str,
                    "status": "missing",
                    "expected_sha": meta.get("sha256"),
                    "actual_sha": None,
                    "expected_rows": meta.get("row_count"),
                    "actual_rows": None,
                })
                continue

            actual_sha = compute_sha256(abs_path)
            if actual_sha != meta.get("sha256"):
                results.append({
                    "file": rel_str,
                    "status": "mismatch",
                    "expected_sha": meta.get("sha256"),
                    "actual_sha": actual_sha,
                    "expected_rows": meta.get("row_count"),
                    "actual_rows": len(read_partition_parquet(abs_path)),
                })
            else:
                results.append({
                    "file": rel_str,
                    "status": "ok",
                    "expected_sha": actual_sha,
                    "actual_sha": actual_sha,
                    "expected_rows": meta.get("row_count"),
                    "actual_rows": meta.get("row_count"),
                })

        return results


def quarantine_file(data_dir: Path, rel_path: Union[str, Path], reason: str = "hash_mismatch") -> Path:
    """
    Quarantine a corrupted or hash-mismatched file by moving it to data/quarantine/.
    """
    data_dir = Path(data_dir).resolve()
    rel_str = Path(rel_path).as_posix()
    source_file = data_dir / rel_str
    if not source_file.exists():
        raise FileNotFoundError(f"Cannot quarantine non-existent file: {source_file}")

    timestamp_str = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    quarantine_dest = data_dir / "quarantine" / f"{timestamp_str}_{reason}" / rel_str
    quarantine_dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(source_file), str(quarantine_dest))
    return quarantine_dest
