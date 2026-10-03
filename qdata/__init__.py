"""
qdata - Market data library and CLI tool.
Ingests, stores, catalogs, and serves market data at 1-min base granularity.
"""

from qdata.adjust import rebuild_adjusted_layer
from qdata.auth import AuthError, get_credential, mask_secret
from qdata.catalog import CatalogManager
from qdata.config import Settings, get_settings
from qdata.doctor import run_doctor
from qdata.providers import (
    AmfiProvider,
    BaseProvider,
    FyersProvider,
    MockProvider,
    UpstoxProvider,
    get_provider,
    register_provider,
)
from qdata.quality import QualityFailureError, QualityReport, validate_bars
from qdata.reader import DataReader
from qdata.store import (
    CANONICAL_COLUMNS,
    CANONICAL_SCHEMA,
    LockHeldError,
    ManifestManager,
    SyncLock,
    atomic_write_parquet,
    compute_sha256,
    normalize_timeframe,
    quarantine_file,
    read_partition_parquet,
)
from qdata.sync import SyncEngine

__version__ = "0.1.0"

__all__ = [
    "__version__",
    "DataReader",
    "SyncEngine",
    "CatalogManager",
    "ManifestManager",
    "SyncLock",
    "LockHeldError",
    "validate_bars",
    "QualityReport",
    "QualityFailureError",
    "rebuild_adjusted_layer",
    "run_doctor",
    "get_provider",
    "register_provider",
    "BaseProvider",
    "MockProvider",
    "UpstoxProvider",
    "FyersProvider",
    "AmfiProvider",
    "get_settings",
    "Settings",
    "AuthError",
    "get_credential",
    "mask_secret",
    "atomic_write_parquet",
    "read_partition_parquet",
    "compute_sha256",
    "normalize_timeframe",
    "quarantine_file",
    "CANONICAL_COLUMNS",
    "CANONICAL_SCHEMA",
]
