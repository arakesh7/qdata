from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from qdata.store import CANONICAL_COLUMNS, normalize_timeframe


@dataclass
class QualityReport:
    status: str = "PASSED"  # PASSED, WARN, FAIL
    total_bars: int = 0
    valid_bars: int = 0
    duplicates_dropped: int = 0
    ohlc_violations: int = 0
    insane_prices: int = 0
    nan_values: int = 0
    gaps_detected: int = 0
    max_gap_seconds: float = 0.0
    details: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class QualityFailureError(Exception):
    """Raised when data quality validation fails critically."""
    def __init__(self, message: str, report: QualityReport):
        super().__init__(message)
        self.report = report


def validate_bars(
    df: pd.DataFrame,
    timeframe: str = "1min",
    strict: bool = False,
) -> Tuple[pd.DataFrame, QualityReport]:
    """
    Validate, dedupe, and run quality checks on market data bars.
    Returns (cleaned_df, report).
    If strict=True and status is FAIL, raises QualityFailureError.
    """
    report = QualityReport()

    if df.empty:
        report.details.append("Empty DataFrame provided.")
        return df, report

    report.total_bars = len(df)
    work_df = df.copy()

    # 1. Ensure required columns
    missing_cols = [c for c in ["ts", "open", "high", "low", "close", "volume"] if c not in work_df.columns]
    if missing_cols:
        report.status = "FAIL"
        report.details.append(f"Missing required columns: {missing_cols}")
        if strict:
            raise QualityFailureError("Missing required columns", report)
        return work_df, report

    # 2. Check and convert ts to UTC
    if not pd.api.types.is_datetime64_any_dtype(work_df["ts"]):
        work_df["ts"] = pd.to_datetime(work_df["ts"], utc=True)
    elif work_df["ts"].dt.tz is None:
        work_df["ts"] = work_df["ts"].dt.tz_localize("UTC")
    else:
        work_df["ts"] = work_df["ts"].dt.tz_convert("UTC")

    # 3. Check for NaNs
    nan_mask = work_df[["ts", "open", "high", "low", "close", "volume"]].isna().any(axis=1)
    nan_count = int(nan_mask.sum())
    if nan_count > 0:
        report.nan_values = nan_count
        report.details.append(f"Found {nan_count} rows with NaN values. Dropping NaN rows.")
        work_df = work_df[~nan_mask].copy()

    # 4. Deduplicate on ts
    # Sort by ingested_at descending so newest ingested bar wins
    sort_cols = ["ts"]
    ascending = [True]
    if "ingested_at" in work_df.columns:
        sort_cols.append("ingested_at")
        ascending.append(False)

    work_df = work_df.sort_values(by=sort_cols, ascending=ascending)
    before_dedupe = len(work_df)
    work_df = work_df.drop_duplicates(subset=["ts"], keep="first")
    report.duplicates_dropped = before_dedupe - len(work_df)
    if report.duplicates_dropped > 0:
        report.details.append(f"Dropped {report.duplicates_dropped} duplicate bars.")

    # Re-sort strictly by timestamp ascending
    work_df = work_df.sort_values(by="ts", ascending=True).reset_index(drop=True)

    # 5. Price sanity checks (positive prices, non-negative volume)
    insane_mask = (
        (work_df["open"] <= 0)
        | (work_df["high"] <= 0)
        | (work_df["low"] <= 0)
        | (work_df["close"] <= 0)
        | (work_df["volume"] < 0)
        | np.isinf(work_df["open"])
        | np.isinf(work_df["high"])
        | np.isinf(work_df["low"])
        | np.isinf(work_df["close"])
        | np.isinf(work_df["volume"])
    )
    insane_count = int(insane_mask.sum())
    if insane_count > 0:
        report.insane_prices = insane_count
        report.details.append(f"Found {insane_count} bars with non-positive price or negative volume.")
        work_df = work_df[~insane_mask].copy()

    # 6. OHLC consistency checks:
    # High >= Low, High >= Open, High >= Close, Low <= Open, Low <= Close
    ohlc_violation_mask = (
        (work_df["high"] < work_df["low"])
        | (work_df["high"] < work_df["open"])
        | (work_df["high"] < work_df["close"])
        | (work_df["low"] > work_df["open"])
        | (work_df["low"] > work_df["close"])
    )
    ohlc_count = int(ohlc_violation_mask.sum())
    if ohlc_count > 0:
        report.ohlc_violations = ohlc_count
        report.details.append(f"Found {ohlc_count} bars with OHLC consistency violations.")
        work_df = work_df[~ohlc_violation_mask].copy()

    # 7. Gap detection
    norm_tf = normalize_timeframe(timeframe)
    if len(work_df) > 1:
        time_diffs = work_df["ts"].diff().dt.total_seconds().dropna()
        if norm_tf == "1min":
            expected_delta = 60.0
        elif norm_tf == "1d":
            expected_delta = 86400.0
        else:
            expected_delta = 60.0

        # Gaps exceeding expected delta
        gap_diffs = time_diffs[time_diffs > expected_delta * 1.5]
        report.gaps_detected = len(gap_diffs)
        report.max_gap_seconds = float(time_diffs.max()) if not time_diffs.empty else 0.0
        if report.gaps_detected > 0:
            report.details.append(
                f"Detected {report.gaps_detected} gaps. Max gap: {report.max_gap_seconds:.0f} seconds."
            )

    report.valid_bars = len(work_df)

    # Determine status
    if report.ohlc_violations > 0 or report.insane_prices > 0:
        report.status = "WARN" if not strict else "FAIL"
    elif report.nan_values > 0 or report.duplicates_dropped > 0:
        report.status = "WARN"
    else:
        report.status = "PASSED"

    if report.valid_bars == 0 and report.total_bars > 0:
        report.status = "FAIL"

    if strict and report.status == "FAIL":
        raise QualityFailureError(f"Quality validation failed: {'; '.join(report.details)}", report)

    return work_df, report
