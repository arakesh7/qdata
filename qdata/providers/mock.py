import hashlib
import time
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from qdata.auth import AuthError, save_cached_token
from qdata.providers.base import BaseProvider, auth_retry_on_401
from qdata.store import CANONICAL_COLUMNS, normalize_timeframe


class MockProvider(BaseProvider):
    """
    Deterministic Mock Provider for testing, offline use, and CI.
    Generates synthetic deterministic market data and supports auth simulation.
    """

    def __init__(
        self,
        data_dir: Optional[Path] = None,
        mock_latest_time: Optional[datetime] = None,
    ):
        super().__init__(data_dir=data_dir)
        self._mock_latest = mock_latest_time or datetime(2026, 9, 25, 15, 30, tzinfo=timezone.utc)
        self.simulate_401_once = False
        self.auth_count = 0
        self.refresh_count = 0

    @property
    def name(self) -> str:
        return "mock"

    def _authenticate(self) -> Dict[str, Any]:
        self.auth_count += 1
        token = {
            "access_token": f"mock_token_{uuid.uuid4().hex[:12]}",
            "token_type": "Bearer",
            "expires_at": time.time() + 3600.0,
            "created_at": time.time(),
            "provider": "mock",
            "auth_count": self.auth_count,
        }
        save_cached_token(self.tokens_dir, self.name, token)
        return token

    def refresh_auth(self) -> Dict[str, Any]:
        self.refresh_count += 1
        return super().refresh_auth()

    def set_mock_latest(self, dt: datetime) -> None:
        if dt.tzinfo is None:
            self._mock_latest = dt.replace(tzinfo=timezone.utc)
        else:
            self._mock_latest = dt.astimezone(timezone.utc)

    @auth_retry_on_401
    def latest_available(self, symbol: str, timeframe: str) -> Optional[datetime]:
        self.rate_limit()
        if self.simulate_401_once:
            self.simulate_401_once = False
            raise AuthError("Simulated 401 Unauthorized for testing auth retry")
        return self._mock_latest

    @auth_retry_on_401
    def fetch(
        self,
        symbol: str,
        timeframe: str,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
    ) -> pd.DataFrame:
        self.rate_limit()
        if self.simulate_401_once:
            self.simulate_401_once = False
            raise AuthError("Simulated 401 Unauthorized for testing auth retry")

        sym = symbol.upper()
        norm_tf = normalize_timeframe(timeframe)

        if end is None:
            end = self._mock_latest
        if end.tzinfo is None:
            end = end.replace(tzinfo=timezone.utc)
        else:
            end = end.astimezone(timezone.utc)

        if start is None:
            if norm_tf == "1d":
                start = end - timedelta(days=30)
            else:
                start = end - timedelta(hours=6)

        if start.tzinfo is None:
            start = start.replace(tzinfo=timezone.utc)
        else:
            start = start.astimezone(timezone.utc)

        if start > end:
            return pd.DataFrame(columns=CANONICAL_COLUMNS)

        # Generate timestamps
        freq = "1min" if norm_tf == "1min" else "1D"
        ts_range = pd.date_range(start=start, end=end, freq=freq, tz=timezone.utc)
        if len(ts_range) == 0:
            return pd.DataFrame(columns=CANONICAL_COLUMNS)

        # Deterministic price series based on symbol seed
        sym_hash = int(hashlib.md5(sym.encode()).hexdigest()[:8], 16)
        base_price = 100.0 + (sym_hash % 900)  # price between 100 and 1000

        rows = []
        ingested_now = datetime.now(timezone.utc)

        for i, ts in enumerate(ts_range):
            # Seed per ts for exact determinism across repeated fetches
            ts_int = int(ts.timestamp())
            rng = np.random.RandomState((sym_hash + ts_int) % (2**31 - 1))

            noise = rng.normal(0, 0.002)
            open_p = round(base_price * (1.0 + (i * 0.0001) + noise), 2)
            delta_1 = round(abs(rng.normal(0, 0.5)), 2)
            delta_2 = round(abs(rng.normal(0, 0.5)), 2)
            close_p = round(max(open_p + rng.normal(0, 0.4), 1.0), 2)
            high_p = round(max(open_p, close_p) + delta_1, 2)
            low_p = round(max(min(open_p, close_p) - delta_2, 0.05), 2)
            vol = round(float(rng.randint(100, 10000)), 0)

            rows.append({
                "ts": ts,
                "open": open_p,
                "high": high_p,
                "low": low_p,
                "close": close_p,
                "volume": vol,
                "provider": "mock",
                "ingested_at": ingested_now,
            })

        df = pd.DataFrame(rows)
        return df

    def health_check(self) -> Dict[str, Any]:
        return {
            "status": "ok",
            "provider": self.name,
            "connected": self.connected,
            "latest_available": self._mock_latest.isoformat(),
        }
