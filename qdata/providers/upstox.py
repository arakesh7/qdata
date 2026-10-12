from __future__ import annotations

import logging
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Tuple
from urllib.parse import quote

import pandas as pd
import requests

from qdata.auth import get_credential, save_cached_token
from qdata.providers.base import BaseProvider, auth_retry_on_401
from qdata.store import CANONICAL_COLUMNS, normalize_timeframe

logger = logging.getLogger(__name__)

BASE_URL = "https://api.upstox.com/v2/historical-candle"
INSTRUMENTS_URL = "https://assets.upstox.com/market-quote/instruments/exchange/complete.csv.gz"

CANDLE_COLUMNS = [
    "timestamp",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "open_interest",
]

INDEX_ALIASES: Dict[str, str] = {
    "NIFTY": "NSE_INDEX|Nifty 50",
    "NIFTY 50": "NSE_INDEX|Nifty 50",
    "NIFTY50": "NSE_INDEX|Nifty 50",
    "NIFTY-50": "NSE_INDEX|Nifty 50",
    "NSE:NIFTY": "NSE_INDEX|Nifty 50",
    "NSE:NIFTY 50": "NSE_INDEX|Nifty 50",
    "BANKNIFTY": "NSE_INDEX|Nifty Bank",
    "NIFTY BANK": "NSE_INDEX|Nifty Bank",
    "NIFTYBANK": "NSE_INDEX|Nifty Bank",
    "NSE:BANKNIFTY": "NSE_INDEX|Nifty Bank",
    "FINNIFTY": "NSE_INDEX|Nifty Fin Service",
    "NIFTY FIN SERVICE": "NSE_INDEX|Nifty Fin Service",
    "NIFTY FINANCIAL SERVICES": "NSE_INDEX|Nifty Fin Service",
    "NIFTY IT": "NSE_INDEX|Nifty IT",
    "NIFTYIT": "NSE_INDEX|Nifty IT",
    "NIFTY 100": "NSE_INDEX|Nifty 100",
    "NIFTY100": "NSE_INDEX|Nifty 100",
    "NIFTY 500": "NSE_INDEX|Nifty 500",
    "NIFTY500": "NSE_INDEX|Nifty 500",
    "NIFTY NEXT 50": "NSE_INDEX|Nifty Next 50",
    "NIFTYNEXT50": "NSE_INDEX|Nifty Next 50",
    "NIFTY MIDCAP 50": "NSE_INDEX|Nifty Midcap 50",
    "NIFTYMIDCAP50": "NSE_INDEX|Nifty Midcap 50",
    "NIFTY MIDCAP 100": "NSE_INDEX|Nifty Midcap 100",
    "NIFTYMIDCAP100": "NSE_INDEX|Nifty Midcap 100",
    "NIFTY SMALLCAP 100": "NSE_INDEX|Nifty Smallcap 100",
    "NIFTYSMALLCAP100": "NSE_INDEX|Nifty Smallcap 100",
    "SENSEX": "BSE_INDEX|SENSEX",
    "BSESENSEX": "BSE_INDEX|SENSEX",
    "BSE:SENSEX": "BSE_INDEX|SENSEX",
    "BANKEX": "BSE_INDEX|BANKEX",
    "BSEBANKEX": "BSE_INDEX|BANKEX",
    "BSE:BANKEX": "BSE_INDEX|BANKEX",
}


class UpstoxProvider(BaseProvider):
    """
    Upstox Market Data Provider for Qdata.

    Current version supports Upstox V2 API for Daily (1D) historical candle data.
    - Public historical endpoints (no authentication or access token required in V2).
    - Supports daily timeframe ('1d') exclusively.
    - Human-readable symbol lookup like Fyers (e.g. 'RELIANCE', 'TCS', 'BSE:RELIANCE', 'SENSEX').
    - Default exchange defaults to NSE if unspecified.
    - Automatic 1-year window chunking with leap-year handling.
    - Resilient network retries with exponential backoff and rate pacing.
    - Output normalized to Qdata canonical schema with UTC timestamps.

    (Note: Future Upstox V3 API will require developer credentials and authentication).
    """

    BASE_URL = BASE_URL
    INSTRUMENTS_URL = INSTRUMENTS_URL
    CANDLE_COLUMNS = CANDLE_COLUMNS
    INDEX_ALIASES = INDEX_ALIASES

    # Class-level cache for instruments master to share across instances and class calls
    _cached_instruments_df: Optional[pd.DataFrame] = None

    def __init__(
        self,
        data_dir: Optional[Path] = None,
        exchange: Optional[str] = None,
        retries: int = 3,
        request_timeout: int = 30,
        retry_delay: float = 1.0,
        max_window_years: int = 1,
        session: Optional[requests.Session] = None,
        instruments_cache_file: Optional[Path] = None,
    ):
        super().__init__(data_dir=data_dir)

        if max_window_years < 1:
            raise ValueError("max_window_years must be >= 1")
        if retries < 1:
            raise ValueError("retries must be >= 1")

        self.default_exchange = (exchange or self.settings.default_exchange or "NSE").upper()
        self.retries = retries
        self.request_timeout = request_timeout
        self.retry_delay = retry_delay
        self.max_window_years = max_window_years
        self._session = session
        self.instruments_cache_file = instruments_cache_file or (
            self.settings.catalog_dir / "upstox_instruments.parquet"
        )
        self._instruments_df: Optional[pd.DataFrame] = None

        # Read credentials if configured (forward-compatibility for future V3 API)
        self.api_key = get_credential("upstox", "api_key")
        self.api_secret = get_credential("upstox", "api_secret")

    @property
    def name(self) -> str:
        return "upstox"

    @property
    def session(self) -> requests.Session:
        if self._session is None:
            self._session = requests.Session()
            self._session.headers.update({"Accept": "application/json"})
        return self._session

    def close(self) -> None:
        super().close()
        if self._session is not None:
            self._session.close()
            self._session = None

    def __enter__(self) -> "UpstoxProvider":
        self.connect()
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        self.close()

    # -------------------------------------------------------------------------
    # Authentication (V2 is open / unauthenticated; stubbed for future V3)
    # -------------------------------------------------------------------------

    def _authenticate(self) -> Dict[str, Any]:
        """
        Upstox V2 historical candles API is publicly accessible without credentials.
        Generates and caches an unauthenticated session record.
        Future V3 API integration will implement full OAuth flow.
        """
        token = {
            "access_token": "upstox_v2_public_access",
            "token_type": "None",
            "api_version": "v2",
            "expires_at": time.time() + (86400.0 * 365.0),
            "created_at": time.time(),
            "provider": self.name,
        }
        save_cached_token(self.tokens_dir, self.name, token)
        return token

    # -------------------------------------------------------------------------
    # Instrument Master Directory & Caching
    # -------------------------------------------------------------------------

    def list_instruments(self, force_refresh: bool = False) -> pd.DataFrame:
        """
        Return the Upstox instruments directory (equity and index instruments).
        Loads from cache if fresh (< 24 hours), or syncs from Upstox master feed.
        """
        if self._instruments_df is not None and not force_refresh:
            return self._instruments_df

        if not force_refresh and self.instruments_cache_file.exists():
            try:
                mtime = self.instruments_cache_file.stat().st_mtime
                if (time.time() - mtime) < 86400.0:
                    df = pd.read_parquet(self.instruments_cache_file)
                    self._instruments_df = df
                    UpstoxProvider._cached_instruments_df = df
                    return df
            except Exception as e:
                logger.warning(f"Could not load cached Upstox instruments: {e}. Refreshing...")

        return self.refresh_instruments()

    def refresh_instruments(self) -> pd.DataFrame:
        """
        Download complete instruments feed from Upstox, filter equities and indices, and cache.
        If network fails, falls back gracefully to cached file or empty DataFrame.
        """
        self.rate_limit()
        try:
            raw_df = pd.read_csv(self.INSTRUMENTS_URL)
            filtered = raw_df[raw_df["instrument_type"].isin(["EQUITY", "INDEX"])][
                ["instrument_key", "tradingsymbol", "exchange", "instrument_type", "name"]
            ].dropna(subset=["instrument_key", "tradingsymbol"]).copy()

            self.instruments_cache_file.parent.mkdir(parents=True, exist_ok=True)
            filtered.to_parquet(self.instruments_cache_file, index=False)
            self._instruments_df = filtered
            UpstoxProvider._cached_instruments_df = filtered
            return filtered
        except Exception as e:
            logger.warning(f"Could not download Upstox instruments master from {self.INSTRUMENTS_URL}: {e}")
            if self.instruments_cache_file.exists():
                try:
                    df = pd.read_parquet(self.instruments_cache_file)
                    self._instruments_df = df
                    UpstoxProvider._cached_instruments_df = df
                    return df
                except Exception:
                    pass

            empty = pd.DataFrame(columns=["instrument_key", "tradingsymbol", "exchange", "instrument_type", "name"])
            self._instruments_df = empty
            return empty

    def lookup_instrument(self, symbol: str, exchange: Optional[str] = None) -> Optional[str]:
        """
        Lookup Upstox instrument key by trading symbol and exchange.
        e.g. lookup_instrument('RELIANCE', 'NSE') -> 'NSE_EQ|INE002A01018'
             lookup_instrument('RELIANCE', 'BSE') -> 'BSE_EQ|INE002A01018'
        """
        clean_sym = symbol.strip().upper()
        if clean_sym.endswith("-EQ"):
            clean_sym = clean_sym[:-3]

        exch = (exchange or getattr(self, "default_exchange", "NSE")).upper()
        if exch.endswith("_EQ"):
            exch = exch[:-3]

        try:
            df = self.list_instruments()
            if not df.empty:
                match = df[(df["tradingsymbol"] == clean_sym) & (df["exchange"].str.startswith(exch))]
                if not match.empty:
                    return str(match["instrument_key"].iloc[0])
        except Exception as e:
            logger.debug(f"Error looking up instrument for {clean_sym} ({exch}): {e}")

        return None

    # -------------------------------------------------------------------------
    # Symbol & Date Helpers
    # -------------------------------------------------------------------------

    @staticmethod
    def _normalize_index_key(key: str) -> str:
        """Normalize known index key casing (e.g. NIFTY 50 -> Nifty 50, SENSEX -> SENSEX)."""
        if key.startswith("NSE_INDEX|") or key.startswith("BSE_INDEX|"):
            prefix, idx_name = key.split("|", 1)
            idx_upper = idx_name.upper().replace("_", " ")
            index_map = {
                "NIFTY 50": "Nifty 50",
                "NIFTY BANK": "Nifty Bank",
                "NIFTY IT": "Nifty IT",
                "NIFTY 100": "Nifty 100",
                "NIFTY 500": "Nifty 500",
                "NIFTY NEXT 50": "Nifty Next 50",
                "NIFTY MIDCAP 50": "Nifty Midcap 50",
                "NIFTY MIDCAP 100": "Nifty Midcap 100",
                "NIFTY SMALLCAP 100": "Nifty Smallcap 100",
                "SENSEX": "SENSEX",
                "BANKEX": "BANKEX",
            }
            if idx_upper in index_map:
                return f"{prefix}|{index_map[idx_upper]}"
        return key

    def format_instrument_key(self, symbol: str, exchange: Optional[str] = None) -> str:
        """
        Normalize symbol / instrument name into an Upstox instrument key.
        Supports:
        - Trading symbol like Fyers (e.g. 'RELIANCE' -> 'NSE_EQ|INE002A01018')
        - Explicit exchange prefixes (e.g. 'BSE:RELIANCE' -> 'BSE_EQ|INE002A01018')
        - Index aliases (e.g. 'NIFTY' -> 'NSE_INDEX|Nifty 50', 'SENSEX' -> 'BSE_INDEX|SENSEX')
        - Direct Upstox instrument keys (e.g. 'NSE_EQ|INE002A01018')
        Default exchange is NSE if unspecified.
        """
        sym = symbol.strip().upper()

        # 1. Already an Upstox instrument key containing '|'
        if "|" in sym:
            return self._normalize_index_key(sym)

        # 2. Check if whole token matches an index alias (e.g. 'NIFTY', 'SENSEX')
        clean_idx = sym.replace("_", " ").strip()
        if clean_idx in self.INDEX_ALIASES:
            return self.INDEX_ALIASES[clean_idx]

        # 3. Parse explicit exchange prefix (e.g. 'BSE:RELIANCE', 'NSE:TCS')
        target_exch = exchange or getattr(self, "default_exchange", "NSE")
        if ":" in sym:
            parts = sym.split(":", 1)
            target_exch = parts[0]
            sym = parts[1]

        # Check if the extracted symbol matches an index alias (e.g. 'BSE:SENSEX')
        if sym in self.INDEX_ALIASES:
            return self.INDEX_ALIASES[sym]

        # Strip standard equity suffix if present (e.g. 'RELIANCE-EQ' -> 'RELIANCE')
        if sym.endswith("-EQ"):
            sym = sym[:-3]

        # 4. Handle known segment prefixes with '_' (e.g. 'NSE_EQ_INE002A01018')
        for prefix in ("NSE_EQ_", "BSE_EQ_", "NSE_INDEX_", "BSE_INDEX_", "MCX_FO_"):
            if sym.startswith(prefix):
                key = prefix[:-1] + "|" + sym[len(prefix):]
                return self._normalize_index_key(key)

        # 5. Lookup trading symbol in Upstox instruments directory
        matched_key = self.lookup_instrument(sym, exchange=target_exch)
        if matched_key:
            return matched_key

        # 6. Fallback format
        exch_clean = target_exch.upper().replace("_EQ", "")
        return f"{exch_clean}_EQ|{sym}"

    @staticmethod
    def _add_years(value: date, years: int) -> date:
        """Add calendar years safely, including leap days."""
        try:
            return value.replace(year=value.year + years)
        except ValueError:
            return value.replace(year=value.year + years, day=28)

    def _date_chunks(
        self,
        start: date,
        end: date,
    ) -> Generator[Tuple[date, date], None, None]:
        """Yield inclusive date ranges within the configured limit."""
        current = start
        while current <= end:
            next_boundary = self._add_years(current, self.max_window_years)
            chunk_end = min(end, next_boundary - timedelta(days=1))
            yield current, chunk_end
            current = chunk_end + timedelta(days=1)

    def _build_url(
        self,
        instrument_key: str,
        start: date,
        end: date,
    ) -> str:
        """
        Build Upstox V2 historical-candle URL:
        https://api.upstox.com/v2/historical-candle/{instrument_key}/day/{to_date}/{from_date}
        Note that Upstox V2 URL path expects {to_date}/{from_date}.
        """
        formatted_key = self.format_instrument_key(instrument_key)
        instrument = quote(formatted_key, safe="")
        return f"{self.BASE_URL}/{instrument}/day/{end.isoformat()}/{start.isoformat()}"

    def _request_candles(
        self,
        instrument_key: str,
        start: date,
        end: date,
    ) -> list:
        """Fetch one date chunk, retrying transient HTTP and network failures."""
        url = self._build_url(instrument_key, start, end)

        for attempt in range(self.retries):
            self.rate_limit()
            try:
                response = self.session.get(url, timeout=self.request_timeout)

                if response.status_code == 429 or response.status_code in (500, 502, 503, 504):
                    if attempt < self.retries - 1:
                        retry_after = response.headers.get("Retry-After")
                        try:
                            delay = float(retry_after) if retry_after else self.retry_delay * (2 ** attempt)
                        except (TypeError, ValueError):
                            delay = self.retry_delay * (2 ** attempt)

                        time.sleep(max(0.0, delay))
                        continue

                response.raise_for_status()
                payload = response.json()

                if payload.get("status") != "success":
                    raise RuntimeError(f"Upstox returned an API error: {payload}")

                return payload.get("data", {}).get("candles", [])

            except (requests.RequestException, ValueError) as exc:
                if attempt == self.retries - 1:
                    raise RuntimeError(
                        f"Request failed for {instrument_key}: {start} to {end}"
                    ) from exc

                time.sleep(self.retry_delay * (2 ** attempt))

        raise RuntimeError("Retry attempts exhausted")

    def _normalize(
        self,
        candles: list,
        start: Optional[date] = None,
        end: Optional[date] = None,
    ) -> pd.DataFrame:
        """Normalize, validate, deduplicate, and convert candles to canonical schema."""
        if not candles:
            return pd.DataFrame(columns=CANONICAL_COLUMNS)

        df = pd.DataFrame(candles, columns=self.CANDLE_COLUMNS)

        # Parse timestamp: Upstox format is 'YYYY-MM-DDTHH:MM:SS+05:30'
        # Extract the calendar date to anchor daily bars at 00:00:00 UTC
        ts_dates = pd.to_datetime(df["timestamp"].astype(str).str.slice(0, 10), utc=True, errors="coerce")
        df["ts"] = ts_dates

        numeric_columns = ["open", "high", "low", "close", "volume"]
        for column in numeric_columns:
            df[column] = pd.to_numeric(df[column], errors="coerce").astype("float64")

        # Filter to the requested date range if provided
        if start is not None:
            df = df.loc[df["ts"].dt.date >= start]
        if end is not None:
            df = df.loc[df["ts"].dt.date <= end]

        if df.empty:
            return pd.DataFrame(columns=CANONICAL_COLUMNS)

        df = df.drop_duplicates(subset=["ts"], keep="last")
        df = df.sort_values("ts").reset_index(drop=True)

        df["provider"] = self.name
        df["ingested_at"] = datetime.now(timezone.utc)

        return df[CANONICAL_COLUMNS]

    # -------------------------------------------------------------------------
    # Core BaseProvider Operations
    # -------------------------------------------------------------------------

    @auth_retry_on_401
    def latest_available(self, symbol: str, timeframe: str) -> Optional[datetime]:
        """
        Return the latest timestamp available from Upstox for symbol and timeframe (UTC).
        Queries recent history (last 14 days) to find the latest trading day bar.
        Only '1d' timeframe is supported in V2 API.
        """
        norm_tf = normalize_timeframe(timeframe)
        if norm_tf != "1d":
            raise ValueError(
                f"Upstox provider (V2 API) only supports daily data ('1d'). Got '{timeframe}'."
            )

        now_utc = datetime.now(timezone.utc)
        end_date = now_utc.date()
        start_date = end_date - timedelta(days=14)

        try:
            candles = self._request_candles(symbol, start=start_date, end=end_date)
            if candles:
                df = self._normalize(candles)
                if not df.empty:
                    return df["ts"].max().to_pydatetime()
        except Exception as e:
            logger.warning(f"Error querying latest available data from Upstox for {symbol}: {e}")

        # Fallback to current UTC midnight
        return now_utc.replace(hour=0, minute=0, second=0, microsecond=0)

    @auth_retry_on_401
    def fetch(
        self,
        symbol: str,
        timeframe: str,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
    ) -> pd.DataFrame:
        """
        Fetch daily market data and return canonical-schema DataFrame.
        Columns: ts, open, high, low, close, volume, provider, ingested_at
        Only '1d' timeframe is supported in V2 API.
        """
        norm_tf = normalize_timeframe(timeframe)
        if norm_tf != "1d":
            raise ValueError(
                f"Upstox provider (V2 API) only supports daily data ('1d'). Got '{timeframe}'."
            )

        now_utc = datetime.now(timezone.utc)
        if end is None:
            end_dt = now_utc
        elif end.tzinfo is None:
            end_dt = end.replace(tzinfo=timezone.utc)
        else:
            end_dt = end.astimezone(timezone.utc)

        if start is None:
            start_dt = end_dt - timedelta(days=30)
        elif start.tzinfo is None:
            start_dt = start.replace(tzinfo=timezone.utc)
        else:
            start_dt = start.astimezone(timezone.utc)

        start_date = start_dt.date()
        end_date = end_dt.date()

        if start_date > end_date:
            return pd.DataFrame(columns=CANONICAL_COLUMNS)

        all_candles: List[Any] = []
        for chunk_start, chunk_end in self._date_chunks(start_date, end_date):
            chunk_candles = self._request_candles(symbol, chunk_start, chunk_end)
            if chunk_candles:
                all_candles.extend(chunk_candles)

        if not all_candles:
            return pd.DataFrame(columns=CANONICAL_COLUMNS)

        return self._normalize(all_candles, start_date, end_date)

    def health_check(self) -> Dict[str, Any]:
        """Verify Upstox provider configuration and endpoint connectivity."""
        has_api_key = bool(self.api_key)
        has_api_secret = bool(self.api_secret)

        api_ok = False
        try:
            url = self._build_url("NSE_INDEX|Nifty 50", date(2024, 1, 1), date(2024, 1, 2))
            resp = self.session.get(url, timeout=min(5, self.request_timeout))
            api_ok = (resp.status_code == 200)
        except Exception:
            api_ok = False

        return {
            "provider": self.name,
            "status": "ok" if api_ok else "unreachable",
            "api_version": "v2",
            "auth_required": False,
            "has_api_key": has_api_key,
            "has_api_secret": has_api_secret,
            "base_url": self.BASE_URL,
            "supported_timeframes": ["1d"],
        }
