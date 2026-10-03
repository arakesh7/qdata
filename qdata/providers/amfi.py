import http.client
import logging
import os
import random
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from qdata.auth import save_cached_token
from qdata.providers.base import BaseProvider, auth_retry_on_401
from qdata.store import CANONICAL_COLUMNS, normalize_timeframe

logger = logging.getLogger(__name__)

SCHEMES_URL = "https://www.amfiindia.com/spages/NAVAll.txt"
NAV_HISTORY_URL = "https://www.amfiindia.com/api/nav-history"

DEFAULT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json, text/plain, */*",
}

# Network resiliency defaults for AMFI
DEFAULT_TIMEOUT: Tuple[float, float] = (15.0, 60.0)  # (connect_timeout, read_timeout)
DEFAULT_MAX_RETRIES: int = 4
DEFAULT_BACKOFF_FACTOR: float = 1.5
DEFAULT_CHUNK_DAYS: int = 730  # 2 years per chunk to avoid AMFI gateway timeouts
DEFAULT_RATE_LIMIT_DELAY: float = 0.3  # Polite pacing delay in seconds



def split_into_date_pairs(
    start_date: Union[str, datetime],
    end_date: Union[str, datetime],
    n_days: int = 1800,
) -> List[Tuple[str, str]]:
    """
    Splits the range between start_date and end_date into intervals of at most n_days.
    AMFI NAV history API enforces a maximum 5-year interval per request (~1825 days).
    We default to 1800 days to account safely for leap year boundaries.

    Returns a list of (from_date_str, to_date_str) pairs formatted as 'YYYY-MM-DD'.
    """
    if isinstance(start_date, str):
        curr_start = datetime.strptime(start_date, "%Y-%m-%d")
    elif isinstance(start_date, datetime):
        curr_start = start_date.replace(hour=0, minute=0, second=0, microsecond=0)
        if curr_start.tzinfo is not None:
            curr_start = curr_start.astimezone(timezone.utc).replace(tzinfo=None)
    else:
        raise TypeError(f"Unsupported start_date type: {type(start_date)}")

    if isinstance(end_date, str):
        target_end = datetime.strptime(end_date, "%Y-%m-%d")
    elif isinstance(end_date, datetime):
        target_end = end_date.replace(hour=0, minute=0, second=0, microsecond=0)
        if target_end.tzinfo is not None:
            target_end = target_end.astimezone(timezone.utc).replace(tzinfo=None)
    else:
        raise TypeError(f"Unsupported end_date type: {type(end_date)}")

    if curr_start > target_end:
        return []

    pairs: List[Tuple[str, str]] = []
    while curr_start <= target_end:
        curr_end = min(curr_start + timedelta(days=n_days - 1), target_end)
        pairs.append((curr_start.strftime("%Y-%m-%d"), curr_end.strftime("%Y-%m-%d")))
        curr_start = curr_end + timedelta(days=1)

    return pairs


def build_scheme_name(base_name: str, plan: str, option: str) -> str:
    """
    Build a clean human-readable mutual fund scheme name from components.
    Filters out empty strings and bare hyphens.
    """
    parts = [p.strip() for p in [base_name, plan, option] if p and p.strip() and p.strip() != "-"]
    return " - ".join(parts) if parts else base_name.strip()


class AmfiProvider(BaseProvider):
    """
    AMFI (Association of Mutual Funds in India) Data Provider.

    Pulls public mutual fund master directory from NAVAll.txt feed
    and historical daily NAV time series from AMFI API.

    Conforms to the BaseProvider contract:
    - Canonical columns: ts, open, high, low, close, volume, provider, ingested_at
    - Timeframe: '1d' (daily NAV)
    - Public access (no API keys required)
    """

    _SCHEME_COLUMNS = [
        "scheme_code",
        "isin_growth",
        "isin_reinv",
        "scheme_name",
        "plan",
        "option",
        "nav",
        "date",
        "fund_house",
    ]

    def __init__(
        self,
        data_dir: Optional[Path] = None,
        schemes_cache_file: Optional[Path] = None,
        session: Optional[requests.Session] = None,
        timeout: Optional[Union[float, Tuple[float, float]]] = None,
        max_retries: Optional[int] = None,
        backoff_factor: Optional[float] = None,
        chunk_days: Optional[int] = None,
        rate_limit_delay: Optional[float] = None,
    ):
        super().__init__(data_dir=data_dir)
        self.timeout = timeout if timeout is not None else self._resolve_timeout()
        self.max_retries = max_retries if max_retries is not None else int(
            os.environ.get("QDATA_AMFI_MAX_RETRIES", DEFAULT_MAX_RETRIES)
        )
        self.backoff_factor = backoff_factor if backoff_factor is not None else float(
            os.environ.get("QDATA_AMFI_BACKOFF_FACTOR", DEFAULT_BACKOFF_FACTOR)
        )
        self.chunk_days = chunk_days if chunk_days is not None else int(
            os.environ.get("QDATA_AMFI_CHUNK_DAYS", DEFAULT_CHUNK_DAYS)
        )
        self.rate_limit_delay = rate_limit_delay if rate_limit_delay is not None else float(
            os.environ.get("QDATA_AMFI_RATE_LIMIT_DELAY", DEFAULT_RATE_LIMIT_DELAY)
        )

        self.session = session or requests.Session()
        self.session.headers.update(DEFAULT_HEADERS)
        self._configure_session_retries(self.session)

        self.schemes_cache_file = schemes_cache_file or (self.settings.catalog_dir / "amfi_schemes.parquet")
        self._schemes_df: Optional[pd.DataFrame] = None

    @staticmethod
    def _resolve_timeout() -> Union[float, Tuple[float, float]]:
        env_timeout = os.environ.get("QDATA_AMFI_TIMEOUT")
        if env_timeout:
            try:
                val = float(env_timeout)
                return (15.0, val)
            except ValueError:
                pass
        return DEFAULT_TIMEOUT

    def _configure_session_retries(self, session: requests.Session) -> None:
        """Mount HTTPAdapter with transport-level retries on the requests session."""
        try:
            retry_strategy = Retry(
                total=self.max_retries,
                backoff_factor=self.backoff_factor,
                status_forcelist=[429, 500, 502, 503, 504],
                allowed_methods=["HEAD", "GET", "OPTIONS"],
                raise_on_status=False,
            )
            adapter = HTTPAdapter(
                max_retries=retry_strategy,
                pool_connections=10,
                pool_maxsize=10,
            )
            session.mount("https://", adapter)
            session.mount("http://", adapter)
        except Exception as e:
            logger.debug(f"Could not mount Retry adapter on session: {e}")

    def rate_limit(self) -> None:
        """Enforce polite rate limiting for AMFI endpoints to prevent throttling or dropped sockets."""
        now = time.time()
        elapsed = now - self._rate_limit_last_called
        if elapsed < self.rate_limit_delay:
            time.sleep(self.rate_limit_delay - elapsed)
        self._rate_limit_last_called = time.time()

    @property
    def name(self) -> str:
        return "amfi"

    def _authenticate(self) -> Dict[str, Any]:
        """
        AMFI provides open public data without credential requirements.
        Generates and caches a long-lived local token structure.
        """
        token = {
            "access_token": "amfi_public_access",
            "token_type": "None",
            "expires_at": time.time() + (86400.0 * 365.0),
            "created_at": time.time(),
            "provider": self.name,
        }
        save_cached_token(self.tokens_dir, self.name, token)
        return token

    # -------------------------------------------------------------------------
    # Schemes Master Feed (NAVAll.txt) Operations
    # -------------------------------------------------------------------------

    def list_all_schemes(self, force_refresh: bool = False) -> pd.DataFrame:
        """
        Return all mutual fund schemes from AMFI master directory.
        Loads from cache if available and not stale (> 24 hours), or syncs from AMFI.
        """
        if self._schemes_df is not None and not force_refresh:
            return self._schemes_df

        if not force_refresh and self.schemes_cache_file.exists():
            try:
                # Check modification age (fresh within 24h)
                mtime = self.schemes_cache_file.stat().st_mtime
                if (time.time() - mtime) < 86400.0:
                    df = pd.read_parquet(self.schemes_cache_file)
                    self._schemes_df = df
                    return df
            except Exception as e:
                logger.warning(f"Could not load cached AMFI schemes parquet: {e}. Refreshing...")

        return self.refresh_schemes()

    def refresh_schemes(self) -> pd.DataFrame:
        """
        Force refresh schemes master list from AMFI NAVAll.txt and cache to disk.
        If network fails and a cache file exists, falls back gracefully to cached data.
        """
        self.rate_limit()
        try:
            raw_lines = self._fetch_raw_nav_lines()
        except Exception as e:
            if self.schemes_cache_file.exists():
                logger.warning(
                    f"Network error refreshing schemes from AMFI ({e}). Falling back to existing cached schemes file."
                )
                try:
                    df = pd.read_parquet(self.schemes_cache_file)
                    self._schemes_df = df
                    return df
                except Exception:
                    pass
            raise

        current_fund_house: Optional[str] = None
        records: List[List[Any]] = []

        for line in raw_lines:
            line = line.strip()
            if not line:
                continue

            # Header row check
            if line.startswith("Scheme Code;"):
                continue

            # Fund house header lines (e.g. "HDFC Mutual Fund", "Axis Mutual Fund")
            if line.endswith("Mutual Fund"):
                current_fund_house = line
                continue

            if ";" not in line:
                continue

            fields = [field.strip() for field in line.split(";")]
            record = self._parse_scheme_row(fields, current_fund_house or "")
            if record:
                records.append(record)

        df = pd.DataFrame(records, columns=self._SCHEME_COLUMNS)
        self._schemes_df = df

        try:
            self.schemes_cache_file.parent.mkdir(parents=True, exist_ok=True)
            df.to_parquet(self.schemes_cache_file, index=False)
        except Exception as e:
            logger.warning(f"Failed to cache AMFI schemes to parquet: {e}")

        return df

    def _fetch_raw_nav_lines(self) -> List[str]:
        """Fetch raw lines of NAVAll.txt feed with retry and exponential backoff."""
        last_err: Optional[Exception] = None
        for attempt in range(1, self.max_retries + 1):
            try:
                self.rate_limit()
                resp = self.session.get(SCHEMES_URL, timeout=self.timeout)
                resp.raise_for_status()
                return resp.content.decode("utf-8", errors="replace").splitlines()
            except Exception as e:
                last_err = e
                if attempt < self.max_retries:
                    delay = self.backoff_factor * (2 ** (attempt - 1)) + random.uniform(0.1, 0.5)
                    logger.warning(
                        f"Failed to fetch AMFI schemes master feed ({e}). Retrying in {delay:.1f}s (attempt {attempt}/{self.max_retries})..."
                    )
                    time.sleep(delay)
                else:
                    logger.error(f"Exhausted retries fetching AMFI schemes master feed: {e}")
        if last_err:
            raise last_err
        return []

    def _parse_scheme_row(self, fields: List[str], fund_house: str) -> Optional[List[Any]]:
        """
        Parse a single semicolon-split AMFI line.
        Handles both current 8-field format and legacy 6-field format.
        """
        MIN_FIELDS = 5
        if len(fields) < MIN_FIELDS:
            return None

        scheme_code = fields[0]
        if not scheme_code.isdigit():
            return None

        isin_growth, isin_reinv = fields[1], fields[2]

        if len(fields) >= 8:
            # Current AMFI format (8 fields)
            base_name, plan, option = fields[3], fields[4], fields[5]
            nav, date_str = fields[6], fields[7]
            scheme_name = build_scheme_name(base_name, plan, option)
        else:
            # Legacy AMFI format (6 fields)
            base_name, plan, option = fields[3], "", ""
            nav, date_str = fields[-2], fields[-1]
            scheme_name = base_name

        return [scheme_code, isin_growth, isin_reinv, scheme_name, plan, option, nav, date_str, fund_house]

    def get_fund_houses(self) -> List[str]:
        """Return sorted list of distinct fund houses."""
        schemes = self.list_all_schemes()
        fh = schemes["fund_house"].dropna().unique().tolist()
        return sorted([f for f in fh if f])

    def get_schemes_by_fund_house(self, fund_house: str) -> pd.DataFrame:
        """Filter schemes by fund house name (case-insensitive substring match)."""
        schemes = self.list_all_schemes()
        fh_lower = fund_house.lower()
        mask = schemes["fund_house"].str.lower().str.contains(fh_lower, na=False)
        return schemes[mask].reset_index(drop=True)

    def search_schemes(self, query: str) -> pd.DataFrame:
        """
        Search schemes by keyword across scheme_name, scheme_code, and ISINs.
        """
        schemes = self.list_all_schemes()
        q = query.strip().lower()
        mask = (
            schemes["scheme_name"].str.lower().str.contains(q, na=False)
            | schemes["scheme_code"].str.lower().str.contains(q, na=False)
            | schemes["isin_growth"].str.lower().str.contains(q, na=False)
            | schemes["isin_reinv"].str.lower().str.contains(q, na=False)
        )
        return schemes[mask].reset_index(drop=True)

    def resolve_scheme_code(self, symbol: str) -> str:
        """
        Resolve a user-provided symbol to an AMFI numeric scheme code.
        Accepts:
        - Direct numeric scheme code (e.g. '120503')
        - ISIN string (e.g. 'INF846K01WO1')
        """
        sym = symbol.strip().upper()
        if sym.isdigit():
            return sym

        try:
            schemes = self.list_all_schemes()
            match = schemes[(schemes["isin_growth"] == sym) | (schemes["isin_reinv"] == sym)]
            if not match.empty:
                return str(match.iloc[0]["scheme_code"])
        except Exception as e:
            logger.debug(f"Could not resolve symbol '{sym}' from schemes cache: {e}")

        return sym

    def get_scheme_info(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Return scheme master record metadata if known."""
        code = self.resolve_scheme_code(symbol)
        try:
            schemes = self.list_all_schemes()
            match = schemes[schemes["scheme_code"] == code]
            if not match.empty:
                return match.iloc[0].to_dict()
        except Exception:
            pass
        return None

    # -------------------------------------------------------------------------
    # BaseProvider Abstract Methods Implementation
    # -------------------------------------------------------------------------

    @auth_retry_on_401
    def latest_available(self, symbol: str, timeframe: str) -> Optional[datetime]:
        """
        Return the latest timestamp available for mutual fund scheme and timeframe (UTC).
        Mutual funds publish NAV at daily frequency ('1d').
        """
        self.rate_limit()
        norm_tf = normalize_timeframe(timeframe)
        if norm_tf != "1d":
            logger.warning(f"AMFI only provides daily NAV data ('1d'), requested '{timeframe}'.")
            return None

        scheme_code = self.resolve_scheme_code(symbol)

        # First check in master schemes list cache
        try:
            info = self.get_scheme_info(scheme_code)
            if info and info.get("date"):
                dt = pd.to_datetime(info["date"], format="%d-%b-%Y", errors="coerce", utc=True)
                if pd.notna(dt):
                    return dt.to_pydatetime()
        except Exception as e:
            logger.debug(f"Could not read latest date from scheme cache: {e}")

        # Fallback: query historical records for recent 30-day window
        now_utc = datetime.now(timezone.utc)
        from_str = (now_utc - timedelta(days=30)).strftime("%Y-%m-%d")
        to_str = now_utc.strftime("%Y-%m-%d")

        try:
            records = self._fetch_historical_records(scheme_code, from_str, to_str)
            if records:
                latest_rec = max(records, key=lambda r: r.get("date", ""))
                dt = pd.to_datetime(latest_rec["date"], format="%Y-%m-%d", errors="coerce", utc=True)
                if pd.notna(dt):
                    return dt.to_pydatetime()
        except Exception as e:
            logger.warning(f"Failed to query recent AMFI history for {scheme_code}: {e}")

        return None

    @auth_retry_on_401
    def fetch(
        self,
        symbol: str,
        timeframe: str,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
    ) -> pd.DataFrame:
        """
        Fetch market data and return canonical-schema DataFrame.
        Columns: ts, open, high, low, close, volume, provider, ingested_at

        Splits requests spanning > 5 years into safe sub-intervals of 1800 days.
        """
        self.rate_limit()
        norm_tf = normalize_timeframe(timeframe)
        if norm_tf != "1d":
            raise ValueError(
                f"AMFI provider only supports daily ('1d') timeframe for mutual fund NAVs, got '{timeframe}'."
            )

        scheme_code = self.resolve_scheme_code(symbol)

        now_utc = datetime.now(timezone.utc)
        target_end = end or now_utc
        if target_end.tzinfo is None:
            target_end = target_end.replace(tzinfo=timezone.utc)
        else:
            target_end = target_end.astimezone(timezone.utc)

        target_start = start or (target_end - timedelta(days=365))
        if target_start.tzinfo is None:
            target_start = target_start.replace(tzinfo=timezone.utc)
        else:
            target_start = target_start.astimezone(timezone.utc)

        if target_start > target_end:
            return pd.DataFrame(columns=CANONICAL_COLUMNS)

        start_str = target_start.strftime("%Y-%m-%d")
        end_str = target_end.strftime("%Y-%m-%d")

        date_pairs = split_into_date_pairs(start_str, end_str, n_days=self.chunk_days)
        nav_records: List[Dict[str, Any]] = []

        total_chunks = len(date_pairs)
        if total_chunks > 1:
            logger.info(
                f"Fetching AMFI NAV history for scheme {scheme_code} from {start_str} to {end_str} "
                f"in {total_chunks} chunks ({self.chunk_days} days/chunk)..."
            )

        for idx, (from_date, to_date) in enumerate(date_pairs, 1):
            if total_chunks > 1:
                logger.info(
                    f"Fetching chunk {idx}/{total_chunks}: {from_date} to {to_date}..."
                )
            self.rate_limit()
            records = self._fetch_historical_records(scheme_code, from_date, to_date)
            nav_records.extend(records)

        if not nav_records:
            return pd.DataFrame(columns=CANONICAL_COLUMNS)

        ingested_now = datetime.now(timezone.utc)
        rows: List[Dict[str, Any]] = []

        for rec in nav_records:
            d_str = rec.get("date")
            nav_val = rec.get("nav")
            if not d_str or nav_val is None:
                continue

            try:
                price = float(nav_val)
            except (ValueError, TypeError):
                continue

            if price <= 0:
                continue

            ts = pd.to_datetime(d_str, format="%Y-%m-%d", utc=True, errors="coerce")
            if pd.isna(ts):
                continue

            rows.append({
                "ts": ts,
                "open": price,
                "high": price,
                "low": price,
                "close": price,
                "volume": 0.0,
                "provider": self.name,
                "ingested_at": ingested_now,
            })

        if not rows:
            return pd.DataFrame(columns=CANONICAL_COLUMNS)

        df = pd.DataFrame(rows)
        # Deduplicate by timestamp (latest wins) and sort chronologically
        df = df.drop_duplicates(subset=["ts"], keep="last").sort_values("ts").reset_index(drop=True)

        # Filter strictly within [target_start, target_end]
        mask = (df["ts"] >= target_start) & (df["ts"] <= target_end)
        df_filtered = df[mask].reset_index(drop=True)

        return df_filtered[CANONICAL_COLUMNS]

    def _fetch_historical_records(
        self,
        scheme_id: str,
        from_date: str,
        to_date: str,
        current_depth: int = 0,
    ) -> List[Dict[str, Any]]:
        """
        Call AMFI nav-history API for a date range with automatic retries,
        exponential backoff, and adaptive chunk subdivision if requests time out.
        """
        params = {
            "query_type": "historical_period",
            "sd_id": scheme_id,
            "from_date": from_date,
            "to_date": to_date,
        }

        last_err: Optional[Exception] = None
        for attempt in range(1, self.max_retries + 1):
            self.rate_limit()
            try:
                resp = self.session.get(NAV_HISTORY_URL, params=params, timeout=self.timeout)
                if resp.status_code == 200:
                    raw_json = resp.json()
                    if "data" in raw_json and "nav_groups" in raw_json["data"]:
                        groups = raw_json["data"]["nav_groups"]
                        all_records: List[Dict[str, Any]] = []
                        for group in groups:
                            records = group.get("historical_records", [])
                            all_records.extend(records)
                        return all_records
                    return []
                elif resp.status_code in (400, 404):
                    # e.g. "No records to display" or fund not active during this window
                    logger.debug(
                        f"AMFI historical API returned {resp.status_code} for {scheme_id} ({from_date} to {to_date})"
                    )
                    return []
                elif resp.status_code in (429, 500, 502, 503, 504):
                    if attempt < self.max_retries:
                        delay = self.backoff_factor * (2 ** (attempt - 1)) + random.uniform(0.1, 0.5)
                        logger.warning(
                            f"AMFI historical API returned status {resp.status_code} for {scheme_id} "
                            f"({from_date} to {to_date}). Retrying in {delay:.1f}s (attempt {attempt}/{self.max_retries})..."
                        )
                        time.sleep(delay)
                        continue
                    resp.raise_for_status()
                else:
                    resp.raise_for_status()
                    return []
            except (requests.exceptions.RequestException, ConnectionResetError, http.client.RemoteDisconnected) as e:
                last_err = e
                if attempt < self.max_retries:
                    delay = self.backoff_factor * (2 ** (attempt - 1)) + random.uniform(0.1, 0.5)
                    logger.warning(
                        f"Network issue fetching AMFI historical NAV for {scheme_id} "
                        f"({from_date} to {to_date}): {e}. Retrying in {delay:.1f}s (attempt {attempt}/{self.max_retries})..."
                    )
                    time.sleep(delay)
                    continue

        # If retries exhausted and the date span is larger than 60 days,
        # adaptively subdivide into two smaller requests!
        try:
            from_dt = datetime.strptime(from_date, "%Y-%m-%d")
            to_dt = datetime.strptime(to_date, "%Y-%m-%d")
            span_days = (to_dt - from_dt).days
        except Exception:
            span_days = 0

        if span_days > 60 and current_depth < 3:
            mid_days = span_days // 2
            mid_dt_1 = from_dt + timedelta(days=mid_days)
            mid_dt_2 = mid_dt_1 + timedelta(days=1)
            mid_str_1 = mid_dt_1.strftime("%Y-%m-%d")
            mid_str_2 = mid_dt_2.strftime("%Y-%m-%d")
            logger.info(
                f"Adaptive chunk subdivision for scheme {scheme_id}: splitting {from_date}..{to_date} "
                f"into {from_date}..{mid_str_1} and {mid_str_2}..{to_date} due to network timeout/error."
            )
            rec1 = self._fetch_historical_records(scheme_id, from_date, mid_str_1, current_depth=current_depth + 1)
            rec2 = self._fetch_historical_records(scheme_id, mid_str_2, to_date, current_depth=current_depth + 1)
            return rec1 + rec2

        logger.error(
            f"Failed to fetch historical NAV from AMFI for {scheme_id} ({from_date} to {to_date}) "
            f"after {self.max_retries} attempts: {last_err}"
        )
        if last_err:
            raise last_err
        return []

    def health_check(self) -> Dict[str, Any]:
        """
        Verify connectivity to AMFI endpoints.
        """
        try:
            resp = self.session.get(
                NAV_HISTORY_URL,
                params={"query_type": "historical_period", "sd_id": "120503", "from_date": "2024-01-01", "to_date": "2024-01-05"},
                timeout=self.timeout,
            )
            api_ok = (resp.status_code == 200)
        except Exception:
            api_ok = False

        has_cache = self.schemes_cache_file.exists()
        status = "ok" if api_ok else "unreachable"

        return {
            "status": status,
            "provider": self.name,
            "connected": self.connected,
            "nav_history_api": "ok" if api_ok else "failed",
            "schemes_cached": has_cache,
            "cache_file": str(self.schemes_cache_file),
        }

    @classmethod
    def get_cli_app(cls) -> Any:
        """
        Dynamically construct and return the Typer sub-app for AMFI-specific CLI commands.
        """
        import typer
        from rich.console import Console
        from rich.table import Table

        sub_app = typer.Typer(
            name="amfi",
            help="Search and inspect AMFI mutual fund schemes master feed.",
            no_args_is_help=True,
        )
        console = Console()

        @sub_app.command(name="search")
        def search_cmd(
            query: str = typer.Argument(..., help="Search query (e.g. 'Axis Bluechip', '120503', 'INF846K01WO1')"),
            limit: int = typer.Option(20, "--limit", help="Max number of results to display."),
            data_dir: Optional[Path] = typer.Option(None, "--data-dir", help="Path to data directory."),
            json_out: bool = typer.Option(False, "--json", help="Output JSON format."),
        ):
            """Search AMFI mutual fund schemes by name, scheme code, or ISIN."""
            p = cls(data_dir=data_dir)
            df = p.search_schemes(query)
            total_count = len(df)
            if limit > 0:
                df = df.head(limit)

            records = df.to_dict(orient="records")
            if json_out:
                import json
                print(json.dumps(records, indent=2))
                raise typer.Exit(code=0)

            table = Table(title=f"AMFI Schemes Matching '{query}' (Found {total_count})")
            table.add_column("Scheme Code", style="cyan bold")
            table.add_column("Scheme Name", style="white")
            table.add_column("Fund House", style="dim")
            table.add_column("Latest NAV", justify="right", style="green")
            table.add_column("Date", style="magenta")

            for r in records:
                table.add_row(
                    str(r.get("scheme_code")),
                    str(r.get("scheme_name")),
                    str(r.get("fund_house", "")),
                    str(r.get("nav", "")),
                    str(r.get("date", "")),
                )
            console.print(table)
            raise typer.Exit(code=0)

        @sub_app.command(name="fund-houses")
        def fund_houses_cmd(
            data_dir: Optional[Path] = typer.Option(None, "--data-dir", help="Path to data directory."),
            json_out: bool = typer.Option(False, "--json", help="Output JSON format."),
        ):
            """List all mutual fund houses available in AMFI."""
            p = cls(data_dir=data_dir)
            houses = p.get_fund_houses()

            if json_out:
                import json
                print(json.dumps(houses, indent=2))
                raise typer.Exit(code=0)

            table = Table(title=f"AMFI Fund Houses ({len(houses)})")
            table.add_column("#", justify="right", style="dim")
            table.add_column("Fund House Name", style="cyan bold")
            for i, h in enumerate(houses, 1):
                table.add_row(str(i), h)
            console.print(table)
            raise typer.Exit(code=0)

        @sub_app.command(name="schemes")
        def schemes_cmd(
            refresh: bool = typer.Option(False, "--refresh", help="Force re-fetch schemes from AMFI."),
            fund_house: Optional[str] = typer.Option(None, "--fund-house", help="Filter by fund house."),
            limit: int = typer.Option(50, "--limit", help="Max number of schemes to display."),
            data_dir: Optional[Path] = typer.Option(None, "--data-dir", help="Path to data directory."),
            json_out: bool = typer.Option(False, "--json", help="Output JSON format."),
        ):
            """List schemes or refresh AMFI master schemes directory cache."""
            p = cls(data_dir=data_dir)
            if refresh:
                console.print("[yellow]Refreshing schemes from AMFI NAVAll.txt feed...[/yellow]")
                df = p.refresh_schemes()
            elif fund_house:
                df = p.get_schemes_by_fund_house(fund_house)
            else:
                df = p.list_all_schemes()

            total_count = len(df)
            if limit > 0:
                df = df.head(limit)

            records = df.to_dict(orient="records")
            if json_out:
                import json
                print(json.dumps(records, indent=2))
                raise typer.Exit(code=0)

            table = Table(title=f"AMFI Schemes (Showing {len(df)} of {total_count})")
            table.add_column("Scheme Code", style="cyan bold")
            table.add_column("Scheme Name", style="white")
            table.add_column("Fund House", style="dim")
            table.add_column("Latest NAV", justify="right", style="green")
            table.add_column("Date", style="magenta")

            for r in records:
                table.add_row(
                    str(r.get("scheme_code")),
                    str(r.get("scheme_name")),
                    str(r.get("fund_house", "")),
                    str(r.get("nav", "")),
                    str(r.get("date", "")),
                )
            console.print(table)
            raise typer.Exit(code=0)

        return sub_app

