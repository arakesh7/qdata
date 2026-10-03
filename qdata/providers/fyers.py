import logging
import os
import threading
import time
import webbrowser
from datetime import datetime, timedelta, timezone
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import parse_qs, urlparse

import pandas as pd

from qdata.auth import (
    AuthError,
    get_cached_token,
    get_credential,
    save_cached_token,
    save_credential_to_file,
)
from qdata.providers.base import BaseProvider, auth_retry_on_401
from qdata.store import CANONICAL_COLUMNS, normalize_timeframe

logger = logging.getLogger(__name__)

try:
    from fyers_apiv3 import fyersModel
    from fyers_apiv3.fyersModel import FyersModel, SessionModel
    FYERS_AVAILABLE = True
except ImportError:
    fyersModel = None
    FyersModel = None
    SessionModel = None
    FYERS_AVAILABLE = False


class OAuthCodeReceiver:
    """
    Lightweight HTTP server to capture the Fyers authorization code
    redirected after user authentication in the browser.
    """

    def __init__(self, host: str = "127.0.0.1", port: int = 80):
        self.host = host
        self.port = port
        self.auth_code: Optional[str] = None
        self._server: Optional[HTTPServer] = None
        self.auth_code_received = threading.Event()

    def start_server(self) -> None:
        receiver = self

        class OAuthCodeHandler(BaseHTTPRequestHandler):
            def do_GET(self) -> None:
                query_components = parse_qs(urlparse(self.path).query)
                if "auth_code" in query_components:
                    receiver.auth_code = query_components["auth_code"][0]
                    message = (
                        b"<html><body><h1>Authentication successful!</h1>"
                        b"<p>You can close this window and return to your terminal.</p>"
                        b"</body></html>"
                    )
                else:
                    message = (
                        b"<html><body><h1>Authentication Failed</h1>"
                        b"<p>auth_code not found in URL. Please try again.</p>"
                        b"</body></html>"
                    )

                self.send_response(200)
                self.send_header("Content-type", "text/html")
                self.end_headers()
                self.wfile.write(message)
                receiver.auth_code_received.set()

            def log_message(self, format: str, *args: Any) -> None:
                # Suppress default server access logs
                pass

        server_address = (self.host, self.port)
        self._server = HTTPServer(server_address, OAuthCodeHandler)
        thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        thread.start()

    def stop_server(self) -> None:
        if self._server:
            try:
                self._server.shutdown()
                self._server.server_close()
            except Exception:
                pass
            self._server = None

    def wait_for_code(self, timeout: int = 60) -> Optional[str]:
        try:
            self.start_server()
            self.auth_code_received.wait(timeout=timeout)
        finally:
            self.stop_server()
        return self.auth_code


class FyersProvider(BaseProvider):
    """
    Fyers API v3 market data provider.
    Ingests 1-min and daily OHLCV bars using Fyers API v3.
    Features:
    - Browser OAuth authentication with auto-redirect capturing.
    - Automatic 100-day chunking for intraday historical data.
    - Timestamp conversion to UTC canonical schema.
    """

    def __init__(
        self,
        data_dir: Optional[Path] = None,
        client_id: Optional[str] = None,
        secret_key: Optional[str] = None,
        redirect_uri: Optional[str] = None,
    ):
        super().__init__(data_dir=data_dir)
        self._explicit_client_id = client_id
        self._explicit_secret_key = secret_key
        self._explicit_redirect_uri = redirect_uri
        self._client: Optional[Any] = None

    @property
    def name(self) -> str:
        return "fyers"

    @property
    def client_id(self) -> Optional[str]:
        return (
            self._explicit_client_id
            or get_credential("fyers", "client_id")
            or get_credential("fyers", "api_key")
        )

    @property
    def secret_key(self) -> Optional[str]:
        return (
            self._explicit_secret_key
            or get_credential("fyers", "secret_key")
            or get_credential("fyers", "api_secret")
        )

    @property
    def redirect_uri(self) -> str:
        return (
            self._explicit_redirect_uri
            or get_credential("fyers", "redirect_uri")
            or "http://localhost.com"
        )

    def _get_client(self) -> Any:
        """Instantiate and cache authenticated FyersModel client."""
        if not FYERS_AVAILABLE:
            raise ImportError(
                "fyers_apiv3 is required to use FyersProvider. Install via 'pip install fyers-apiv3'."
            )
        token_info = self.ensure_auth()
        token = token_info.get("access_token")
        if not token:
            raise AuthError("No valid access token available for Fyers.")

        if self._client is None or getattr(self._client, "token", None) != token:
            self._client = FyersModel(
                token=token,
                is_async=False,
                client_id=self.client_id or "",
                log_path="",
            )
        return self._client

    # -------------------------------------------------------------------------
    # Authentication Implementation
    # -------------------------------------------------------------------------

    def _authenticate(self) -> Dict[str, Any]:
        """
        Authenticate with Fyers API v3 using OAuth 2.0.
        Checks for cached token first, otherwise launches browser flow.
        """
        if not FYERS_AVAILABLE:
            raise ImportError(
                "fyers_apiv3 is required to use FyersProvider. Install via 'pip install fyers-apiv3'."
            )

        if not self.client_id or not self.secret_key:
            raise AuthError(
                "Fyers credentials missing. Configure via 'qdata auth login fyers' "
                "or set QDATA_FYERS_CLIENT_ID and QDATA_FYERS_SECRET_KEY env vars."
            )

        # Check existing cached token
        cached = get_cached_token(self.tokens_dir, self.name)
        if cached and "access_token" in cached:
            return cached

        # Generate auth URL via SessionModel
        session = SessionModel(
            client_id=self.client_id,
            secret_key=self.secret_key,
            redirect_uri=self.redirect_uri,
            response_type="code",
            state="qdata_fyers_auth",
            grant_type="authorization_code",
            nonce="",
        )
        auth_url = session.generate_authcode()
        print(f"\nOpening browser for Fyers authentication: {auth_url}\n")

        # Parse host and port from redirect_uri
        parsed_uri = urlparse(self.redirect_uri)
        host = parsed_uri.hostname or "127.0.0.1"
        if host in ("localhost.com", "localhost"):
            host = "127.0.0.1"
        port = parsed_uri.port or (443 if parsed_uri.scheme == "https" else 80)

        auth_code: Optional[str] = None
        server: Optional[OAuthCodeReceiver] = None

        try:
            server = OAuthCodeReceiver(host=host, port=port)
            webbrowser.open(auth_url)
            auth_code = server.wait_for_code(timeout=60)
        except (OSError, PermissionError) as e:
            # Cannot bind local port (e.g. non-admin Windows port 80 restriction)
            logger.warning(f"Could not bind local listener on {host}:{port} ({e}).")
            webbrowser.open(auth_url)
        finally:
            if server:
                server.stop_server()

        # Fallback to manual entry if automatic capture timed out or failed
        if not auth_code:
            print("\nCould not automatically capture the authorization callback.")
            print("Please authenticate in your browser, copy either the 'auth_code' parameter")
            print("or the entire redirected URL from your browser address bar, and paste it below:\n")
            user_input = input("Paste auth_code or full redirect URL: ").strip()
            if "auth_code=" in user_input:
                parsed_query = parse_qs(urlparse(user_input).query)
                auth_code = parsed_query.get("auth_code", [""])[0]
            else:
                auth_code = user_input

        if not auth_code:
            raise AuthError("Failed to obtain Fyers authorization code.")

        # Generate access token
        session.set_token(auth_code)
        response = session.generate_token()

        if response.get("s") == "ok" and "access_token" in response:
            access_token = response["access_token"]
            # Fyers tokens are typically valid for ~24 hours (until next trading day reset)
            expires_at = time.time() + 86400
            token_data = {
                "access_token": access_token,
                "token_type": "Bearer",
                "expires_at": expires_at,
                "provider": self.name,
                "client_id": self.client_id,
            }
            save_cached_token(self.tokens_dir, self.name, token_data)
            return token_data
        else:
            err_msg = response.get("message") or response.get("err_msg") or str(response)
            raise AuthError(f"Fyers token generation failed: {err_msg}")

    # -------------------------------------------------------------------------
    # Formatting & Timeframe Utilities
    # -------------------------------------------------------------------------

    def format_symbol(self, symbol: str, exchange: Optional[str] = None) -> str:
        """
        Format symbol to Fyers API syntax:
        - If already formatted (e.g. 'NSE:RELIANCE-EQ' or 'MCX:CRUDEOIL-FUT'), return as-is.
        - Otherwise format as '{EXCHANGE}:{SYMBOL}-EQ' (e.g. 'NSE:RELIANCE-EQ').
        """
        sym = symbol.strip().upper()
        if ":" in sym:
            return sym

        exch = (exchange or self.settings.default_exchange).upper()
        return f"{exch}:{sym}-EQ"

    def map_resolution(self, timeframe: str) -> str:
        """
        Map canonical timeframe string to Fyers API resolution.
        - '1min', '1m' -> '1'
        - '1d', 'd', 'daily' -> 'D'
        - '5min' -> '5', etc.
        """
        norm = normalize_timeframe(timeframe)
        if norm in ("1min", "1m"):
            return "1"
        if norm in ("1d", "d", "daily"):
            return "D"
        # Extract digits if like '5min' -> '5'
        digits = "".join(filter(str.isdigit, norm))
        if digits:
            return digits
        return norm

    def split_into_chunks(
        self,
        start: datetime,
        end: datetime,
        max_days: int = 99,
    ) -> List[Tuple[str, str]]:
        """
        Split a date range into sub-intervals of at most max_days.
        Fyers historical intraday API has a 100-day limit per request.
        Returns list of (start_str, end_str) in 'YYYY-MM-DD' format.
        """
        start_dt = start.date() if isinstance(start, datetime) else start
        end_dt = end.date() if isinstance(end, datetime) else end

        pairs = []
        curr = start_dt
        while curr <= end_dt:
            curr_end = min(curr + timedelta(days=max_days), end_dt)
            pairs.append((curr.strftime("%Y-%m-%d"), curr_end.strftime("%Y-%m-%d")))
            curr = curr_end + timedelta(days=1)

        return pairs

    # -------------------------------------------------------------------------
    # Provider Core Operations
    # -------------------------------------------------------------------------

    @auth_retry_on_401
    def latest_available(self, symbol: str, timeframe: str) -> Optional[datetime]:
        """
        Return the latest timestamp available from Fyers for symbol & timeframe.
        Queries recent history (last 5 days) to find the most recent trading bar.
        """
        self.rate_limit()
        fyers_sym = self.format_symbol(symbol)
        res = self.map_resolution(timeframe)

        now_utc = datetime.now(timezone.utc)
        start_date = (now_utc - timedelta(days=7)).strftime("%Y-%m-%d")
        end_date = now_utc.strftime("%Y-%m-%d")

        is_futures = "-FUT" in fyers_sym.upper()
        cont_flag = "1" if is_futures else "0"

        client = self._get_client()
        data = {
            "symbol": fyers_sym,
            "resolution": res,
            "date_format": "1",
            "range_from": start_date,
            "range_to": end_date,
            "cont_flag": cont_flag,
        }

        try:
            resp = client.history(data=data)
            if resp and resp.get("s") == "ok" and resp.get("candles"):
                candles = resp["candles"]
                latest_epoch = candles[-1][0]
                return pd.to_datetime(latest_epoch, unit="s", utc=True).to_pydatetime()
        except Exception as e:
            logger.warning(f"Error querying latest available data from Fyers: {e}")

        return now_utc

    @auth_retry_on_401
    def fetch(
        self,
        symbol: str,
        timeframe: str,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
    ) -> pd.DataFrame:
        """
        Fetch market data from Fyers and return canonical DataFrame.
        Handles date chunking (max 100 days per request) and converts candles to UTC.
        """
        fyers_sym = self.format_symbol(symbol)
        res = self.map_resolution(timeframe)

        now_utc = datetime.now(timezone.utc)
        if end is None:
            end = now_utc
        elif end.tzinfo is None:
            end = end.replace(tzinfo=timezone.utc)
        else:
            end = end.astimezone(timezone.utc)

        if start is None:
            # Default to 30 days ago if no start given
            start = end - timedelta(days=30)
        elif start.tzinfo is None:
            start = start.replace(tzinfo=timezone.utc)
        else:
            start = start.astimezone(timezone.utc)

        if start > end:
            return pd.DataFrame(columns=CANONICAL_COLUMNS)

        client = self._get_client()
        chunks = self.split_into_chunks(start, end, max_days=99)
        all_candles: List[List[Any]] = []

        is_futures = "-FUT" in fyers_sym.upper()
        cont_flag = "1" if is_futures else "0"

        for pair_start, pair_end in chunks:
            self.rate_limit()
            req_data = {
                "symbol": fyers_sym,
                "resolution": res,
                "date_format": "1",
                "range_from": pair_start,
                "range_to": pair_end,
                "cont_flag": cont_flag,
            }

            resp = client.history(data=req_data)
            if not resp:
                continue

            status = resp.get("s")
            if status == "ok":
                candles = resp.get("candles", [])
                if candles:
                    all_candles.extend(candles)
            elif status == "error":
                msg = resp.get("message", "")
                if "token" in msg.lower() or "auth" in msg.lower() or "session" in msg.lower():
                    raise AuthError(f"Fyers session error: {msg}")
                logger.warning(f"Fyers history error for {fyers_sym} [{pair_start} -> {pair_end}]: {msg}")
            else:
                # No data or weekend
                pass

        if not all_candles:
            return pd.DataFrame(columns=CANONICAL_COLUMNS)

        df = pd.DataFrame(all_candles, columns=["ts", "open", "high", "low", "close", "volume"])
        df["ts"] = pd.to_datetime(df["ts"], unit="s", utc=True)

        # Filter strictly within requested range [start, end]
        df = df[(df["ts"] >= start) & (df["ts"] <= end)].copy()
        if df.empty:
            return pd.DataFrame(columns=CANONICAL_COLUMNS)

        # Deduplicate broker feed artifact timestamps (e.g. 15:30 closing tick), keeping the last candle
        df = df.drop_duplicates(subset=["ts"], keep="last")

        df["open"] = df["open"].astype("float64")
        df["high"] = df["high"].astype("float64")
        df["low"] = df["low"].astype("float64")
        df["close"] = df["close"].astype("float64")
        df["volume"] = df["volume"].astype("float64")
        df["provider"] = self.name
        df["ingested_at"] = datetime.now(timezone.utc)

        df = df[CANONICAL_COLUMNS].sort_values("ts").reset_index(drop=True)
        return df

    def health_check(self) -> Dict[str, Any]:
        """Return health and credential status for Fyers."""
        has_client_id = bool(self.client_id)
        has_secret = bool(self.secret_key)
        cached = get_cached_token(self.tokens_dir, self.name)
        has_token = cached is not None and "access_token" in cached

        profile_ok = False
        profile_data: Optional[Dict[str, Any]] = None

        if has_token:
            try:
                client = self._get_client()
                prof_resp = client.get_profile()
                if prof_resp and prof_resp.get("s") == "ok":
                    profile_ok = True
                    profile_data = {
                        "name": prof_resp.get("data", {}).get("name"),
                        "fy_id": prof_resp.get("data", {}).get("fy_id"),
                        "email": prof_resp.get("data", {}).get("email_id"),
                    }
            except Exception:
                pass

        return {
            "provider": self.name,
            "status": "ok" if profile_ok else ("configured" if (has_client_id and has_secret) else "missing_credentials"),
            "has_client_id": has_client_id,
            "has_secret_key": has_secret,
            "has_cached_token": has_token,
            "authenticated": profile_ok,
            "profile": profile_data,
        }
