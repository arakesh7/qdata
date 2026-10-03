from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd

from qdata.auth import AuthError, get_credential
from qdata.providers.base import BaseProvider, auth_retry_on_401
from qdata.store import CANONICAL_COLUMNS


class UpstoxProvider(BaseProvider):
    """
    Upstox Provider stub conforming to the BaseProvider interface.
    Implements credential checking and standard endpoints.
    Actual external HTTP requests require live Upstox developer credentials.
    """

    BASE_URL = "https://api.upstox.com/v2"

    def __init__(self, data_dir: Optional[Path] = None):
        super().__init__(data_dir=data_dir)
        self.api_key = get_credential("upstox", "api_key")
        self.api_secret = get_credential("upstox", "api_secret")

    @property
    def name(self) -> str:
        return "upstox"

    def _authenticate(self) -> Dict[str, Any]:
        """Verify Upstox credentials are configured."""
        if not self.api_key:
            raise AuthError(
                "Upstox credentials not configured. Set QDATA_UPSTOX_API_KEY env var "
                "or run 'qdata auth login upstox'."
            )
        # Placeholder session structure for live API integration
        return {
            "access_token": "upstox_session_token_stub",
            "token_type": "Bearer",
            "expires_at": datetime.now(timezone.utc).timestamp() + 86400,
            "provider": self.name,
        }

    @auth_retry_on_401
    def latest_available(self, symbol: str, timeframe: str) -> Optional[datetime]:
        self.rate_limit()
        if not self.api_key:
            raise AuthError("Upstox credentials required to query latest available data.")
        return datetime.now(timezone.utc)

    @auth_retry_on_401
    def fetch(
        self,
        symbol: str,
        timeframe: str,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
    ) -> pd.DataFrame:
        self.rate_limit()
        if not self.api_key:
            raise AuthError("Upstox credentials required to fetch market data.")
        raise NotImplementedError("Live Upstox API network integration is not configured in this environment.")

    def health_check(self) -> Dict[str, Any]:
        has_creds = bool(self.api_key)
        return {
            "status": "configured" if has_creds else "missing_credentials",
            "provider": self.name,
            "has_api_key": has_creds,
            "base_url": self.BASE_URL,
        }
