import functools
import time
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Optional, TypeVar

import pandas as pd

from qdata.auth import AuthError, clear_cached_token, get_cached_token, save_cached_token
from qdata.config import get_settings

F = TypeVar("F", bound=Callable[..., Any])


def auth_retry_on_401(method: F) -> F:
    """
    Decorator for provider methods: calls ensure_auth() before execution.
    If an AuthError or 401 occurs, calls refresh_auth() once and retries.
    If it fails again, raises AuthError.
    """
    @functools.wraps(method)
    def wrapper(self: "BaseProvider", *args: Any, **kwargs: Any) -> Any:
        self.ensure_auth()
        try:
            return method(self, *args, **kwargs)
        except AuthError as e:
            # Single retry on 401 / AuthError
            self.refresh_auth()
            return method(self, *args, **kwargs)

    return wrapper  # type: ignore


class BaseProvider(ABC):
    """
    Abstract base class for all market data providers.
    """

    def __init__(self, data_dir: Optional[Path] = None):
        self.settings = get_settings(override_data_dir=data_dir)
        self.data_dir = self.settings.data_dir
        self.tokens_dir = self.settings.tokens_dir
        self.connected = False
        self._rate_limit_last_called = 0.0

    @property
    @abstractmethod
    def name(self) -> str:
        """Provider identifier (e.g. 'mock', 'upstox')."""
        pass

    def connect(self) -> None:
        """Establish session or connection to provider."""
        self.ensure_auth()
        self.connected = True

    def close(self) -> None:
        """Close session or connection."""
        self.connected = False

    def ensure_auth(self) -> Dict[str, Any]:
        """
        Lazy auth verification called before any public provider method.
        Returns cached token if valid (not within 60s of expiry),
        or triggers authentication flow.
        """
        cached = get_cached_token(self.tokens_dir, self.name)
        if cached:
            return cached
        return self._authenticate()

    def refresh_auth(self) -> Dict[str, Any]:
        """
        Forcibly refresh auth credentials.
        Clears cached token, re-authenticates, and caches new token.
        """
        clear_cached_token(self.tokens_dir, self.name)
        return self._authenticate()

    @abstractmethod
    def _authenticate(self) -> Dict[str, Any]:
        """Internal authentication implementation."""
        pass

    @abstractmethod
    def latest_available(self, symbol: str, timeframe: str) -> Optional[datetime]:
        """Return the latest timestamp available from provider for symbol and timeframe (UTC)."""
        pass

    @abstractmethod
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
        """
        pass

    @abstractmethod
    def health_check(self) -> Dict[str, Any]:
        """Return health status dictionary."""
        pass

    def rate_limit(self) -> None:
        """Enforce provider rate limits (polite pacing)."""
        now = time.time()
        elapsed = now - self._rate_limit_last_called
        # Default minimum delay 100ms
        if elapsed < 0.1:
            time.sleep(0.1 - elapsed)
        self._rate_limit_last_called = time.time()

    @classmethod
    def get_cli_app(cls) -> Optional[Any]:
        """
        Optional Typer sub-app exposing provider-specific CLI commands.
        Subclasses may override to return a typer.Typer instance which
        qdata CLI dynamically discovers and mounts as 'qdata <provider_name>'.
        """
        return None

