import os
import time
from datetime import datetime, timezone
from pathlib import Path
import pytest

from qdata.auth import (
    AuthError,
    clear_cached_token,
    get_cached_token,
    get_credential,
    save_cached_token,
    save_credential_to_file,
)
from qdata.providers.mock import MockProvider


def test_credential_precedence(tmp_path: Path):
    cred_file = tmp_path / "credentials.toml"

    # 1. Test credentials.toml
    save_credential_to_file("upstox", "api_key", "file_key_123", credentials_file=cred_file)
    val = get_credential("upstox", "api_key", credentials_file=cred_file)
    assert val == "file_key_123"

    # 2. Test Environment variable overrides file
    os.environ["QDATA_UPSTOX_API_KEY"] = "env_key_456"
    try:
        val_env = get_credential("upstox", "api_key", credentials_file=cred_file)
        assert val_env == "env_key_456"
    finally:
        del os.environ["QDATA_UPSTOX_API_KEY"]


def test_token_cache_and_expiration(tmp_data_dir: Path):
    tokens_dir = tmp_data_dir / ".tokens"

    # Token valid for 1 hour
    token_data = {
        "access_token": "valid_tok_abc",
        "expires_at": time.time() + 3600,
    }
    save_cached_token(tokens_dir, "mock", token_data)

    cached = get_cached_token(tokens_dir, "mock")
    assert cached is not None
    assert cached["access_token"] == "valid_tok_abc"

    # Token within 50s of expiry (<60s threshold) -> should return None
    expiring_token = {
        "access_token": "expiring_tok",
        "expires_at": time.time() + 45,
    }
    save_cached_token(tokens_dir, "mock", expiring_token)
    assert get_cached_token(tokens_dir, "mock") is None

    # Clear cached token
    clear_cached_token(tokens_dir, "mock")
    assert get_cached_token(tokens_dir, "mock") is None


def test_auth_refresh_retry_on_401(tmp_data_dir: Path):
    """
    Mock provider exercises auth-refresh-retry on 401 without real credentials.
    """
    mock = MockProvider(data_dir=tmp_data_dir)
    mock.connect()
    assert mock.auth_count == 1

    # Simulate 401 on next call
    mock.simulate_401_once = True

    # Calling fetch should catch 401, call refresh_auth(), and succeed on retry
    end_dt = datetime(2026, 9, 25, 12, 0, tzinfo=timezone.utc)
    df = mock.fetch("AAPL", "1min", end=end_dt)
    assert not df.empty
    # refresh_count should have incremented
    assert mock.refresh_count == 1
    assert mock.auth_count == 2
