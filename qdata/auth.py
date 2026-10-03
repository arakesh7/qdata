import json
import os
import stat
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import keyring
import tomli_w

if sys.version_info >= (3, 11):
    import tomllib
else:
    import toml as tomllib  # type: ignore

from qdata.config import get_settings


class AuthError(Exception):
    """Raised when authentication fails."""
    pass


def mask_secret(secret: Optional[str]) -> str:
    """Mask secret strings for safe display."""
    if not secret:
        return "<not set>"
    if len(secret) <= 6:
        return "******"
    return f"{secret[:3]}***{secret[-3:]}"


def check_credentials_file_permissions(file_path: Path) -> Tuple[bool, str]:
    """
    Check if credentials file has secure permissions (chmod 600/400).
    On Windows, permissions behave differently, but we check if readable.
    """
    if not file_path.exists():
        return True, "File does not exist yet."

    if os.name == "posix":
        st = os.stat(file_path)
        mode = stat.S_IMODE(st.st_mode)
        # Check if group or others have read/write/execute permissions (0o077)
        if mode & 0o077 != 0:
            return False, f"Insecure file permissions: {oct(mode)} (expected 0600 or 0400). Run: chmod 600 {file_path}"
    return True, "Permissions are secure."


def get_credential(provider: str, key: str, credentials_file: Optional[Path] = None) -> Optional[str]:
    """
    Resolve credential by precedence:
    1. Environment variables: QDATA_<PROVIDER>_<KEY>
    2. OS keyring: service="qdata:<provider>", username=key
    3. ~/.config/qdata/credentials.toml: [<provider>] key = ...
    """
    prov_upper = provider.upper()
    key_upper = key.upper()
    env_var_name = f"QDATA_{prov_upper}_{key_upper}"
    val = os.getenv(env_var_name)
    if val:
        return val

    # 2. OS Keyring
    try:
        kr_val = keyring.get_password(f"qdata:{provider.lower()}", key)
        if kr_val:
            return kr_val
    except Exception:
        # Keyring might not be supported or available in headless/CI environments
        pass

    # 3. credentials.toml
    cred_file = credentials_file or get_settings().resolved_credentials_file
    if cred_file.exists():
        try:
            with open(cred_file, "rb") as f:
                data = tomllib.load(f)
            prov_data = data.get(provider.lower(), {})
            if key in prov_data:
                return str(prov_data[key])
        except Exception:
            pass

    return None


def get_provider_credentials(provider: str, credentials_file: Optional[Path] = None) -> Dict[str, str]:
    """Retrieve all known credentials for a provider across all sources."""
    prov = provider.lower()
    creds: Dict[str, str] = {}

    # From credentials.toml
    cred_file = credentials_file or get_settings().resolved_credentials_file
    if cred_file.exists():
        try:
            with open(cred_file, "rb") as f:
                data = tomllib.load(f)
            prov_data = data.get(prov, {})
            for k, v in prov_data.items():
                creds[k] = str(v)
        except Exception:
            pass

    # From env vars (overrides file)
    prefix = f"QDATA_{prov.upper()}_"
    for env_k, env_v in os.environ.items():
        if env_k.startswith(prefix):
            k = env_k[len(prefix):].lower()
            creds[k] = env_v

    return creds


def save_credential_to_file(
    provider: str,
    key: str,
    value: str,
    credentials_file: Optional[Path] = None,
) -> Path:
    """Save credential into credentials.toml and enforce chmod 600 on POSIX."""
    cred_file = credentials_file or get_settings().resolved_credentials_file
    cred_file.parent.mkdir(parents=True, exist_ok=True)

    data: Dict[str, Any] = {}
    if cred_file.exists():
        try:
            with open(cred_file, "rb") as f:
                data = tomllib.load(f)
        except Exception:
            data = {}

    prov = provider.lower()
    if prov not in data:
        data[prov] = {}
    data[prov][key] = value

    temp_file = cred_file.with_suffix(".tmp")
    with open(temp_file, "wb") as f:
        tomli_w.dump(data, f)
        f.flush()
        os.fsync(f.fileno())

    if os.name == "posix":
        os.chmod(temp_file, 0o600)

    os.replace(temp_file, cred_file)
    if os.name == "posix":
        os.chmod(cred_file, 0o600)

    return cred_file


def save_credential_to_keyring(provider: str, key: str, value: str) -> None:
    """Save credential into OS keyring."""
    keyring.set_password(f"qdata:{provider.lower()}", key, value)


def get_cached_token(tokens_dir: Path, provider: str) -> Optional[Dict[str, Any]]:
    """
    Get cached session token if present and not within 60s of expiry.
    Returns None if missing, expired, or invalid.
    """
    token_file = tokens_dir / f"{provider.lower()}.json"
    if not token_file.exists():
        return None

    try:
        with open(token_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        expires_at = data.get("expires_at")
        if expires_at is not None:
            # Refreshed when within 60s of expiry
            if time.time() >= float(expires_at) - 60.0:
                return None

        return data
    except Exception:
        return None


def save_cached_token(tokens_dir: Path, provider: str, token_data: Dict[str, Any]) -> Path:
    """
    Cache session token in data/.tokens/<provider>.json (chmod 600 on POSIX).
    """
    tokens_dir.mkdir(parents=True, exist_ok=True)
    if os.name == "posix":
        try:
            os.chmod(tokens_dir, 0o700)
        except Exception:
            pass

    token_file = tokens_dir / f"{provider.lower()}.json"
    temp_file = token_file.with_suffix(".tmp")

    with open(temp_file, "w", encoding="utf-8") as f:
        json.dump(token_data, f, indent=2)
        f.flush()
        os.fsync(f.fileno())

    if os.name == "posix":
        try:
            os.chmod(temp_file, 0o600)
        except Exception:
            pass

    os.replace(temp_file, token_file)
    if os.name == "posix":
        try:
            os.chmod(token_file, 0o600)
        except Exception:
            pass

    return token_file


def clear_cached_token(tokens_dir: Path, provider: str) -> bool:
    """Clear cached token file for provider."""
    token_file = tokens_dir / f"{provider.lower()}.json"
    if token_file.exists():
        try:
            token_file.unlink()
            return True
        except Exception:
            return False
    return False
