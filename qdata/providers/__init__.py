from pathlib import Path
from typing import Dict, Optional, Type

from qdata.providers.amfi import AmfiProvider
from qdata.providers.base import BaseProvider
from qdata.providers.fyers import FyersProvider
from qdata.providers.mock import MockProvider
from qdata.providers.upstox import UpstoxProvider

_PROVIDERS: Dict[str, Type[BaseProvider]] = {
    "mock": MockProvider,
    "upstox": UpstoxProvider,
    "fyers": FyersProvider,
    "amfi": AmfiProvider,
}


def _discover_entry_points() -> None:
    """Discover third-party providers via Python entry points (group 'qdata.providers')."""
    try:
        from importlib.metadata import entry_points
        eps = entry_points(group="qdata.providers")
        for ep in eps:
            k = ep.name.lower()
            if k not in _PROVIDERS:
                try:
                    cls = ep.load()
                    _PROVIDERS[k] = cls
                except Exception:
                    pass
    except Exception:
        pass


def list_providers() -> Dict[str, Type[BaseProvider]]:
    """Return dictionary of all registered providers."""
    _discover_entry_points()
    return dict(_PROVIDERS)


def get_provider(name: str, data_dir: Optional[Path] = None, **kwargs) -> BaseProvider:
    """Factory to instantiate provider by name."""
    _discover_entry_points()
    prov_key = name.lower()
    if prov_key not in _PROVIDERS:
        raise ValueError(f"Unknown provider '{name}'. Available: {list(_PROVIDERS.keys())}")
    cls = _PROVIDERS[prov_key]
    return cls(data_dir=data_dir, **kwargs)


def register_provider(name: str, cls: Type[BaseProvider]) -> None:
    """Register custom provider class."""
    _PROVIDERS[name.lower()] = cls


__all__ = [
    "BaseProvider",
    "MockProvider",
    "UpstoxProvider",
    "FyersProvider",
    "AmfiProvider",
    "get_provider",
    "register_provider",
    "list_providers",
]


