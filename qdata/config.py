from pathlib import Path
from typing import Optional
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="QDATA_",
        env_file=".env",
        extra="ignore",
    )

    data_dir: Path = Path("data")
    config_dir: Path = Path.home() / ".config" / "qdata"
    credentials_file: Optional[Path] = None
    default_provider: str = "mock"
    default_exchange: str = "NSE"
    default_currency: str = "INR"
    default_asset_class: str = "EQUITY"
    log_level: str = "INFO"

    @property
    def resolved_credentials_file(self) -> Path:
        if self.credentials_file:
            return Path(self.credentials_file).expanduser().resolve()
        return (self.config_dir / "credentials.toml").expanduser().resolve()

    @property
    def catalog_dir(self) -> Path:
        return self.data_dir / "catalog"

    @property
    def raw_dir(self) -> Path:
        return self.data_dir / "raw"

    @property
    def adjusted_dir(self) -> Path:
        return self.data_dir / "adjusted"

    @property
    def tokens_dir(self) -> Path:
        return self.data_dir / ".tokens"

    @property
    def lock_file(self) -> Path:
        return self.data_dir / ".sync.lock"


_settings: Optional[Settings] = None


def get_settings(override_data_dir: Optional[Path] = None) -> Settings:
    global _settings
    if _settings is None:
        _settings = Settings()
    if override_data_dir is not None:
        # Create a copy with the overridden data_dir
        return _settings.model_copy(update={"data_dir": Path(override_data_dir).resolve()})
    return _settings


def reset_settings(new_settings: Optional[Settings] = None) -> None:
    global _settings
    _settings = new_settings
