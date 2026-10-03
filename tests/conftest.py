import os
from pathlib import Path
import pytest

from qdata.config import reset_settings, get_settings


@pytest.fixture
def tmp_data_dir(tmp_path: Path):
    """Provides a fresh isolated data directory for tests."""
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    # Ensure settings points to tmp_data_dir
    os.environ["QDATA_DATA_DIR"] = str(data_dir)
    reset_settings()
    yield data_dir
    # Cleanup
    if "QDATA_DATA_DIR" in os.environ:
        del os.environ["QDATA_DATA_DIR"]
    reset_settings()
