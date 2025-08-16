from collections.abc import Generator
from typing import Any
from unittest.mock import patch
import pytest
from homeassistant.components.prusalink import DOMAIN
from homeassistant.core import HomeAssistant
from tests.common import MockConfigEntry

@pytest.fixture
def mock_config_entry(hass: HomeAssistant) -> MockConfigEntry:
    ...

@pytest.fixture
def mock_version_api() -> Generator[dict[str, Any], None, None]:
    ...

@pytest.fixture
def mock_info_api() -> Generator[dict[str, Any], None, None]:
    ...

@pytest.fixture
def mock_get_legacy_printer() -> Generator[dict[str, Any], None, None]:
    ...

@pytest.fixture
def mock_get_status_idle() -> Generator[dict[str, Any], None, None]:
    ...

@pytest.fixture
def mock_get_status_printing() -> Generator[dict[str, Any], None, None]:
    ...

@pytest.fixture
def mock_job_api_idle() -> Generator[dict[str, Any], None, None]:
    ...

@pytest.fixture
def mock_job_api_idle_mk3() -> Generator[dict[str, Any], None, None]:
    ...

@pytest.fixture
def mock_job_api_printing() -> Generator[dict[str, Any], None, None]:
    ...

@pytest.fixture
def mock_job_api_paused(mock_get_status_printing, mock_job_api_printing) -> None:
    ...

@pytest.fixture
def mock_api(mock_version_api, mock_info_api, mock_get_legacy_printer, mock_get_status_idle, mock_job_api_idle) -> None:
    ...
