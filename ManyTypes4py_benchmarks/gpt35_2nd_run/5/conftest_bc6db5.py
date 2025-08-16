from typing import Any, Generator
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, patch
from pydrawise.schema import Controller, ControllerHardware, ControllerWaterUseSummary, CustomSensorTypeEnum, LocalizedValueType, ScheduledZoneRun, ScheduledZoneRuns, Sensor, SensorModel, SensorStatus, UnitsSummary, User, Zone
import pytest
from homeassistant.components.hydrawise.const import DOMAIN
from homeassistant.const import CONF_API_KEY, CONF_PASSWORD, CONF_USERNAME
from homeassistant.core import HomeAssistant
from homeassistant.util import dt as dt_util
from tests.common import MockConfigEntry

@pytest.fixture
def mock_setup_entry() -> Generator[Any, Any, Any]:
    ...

@pytest.fixture
def mock_legacy_pydrawise(user: User, controller: Controller, zones: Any) -> Generator[Any, Any, Any]:
    ...

@pytest.fixture
def mock_pydrawise(user: User, controller: Controller, zones: Any, sensors: Any, controller_water_use_summary: ControllerWaterUseSummary) -> Generator[Any, Any, Any]:
    ...

@pytest.fixture
def mock_auth() -> Generator[Any, Any, Any]:
    ...

@pytest.fixture
def user() -> User:
    ...

@pytest.fixture
def controller() -> Controller:
    ...

@pytest.fixture
def sensors() -> Any:
    ...

@pytest.fixture
def zones() -> Any:
    ...

@pytest.fixture
def controller_water_use_summary() -> ControllerWaterUseSummary:
    ...

@pytest.fixture
def mock_config_entry_legacy() -> MockConfigEntry:
    ...

@pytest.fixture
def mock_config_entry() -> MockConfigEntry:
    ...

@pytest.fixture
async def mock_added_config_entry(mock_add_config_entry: Any) -> Any:
    ...

@pytest.fixture
async def mock_add_config_entry(hass: HomeAssistant, mock_config_entry: MockConfigEntry, mock_pydrawise: Any) -> Any:
    ...
