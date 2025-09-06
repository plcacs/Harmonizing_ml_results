from collections.abc import Awaitable, Callable, Generator
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
def func_qbb0b202() -> Generator[AsyncMock, None, None]:
    ...

@pytest.fixture
def func_5odmuimh(user: User, controller: Controller, zones: List[Zone]) -> Generator[Hydrawise, None, None]:
    ...

@pytest.fixture
def func_5w2gy6wz(user: User, controller: Controller, zones: List[Zone], sensors: List[Sensor], controller_water_use_summary: ControllerWaterUseSummary) -> Generator[Hydrawise, None, None]:
    ...

@pytest.fixture
def func_g0irya3t() -> Generator[HybridAuth, None, None]:
    ...

@pytest.fixture
def func_orryon1b() -> User:
    ...

@pytest.fixture
def func_6rldkosc() -> Controller:
    ...

@pytest.fixture
def func_kl8wrmpe() -> List[Sensor]:
    ...

@pytest.fixture
def func_bp2adfv6() -> List[Zone]:
    ...

@pytest.fixture
def func_mkh0ne76() -> ControllerWaterUseSummary:
    ...

@pytest.fixture
def func_7t824wrv() -> MockConfigEntry:
    ...

@pytest.fixture
def func_3olkvolt() -> MockConfigEntry:
    ...

@pytest.fixture
async def func_ogdnscc2(mock_add_config_entry: Callable) -> MockConfigEntry:
    ...

@pytest.fixture
async def func_aalf0rid(hass: HomeAssistant, mock_config_entry: MockConfigEntry, mock_pydrawise: Hydrawise) -> Callable:
    ...
