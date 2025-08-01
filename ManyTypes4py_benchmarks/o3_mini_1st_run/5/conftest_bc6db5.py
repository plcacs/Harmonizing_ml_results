from collections.abc import Awaitable, Callable
from datetime import datetime, timedelta
from typing import Any, AsyncGenerator, Callable as TCallable, Generator, List

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from pydrawise.schema import (
    Controller,
    ControllerHardware,
    ControllerWaterUseSummary,
    CustomSensorTypeEnum,
    LocalizedValueType,
    ScheduledZoneRun,
    ScheduledZoneRuns,
    Sensor,
    SensorModel,
    SensorStatus,
    UnitsSummary,
    User,
    Zone,
)

from homeassistant.components.hydrawise.const import DOMAIN
from homeassistant.const import CONF_API_KEY, CONF_PASSWORD, CONF_USERNAME
from homeassistant.core import HomeAssistant
from homeassistant.util import dt as dt_util
from tests.common import MockConfigEntry


@pytest.fixture
def mock_setup_entry() -> Generator[MagicMock, None, None]:
    """Override async_setup_entry."""
    with patch("homeassistant.components.hydrawise.async_setup_entry", return_value=True) as mock_setup_entry:
        yield mock_setup_entry


@pytest.fixture
def mock_legacy_pydrawise(user: User, controller: Controller, zones: List[Zone]) -> Generator[Any, None, None]:
    """Mock LegacyHydrawiseAsync."""
    with patch("pydrawise.legacy.LegacyHydrawiseAsync", autospec=True) as mock_pydrawise:
        user.controllers = [controller]
        controller.zones = zones
        mock_pydrawise.return_value.get_user.return_value = user
        yield mock_pydrawise.return_value


@pytest.fixture
def mock_pydrawise(
    user: User,
    controller: Controller,
    zones: List[Zone],
    sensors: List[Sensor],
    controller_water_use_summary: ControllerWaterUseSummary,
) -> Generator[Any, None, None]:
    """Mock Hydrawise."""
    with patch("pydrawise.hybrid.HybridClient", autospec=True) as mock_pydrawise:
        user.controllers = [controller]
        controller.sensors = sensors
        mock_pydrawise.return_value.get_user.return_value = user
        mock_pydrawise.return_value.get_zones.return_value = zones
        mock_pydrawise.return_value.get_water_use_summary.return_value = controller_water_use_summary
        yield mock_pydrawise.return_value


@pytest.fixture
def mock_auth() -> Generator[Any, None, None]:
    """Mock pydrawise HybridAuth."""
    with patch("pydrawise.auth.HybridAuth", autospec=True) as mock_auth:
        yield mock_auth.return_value


@pytest.fixture
def user() -> User:
    """Hydrawise User fixture."""
    return User(customer_id=12345, email="asdf@asdf.com", units=UnitsSummary(units_name="imperial"))


@pytest.fixture
def controller() -> Controller:
    """Hydrawise Controller fixture."""
    return Controller(
        id=52496,
        name="Home Controller",
        hardware=ControllerHardware(serial_number="0310b36090"),
        last_contact_time=datetime.fromtimestamp(1693292420),
        online=True,
        sensors=[],
    )


@pytest.fixture
def sensors() -> List[Sensor]:
    """Hydrawise sensor fixtures."""
    return [
        Sensor(
            id=337844,
            name="Rain sensor ",
            model=SensorModel(
                id=3318,
                name="Rain Sensor (normally closed wire)",
                active=True,
                off_level=1,
                off_timer=0,
                divisor=0.0,
                flow_rate=0.0,
                sensor_type=CustomSensorTypeEnum.LEVEL_CLOSED,
            ),
            status=SensorStatus(water_flow=None, active=False),
        ),
        Sensor(
            id=337845,
            name="Flow meter",
            model=SensorModel(
                id=3324,
                name="1, 1½ or 2 inch NPT Flow Meter",
                active=True,
                off_level=0,
                off_timer=0,
                divisor=0.52834,
                flow_rate=3.7854,
                sensor_type=CustomSensorTypeEnum.FLOW,
            ),
            status=SensorStatus(water_flow=LocalizedValueType(value=577.0044752010709, unit="gal"), active=False),
        ),
    ]


@pytest.fixture
def zones() -> List[Zone]:
    """Hydrawise zone fixtures."""
    return [
        Zone(
            name="Zone One",
            id=5965394,
            scheduled_runs=ScheduledZoneRuns(
                summary="",
                current_run=None,
                next_run=ScheduledZoneRun(
                    start_time=dt_util.now() + timedelta(seconds=330597),
                    end_time=dt_util.now() + timedelta(seconds=330597) + timedelta(seconds=1800),
                    normal_duration=timedelta(seconds=1800),
                    duration=timedelta(seconds=1800),
                ),
            ),
        ),
        Zone(
            name="Zone Two",
            id=5965395,
            scheduled_runs=ScheduledZoneRuns(
                current_run=ScheduledZoneRun(remaining_time=timedelta(seconds=1788))
            ),
        ),
    ]


@pytest.fixture
def controller_water_use_summary() -> ControllerWaterUseSummary:
    """Mock water use summary for the controller."""
    return ControllerWaterUseSummary(
        total_use=345.6,
        total_active_use=332.6,
        total_inactive_use=13.0,
        active_use_by_zone_id={5965394: 120.1, 5965395: 0.0},
        total_active_time=timedelta(seconds=123),
        active_time_by_zone_id={5965394: timedelta(seconds=123), 5965395: timedelta()},
        unit="gal",
    )


@pytest.fixture
def mock_config_entry_legacy() -> MockConfigEntry:
    """Mock ConfigEntry."""
    return MockConfigEntry(
        title="Hydrawise",
        domain=DOMAIN,
        data={CONF_API_KEY: "abc123"},
        unique_id="hydrawise-customerid",
        version=1,
    )


@pytest.fixture
def mock_config_entry() -> MockConfigEntry:
    """Mock ConfigEntry."""
    return MockConfigEntry(
        title="Hydrawise",
        domain=DOMAIN,
        data={CONF_USERNAME: "asfd@asdf.com", CONF_PASSWORD: "__password__", CONF_API_KEY: "abc123"},
        unique_id="hydrawise-customerid",
        version=1,
        minor_version=2,
    )


@pytest.fixture
async def mock_added_config_entry(
    mock_add_config_entry: TCallable[[], Awaitable[MockConfigEntry]]
) -> AsyncGenerator[MockConfigEntry, None]:
    """Mock ConfigEntry that's been added to HA."""
    config_entry = await mock_add_config_entry()
    yield config_entry


@pytest.fixture
def mock_add_config_entry(
    hass: HomeAssistant, mock_config_entry: MockConfigEntry, mock_pydrawise: Any
) -> TCallable[[], Awaitable[MockConfigEntry]]:
    """Callable that creates a mock ConfigEntry that's been added to HA."""

    async def callback() -> MockConfigEntry:
        mock_config_entry.add_to_hass(hass)
        await hass.config_entries.async_setup(mock_config_entry.entry_id)
        await hass.async_block_till_done()
        assert DOMAIN in hass.config_entries.async_domains()
        return mock_config_entry

    return callback