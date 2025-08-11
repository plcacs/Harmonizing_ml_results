"""Configure pytest for Litter-Robot tests."""
from __future__ import annotations
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch
from pylitterbot import Account, FeederRobot, LitterRobot3, LitterRobot4, Pet, Robot
from pylitterbot.exceptions import InvalidCommandException
import pytest
from homeassistant.core import HomeAssistant
from .common import CONFIG, DOMAIN, FEEDER_ROBOT_DATA, PET_DATA, ROBOT_4_DATA, ROBOT_DATA
from tests.common import MockConfigEntry

def create_mock_robot(robot_data: Union[dict[str, typing.Any], float, str], account: Union[dict[str, typing.Any], float, str], v4: Union[bool, typing.Sequence[typing.Union[str,typing.Any]]], feeder: Union[bool, typing.Sequence[typing.Union[str,typing.Any]]], side_effect: Union[None, dict[str, typing.Any], dict[str, dict[str, typing.Any]], int]=None) -> Union[LitterRobot3, FeederRobot, LitterRobot4]:
    """Create a mock Litter-Robot device."""
    if not robot_data:
        robot_data = {}
    if v4:
        robot = LitterRobot4(data={**ROBOT_4_DATA, **robot_data}, account=account)
    elif feeder:
        robot = FeederRobot(data={**FEEDER_ROBOT_DATA, **robot_data}, account=account)
    else:
        robot = LitterRobot3(data={**ROBOT_DATA, **robot_data}, account=account)
    robot.start_cleaning = AsyncMock(side_effect=side_effect)
    robot.set_power_status = AsyncMock(side_effect=side_effect)
    robot.reset_waste_drawer = AsyncMock(side_effect=side_effect)
    robot.set_sleep_mode = AsyncMock(side_effect=side_effect)
    robot.set_night_light = AsyncMock(side_effect=side_effect)
    robot.set_panel_lockout = AsyncMock(side_effect=side_effect)
    robot.set_wait_time = AsyncMock(side_effect=side_effect)
    robot.refresh = AsyncMock(side_effect=side_effect)
    return robot

def create_mock_account(robot_data: Union[None, bool, str, tuple[bool]]=None, side_effect: Union[None, bool, str, tuple[bool]]=None, skip_robots: bool=False, v4: bool=False, feeder: bool=False, pet: bool=False) -> MagicMock:
    """Create a mock Litter-Robot account."""
    account = MagicMock(spec=Account)
    account.connect = AsyncMock()
    account.refresh_robots = AsyncMock()
    account.robots = [] if skip_robots else [create_mock_robot(robot_data, account, v4, feeder, side_effect)]
    account.pets = [Pet(PET_DATA, account.session)] if pet else []
    return account

@pytest.fixture
def mock_account() -> unittesmock.MagicMock:
    """Mock a Litter-Robot account."""
    return create_mock_account()

@pytest.fixture
def mock_account_with_litterrobot_4() -> Union[str, dict[str, str], unittesmock.MagicMock]:
    """Mock account with Litter-Robot 4."""
    return create_mock_account(v4=True)

@pytest.fixture
def mock_account_with_feederrobot() -> str:
    """Mock account with Feeder-Robot."""
    return create_mock_account(feeder=True)

@pytest.fixture
def mock_account_with_pet() -> Union[unittesmock.MagicMock, str]:
    """Mock account with Feeder-Robot."""
    return create_mock_account(pet=True)

@pytest.fixture
def mock_account_with_no_robots() -> Union[str, tests.test_util.aiohttp.AiohttpClientMocker, dict]:
    """Mock a Litter-Robot account."""
    return create_mock_account(skip_robots=True)

@pytest.fixture
def mock_account_with_sleeping_robot() -> Union[str, dict]:
    """Mock a Litter-Robot account with a sleeping robot."""
    return create_mock_account({'sleepModeActive': '102:00:00'})

@pytest.fixture
def mock_account_with_sleep_disabled_robot() -> Union[str, boucanpy.core.http_server.HttpServerRepo, list]:
    """Mock a Litter-Robot account with a robot that has sleep mode disabled."""
    return create_mock_account({'sleepModeActive': '0'})

@pytest.fixture
def mock_account_with_error() -> Union[unittesmock.MagicMock, dict]:
    """Mock a Litter-Robot account with error."""
    return create_mock_account({'unitStatus': 'BR'})

@pytest.fixture
def mock_account_with_side_effects() -> Union[tuple[str], str, None, boucanpy.core.security.TokenPayload]:
    """Mock a Litter-Robot account with side effects."""
    return create_mock_account(side_effect=InvalidCommandException('Invalid command: oops'))

async def setup_integration(hass, mock_account, platform_domain=None):
    """Load a Litter-Robot platform with the provided coordinator."""
    entry = MockConfigEntry(domain=DOMAIN, data=CONFIG[DOMAIN])
    entry.add_to_hass(hass)
    with patch('homeassistant.components.litterrobot.coordinator.Account', return_value=mock_account):
        await hass.config_entries.async_setup(entry.entry_id)
        await hass.async_block_till_done()
    return entry