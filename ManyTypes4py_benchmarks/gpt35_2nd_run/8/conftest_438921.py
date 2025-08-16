from __future__ import annotations
from typing import Any, Dict, List
from unittest.mock import AsyncMock, MagicMock, patch
from pylitterbot import Account, FeederRobot, LitterRobot3, LitterRobot4, Pet, Robot
from pylitterbot.exceptions import InvalidCommandException
import pytest
from homeassistant.core import HomeAssistant
from .common import CONFIG, DOMAIN, FEEDER_ROBOT_DATA, PET_DATA, ROBOT_4_DATA, ROBOT_DATA
from tests.common import MockConfigEntry

def create_mock_robot(robot_data: Dict[str, Any], account: Account, v4: bool, feeder: bool, side_effect: Any = None) -> Robot:
    ...

def create_mock_account(robot_data: Dict[str, Any] = None, side_effect: Any = None, skip_robots: bool = False, v4: bool = False, feeder: bool = False, pet: bool = False) -> Account:
    ...

@pytest.fixture
def mock_account() -> Account:
    ...

@pytest.fixture
def mock_account_with_litterrobot_4() -> Account:
    ...

@pytest.fixture
def mock_account_with_feederrobot() -> Account:
    ...

@pytest.fixture
def mock_account_with_pet() -> Account:
    ...

@pytest.fixture
def mock_account_with_no_robots() -> Account:
    ...

@pytest.fixture
def mock_account_with_sleeping_robot() -> Account:
    ...

@pytest.fixture
def mock_account_with_sleep_disabled_robot() -> Account:
    ...

@pytest.fixture
def mock_account_with_error() -> Account:
    ...

@pytest.fixture
def mock_account_with_side_effects() -> Account:
    ...

async def setup_integration(hass: HomeAssistant, mock_account: Account, platform_domain: str = None) -> MockConfigEntry:
    ...
