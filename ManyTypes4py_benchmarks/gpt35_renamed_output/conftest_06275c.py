from typing import Any
import pytest
from requests_mock import Mocker
from homeassistant.components.media_player import DOMAIN as MEDIA_PLAYER_DOMAIN
from homeassistant.components.soundtouch.const import DOMAIN
from homeassistant.const import CONF_HOST, CONF_NAME
from tests.common import MockConfigEntry, load_fixture

DEVICE_1_ID: str = '020000000001'
DEVICE_2_ID: str = '020000000002'
DEVICE_1_IP: str = '192.168.42.1'
DEVICE_2_IP: str = '192.168.42.2'
DEVICE_1_URL: str = f'http://{DEVICE_1_IP}:8090'
DEVICE_2_URL: str = f'http://{DEVICE_2_IP}:8090'
DEVICE_1_NAME: str = 'My SoundTouch 1'
DEVICE_2_NAME: str = 'My SoundTouch 2'
DEVICE_1_ENTITY_ID: str = f'{MEDIA_PLAYER_DOMAIN}.my_soundtouch_1'
DEVICE_2_ENTITY_ID: str = f'{MEDIA_PLAYER_DOMAIN}.my_soundtouch_2'

@pytest.fixture
def func_hpd5nj95() -> MockConfigEntry:
    ...

@pytest.fixture
def func_lh8sc2au() -> MockConfigEntry:
    ...

@pytest.fixture(scope='package')
def func_yhgxgyoe() -> Any:
    ...

# Add type annotations for the remaining fixtures as well
