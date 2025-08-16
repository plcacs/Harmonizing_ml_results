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
def device1_config() -> MockConfigEntry:
    ...

@pytest.fixture
def device2_config() -> MockConfigEntry:
    ...

@pytest.fixture(scope='package')
def device1_info() -> Any:
    ...

@pytest.fixture(scope='package')
def device1_now_playing_aux() -> Any:
    ...

@pytest.fixture(scope='package')
def device1_now_playing_bluetooth() -> Any:
    ...

@pytest.fixture(scope='package')
def device1_now_playing_radio() -> Any:
    ...

@pytest.fixture(scope='package')
def device1_now_playing_standby() -> Any:
    ...

@pytest.fixture(scope='package')
def device1_now_playing_upnp() -> Any:
    ...

@pytest.fixture(scope='package')
def device1_now_playing_upnp_paused() -> Any:
    ...

@pytest.fixture(scope='package')
def device1_presets() -> Any:
    ...

@pytest.fixture(scope='package')
def device1_volume() -> Any:
    ...

@pytest.fixture(scope='package')
def device1_volume_muted() -> Any:
    ...

@pytest.fixture(scope='package')
def device1_zone_master() -> Any:
    ...

@pytest.fixture(scope='package')
def device2_info() -> Any:
    ...

@pytest.fixture(scope='package')
def device2_volume() -> Any:
    ...

@pytest.fixture(scope='package')
def device2_now_playing_standby() -> Any:
    ...

@pytest.fixture(scope='package')
def device2_zone_slave() -> Any:
    ...

@pytest.fixture(scope='package')
def zone_empty() -> Any:
    ...

@pytest.fixture
def device1_requests_mock(requests_mock: Mocker, device1_info: Any, device1_volume: Any, device1_presets: Any, device1_zone_master: Any) -> Mocker:
    ...

@pytest.fixture
def device1_requests_mock_standby(device1_requests_mock: Mocker, device1_now_playing_standby: Any) -> None:
    ...

@pytest.fixture
def device1_requests_mock_aux(device1_requests_mock: Mocker, device1_now_playing_aux: Any) -> None:
    ...

@pytest.fixture
def device1_requests_mock_bluetooth(device1_requests_mock: Mocker, device1_now_playing_bluetooth: Any) -> None:
    ...

@pytest.fixture
def device1_requests_mock_radio(device1_requests_mock: Mocker, device1_now_playing_radio: Any) -> None:
    ...

@pytest.fixture
def device1_requests_mock_upnp(device1_requests_mock: Mocker, device1_now_playing_upnp: Any) -> None:
    ...

@pytest.fixture
def device1_requests_mock_upnp_paused(device1_requests_mock: Mocker, device1_now_playing_upnp_paused: Any) -> None:
    ...

@pytest.fixture
def device1_requests_mock_key(device1_requests_mock: Mocker) -> Any:
    ...

@pytest.fixture
def device1_requests_mock_volume(device1_requests_mock: Mocker) -> Any:
    ...

@pytest.fixture
def device1_requests_mock_select(device1_requests_mock: Mocker) -> Any:
    ...

@pytest.fixture
def device1_requests_mock_set_zone(device1_requests_mock: Mocker) -> Any:
    ...

@pytest.fixture
def device1_requests_mock_add_zone_slave(device1_requests_mock: Mocker) -> Any:
    ...

@pytest.fixture
def device1_requests_mock_remove_zone_slave(device1_requests_mock: Mocker) -> Any:
    ...

@pytest.fixture
def device1_requests_mock_dlna(device1_requests_mock: Mocker) -> Any:
    ...

@pytest.fixture
def device2_requests_mock_standby(requests_mock: Mocker, device2_info: Any, device2_volume: Any, device2_now_playing_standby: Any, device2_zone_slave: Any) -> Mocker:
    ...
