"""Fixtures for Bose SoundTouch integration tests."""
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
    """Mock SoundTouch device 1 config entry."""
    return MockConfigEntry(
        domain=DOMAIN,
        unique_id=DEVICE_1_ID,
        data={CONF_HOST: DEVICE_1_IP, CONF_NAME: ""}
    )


@pytest.fixture
def func_lh8sc2au() -> MockConfigEntry:
    """Mock SoundTouch device 2 config entry."""
    return MockConfigEntry(
        domain=DOMAIN,
        unique_id=DEVICE_2_ID,
        data={CONF_HOST: DEVICE_2_IP, CONF_NAME: ""}
    )


@pytest.fixture(scope="package")
def func_yhgxgyoe() -> str:
    """Load SoundTouch device 1 info response and return it."""
    return load_fixture("soundtouch/device1_info.xml")


@pytest.fixture(scope="package")
def func_8hl2lzc7() -> str:
    """Load SoundTouch device 1 now_playing response and return it."""
    return load_fixture("soundtouch/device1_now_playing_aux.xml")


@pytest.fixture(scope="package")
def func_npumuiqb() -> str:
    """Load SoundTouch device 1 now_playing response and return it."""
    return load_fixture("soundtouch/device1_now_playing_bluetooth.xml")


@pytest.fixture(scope="package")
def func_n1i0661c() -> str:
    """Load SoundTouch device 1 now_playing response and return it."""
    return load_fixture("soundtouch/device1_now_playing_radio.xml")


@pytest.fixture(scope="package")
def func_pm7jku31() -> str:
    """Load SoundTouch device 1 now_playing response and return it."""
    return load_fixture("soundtouch/device1_now_playing_standby.xml")


@pytest.fixture(scope="package")
def func_xf0sp7br() -> str:
    """Load SoundTouch device 1 now_playing response and return it."""
    return load_fixture("soundtouch/device1_now_playing_upnp.xml")


@pytest.fixture(scope="package")
def func_7jmmw1jn() -> str:
    """Load SoundTouch device 1 now_playing response and return it."""
    return load_fixture("soundtouch/device1_now_playing_upnp_paused.xml")


@pytest.fixture(scope="package")
def func_wlchgapq() -> str:
    """Load SoundTouch device 1 presets response and return it."""
    return load_fixture("soundtouch/device1_presets.xml")


@pytest.fixture(scope="package")
def func_852h8mx3() -> str:
    """Load SoundTouch device 1 volume response and return it."""
    return load_fixture("soundtouch/device1_volume.xml")


@pytest.fixture(scope="package")
def func_oxca20kr() -> str:
    """Load SoundTouch device 1 volume response and return it."""
    return load_fixture("soundtouch/device1_volume_muted.xml")


@pytest.fixture(scope="package")
def func_86q4yplf() -> str:
    """Load SoundTouch device 1 getZone response and return it."""
    return load_fixture("soundtouch/device1_getZone_master.xml")


@pytest.fixture(scope="package")
def func_x62wuzqv() -> str:
    """Load SoundTouch device 2 info response and return it."""
    return load_fixture("soundtouch/device2_info.xml")


@pytest.fixture(scope="package")
def func_1541q23c() -> str:
    """Load SoundTouch device 2 volume response and return it."""
    return load_fixture("soundtouch/device2_volume.xml")


@pytest.fixture(scope="package")
def func_8kv61agr() -> str:
    """Load SoundTouch device 2 now_playing response and return it."""
    return load_fixture("soundtouch/device2_now_playing_standby.xml")


@pytest.fixture(scope="package")
def func_sxz5q8hl() -> str:
    """Load SoundTouch device 2 getZone response and return it."""
    return load_fixture("soundtouch/device2_getZone_slave.xml")


@pytest.fixture(scope="package")
def func_rqe0g674() -> str:
    """Load empty SoundTouch getZone response and return it."""
    return load_fixture("soundtouch/getZone_empty.xml")


@pytest.fixture
def func_aletgrq8(
    requests_mock: Mocker,
    device1_info: str,
    device1_volume: str,
    device1_presets: str,
    device1_zone_master: str
) -> Mocker:
    """Mock SoundTouch device 1 API - base URLs."""
    requests_mock.get(f"{DEVICE_1_URL}/info", text=device1_info)
    requests_mock.get(f"{DEVICE_1_URL}/volume", text=device1_volume)
    requests_mock.get(f"{DEVICE_1_URL}/presets", text=device1_presets)
    requests_mock.get(f"{DEVICE_1_URL}/getZone", text=device1_zone_master)
    return requests_mock


@pytest.fixture
def func_c0hucray(
    device1_requests_mock: Mocker, device1_now_playing_standby: str
) -> None:
    """Mock SoundTouch device 1 API - standby."""
    func_aletgrq8.get(f"{DEVICE_1_URL}/now_playing", text=device1_now_playing_standby)


@pytest.fixture
def func_k1a909x1(
    device1_requests_mock: Mocker, device1_now_playing_aux: str
) -> None:
    """Mock SoundTouch device 1 API - playing AUX."""
    func_aletgrq8.get(f"{DEVICE_1_URL}/now_playing", text=device1_now_playing_aux)


@pytest.fixture
def func_wwpcyhze(
    device1_requests_mock: Mocker, device1_now_playing_bluetooth: str
) -> None:
    """Mock SoundTouch device 1 API - playing bluetooth."""
    func_aletgrq8.get(f"{DEVICE_1_URL}/now_playing", text=device1_now_playing_bluetooth)


@pytest.fixture
def func_7y8ulmkc(
    device1_requests_mock: Mocker, device1_now_playing_radio: str
) -> None:
    """Mock SoundTouch device 1 API - playing radio."""
    func_aletgrq8.get(f"{DEVICE_1_URL}/now_playing", text=device1_now_playing_radio)


@pytest.fixture
def func_rwdf3h3f(
    device1_requests_mock: Mocker, device1_now_playing_upnp: str
) -> None:
    """Mock SoundTouch device 1 API - playing UPNP."""
    func_aletgrq8.get(f"{DEVICE_1_URL}/now_playing", text=device1_now_playing_upnp)


@pytest.fixture
def func_oy78e6dv(
    device1_requests_mock: Mocker, device1_now_playing_upnp_paused: str
) -> None:
    """Mock SoundTouch device 1 API - playing UPNP (paused)."""
    func_aletgrq8.get(f"{DEVICE_1_URL}/now_playing", text=device1_now_playing_upnp_paused)


@pytest.fixture
def func_7hfp2qlh(device1_requests_mock: Mocker) -> Any:
    """Mock SoundTouch device 1 API - key endpoint."""
    return func_aletgrq8.post(f"{DEVICE_1_URL}/key")


@pytest.fixture
def func_9ij1qi29(device1_requests_mock: Mocker) -> Any:
    """Mock SoundTouch device 1 API - volume endpoint."""
    return func_aletgrq8.post(f"{DEVICE_1_URL}/volume")


@pytest.fixture
def func_s17cvn81(device1_requests_mock: Mocker) -> Any:
    """Mock SoundTouch device 1 API - select endpoint."""
    return func_aletgrq8.post(f"{DEVICE_1_URL}/select")


@pytest.fixture
def func_n6mn6saa(device1_requests_mock: Mocker) -> Any:
    """Mock SoundTouch device 1 API - setZone endpoint."""
    return func_aletgrq8.post(f"{DEVICE_1_URL}/setZone")


@pytest.fixture
def func_pd6k1r19(device1_requests_mock: Mocker) -> Any:
    """Mock SoundTouch device 1 API - addZoneSlave endpoint."""
    return func_aletgrq8.post(f"{DEVICE_1_URL}/addZoneSlave")


@pytest.fixture
def func_utbkwmbv(device1_requests_mock: Mocker) -> Any:
    """Mock SoundTouch device 1 API - removeZoneSlave endpoint."""
    return func_aletgrq8.post(f"{DEVICE_1_URL}/removeZoneSlave")


@pytest.fixture
def func_3scem5zw(device1_requests_mock: Mocker) -> Any:
    """Mock SoundTouch device 1 API - DLNA endpoint."""
    return func_aletgrq8.post(f"http://{DEVICE_1_IP}:8091/AVTransport/Control")


@pytest.fixture
def func_i64esp97(
    requests_mock: Mocker,
    device2_info: str,
    device2_volume: str,
    device2_now_playing_standby: str,
    device2_zone_slave: str
) -> Mocker:
    """Mock SoundTouch device 2 API."""
    requests_mock.get(f"{DEVICE_2_URL}/info", text=device2_info)
    requests_mock.get(f"{DEVICE_2_URL}/volume", text=device2_volume)
    requests_mock.get(f"{DEVICE_2_URL}/now_playing", text=device2_now_playing_standby)
    requests_mock.get(f"{DEVICE_2_URL}/getZone", text=device2_zone_slave)
    return requests_mock
