"""Test fixtures for rainbird."""
from __future__ import annotations
from collections.abc import Generator
from http import HTTPStatus
import json
from typing import Any, Dict, List, Optional, Union, Callable, Awaitable
from unittest.mock import patch
from pyrainbird import encryption
import pytest
from homeassistant.components.rainbird import DOMAIN
from homeassistant.components.rainbird.const import ATTR_DURATION, DEFAULT_TRIGGER_TIME_MINUTES
from homeassistant.const import EVENT_HOMEASSISTANT_CLOSE, Platform
from homeassistant.core import HomeAssistant
from tests.common import MockConfigEntry
from tests.test_util.aiohttp import AiohttpClientMocker, AiohttpClientMockResponse

HOST: str = 'example.com'
URL: str = 'http://example.com/stick'
PASSWORD: str = 'password'
SERIAL_NUMBER: int = 1263613994342
MAC_ADDRESS: str = '4C:A1:61:00:11:22'
MAC_ADDRESS_UNIQUE_ID: str = '4c:a1:61:00:11:22'
SERIAL_RESPONSE: str = '850000012635436566'
ZERO_SERIAL_RESPONSE: str = '850000000000000000'
MODEL_AND_VERSION_RESPONSE: str = '820005090C'
AVAILABLE_STATIONS_RESPONSE: str = '83017F000000'
EMPTY_STATIONS_RESPONSE: str = '830000000000'
ZONE_3_ON_RESPONSE: str = 'BF0004000000'
ZONE_5_ON_RESPONSE: str = 'BF0010000000'
ZONE_OFF_RESPONSE: str = 'BF0000000000'
ZONE_STATE_OFF_RESPONSE: str = 'BF0000000000'
RAIN_SENSOR_OFF: str = 'BE00'
RAIN_SENSOR_ON: str = 'BE01'
RAIN_DELAY: str = 'B60010'
RAIN_DELAY_OFF: str = 'B60000'
ACK_ECHO: str = '0106'
WIFI_PARAMS_RESPONSE: Dict[str, Union[str, int]] = {
    'macAddress': MAC_ADDRESS,
    'localIpAddress': '1.1.1.38',
    'localNetmask': '255.255.255.0',
    'localGateway': '1.1.1.1',
    'rssi': -61,
    'wifiSsid': 'wifi-ssid-name',
    'wifiPassword': 'wifi-password-name',
    'wifiSecurity': 'wpa2-aes',
    'apTimeoutNoLan': 20,
    'apTimeoutIdle': 20,
    'apSecurity': 'unknown',
    'stickVersion': 'Rain Bird Stick Rev C/1.63'
}
CONFIG: Dict[str, Any] = {DOMAIN: {'host': HOST, 'password': PASSWORD, 'trigger_time': {'minutes': 6}}}
CONFIG_ENTRY_DATA_OLD_FORMAT: Dict[str, Any] = {'host': HOST, 'password': PASSWORD, 'serial_number': SERIAL_NUMBER}
CONFIG_ENTRY_DATA: Dict[str, Any] = {'host': HOST, 'password': PASSWORD, 'serial_number': SERIAL_NUMBER, 'mac': MAC_ADDRESS}


@pytest.fixture
def platforms() -> List[Platform]:
    """Fixture to specify platforms to test."""
    return []


@pytest.fixture
async def config_entry_unique_id() -> str:
    """Fixture for config entry unique id."""
    return MAC_ADDRESS_UNIQUE_ID


@pytest.fixture
async def serial_number() -> int:
    """Fixture for serial number used in the config entry data."""
    return SERIAL_NUMBER


@pytest.fixture
async def config_entry_data(serial_number: int) -> Dict[str, Any]:
    """Fixture for MockConfigEntry data."""
    return {**CONFIG_ENTRY_DATA, 'serial_number': serial_number}


@pytest.fixture
async def config_entry(
    config_entry_data: Dict[str, Any], config_entry_unique_id: str
) -> Optional[MockConfigEntry]:
    """Fixture for MockConfigEntry."""
    if config_entry_data is None:
        return None
    return MockConfigEntry(
        unique_id=config_entry_unique_id,
        domain=DOMAIN,
        data=config_entry_data,
        options={ATTR_DURATION: DEFAULT_TRIGGER_TIME_MINUTES},
    )


@pytest.fixture(autouse=True)
async def add_config_entry(hass: HomeAssistant, config_entry: Optional[MockConfigEntry]) -> None:
    """Fixture to add the config entry."""
    if config_entry:
        config_entry.add_to_hass(hass)


@pytest.fixture(autouse=True)
def setup_platforms(hass: HomeAssistant, platforms: List[Platform]) -> Generator[None, None, None]:
    """Fixture for setting up the default platforms."""
    with patch(f'homeassistant.components.{DOMAIN}.PLATFORMS', platforms):
        yield


@pytest.fixture(autouse=True)
def aioclient_mock(hass: HomeAssistant) -> Generator[AiohttpClientMocker, None, None]:
    """Context manager to mock aiohttp client."""
    mocker: AiohttpClientMocker = AiohttpClientMocker()

    def create_session() -> Any:
        session = mocker.create_session(hass.loop)

        async def close_session(event: Any) -> None:
            """Close session."""
            await session.close()

        hass.bus.async_listen_once(EVENT_HOMEASSISTANT_CLOSE, close_session)
        return session

    with patch(
        'homeassistant.components.rainbird.async_create_clientsession',
        side_effect=create_session,
    ), patch(
        'homeassistant.components.rainbird.config_flow.async_create_clientsession',
        side_effect=create_session,
    ):
        yield mocker


def rainbird_json_response(result: Any) -> str:
    """Create a fake API response."""
    return encryption.encrypt(f'{{"jsonrpc": "2.0", "result": {json.dumps(result)}, "id": 1}} ', PASSWORD)


def mock_json_response(result: Any) -> AiohttpClientMockResponse:
    """Create a fake AiohttpClientMockResponse."""
    return AiohttpClientMockResponse('POST', URL, response=rainbird_json_response(result))


def mock_response(data: Any) -> AiohttpClientMockResponse:
    """Create a fake AiohttpClientMockResponse."""
    return mock_json_response({'data': data})


def mock_response_error(status: Union[int, HTTPStatus] = HTTPStatus.SERVICE_UNAVAILABLE) -> AiohttpClientMockResponse:
    """Create a fake AiohttpClientMockResponse."""
    return AiohttpClientMockResponse('POST', URL, status=status)


@pytest.fixture(name='stations_response')
def mock_station_response() -> str:
    """Mock response to return available stations."""
    return AVAILABLE_STATIONS_RESPONSE


@pytest.fixture(name='zone_state_response')
def mock_zone_state_response() -> str:
    """Mock response to return zone states."""
    return ZONE_STATE_OFF_RESPONSE


@pytest.fixture(name='rain_response')
def mock_rain_response() -> str:
    """Mock response to return rain sensor state."""
    return RAIN_SENSOR_OFF


@pytest.fixture(name='rain_delay_response')
def mock_rain_delay_response() -> str:
    """Mock response to return rain delay state."""
    return RAIN_DELAY_OFF


@pytest.fixture(name='model_and_version_response')
def mock_model_and_version_response() -> str:
    """Mock response to return rain delay state."""
    return MODEL_AND_VERSION_RESPONSE


@pytest.fixture(name='api_responses')
def mock_api_responses(
    model_and_version_response: str,
    stations_response: str,
    zone_state_response: str,
    rain_response: str,
    rain_delay_response: str,
) -> List[str]:
    """Fixture to set up a list of fake API responsees for tests to extend.

    These are returned in the order they are requested by the update coordinator.
    """
    return [
        model_and_version_response,
        stations_response,
        zone_state_response,
        rain_response,
        rain_delay_response,
    ]


@pytest.fixture(name='responses')
def mock_responses(api_responses: List[str]) -> List[AiohttpClientMockResponse]:
    """Fixture to set up a list of fake API responsees for tests to extend."""
    return [mock_response(api_response) for api_response in api_responses]


@pytest.fixture(autouse=True)
def handle_responses(aioclient_mock: AiohttpClientMocker, responses: List[AiohttpClientMockResponse]) -> Generator[None, None, None]:
    """Fixture for command mocking for fake responses to the API url."""

    async def handle(method: str, url: str, data: Any) -> AiohttpClientMockResponse:
        return responses.pop(0)

    aioclient_mock.post(URL, side_effect=handle)
    yield
