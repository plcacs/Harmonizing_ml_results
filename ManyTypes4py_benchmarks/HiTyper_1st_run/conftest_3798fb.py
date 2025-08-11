"""Test fixtures for rainbird."""
from __future__ import annotations
from collections.abc import Generator
from http import HTTPStatus
import json
from typing import Any
from unittest.mock import patch
from pyrainbird import encryption
import pytest
from homeassistant.components.rainbird import DOMAIN
from homeassistant.components.rainbird.const import ATTR_DURATION, DEFAULT_TRIGGER_TIME_MINUTES
from homeassistant.const import EVENT_HOMEASSISTANT_CLOSE, Platform
from homeassistant.core import HomeAssistant
from tests.common import MockConfigEntry
from tests.test_util.aiohttp import AiohttpClientMocker, AiohttpClientMockResponse
HOST = 'example.com'
URL = 'http://example.com/stick'
PASSWORD = 'password'
SERIAL_NUMBER = 1263613994342
MAC_ADDRESS = '4C:A1:61:00:11:22'
MAC_ADDRESS_UNIQUE_ID = '4c:a1:61:00:11:22'
SERIAL_RESPONSE = '850000012635436566'
ZERO_SERIAL_RESPONSE = '850000000000000000'
MODEL_AND_VERSION_RESPONSE = '820005090C'
AVAILABLE_STATIONS_RESPONSE = '83017F000000'
EMPTY_STATIONS_RESPONSE = '830000000000'
ZONE_3_ON_RESPONSE = 'BF0004000000'
ZONE_5_ON_RESPONSE = 'BF0010000000'
ZONE_OFF_RESPONSE = 'BF0000000000'
ZONE_STATE_OFF_RESPONSE = 'BF0000000000'
RAIN_SENSOR_OFF = 'BE00'
RAIN_SENSOR_ON = 'BE01'
RAIN_DELAY = 'B60010'
RAIN_DELAY_OFF = 'B60000'
ACK_ECHO = '0106'
WIFI_PARAMS_RESPONSE = {'macAddress': MAC_ADDRESS, 'localIpAddress': '1.1.1.38', 'localNetmask': '255.255.255.0', 'localGateway': '1.1.1.1', 'rssi': -61, 'wifiSsid': 'wifi-ssid-name', 'wifiPassword': 'wifi-password-name', 'wifiSecurity': 'wpa2-aes', 'apTimeoutNoLan': 20, 'apTimeoutIdle': 20, 'apSecurity': 'unknown', 'stickVersion': 'Rain Bird Stick Rev C/1.63'}
CONFIG = {DOMAIN: {'host': HOST, 'password': PASSWORD, 'trigger_time': {'minutes': 6}}}
CONFIG_ENTRY_DATA_OLD_FORMAT = {'host': HOST, 'password': PASSWORD, 'serial_number': SERIAL_NUMBER}
CONFIG_ENTRY_DATA = {'host': HOST, 'password': PASSWORD, 'serial_number': SERIAL_NUMBER, 'mac': MAC_ADDRESS}

@pytest.fixture
def platforms() -> list:
    """Fixture to specify platforms to test."""
    return []

@pytest.fixture
async def config_entry_unique_id():
    """Fixture for config entry unique id."""
    return MAC_ADDRESS_UNIQUE_ID

@pytest.fixture
async def serial_number():
    """Fixture for serial number used in the config entry data."""
    return SERIAL_NUMBER

@pytest.fixture
async def config_entry_data(serial_number):
    """Fixture for MockConfigEntry data."""
    return {**CONFIG_ENTRY_DATA, 'serial_number': serial_number}

@pytest.fixture
async def config_entry(config_entry_data, config_entry_unique_id):
    """Fixture for MockConfigEntry."""
    if config_entry_data is None:
        return None
    return MockConfigEntry(unique_id=config_entry_unique_id, domain=DOMAIN, data=config_entry_data, options={ATTR_DURATION: DEFAULT_TRIGGER_TIME_MINUTES})

@pytest.fixture(autouse=True)
async def add_config_entry(hass, config_entry):
    """Fixture to add the config entry."""
    if config_entry:
        config_entry.add_to_hass(hass)

@pytest.fixture(autouse=True)
def setup_platforms(hass: Union[str, list[str]], platforms: str) -> typing.Generator:
    """Fixture for setting up the default platforms."""
    with patch(f'homeassistant.components.{DOMAIN}.PLATFORMS', platforms):
        yield

@pytest.fixture(autouse=True)
def aioclient_mock(hass: Union[homeassistancore.HomeAssistant, aiohttp.ClientSession, tests.async_mock.Mock]) -> typing.Generator[AiohttpClientMocker]:
    """Context manager to mock aiohttp client."""
    mocker = AiohttpClientMocker()

    def create_session():
        session = mocker.create_session(hass.loop)

        async def close_session(event):
            """Close session."""
            await session.close()
        hass.bus.async_listen_once(EVENT_HOMEASSISTANT_CLOSE, close_session)
        return session
    with patch('homeassistant.components.rainbird.async_create_clientsession', side_effect=create_session), patch('homeassistant.components.rainbird.config_flow.async_create_clientsession', side_effect=create_session):
        yield mocker

def rainbird_json_response(result: Union[int, dict, aiohttp.clienClientResponse]):
    """Create a fake API response."""
    return encryption.encrypt(f'{{"jsonrpc": "2.0", "result": {json.dumps(result)}, "id": 1}} ', PASSWORD)

def mock_json_response(result: Union[preacher.core.requesExecutionReport, dict, str]) -> AiohttpClientMockResponse:
    """Create a fake AiohttpClientMockResponse."""
    return AiohttpClientMockResponse('POST', URL, response=rainbird_json_response(result))

def mock_response(data: Union[dict, dict[str, typing.Any], typing.Mapping]) -> Union[str, dict]:
    """Create a fake AiohttpClientMockResponse."""
    return mock_json_response({'data': data})

def mock_response_error(status: Any=HTTPStatus.SERVICE_UNAVAILABLE) -> AiohttpClientMockResponse:
    """Create a fake AiohttpClientMockResponse."""
    return AiohttpClientMockResponse('POST', URL, status=status)

@pytest.fixture(name='stations_response')
def mock_station_response() -> Union[list[str], list, str]:
    """Mock response to return available stations."""
    return AVAILABLE_STATIONS_RESPONSE

@pytest.fixture(name='zone_state_response')
def mock_zone_state_response() -> Union[dict[str, typing.Union[int,None]], dict]:
    """Mock response to return zone states."""
    return ZONE_STATE_OFF_RESPONSE

@pytest.fixture(name='rain_response')
def mock_rain_response() -> Union[int, str]:
    """Mock response to return rain sensor state."""
    return RAIN_SENSOR_OFF

@pytest.fixture(name='rain_delay_response')
def mock_rain_delay_response() -> Union[int, float]:
    """Mock response to return rain delay state."""
    return RAIN_DELAY_OFF

@pytest.fixture(name='model_and_version_response')
def mock_model_and_version_response() -> Union[str, list[str], dict[str, str]]:
    """Mock response to return rain delay state."""
    return MODEL_AND_VERSION_RESPONSE

@pytest.fixture(name='api_responses')
def mock_api_responses(model_and_version_response: Union[dict[str, typing.Any], int, None, dict], stations_response: Union[dict[str, typing.Any], int, None, dict], zone_state_response: Union[dict[str, typing.Any], int, None, dict], rain_response: Union[dict[str, typing.Any], int, None, dict], rain_delay_response: Union[dict[str, typing.Any], int, None, dict]) -> list[typing.Union[dict[str, typing.Any],int,None,dict]]:
    """Fixture to set up a list of fake API responsees for tests to extend.

    These are returned in the order they are requested by the update coordinator.
    """
    return [model_and_version_response, stations_response, zone_state_response, rain_response, rain_delay_response]

@pytest.fixture(name='responses')
def mock_responses(api_responses: Union[django.http.HttpRequest, aiohttp.web.Request, aiohttp.web.Response]) -> list:
    """Fixture to set up a list of fake API responsees for tests to extend."""
    return [mock_response(api_response) for api_response in api_responses]

@pytest.fixture(autouse=True)
def handle_responses(aioclient_mock: Any, responses: Union[dict[str, dict], aiohttp.web.Response, mock.Mock]) -> None:
    """Fixture for command mocking for fake responses to the API url."""

    async def handle(method, url, data):
        return responses.pop(0)
    aioclient_mock.post(URL, side_effect=handle)