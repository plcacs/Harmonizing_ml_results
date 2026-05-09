from __future__ import annotations
from collections.abc import Generator
from http import HTTPStatus
import json
from typing import Any, Generator

@pytest.fixture
async def platforms() -> list[str]:  # type hint added
    """Fixture to specify platforms to test."""
    return []

@pytest.fixture
async def config_entry_unique_id() -> str:  # type hint added
    """Fixture for config entry unique id."""
    return MAC_ADDRESS_UNIQUE_ID

@pytest.fixture
async def serial_number() -> int:  # type hint added
    """Fixture for serial number used in the config entry data."""
    return SERIAL_NUMBER

@pytest.fixture
async def config_entry_data(serial_number: int) -> dict[str, Any]:  # type hint added
    """Fixture for MockConfigEntry data."""
    return {**CONFIG_ENTRY_DATA, 'serial_number': serial_number}

@pytest.fixture
async def config_entry(config_entry_data: dict[str, Any], config_entry_unique_id: str) -> MockConfigEntry:
    """Fixture for MockConfigEntry."""
    if config_entry_data is None:
        return None
    return MockConfigEntry(unique_id=config_entry_unique_id, domain=DOMAIN, data=config_entry_data, options={ATTR_DURATION: DEFAULT_TRIGGER_TIME_MINUTES})

@pytest.fixture(autouse=True)
async def add_config_entry(hass: HomeAssistant, config_entry: MockConfigEntry) -> None:
    """Fixture to add the config entry."""
    if config_entry:
        config_entry.add_to_hass(hass)

@pytest.fixture(autouse=True)
def setup_platforms(hass: HomeAssistant, platforms: list[str]) -> Generator[None, None, None]:
    """Fixture for setting up the default platforms."""
    with patch(f'homeassistant.components.{DOMAIN}.PLATFORMS', platforms):
        yield

@pytest.fixture(autouse=True)
def aioclient_mock(hass: HomeAssistant) -> AiohttpClientMocker:
    """Context manager to mock aiohttp client."""
    mocker = AiohttpClientMocker()

    def create_session() -> AiohttpSession:
        session = mocker.create_session(hass.loop)

        async def close_session(event: Event) -> None:
            """Close session."""
            await session.close()
        hass.bus.async_listen_once(EVENT_HOMEASSISTANT_CLOSE, close_session)
        return session
    with patch('homeassistant.components.rainbird.async_create_clientsession', side_effect=create_session), patch('homeassistant.components.rainbird.config_flow.async_create_clientsession', side_effect=create_session):
        yield mocker

def rainbird_json_response(result: Any) -> str:  # type hint added
    """Create a fake API response."""
    return encryption.encrypt(f'{{"jsonrpc": "2.0", "result": {json.dumps(result)}, "id": 1}} ', PASSWORD)

def mock_json_response(result: Any) -> AiohttpClientMockResponse:  # type hint added
    """Create a fake AiohttpClientMockResponse."""
    return AiohttpClientMockResponse('POST', URL, response=rainbird_json_response(result))

def mock_response(data: Any) -> AiohttpClientMockResponse:  # type hint added
    """Create a fake AiohttpClientMockResponse."""
    return mock_json_response({'data': data})

def mock_response_error(status: HTTPStatus = HTTPStatus.SERVICE_UNAVAILABLE) -> AiohttpClientMockResponse:  # type hint added
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
def mock_api_responses(model_and_version_response: str, stations_response: str, zone_state_response: str, rain_response: str, rain_delay_response: str) -> list[AiohttpClientMockResponse]:
    """Fixture to set up a list of fake API responsees for tests to extend.

    These are returned in the order they are requested by the update coordinator.
    """
    return [mock_response(api_response) for api_response in [model_and_version_response, stations_response, zone_state_response, rain_response, rain_delay_response]]

@pytest.fixture(name='responses')
def mock_responses(api_responses: list[AiohttpClientMockResponse]) -> list[AiohttpClientMockResponse]:
    """Fixture to set up a list of fake API responsees for tests to extend."""
    return api_responses

@pytest.fixture(autouse=True)
def handle_responses(aioclient_mock: AiohttpClientMocker, responses: list[AiohttpClientMockResponse]) -> None:
    """Fixture for command mocking for fake responses to the API url."""

    async def handle(method: str, url: str, data: Any) -> AiohttpClientMockResponse:
        return responses.pop(0)
    aioclient_mock.post(URL, side_effect=handle)
