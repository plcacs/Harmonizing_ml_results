from aiohttp.client_exceptions import ClientError
from homeassistant.core import HomeAssistant
from homeassistant.setup import async_setup_component
from tests.common import MockConfigEntry, load_fixture
from tests.test_util.aiohttp import AiohttpClientMocker
from typing import Any, Dict

URL: str = 'http://192.168.1.189:7887/test'
API_KEY: str = 'MOCK_API_KEY'
MOCK_REAUTH_INPUT: Dict[str, str] = {CONF_API_KEY: 'test-api-key-reauth'}
MOCK_USER_INPUT: Dict[str, Any] = {CONF_URL: URL, CONF_API_KEY: API_KEY, CONF_VERIFY_SSL: False}
CONF_DATA: Dict[str, Any] = {CONF_URL: URL, CONF_API_KEY: API_KEY, CONF_VERIFY_SSL: False}

def mock_connection(aioclient_mock: AiohttpClientMocker, url: str = URL, error: bool = False, invalid_auth: bool = False, windows: bool = False, single_return: bool = False) -> None:
    ...

def mock_calendar(aioclient_mock: AiohttpClientMocker, url: str = URL) -> None:
    ...

def mock_connection_error(aioclient_mock: AiohttpClientMocker, url: str = URL) -> None:
    ...

def mock_connection_invalid_auth(aioclient_mock: AiohttpClientMocker, url: str = URL) -> None:
    ...

def mock_connection_server_error(aioclient_mock: AiohttpClientMocker, url: str = URL) -> None:
    ...

async def setup_integration(hass: HomeAssistant, aioclient_mock: AiohttpClientMocker, url: str = URL, api_key: str = API_KEY, unique_id: str = None, skip_entry_setup: bool = False, connection_error: bool = False, invalid_auth: bool = False, windows: bool = False, single_return: bool = False) -> MockConfigEntry:
    ...

def patch_async_setup_entry(return_value: bool = True) -> Any:
    ...

def create_entry(hass: HomeAssistant) -> MockConfigEntry:
    ...
