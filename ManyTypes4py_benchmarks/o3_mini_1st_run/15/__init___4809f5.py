from typing import Any, Callable, ContextManager, Coroutine
from unittest.mock import patch
import xml.etree.ElementTree as ET
from pylgnetcast import AccessTokenError, LgNetCastClient, SessionIdError
import requests
from homeassistant.components.lg_netcast import DOMAIN
from homeassistant.components.media_player import DOMAIN as MP_DOMAIN
from homeassistant.const import CONF_ACCESS_TOKEN, CONF_HOST, CONF_ID, CONF_MODEL, CONF_NAME
from homeassistant.core import HomeAssistant
from tests.common import MockConfigEntry

FAIL_TO_BIND_IP: str = '1.2.3.4'
IP_ADDRESS: str = '192.168.1.239'
DEVICE_TYPE: str = 'TV'
MODEL_NAME: str = 'MockLGModelName'
FRIENDLY_NAME: str = 'LG Smart TV'
UNIQUE_ID: str = '1234'
ENTITY_ID: str = f'{MP_DOMAIN}.{MODEL_NAME.lower()}'
FAKE_SESSION_ID: str = '987654321'
FAKE_PIN: str = '123456'


def _patched_lgnetcast_client(*args: Any, session_error: bool = False, fail_connection: bool = True,
                              invalid_details: bool = False, always_404: bool = False, no_unique_id: bool = False,
                              **kwargs: Any) -> LgNetCastClient:
    client: LgNetCastClient = LgNetCastClient(*args, **kwargs)

    def _get_fake_session_id() -> str:
        if not client.access_token:
            raise AccessTokenError('Fake Access Token Requested')
        if session_error:
            raise SessionIdError('Can not get session id from TV.')
        return FAKE_SESSION_ID

    def _get_fake_query_device_info() -> Any:
        if fail_connection:
            raise requests.exceptions.ConnectTimeout('Mocked Failed Connection')
        if always_404:
            return None
        if invalid_details:
            raise ET.ParseError('Mocked Parsed Error')
        return {'uuid': UNIQUE_ID if not no_unique_id else None, 'model_name': MODEL_NAME, 'friendly_name': FRIENDLY_NAME}

    client._get_session_id = _get_fake_session_id  # type: ignore
    client.query_device_info = _get_fake_query_device_info  # type: ignore
    return client


def _patch_lg_netcast(*, session_error: bool = False, fail_connection: bool = False,
                      invalid_details: bool = False, always_404: bool = False,
                      no_unique_id: bool = False) -> ContextManager[Any]:
    def _generate_fake_lgnetcast_client(*args: Any, **kwargs: Any) -> LgNetCastClient:
        return _patched_lgnetcast_client(*args, session_error=session_error, fail_connection=fail_connection,
                                         invalid_details=invalid_details, always_404=always_404, no_unique_id=no_unique_id,
                                         **kwargs)
    return patch('homeassistant.components.lg_netcast.config_flow.LgNetCastClient', new=_generate_fake_lgnetcast_client)


async def setup_lgnetcast(hass: HomeAssistant, unique_id: str = UNIQUE_ID) -> MockConfigEntry:
    config_entry_data = {
        CONF_HOST: IP_ADDRESS,
        CONF_ACCESS_TOKEN: FAKE_PIN,
        CONF_NAME: MODEL_NAME,
        CONF_MODEL: MODEL_NAME,
        CONF_ID: unique_id
    }
    config_entry: MockConfigEntry = MockConfigEntry(domain=DOMAIN, data=config_entry_data, title=MODEL_NAME, unique_id=unique_id)
    config_entry.add_to_hass(hass)
    await hass.config_entries.async_setup(config_entry.entry_id)
    await hass.async_block_till_done()
    return config_entry