from __future__ import annotations
import asyncio
import json
from dataclasses import dataclass, field
from typing import Any, AsyncGenerator, List, Optional
from unittest.mock import AsyncMock, patch

from aiohttp import ClientSession
from anova_wifi import AnovaApi, AnovaWebsocketHandler, InvalidLogin, NoDevicesFound, WebsocketFailure
import pytest
from homeassistant.core import HomeAssistant

DUMMY_ID: str = 'anova_id'

@dataclass
class MockedanovaWebsocketMessage:
    input_data: dict[str, Any]
    data: str = field(init=False)

    def __post_init__(self) -> None:
        self.data = json.dumps(self.input_data)

class MockedAnovaWebsocketStream:
    def __init__(self, messages: List[MockedanovaWebsocketMessage]) -> None:
        self.messages: List[MockedanovaWebsocketMessage] = messages

    def __aiter__(self) -> MockedAnovaWebsocketStream:
        return self

    async def __anext__(self) -> MockedanovaWebsocketMessage:
        if self.messages:
            return self.messages.pop(0)
        raise StopAsyncIteration

    def clear(self) -> None:
        self.messages.clear()

class MockedAnovaWebsocketHandler(AnovaWebsocketHandler):
    def __init__(
        self,
        firebase_jwt: str,
        jwt: str,
        session: ClientSession,
        connect_messages: List[MockedanovaWebsocketMessage],
        post_connect_messages: List[MockedanovaWebsocketMessage]
    ) -> None:
        super().__init__(firebase_jwt, jwt, session)
        self.connect_messages: List[MockedanovaWebsocketMessage] = connect_messages
        self.post_connect_messages: List[MockedanovaWebsocketMessage] = post_connect_messages

    async def connect(self) -> None:
        self.ws = MockedAnovaWebsocketStream(self.connect_messages)
        await self.message_listener()
        self.ws = MockedAnovaWebsocketStream(self.post_connect_messages)
        asyncio.ensure_future(self.message_listener())

def anova_api_mock(
    connect_messages: Optional[List[MockedanovaWebsocketMessage]] = None,
    post_connect_messages: Optional[List[MockedanovaWebsocketMessage]] = None
) -> AsyncMock:
    api_mock: AsyncMock = AsyncMock()

    async def authenticate_side_effect() -> None:
        api_mock.jwt = 'my_test_jwt'
        api_mock._firebase_jwt = 'my_test_firebase_jwt'

    async def create_websocket_side_effect() -> None:
        api_mock.websocket_handler = MockedAnovaWebsocketHandler(
            firebase_jwt=api_mock._firebase_jwt,
            jwt=api_mock.jwt,
            session=AsyncMock(spec=ClientSession),
            connect_messages=connect_messages if connect_messages is not None else [
                MockedanovaWebsocketMessage({'command': 'EVENT_APC_WIFI_LIST', 'payload': [{
                    'cookerId': DUMMY_ID,
                    'type': 'a5',
                    'pairedAt': '2023-08-12T02:33:20.917716Z',
                    'name': 'Anova Precision Cooker'
                }]})
            ],
            post_connect_messages=post_connect_messages if post_connect_messages is not None else [
                MockedanovaWebsocketMessage({
                    'command': 'EVENT_APC_STATE',
                    'payload': {
                        'cookerId': DUMMY_ID,
                        'state': {
                            'boot-id': '8620610049456548422',
                            'job': {
                                'cook-time-seconds': 0,
                                'id': '8759286e3125b0c547',
                                'mode': 'IDLE',
                                'ota-url': '',
                                'target-temperature': 54.72,
                                'temperature-unit': 'F'
                            },
                            'job-status': {
                                'cook-time-remaining': 0,
                                'job-start-systick': 599679,
                                'provisioning-pairing-code': 7514,
                                'state': '',
                                'state-change-systick': 599679
                            },
                            'pin-info': {
                                'device-safe': 0,
                                'water-leak': 0,
                                'water-level-critical': 0,
                                'water-temp-too-high': 0
                            },
                            'system-info': {
                                'class': 'A5',
                                'firmware-version': '2.2.0',
                                'type': 'RA2L1-128'
                            },
                            'system-info-details': {
                                'firmware-version-raw': 'VM178_A_02.02.00_MKE15-128',
                                'systick': 607026,
                                'version-string': 'VM171_A_02.02.00 RA2L1-128'
                            },
                            'temperature-info': {
                                'heater-temperature': 22.37,
                                'triac-temperature': 36.04,
                                'water-temperature': 18.33
                            }
                        }
                    }
                })
            ]
        )
        await api_mock.websocket_handler.connect()
        if not api_mock.websocket_handler.devices:
            raise NoDevicesFound('No devices were found on the websocket.')

    api_mock.authenticate.side_effect = authenticate_side_effect
    api_mock.create_websocket.side_effect = create_websocket_side_effect
    return api_mock

@pytest.fixture
async def anova_api(hass: HomeAssistant) -> AsyncGenerator[AnovaApi, None]:
    api_mock: AsyncMock = anova_api_mock()
    with patch('homeassistant.components.anova.AnovaApi', return_value=api_mock), patch(
        'homeassistant.components.anova.config_flow.AnovaApi', return_value=api_mock
    ):
        api: AnovaApi = AnovaApi(None, 'sample@gmail.com', 'sample')
        yield api

@pytest.fixture
async def anova_api_no_devices(hass: HomeAssistant) -> AsyncGenerator[AnovaApi, None]:
    api_mock: AsyncMock = anova_api_mock(connect_messages=[], post_connect_messages=[])
    with patch('homeassistant.components.anova.AnovaApi', return_value=api_mock), patch(
        'homeassistant.components.anova.config_flow.AnovaApi', return_value=api_mock
    ):
        api: AnovaApi = AnovaApi(None, 'sample@gmail.com', 'sample')
        yield api

@pytest.fixture
async def anova_api_wrong_login(hass: HomeAssistant) -> AsyncGenerator[AnovaApi, None]:
    api_mock: AsyncMock = anova_api_mock()

    async def authenticate_side_effect() -> None:
        raise InvalidLogin

    api_mock.authenticate.side_effect = authenticate_side_effect
    with patch('homeassistant.components.anova.AnovaApi', return_value=api_mock):
        api: AnovaApi = AnovaApi(None, 'sample@gmail.com', 'sample')
        yield api

@pytest.fixture
async def anova_api_no_data(hass: HomeAssistant) -> AsyncGenerator[AnovaApi, None]:
    api_mock: AsyncMock = anova_api_mock(post_connect_messages=[])
    with patch('homeassistant.components.anova.AnovaApi', return_value=api_mock):
        api: AnovaApi = AnovaApi(None, 'sample@gmail.com', 'sample')
        yield api

@pytest.fixture
async def anova_api_websocket_failure(hass: HomeAssistant) -> AsyncGenerator[AnovaApi, None]:
    api_mock: AsyncMock = anova_api_mock()

    async def create_websocket_side_effect() -> None:
        raise WebsocketFailure

    api_mock.create_websocket.side_effect = create_websocket_side_effect
    with patch('homeassistant.components.anova.AnovaApi', return_value=api_mock):
        api: AnovaApi = AnovaApi(None, 'sample@gmail.com', 'sample')
        yield api