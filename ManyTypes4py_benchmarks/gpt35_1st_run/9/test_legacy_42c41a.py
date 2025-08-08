import asyncio
from collections.abc import Callable, Coroutine, Mapping
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, Mock, patch
import pytest
import voluptuous as vol
import yaml
from homeassistant import config as hass_config
from homeassistant.components import notify
from homeassistant.const import SERVICE_RELOAD, Platform
from homeassistant.core import HomeAssistant
from homeassistant.helpers.discovery import async_load_platform
from homeassistant.helpers.reload import async_setup_reload_service
from homeassistant.helpers.typing import ConfigType, DiscoveryInfoType
from homeassistant.setup import async_setup_component
from tests.common import MockPlatform, mock_platform

class NotificationService(notify.BaseNotificationService):
    def __init__(self, hass: HomeAssistant, target_list: Mapping[str, int] = None, name: str = 'notify') -> None:
        async def _async_make_reloadable(hass: HomeAssistant) -> None:
            await async_setup_reload_service(hass, name, [notify.DOMAIN])
        self.hass = hass
        self.target_list = target_list or {'a': 1, 'b': 2}
        hass.async_create_task(_async_make_reloadable(hass))

    @property
    def targets(self) -> Mapping[str, int]:
        return self.target_list

class MockNotifyPlatform(MockPlatform):
    def __init__(self, async_get_service: Callable = None, get_service: Callable = None) -> None:
        super().__init__()
        if get_service:
            self.get_service = get_service
        if async_get_service:
            self.async_get_service = async_get_service

def mock_notify_platform(hass: HomeAssistant, tmp_path: Path, integration: str = 'notify', async_get_service: Callable = None, get_service: Callable = None) -> MockNotifyPlatform:
    loaded_platform = MockNotifyPlatform(async_get_service, get_service)
    mock_platform(hass, f'{integration}.notify', loaded_platform)
    return loaded_platform

async def help_setup_notify(hass: HomeAssistant, tmp_path: Path, targets: Mapping[str, int] = None) -> MagicMock:
    send_message_mock = MagicMock()

    class _TestNotifyService(notify.BaseNotificationService):
        def __init__(self, targets: Mapping[str, int]) -> None:
            self._targets = targets
            super().__init__()

        @property
        def targets(self) -> Mapping[str, int]:
            return self._targets

        def send_message(self, message: str, **kwargs: Any) -> None:
            send_message_mock(message, kwargs)

    async def async_get_service(hass: HomeAssistant, config: ConfigType, discovery_info: DiscoveryInfoType = None) -> _TestNotifyService:
        return _TestNotifyService(targets)
    mock_notify_platform(hass, tmp_path, 'test', async_get_service=async_get_service)
    await async_setup_component(hass, 'notify', {'notify': [{'platform': 'test'}]})
    await hass.async_block_till_done()
    return send_message_mock

async def test_same_targets(hass: HomeAssistant) -> None:
    test = NotificationService(hass)
    await test.async_setup(hass, 'notify', 'test')
    await test.async_register_services()
    await hass.async_block_till_done()
    assert hasattr(test, 'registered_targets')
    assert test.registered_targets == {'test_a': 1, 'test_b': 2}
    await test.async_register_services()
    await hass.async_block_till_done()
    assert test.registered_targets == {'test_a': 1, 'test_b': 2}

# Add type annotations for the remaining test functions
