"""The tests for legacy notify services."""
import asyncio
from collections.abc import Callable, Coroutine, Mapping
from pathlib import Path
from typing import Any, Optional, Dict, List, Tuple, Set
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
    """A test class for legacy notification services."""

    def __init__(self, hass: HomeAssistant, target_list: Optional[Dict[str, Any]] = None, name: str = 'notify') -> None:
        """Initialize the service."""

        async def _async_make_reloadable(hass: HomeAssistant) -> None:
            """Initialize the reload service."""
            await async_setup_reload_service(hass, name, [notify.DOMAIN])
        self.hass = hass
        self.target_list = target_list or {'a': 1, 'b': 2}
        hass.async_create_task(_async_make_reloadable(hass))

    @property
    def targets(self) -> Dict[str, Any]:
        """Return a dictionary of devices."""
        return self.target_list

class MockNotifyPlatform(MockPlatform):
    """Help to set up a legacy test notify service."""

    def __init__(
        self,
        async_get_service: Optional[Callable[[HomeAssistant, ConfigType, Optional[DiscoveryInfoType]], Coroutine[Any, Any, Optional[notify.BaseNotificationService]]] = None,
        get_service: Optional[Callable[[HomeAssistant, ConfigType, Optional[DiscoveryInfoType]], Optional[notify.BaseNotificationService]] = None
    ) -> None:
        """Return a legacy notify service."""
        super().__init__()
        if get_service:
            self.get_service = get_service
        if async_get_service:
            self.async_get_service = async_get_service

def mock_notify_platform(
    hass: HomeAssistant,
    tmp_path: Path,
    integration: str = 'notify',
    async_get_service: Optional[Callable[[HomeAssistant, ConfigType, Optional[DiscoveryInfoType]], Coroutine[Any, Any, Optional[notify.BaseNotificationService]]] = None,
    get_service: Optional[Callable[[HomeAssistant, ConfigType, Optional[DiscoveryInfoType]], Optional[notify.BaseNotificationService]] = None
) -> MockNotifyPlatform:
    """Specialize the mock platform for legacy notify service."""
    loaded_platform = MockNotifyPlatform(async_get_service, get_service)
    mock_platform(hass, f'{integration}.notify', loaded_platform)
    return loaded_platform

async def help_setup_notify(
    hass: HomeAssistant,
    tmp_path: Path,
    targets: Optional[Dict[str, Any]] = None
) -> MagicMock:
    """Help set up a platform notify service."""
    send_message_mock = MagicMock()

    class _TestNotifyService(notify.BaseNotificationService):

        def __init__(self, targets: Dict[str, Any]) -> None:
            """Initialize service."""
            self._targets = targets
            super().__init__()

        @property
        def targets(self) -> Dict[str, Any]:
            """Return a dictionary of registered targets."""
            return self._targets

        def send_message(self, message: str, **kwargs: Any) -> None:
            """Send a message."""
            send_message_mock(message, kwargs)

    async def async_get_service(
        hass: HomeAssistant,
        config: ConfigType,
        discovery_info: Optional[DiscoveryInfoType] = None
    ) -> _TestNotifyService:
        """Get notify service for mocked platform."""
        return _TestNotifyService(targets or {})
    mock_notify_platform(hass, tmp_path, 'test', async_get_service=async_get_service)
    await async_setup_component(hass, 'notify', {'notify': [{'platform': 'test'}]})
    await hass.async_block_till_done()
    return send_message_mock

async def test_same_targets(hass: HomeAssistant) -> None:
    """Test not changing the targets in a legacy notify service."""
    test = NotificationService(hass)
    await test.async_setup(hass, 'notify', 'test')
    await test.async_register_services()
    await hass.async_block_till_done()
    assert hasattr(test, 'registered_targets')
    assert test.registered_targets == {'test_a': 1, 'test_b': 2}
    await test.async_register_services()
    await hass.async_block_till_done()
    assert test.registered_targets == {'test_a': 1, 'test_b': 2}

async def test_change_targets(hass: HomeAssistant) -> None:
    """Test changing the targets in a legacy notify service."""
    test = NotificationService(hass)
    await test.async_setup(hass, 'notify', 'test')
    await test.async_register_services()
    await hass.async_block_till_done()
    assert hasattr(test, 'registered_targets')
    assert test.registered_targets == {'test_a': 1, 'test_b': 2}
    test.target_list = {'a': 0}
    await test.async_register_services()
    await hass.async_block_till_done()
    assert test.target_list == {'a': 0}
    assert test.registered_targets == {'test_a': 0}

async def test_add_targets(hass: HomeAssistant) -> None:
    """Test adding the targets in a legacy notify service."""
    test = NotificationService(hass)
    await test.async_setup(hass, 'notify', 'test')
    await test.async_register_services()
    await hass.async_block_till_done()
    assert hasattr(test, 'registered_targets')
    assert test.registered_targets == {'test_a': 1, 'test_b': 2}
    test.target_list = {'a': 1, 'b': 2, 'c': 3}
    await test.async_register_services()
    await hass.async_block_till_done()
    assert test.target_list == {'a': 1, 'b': 2, 'c': 3}
    assert test.registered_targets == {'test_a': 1, 'test_b': 2, 'test_c': 3}

async def test_remove_targets(hass: HomeAssistant) -> None:
    """Test removing targets from the targets in a legacy notify service."""
    test = NotificationService(hass)
    await test.async_setup(hass, 'notify', 'test')
    await test.async_register_services()
    await hass.async_block_till_done()
    assert hasattr(test, 'registered_targets')
    assert test.registered_targets == {'test_a': 1, 'test_b': 2}
    test.target_list = {'c': 1}
    await test.async_register_services()
    await hass.async_block_till_done()
    assert test.target_list == {'c': 1}
    assert test.registered_targets == {'test_c': 1}

async def test_invalid_platform(hass: HomeAssistant, caplog: pytest.LogCaptureFixture, tmp_path: Path) -> None:
    """Test service setup with an invalid platform."""
    mock_notify_platform(hass, tmp_path, 'testnotify1')
    await async_setup_component(hass, 'notify', {'notify': [{'platform': 'testnotify1'}]})
    await hass.async_block_till_done()
    assert 'Invalid notify platform' in caplog.text
    caplog.clear()
    mock_notify_platform(hass, tmp_path, 'testnotify2')
    await async_load_platform(hass, 'notify', 'testnotify2', {}, hass_config={'notify': [{'platform': 'testnotify2'}]})
    await hass.async_block_till_done()
    assert 'Invalid notify platform' in caplog.text

async def test_invalid_service(hass: HomeAssistant, caplog: pytest.LogCaptureFixture, tmp_path: Path) -> None:
    """Test service setup with an invalid service object or platform."""

    def get_service(hass: HomeAssistant, config: ConfigType, discovery_info: Optional[DiscoveryInfoType] = None) -> None:
        """Return None for an invalid notify service."""
        return None
    mock_notify_platform(hass, tmp_path, 'testnotify', get_service=get_service)
    await async_load_platform(hass, 'notify', 'testnotify', {}, hass_config={'notify': [{'platform': 'testnotify'}]})
    await hass.async_block_till_done()
    assert 'Failed to initialize notification service testnotify' in caplog.text
    caplog.clear()
    await async_load_platform(hass, 'notify', 'testnotifyinvalid', {'notify': [{'platform': 'testnotifyinvalid'}]}, hass_config={'notify': [{'platform': 'testnotifyinvalid'}]})
    await hass.async_block_till_done()
    assert 'Unknown notification service specified' in caplog.text

async def test_platform_setup_with_error(hass: HomeAssistant, caplog: pytest.LogCaptureFixture, tmp_path: Path) -> None:
    """Test service setup with an invalid setup."""

    async def async_get_service(hass: HomeAssistant, config: ConfigType, discovery_info: Optional[DiscoveryInfoType] = None) -> None:
        """Return None for an invalid notify service."""
        raise Exception('Setup error')
    mock_notify_platform(hass, tmp_path, 'testnotify', async_get_service=async_get_service)
    await async_load_platform(hass, 'notify', 'testnotify', {}, hass_config={'notify': [{'platform': 'testnotify'}]})
    await hass.async_block_till_done()
    assert 'Error setting up platform testnotify' in caplog.text

async def test_reload_with_notify_builtin_platform_reload(hass: HomeAssistant, tmp_path: Path) -> None:
    """Test reload using the legacy notify platform reload method."""

    async def async_get_service(hass: HomeAssistant, config: ConfigType, discovery_info: Optional[DiscoveryInfoType] = None) -> NotificationService:
        """Get notify service for mocked platform."""
        targetlist = {'a': 1, 'b': 2}
        return NotificationService(hass, targetlist, 'testnotify')
    mock_notify_platform(hass, tmp_path, 'testnotify', async_get_service=async_get_service)
    await notify.async_reload(hass, 'testnotify')
    await async_setup_component(hass, 'notify', {'notify': [{'platform': 'testnotify'}]})
    await hass.async_block_till_done()
    assert hass.services.has_service(notify.DOMAIN, 'testnotify_a')
    assert hass.services.has_service(notify.DOMAIN, 'testnotify_b')
    await notify.async_reload(hass, 'testnotify')
    assert hass.services.has_service(notify.DOMAIN, 'testnotify_a')
    assert hass.services.has_service(notify.DOMAIN, 'testnotify_b')

async def test_setup_platform_and_reload(hass: HomeAssistant, tmp_path: Path) -> None:
    """Test service setup and reload."""
    get_service_called = Mock()

    async def async_get_service(hass: HomeAssistant, config: ConfigType, discovery_info: Optional[DiscoveryInfoType] = None) -> NotificationService:
        """Get notify service for mocked platform."""
        get_service_called(config, discovery_info)
        targetlist = {'a': 1, 'b': 2}
        return NotificationService(hass, targetlist, 'testnotify')

    async def async_get_service2(hass: HomeAssistant, config: ConfigType, discovery_info: Optional[DiscoveryInfoType] = None) -> NotificationService:
        """Get legacy notify service for mocked platform."""
        get_service_called(config, discovery_info)
        targetlist = {'c': 3, 'd': 4}
        return NotificationService(hass, targetlist, 'testnotify2')
    mock_notify_platform(hass, tmp_path, 'testnotify', async_get_service=async_get_service)
    mock_notify_platform(hass, tmp_path, 'testnotify2', async_get_service=async_get_service2)
    await async_setup_component(hass, 'notify', {'notify': [{'platform': 'testnotify'}]})
    await hass.async_block_till_done()
    assert hass.services.has_service('testnotify', SERVICE_RELOAD)
    assert hass.services.has_service(notify.DOMAIN, 'testnotify_a')
    assert hass.services.has_service(notify.DOMAIN, 'testnotify_b')
    assert get_service_called.call_count == 1
    assert get_service_called.call_args[0][0] == {'platform': 'testnotify'}
    assert get_service_called.call_args[0][1] is None
    get_service_called.reset_mock()
    await async_load_platform(hass, 'notify', 'testnotify2', {}, hass_config={'notify': [{'platform': 'testnotify'}]})
    await hass.async_block_till_done()
    assert hass.services.has_service('testnotify2', SERVICE_RELOAD)
    assert hass.services.has_service(notify.DOMAIN, 'testnotify2_c')
    assert hass.services.has_service(notify.DOMAIN, 'testnotify2_d')
    assert get_service_called.call_count == 1
    assert get_service_called.call_args[0][0] == {}
    assert get_service_called.call_args[0][1] == {}
    get_service_called.reset_mock()
    new_yaml_config_file = tmp_path / 'configuration.yaml'
    new_yaml_config = yaml.dump({'notify': [{'platform': 'testnotify'}]})
    new_yaml_config_file.write_text(new_yaml_config)
    with patch.object(hass_config, 'YAML_CONFIG_FILE', new_yaml_config_file):
        await hass.services.async_call('testnotify', SERVICE_RELOAD, {}, blocking=True)
        await hass.services.async_call('testnotify2', SERVICE_RELOAD, {}, blocking=True)
        await hass.async_block_till_done()
    assert hass.services.has_service(notify.DOMAIN, 'testnotify_a')
    assert hass.services.has_service(notify.DOMAIN, 'testnotify_b')
    assert get_service_called.call_count == 1
    assert get_service_called.call_args[0][0] == {'platform': 'testnotify'}
    assert get_service_called.call_args[0][1] is None
    assert not hass.services.has_service(notify.DOMAIN, 'testnotify2_c')
    assert not hass.services.has_service(notify.DOMAIN, 'testnotify2_d')

async def test_setup_platform_before_notify_setup(hass: HomeAssistant, tmp_path: Path) -> None:
    """Test trying to setup a platform before legacy notify service is setup."""
    get_service_called = Mock()

    async def async_get_service(hass: HomeAssistant, config: ConfigType, discovery_info: Optional[DiscoveryInfoType] = None) -> NotificationService:
        """Get notify service for mocked platform."""
        get_service_called(config, discovery_info)
        targetlist = {'a': 1, 'b': 2}
        return NotificationService(hass, targetlist, 'testnotify')

    async def async_get_service2(hass: HomeAssistant, config: ConfigType, discovery_info: Optional[DiscoveryInfoType] = None) -> NotificationService:
        """Get notify service for mocked platform."""
        get_service_called(config, discovery_info)
        targetlist = {'c': 3, 'd': 4}
        return NotificationService(hass, targetlist, 'testnotify2')
    mock_notify_platform(hass, tmp_path, 'testnotify', async_get_service=async_get_service)
    mock_notify_platform(hass, tmp_path, 'testnotify2', async_get_service=async_get_service2)
    hass_config = {'notify': [{'platform': 'testnotify'}]}
    load_coro = async_load_platform(hass, Platform.NOTIFY, 'testnotify2', {}, hass_config=hass_config)
    setup_coro = async_setup_component(hass, 'notify', hass_config)
    load_task = asyncio.create_task(load_coro)
    setup_task = asyncio.create_task(setup_coro)
    await asyncio.gather(load_task, setup_task)
    await hass.async_block_till_done()
    assert hass.services.has_service(notify.DOMAIN, 'testnotify_a')
    assert hass.services.has_service(notify.DOMAIN, 'testnotify_b')
    assert hass.services.has_service(notify.DOMAIN, 'testnotify2_c')
    assert hass.services.has_service(notify.DOMAIN, 'testnotify2_d')

async def test_setup_platform_after_notify_setup(hass: HomeAssistant, tmp_path: Path) -> None:
    """Test trying to setup a platform after legacy notify service is set up."""
    get_service_called = Mock()

    async def async_get_service(hass: HomeAssistant, config: ConfigType, discovery_info: Optional[DiscoveryInfoType] = None) -> NotificationService:
        """Get notify service for mocked platform."""
        get_service_called(config, discovery_info)
        targetlist = {'a': 1, 'b': 2}
        return NotificationService(hass, targetlist, 'testnotify')

    async def async_get_service2(hass: HomeAssistant, config: ConfigType, discovery_info: Optional[DiscoveryInfoType] = None) -> NotificationService:
        """Get notify service for mocked platform."""
        get_service_called(config, discovery_info)
        targetlist = {'c': 3, 'd': 4}
        return NotificationService