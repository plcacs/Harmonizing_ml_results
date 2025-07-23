"""Fixtures for component testing."""
from __future__ import annotations
import asyncio
from collections.abc import AsyncGenerator, Callable, Generator, Iterable
from functools import lru_cache
from importlib.util import find_spec
from pathlib import Path
import re
import string
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Set, Tuple, Union
from unittest.mock import AsyncMock, MagicMock, patch
from aiohasupervisor.models import Discovery, Repository, ResolutionInfo, StoreAddon, StoreInfo
import pytest
import voluptuous as vol
from homeassistant.components import repairs
from homeassistant.config_entries import DISCOVERY_SOURCES, ConfigEntriesFlowManager, FlowResult, OptionsFlowManager
from homeassistant.const import STATE_OFF, STATE_ON
from homeassistant.core import Context, HomeAssistant, ServiceRegistry, ServiceResponse
from homeassistant.data_entry_flow import FlowContext, FlowHandler, FlowManager, FlowResultType, section
from homeassistant.exceptions import HomeAssistantError
from homeassistant.helpers import issue_registry as ir
from homeassistant.helpers.translation import async_get_translations
from homeassistant.util import yaml as yaml_util
from tests.common import QualityScaleStatus, get_quality_scale

if TYPE_CHECKING:
    from homeassistant.components.hassio import AddonManager
    from .conversation import MockAgent
    from .device_tracker.common import MockScanner
    from .light.common import MockLight
    from .sensor.common import MockSensor
    from .switch.common import MockSwitch

RE_REQUEST_DOMAIN = re.compile('.*tests\\/components\\/([^/]+)\\/.*')

@pytest.fixture(scope='session', autouse=find_spec('zeroconf') is not None)
def patch_zeroconf_multiple_catcher() -> Generator[None, None, None]:
    """If installed, patch zeroconf wrapper that detects if multiple instances are used."""
    with patch('homeassistant.components.zeroconf.install_multiple_zeroconf_catcher', side_effect=lambda zc: None):
        yield

@pytest.fixture(scope='session', autouse=True)
def prevent_io() -> Generator[None, None, None]:
    """Fixture to prevent certain I/O from happening."""
    with patch('homeassistant.components.http.ban.load_yaml_config_file'):
        yield

@pytest.fixture
def entity_registry_enabled_by_default() -> Generator[None, None, None]:
    """Test fixture that ensures all entities are enabled in the registry."""
    with patch('homeassistant.helpers.entity.Entity.entity_registry_enabled_default', return_value=True), patch('homeassistant.components.device_tracker.config_entry.ScannerEntity.entity_registry_enabled_default', return_value=True):
        yield

@pytest.fixture(name='stub_blueprint_populate')
def stub_blueprint_populate_fixture() -> Generator[None, None, None]:
    """Stub copying the blueprints to the config folder."""
    from .blueprint.common import stub_blueprint_populate_fixture_helper
    yield from stub_blueprint_populate_fixture_helper()

@pytest.fixture(name='mock_tts_get_cache_files')
def mock_tts_get_cache_files_fixture() -> Generator[None, None, None]:
    """Mock the list TTS cache function."""
    from .tts.common import mock_tts_get_cache_files_fixture_helper
    yield from mock_tts_get_cache_files_fixture_helper()

@pytest.fixture(name='mock_tts_init_cache_dir')
def mock_tts_init_cache_dir_fixture(init_tts_cache_dir_side_effect: Any) -> Generator[None, None, None]:
    """Mock the TTS cache dir in memory."""
    from .tts.common import mock_tts_init_cache_dir_fixture_helper
    yield from mock_tts_init_cache_dir_fixture_helper(init_tts_cache_dir_side_effect)

@pytest.fixture(name='init_tts_cache_dir_side_effect')
def init_tts_cache_dir_side_effect_fixture() -> Any:
    """Return the cache dir."""
    from .tts.common import init_tts_cache_dir_side_effect_fixture_helper
    return init_tts_cache_dir_side_effect_fixture_helper()

@pytest.fixture(name='mock_tts_cache_dir')
def mock_tts_cache_dir_fixture(tmp_path: Path, mock_tts_init_cache_dir: Any, mock_tts_get_cache_files: Any, request: Any) -> Generator[None, None, None]:
    """Mock the TTS cache dir with empty dir."""
    from .tts.common import mock_tts_cache_dir_fixture_helper
    yield from mock_tts_cache_dir_fixture_helper(tmp_path, mock_tts_init_cache_dir, mock_tts_get_cache_files, request)

@pytest.fixture(name='tts_mutagen_mock')
def tts_mutagen_mock_fixture() -> Generator[None, None, None]:
    """Mock writing tags."""
    from .tts.common import tts_mutagen_mock_fixture_helper
    yield from tts_mutagen_mock_fixture_helper()

@pytest.fixture(name='mock_conversation_agent')
def mock_conversation_agent_fixture(hass: HomeAssistant) -> Any:
    """Mock a conversation agent."""
    from .conversation.common import mock_conversation_agent_fixture_helper
    return mock_conversation_agent_fixture_helper(hass)

@pytest.fixture(scope='session', autouse=find_spec('ffmpeg') is not None)
def prevent_ffmpeg_subprocess() -> Generator[None, None, None]:
    """If installed, prevent ffmpeg from creating a subprocess."""
    with patch('homeassistant.components.ffmpeg.FFVersion.get_version', return_value='6.0'):
        yield

@pytest.fixture
def mock_light_entities() -> List[Any]:
    """Return mocked light entities."""
    from .light.common import MockLight
    return [MockLight('Ceiling', STATE_ON), MockLight('Ceiling', STATE_OFF), MockLight(None, STATE_OFF)]

@pytest.fixture
def mock_sensor_entities() -> List[Any]:
    """Return mocked sensor entities."""
    from .sensor.common import get_mock_sensor_entities
    return get_mock_sensor_entities()

@pytest.fixture
def mock_switch_entities() -> List[Any]:
    """Return mocked toggle entities."""
    from .switch.common import get_mock_switch_entities
    return get_mock_switch_entities()

@pytest.fixture
def mock_legacy_device_scanner() -> Any:
    """Return mocked legacy device scanner entity."""
    from .device_tracker.common import MockScanner
    return MockScanner()

@pytest.fixture
def mock_legacy_device_tracker_setup() -> Any:
    """Return setup callable for legacy device tracker setup."""
    from .device_tracker.common import mock_legacy_device_tracker_setup
    return mock_legacy_device_tracker_setup

@pytest.fixture(name='addon_manager')
def addon_manager_fixture(hass: HomeAssistant, supervisor_client: Any) -> Any:
    """Return an AddonManager instance."""
    from .hassio.common import mock_addon_manager
    return mock_addon_manager(hass)

@pytest.fixture(name='discovery_info')
def discovery_info_fixture() -> List[Any]:
    """Return the discovery info from the supervisor."""
    return []

@pytest.fixture(name='discovery_info_side_effect')
def discovery_info_side_effect_fixture() -> Optional[Callable[..., Any]]:
    """Return the discovery info from the supervisor."""
    return None

@pytest.fixture(name='get_addon_discovery_info')
def get_addon_discovery_info_fixture(supervisor_client: Any, discovery_info: List[Any], discovery_info_side_effect: Optional[Callable[..., Any]]) -> Any:
    """Mock get add-on discovery info."""
    supervisor_client.discovery.list.return_value = discovery_info
    supervisor_client.discovery.list.side_effect = discovery_info_side_effect
    return supervisor_client.discovery.list

@pytest.fixture(name='get_discovery_message_side_effect')
def get_discovery_message_side_effect_fixture() -> Optional[Callable[..., Any]]:
    """Side effect for getting a discovery message by uuid."""
    return None

@pytest.fixture(name='get_discovery_message')
def get_discovery_message_fixture(supervisor_client: Any, get_discovery_message_side_effect: Optional[Callable[..., Any]]) -> Any:
    """Mock getting a discovery message by uuid."""
    supervisor_client.discovery.get.side_effect = get_discovery_message_side_effect
    return supervisor_client.discovery.get

@pytest.fixture(name='addon_store_info_side_effect')
def addon_store_info_side_effect_fixture() -> Optional[Callable[..., Any]]:
    """Return the add-on store info side effect."""
    return None

@pytest.fixture(name='addon_store_info')
def addon_store_info_fixture(supervisor_client: Any, addon_store_info_side_effect: Optional[Callable[..., Any]]) -> Any:
    """Mock Supervisor add-on store info."""
    from .hassio.common import mock_addon_store_info
    return mock_addon_store_info(supervisor_client, addon_store_info_side_effect)

@pytest.fixture(name='addon_info_side_effect')
def addon_info_side_effect_fixture() -> Optional[Callable[..., Any]]:
    """Return the add-on info side effect."""
    return None

@pytest.fixture(name='addon_info')
def addon_info_fixture(supervisor_client: Any, addon_info_side_effect: Optional[Callable[..., Any]]) -> Any:
    """Mock Supervisor add-on info."""
    from .hassio.common import mock_addon_info
    return mock_addon_info(supervisor_client, addon_info_side_effect)

@pytest.fixture(name='addon_not_installed')
def addon_not_installed_fixture(addon_store_info: Any, addon_info: Any) -> Any:
    """Mock add-on not installed."""
    from .hassio.common import mock_addon_not_installed
    return mock_addon_not_installed(addon_store_info, addon_info)

@pytest.fixture(name='addon_installed')
def addon_installed_fixture(addon_store_info: Any, addon_info: Any) -> Any:
    """Mock add-on already installed but not running."""
    from .hassio.common import mock_addon_installed
    return mock_addon_installed(addon_store_info, addon_info)

@pytest.fixture(name='addon_running')
def addon_running_fixture(addon_store_info: Any, addon_info: Any) -> Any:
    """Mock add-on already running."""
    from .hassio.common import mock_addon_running
    return mock_addon_running(addon_store_info, addon_info)

@pytest.fixture(name='install_addon_side_effect')
def install_addon_side_effect_fixture(addon_store_info: Any, addon_info: Any) -> Optional[Callable[..., Any]]:
    """Return the install add-on side effect."""
    from .hassio.common import mock_install_addon_side_effect
    return mock_install_addon_side_effect(addon_store_info, addon_info)

@pytest.fixture(name='install_addon')
def install_addon_fixture(supervisor_client: Any, install_addon_side_effect: Optional[Callable[..., Any]]) -> Any:
    """Mock install add-on."""
    supervisor_client.store.install_addon.side_effect = install_addon_side_effect
    return supervisor_client.store.install_addon

@pytest.fixture(name='start_addon_side_effect')
def start_addon_side_effect_fixture(addon_store_info: Any, addon_info: Any) -> Optional[Callable[..., Any]]:
    """Return the start add-on options side effect."""
    from .hassio.common import mock_start_addon_side_effect
    return mock_start_addon_side_effect(addon_store_info, addon_info)

@pytest.fixture(name='start_addon')
def start_addon_fixture(supervisor_client: Any, start_addon_side_effect: Optional[Callable[..., Any]]) -> Any:
    """Mock start add-on."""
    supervisor_client.addons.start_addon.side_effect = start_addon_side_effect
    return supervisor_client.addons.start_addon

@pytest.fixture(name='restart_addon_side_effect')
def restart_addon_side_effect_fixture() -> Optional[Callable[..., Any]]:
    """Return the restart add-on options side effect."""
    return None

@pytest.fixture(name='restart_addon')
def restart_addon_fixture(supervisor_client: Any, restart_addon_side_effect: Optional[Callable[..., Any]]) -> Any:
    """Mock restart add-on."""
    supervisor_client.addons.restart_addon.side_effect = restart_addon_side_effect
    return supervisor_client.addons.restart_addon

@pytest.fixture(name='stop_addon')
def stop_addon_fixture(supervisor_client: Any) -> Any:
    """Mock stop add-on."""
    return supervisor_client.addons.stop_addon

@pytest.fixture(name='addon_options')
def addon_options_fixture(addon_info: Any) -> Any:
    """Mock add-on options."""
    return addon_info.return_value.options

@pytest.fixture(name='set_addon_options_side_effect')
def set_addon_options_side_effect_fixture(addon_options: Any) -> Optional[Callable[..., Any]]:
    """Return the set add-on options side effect."""
    from .hassio.common import mock_set_addon_options_side_effect
    return mock_set_addon_options_side_effect(addon_options)

@pytest.fixture(name='set_addon_options')
def set_addon_options_fixture(supervisor_client: Any, set_addon_options_side_effect: Optional[Callable[..., Any]]) -> Any:
    """Mock set add-on options."""
    supervisor_client.addons.set_addon_options.side_effect = set_addon_options_side_effect
    return supervisor_client.addons.set_addon_options

@pytest.fixture(name='uninstall_addon')
def uninstall_addon_fixture(supervisor_client: Any) -> Any:
    """Mock uninstall add-on."""
    return supervisor_client.addons.uninstall_addon

@pytest.fixture(name='create_backup')
def create_backup_fixture() -> Generator[None, None, None]:
    """Mock create backup."""
    from .hassio.common import mock_create_backup
    yield from mock_create_backup()

@pytest.fixture(name='update_addon')
def update_addon_fixture(supervisor_client: Any) -> Any:
    """Mock update add-on."""
    return supervisor_client.store.update_addon

@pytest.fixture(name='store_addons')
def store_addons_fixture() -> List[Any]:
    """Mock store addons list."""
    return []

@pytest.fixture(name='store_repositories')
def store_repositories_fixture() -> List[Any]:
    """Mock store repositories list."""
    return []

@pytest.fixture(name='store_info')
def store_info_fixture(supervisor_client: Any, store_addons: List[Any], store_repositories: List[Any]) -> Any:
    """Mock store info."""
    supervisor_client.store.info.return_value = StoreInfo(addons=store_addons, repositories=store_repositories)
    return supervisor_client.store.info

@pytest.fixture(name='addon_stats')
def addon_stats_fixture(supervisor_client: Any) -> Any:
    """Mock addon stats info."""
    from .hassio.common import mock_addon_stats
    return mock_addon_stats(supervisor_client)

@pytest.fixture(name='addon_changelog')
def addon_changelog_fixture(supervisor_client: Any) -> Any:
    """Mock addon changelog."""
    supervisor_client.store.addon_changelog.return_value = ''
    return supervisor_client.store.addon_changelog

@pytest.fixture(name='supervisor_is_connected')
def supervisor_is_connected_fixture(supervisor_client: Any) -> Any:
    """Mock supervisor is connected."""
    supervisor_client.supervisor.ping.return_value = None
    return supervisor_client.supervisor.ping

@pytest.fixture(name='resolution_info')
def resolution_info_fixture(supervisor_client: Any) -> Any:
    """Mock resolution info from supervisor."""
    supervisor_client.resolution.info.return_value = ResolutionInfo(suggestions=[], unsupported=[], unhealthy=[], issues=[], checks=[])
    return supervisor_client.resolution.info

@pytest.fixture(name='resolution_suggestions_for_issue')
def resolution_suggestions_for_issue_fixture(supervisor_client: Any) -> Any:
    """Mock suggestions by issue from supervisor resolution."""
    supervisor_client.resolution.suggestions_for_issue.return_value = []
    return supervisor_client.resolution.suggestions_for_issue

@pytest.fixture(name='supervisor_client')
def supervisor_client_fixture() -> Generator[Any, None, None]:
    """Mock the supervisor client."""
    mounts_info_mock = AsyncMock(spec_set=['default_backup_mount', 'mounts'])
    mounts_info_mock.mounts = []
    supervisor_client = AsyncMock()
    supervisor_client.addons = AsyncMock()
    supervisor_client.discovery = AsyncMock()
    supervisor_client.homeassistant = AsyncMock()
    supervisor_client.host = AsyncMock()
    supervisor_client.jobs = AsyncMock()
    supervisor_client.mounts.info.return_value = mounts_info_mock
    supervisor_client.os = AsyncMock()
    supervisor_client.resolution = AsyncMock()
    supervisor_client.supervisor = AsyncMock()
    with patch('homeassistant.components.hassio.get_supervisor_client', return_value=supervisor_client), patch('homeassistant.components.hassio.handler.get_supervisor_client', return_value=supervisor_client), patch('homeassistant.components.hassio.add