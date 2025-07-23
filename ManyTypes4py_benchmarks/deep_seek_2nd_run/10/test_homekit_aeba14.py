"""Tests for the HomeKit component."""
from __future__ import annotations

import asyncio
from typing import Any, Dict, List, Optional, Set, Tuple, Union, cast
from unittest.mock import ANY, AsyncMock, MagicMock, Mock, patch
from uuid import uuid1

from pyhap.accessory import Accessory
from pyhap.const import CATEGORY_CAMERA, CATEGORY_TELEVISION
import pytest

from homeassistant import config as hass_config
from homeassistant.components import homekit as homekit_base, zeroconf
from homeassistant.components.binary_sensor import BinarySensorDeviceClass
from homeassistant.components.event import EventDeviceClass
from homeassistant.components.homekit import (
    MAX_DEVICES,
    STATUS_READY,
    STATUS_RUNNING,
    STATUS_STOPPED,
    STATUS_WAIT,
    HomeKit,
)
from homeassistant.components.homekit.accessories import HomeBridge
from homeassistant.components.homekit.const import (
    BRIDGE_NAME,
    BRIDGE_SERIAL_NUMBER,
    CONF_ADVERTISE_IP,
    DEFAULT_PORT,
    DOMAIN,
    HOMEKIT_MODE_ACCESSORY,
    HOMEKIT_MODE_BRIDGE,
    SERVICE_HOMEKIT_RESET_ACCESSORY,
    SERVICE_HOMEKIT_UNPAIR,
)
from homeassistant.components.homekit.models import HomeKitEntryData
from homeassistant.components.homekit.type_triggers import DeviceTriggerAccessory
from homeassistant.components.homekit.util import get_persist_fullpath_for_entry_id
from homeassistant.components.light import ATTR_COLOR_MODE, ATTR_SUPPORTED_COLOR_MODES, ColorMode
from homeassistant.components.sensor import SensorDeviceClass
from homeassistant.components.switch import SwitchDeviceClass
from homeassistant.config_entries import SOURCE_IMPORT, SOURCE_ZEROCONF
from homeassistant.const import (
    ATTR_DEVICE_CLASS,
    ATTR_DEVICE_ID,
    ATTR_ENTITY_ID,
    ATTR_UNIT_OF_MEASUREMENT,
    CONF_NAME,
    CONF_PORT,
    EVENT_HOMEASSISTANT_STARTED,
    PERCENTAGE,
    SERVICE_RELOAD,
    STATE_ON,
    EntityCategory,
)
from homeassistant.core import HomeAssistant, State
from homeassistant.exceptions import HomeAssistantError
from homeassistant.helpers import device_registry as dr, entity_registry as er, instance_id
from homeassistant.helpers.entityfilter import (
    CONF_EXCLUDE_DOMAINS,
    CONF_EXCLUDE_ENTITIES,
    CONF_EXCLUDE_ENTITY_GLOBS,
    CONF_INCLUDE_DOMAINS,
    CONF_INCLUDE_ENTITIES,
    CONF_INCLUDE_ENTITY_GLOBS,
    EntityFilter,
    convert_filter,
)
from homeassistant.setup import async_setup_component

from .util import PATH_HOMEKIT, async_init_entry, async_init_integration
from tests.common import MockConfigEntry, get_fixture_path

IP_ADDRESS = '127.0.0.1'
DEFAULT_LISTEN = ['0.0.0.0', '::']

def generate_filter(
    include_domains: List[str],
    include_entities: List[str],
    exclude_domains: List[str],
    exclude_entites: List[str],
    include_globs: Optional[List[str]] = None,
    exclude_globs: Optional[List[str]] = None,
) -> EntityFilter:
    """Generate an entity filter using the standard method."""
    return convert_filter({
        CONF_INCLUDE_DOMAINS: include_domains,
        CONF_INCLUDE_ENTITIES: include_entities,
        CONF_EXCLUDE_DOMAINS: exclude_domains,
        CONF_EXCLUDE_ENTITIES: exclude_entites,
        CONF_INCLUDE_ENTITY_GLOBS: include_globs or [],
        CONF_EXCLUDE_ENTITY_GLOBS: exclude_globs or [],
    })

@pytest.fixture(autouse=True)
def always_patch_driver(hk_driver: MagicMock) -> None:
    """Load the hk_driver fixture."""
    pass

@pytest.fixture(autouse=True)
def patch_source_ip() -> None:
    """Patch homeassistant and pyhap functions for getting local address."""
    with patch('pyhap.util.get_local_address', return_value='10.10.10.10'):
        yield

def _mock_homekit(
    hass: HomeAssistant,
    entry: MockConfigEntry,
    homekit_mode: str,
    entity_filter: Optional[EntityFilter] = None,
    devices: Optional[List[str]] = None,
) -> HomeKit:
    return HomeKit(
        hass=hass,
        name=BRIDGE_NAME,
        port=DEFAULT_PORT,
        ip_address=None,
        entity_filter=entity_filter or generate_filter([], [], [], []),
        exclude_accessory_mode=False,
        entity_config={},
        homekit_mode=homekit_mode,
        advertise_ips=None,
        entry_id=entry.entry_id,
        entry_title=entry.title,
        devices=devices or [],
    )

def _mock_homekit_bridge(hass: HomeAssistant, entry: MockConfigEntry) -> HomeKit:
    homekit = _mock_homekit(hass, entry, HOMEKIT_MODE_BRIDGE)
    homekit.driver = MagicMock()
    homekit.iid_storage = MagicMock()
    return homekit

def _mock_accessories(accessory_count: int) -> Dict[int, MagicMock]:
    accessories = {}
    for idx in range(accessory_count + 1):
        accessories[idx + 1000] = MagicMock(async_stop=AsyncMock())
    return accessories

def _mock_pyhap_bridge() -> MagicMock:
    return MagicMock(
        aid=1,
        accessories=_mock_accessories(10),
        display_name='HomeKit Bridge',
    )

@pytest.mark.usefixtures('mock_async_zeroconf')
async def test_setup_min(hass: HomeAssistant) -> None:
    """Test async_setup with min config options."""
    entry = MockConfigEntry(
        domain=DOMAIN,
        data={CONF_NAME: BRIDGE_NAME, CONF_PORT: DEFAULT_PORT},
        options={},
    )
    entry.add_to_hass(hass)
    with patch(f'{PATH_HOMEKIT}.HomeKit') as mock_homekit, patch(
        'homeassistant.components.network.async_get_source_ip',
        return_value='1.2.3.4',
    ):
        mock_homekit.return_value = homekit = Mock()
        type(homekit).async_start = AsyncMock()
        assert await hass.config_entries.async_setup(entry.entry_id)
        await hass.async_block_till_done()
    mock_homekit.assert_any_call(
        hass,
        BRIDGE_NAME,
        DEFAULT_PORT,
        DEFAULT_LISTEN,
        ANY,
        ANY,
        {},
        HOMEKIT_MODE_BRIDGE,
        ['1.2.3.4', '10.10.10.10'],
        entry.entry_id,
        entry.title,
        devices=[],
    )
    hass.bus.async_fire(EVENT_HOMEASSISTANT_STARTED)
    await hass.async_block_till_done()
    assert mock_homekit().async_start.called is True

# ... (rest of the test functions with similar type annotations)

@pytest.mark.usefixtures('mock_async_zeroconf')
async def test_homekit_start_in_accessory_mode_missing_entity(
    hass: HomeAssistant,
    hk_driver: MagicMock,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Test HomeKit start method in accessory mode when entity is not available."""
    entry = await async_init_integration(hass)
    homekit = _mock_homekit(hass, entry, HOMEKIT_MODE_ACCESSORY)
    homekit.bridge = Mock()
    homekit.bridge.accessories = []
    homekit.driver = hk_driver
    homekit.driver.accessory = Accessory(hk_driver, 'any')
    with patch(f'{PATH_HOMEKIT}.HomeKit.add_bridge_accessory'), patch(
        f'{PATH_HOMEKIT}.async_show_setup_message'
    ), patch('pyhap.accessory_driver.AccessoryDriver.async_start'):
        await homekit.async_start()
    await hass.async_block_till_done()
    assert homekit.status == STATUS_WAIT
    assert 'entity not available' in caplog.text

@pytest.mark.usefixtures('mock_async_zeroconf')
async def test_wait_for_port_to_free(
    hass: HomeAssistant,
    hk_driver: MagicMock,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Test we wait for the port to free before declaring unload success."""
    entry = MockConfigEntry(
        domain=DOMAIN,
        data={CONF_NAME: BRIDGE_NAME, CONF_PORT: DEFAULT_PORT},
        options={},
    )
    entry.add_to_hass(hass)
    with patch('pyhap.accessory_driver.AccessoryDriver.async_start'), patch(
        f'{PATH_HOMEKIT}.HomeKit.async_stop'
    ), patch(f'{PATH_HOMEKIT}.async_port_is_available', return_value=True) as port_mock:
        assert await hass.config_entries.async_setup(entry.entry_id)
        await hass.async_block_till_done()
        assert await hass.config_entries.async_unload(entry.entry_id)
        await hass.async_block_till_done()
        assert 'Waiting for the HomeKit server to shutdown' not in caplog.text
        assert port_mock.called
    with patch('pyhap.accessory_driver.AccessoryDriver.async_start'), patch(
        f'{PATH_HOMEKIT}.HomeKit.async_stop'
    ), patch.object(homekit_base, 'PORT_CLEANUP_CHECK_INTERVAL_SECS', 0), patch(
        f'{PATH_HOMEKIT}.async_port_is_available', return_value=False
    ) as port_mock:
        assert await hass.config_entries.async_setup(entry.entry_id)
        await hass.async_block_till_done()
        assert await hass.config_entries.async_unload(entry.entry_id)
        await hass.async_block_till_done()
        assert 'Waiting for the HomeKit server to shutdown' in caplog.text
        assert port_mock.called
