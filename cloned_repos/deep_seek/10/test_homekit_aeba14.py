"""Tests for the HomeKit component."""
from __future__ import annotations
import asyncio
from typing import Any, Dict, List, Optional, Set, Tuple, Union
from unittest.mock import ANY, AsyncMock, MagicMock, Mock, patch
from uuid import UUID, uuid1
from pyhap.accessory import Accessory
from pyhap.const import CATEGORY_CAMERA, CATEGORY_TELEVISION
import pytest
from homeassistant import config as hass_config
from homeassistant.components import homekit as homekit_base, zeroconf
from homeassistant.components.binary_sensor import BinarySensorDeviceClass
from homeassistant.components.event import EventDeviceClass
from homeassistant.components.homekit import MAX_DEVICES, STATUS_READY, STATUS_RUNNING, STATUS_STOPPED, STATUS_WAIT, HomeKit
from homeassistant.components.homekit.accessories import HomeBridge
from homeassistant.components.homekit.const import BRIDGE_NAME, BRIDGE_SERIAL_NUMBER, CONF_ADVERTISE_IP, DEFAULT_PORT, DOMAIN, HOMEKIT_MODE_ACCESSORY, HOMEKIT_MODE_BRIDGE, SERVICE_HOMEKIT_RESET_ACCESSORY, SERVICE_HOMEKIT_UNPAIR
from homeassistant.components.homekit.models import HomeKitEntryData
from homeassistant.components.homekit.type_triggers import DeviceTriggerAccessory
from homeassistant.components.homekit.util import get_persist_fullpath_for_entry_id
from homeassistant.components.light import ATTR_COLOR_MODE, ATTR_SUPPORTED_COLOR_MODES, ColorMode
from homeassistant.components.sensor import SensorDeviceClass
from homeassistant.components.switch import SwitchDeviceClass
from homeassistant.config_entries import SOURCE_IMPORT, SOURCE_ZEROCONF, ConfigEntry
from homeassistant.const import ATTR_DEVICE_CLASS, ATTR_DEVICE_ID, ATTR_ENTITY_ID, ATTR_UNIT_OF_MEASUREMENT, CONF_NAME, CONF_PORT, EVENT_HOMEASSISTANT_STARTED, PERCENTAGE, SERVICE_RELOAD, STATE_ON, EntityCategory
from homeassistant.core import HomeAssistant, State
from homeassistant.exceptions import HomeAssistantError
from homeassistant.helpers import device_registry as dr, entity_registry as er, instance_id
from homeassistant.helpers.entityfilter import CONF_EXCLUDE_DOMAINS, CONF_EXCLUDE_ENTITIES, CONF_EXCLUDE_ENTITY_GLOBS, CONF_INCLUDE_DOMAINS, CONF_INCLUDE_ENTITIES, CONF_INCLUDE_ENTITY_GLOBS, EntityFilter, convert_filter
from homeassistant.setup import async_setup_component
from .util import PATH_HOMEKIT, async_init_entry, async_init_integration
from tests.common import MockConfigEntry, get_fixture_path

IP_ADDRESS: str = '127.0.0.1'
DEFAULT_LISTEN: List[str] = ['0.0.0.0', '::']

def generate_filter(
    include_domains: List[str],
    include_entities: List[str],
    exclude_domains: List[str],
    exclude_entites: List[str],
    include_globs: Optional[List[str]] = None,
    exclude_globs: Optional[List[str]] = None
) -> EntityFilter:
    """Generate an entity filter using the standard method."""
    return convert_filter({
        CONF_INCLUDE_DOMAINS: include_domains,
        CONF_INCLUDE_ENTITIES: include_entities,
        CONF_EXCLUDE_DOMAINS: exclude_domains,
        CONF_EXCLUDE_ENTITIES: exclude_entites,
        CONF_INCLUDE_ENTITY_GLOBS: include_globs or [],
        CONF_EXCLUDE_ENTITY_GLOBS: exclude_globs or []
    })

@pytest.fixture(autouse=True)
def always_patch_driver(hk_driver: Any) -> None:
    """Load the hk_driver fixture."""

@pytest.fixture(autouse=True)
def patch_source_ip() -> Any:
    """Patch homeassistant and pyhap functions for getting local address."""
    with patch('pyhap.util.get_local_address', return_value='10.10.10.10'):
        yield

def _mock_homekit(
    hass: HomeAssistant,
    entry: ConfigEntry,
    homekit_mode: str,
    entity_filter: Optional[EntityFilter] = None,
    devices: Optional[List[str]] = None
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
        devices=devices or []
    )

def _mock_homekit_bridge(hass: HomeAssistant, entry: ConfigEntry) -> HomeKit:
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
    return MagicMock(aid=1, accessories=_mock_accessories(10), display_name='HomeKit Bridge')

@pytest.mark.usefixtures('mock_async_zeroconf')
async def test_setup_min(hass: HomeAssistant) -> None:
    """Test async_setup with min config options."""
    entry = MockConfigEntry(domain=DOMAIN, data={CONF_NAME: BRIDGE_NAME, CONF_PORT: DEFAULT_PORT}, options={})
    entry.add_to_hass(hass)
    with patch(f'{PATH_HOMEKIT}.HomeKit') as mock_homekit, patch('homeassistant.components.network.async_get_source_ip', return_value='1.2.3.4'):
        mock_homekit.return_value = homekit = Mock()
        type(homekit).async_start = AsyncMock()
        assert await hass.config_entries.async_setup(entry.entry_id)
        await hass.async_block_till_done()
    mock_homekit.assert_any_call(hass, BRIDGE_NAME, DEFAULT_PORT, DEFAULT_LISTEN, ANY, ANY, {}, HOMEKIT_MODE_BRIDGE, ['1.2.3.4', '10.10.10.10'], entry.entry_id, entry.title, devices=[])
    hass.bus.async_fire(EVENT_HOMEASSISTANT_STARTED)
    await hass.async_block_till_done()
    assert mock_homekit().async_start.called is True

@patch(f'{PATH_HOMEKIT}.async_port_is_available', return_value=True)
@pytest.mark.usefixtures('mock_async_zeroconf')
async def test_removing_entry(port_mock: MagicMock, hass: HomeAssistant) -> None:
    """Test removing a config entry."""
    entry = MockConfigEntry(domain=DOMAIN, data={CONF_NAME: BRIDGE_NAME, CONF_PORT: DEFAULT_PORT}, options={})
    entry.add_to_hass(hass)
    with patch(f'{PATH_HOMEKIT}.HomeKit') as mock_homekit, patch('homeassistant.components.network.async_get_source_ip', return_value='1.2.3.4'):
        mock_homekit.return_value = homekit = Mock()
        type(homekit).async_start = AsyncMock()
        assert await hass.config_entries.async_setup(entry.entry_id)
        await hass.async_block_till_done()
    mock_homekit.assert_any_call(hass, BRIDGE_NAME, DEFAULT_PORT, DEFAULT_LISTEN, ANY, ANY, {}, HOMEKIT_MODE_BRIDGE, ['1.2.3.4', '10.10.10.10'], entry.entry_id, entry.title, devices=[])
    hass.bus.async_fire(EVENT_HOMEASSISTANT_STARTED)
    await hass.async_block_till_done()
    assert mock_homekit().async_start.called is True
    await hass.config_entries.async_remove(entry.entry_id)
    await hass.async_block_till_done()

@pytest.mark.usefixtures('mock_async_zeroconf')
async def test_homekit_setup(hass: HomeAssistant, hk_driver: Any) -> None:
    """Test setup of bridge and driver."""
    entry = MockConfigEntry(domain=DOMAIN, data={CONF_NAME: 'mock_name', CONF_PORT: 12345}, source=SOURCE_IMPORT)
    homekit = HomeKit(hass, BRIDGE_NAME, DEFAULT_PORT, IP_ADDRESS, True, {}, {}, HOMEKIT_MODE_BRIDGE, advertise_ips=None, entry_id=entry.entry_id, entry_title=entry.title)
    hass.states.async_set('light.demo', 'on')
    hass.states.async_set('light.demo2', 'on')
    zeroconf_mock = MagicMock()
    uuid = await instance_id.async_get(hass)
    with patch(f'{PATH_HOMEKIT}.HomeDriver', return_value=hk_driver) as mock_driver:
        homekit.iid_storage = MagicMock()
        await hass.async_add_executor_job(homekit.setup, zeroconf_mock, uuid)
    path = get_persist_fullpath_for_entry_id(hass, entry.entry_id)
    mock_driver.assert_called_with(hass, entry.entry_id, BRIDGE_NAME, entry.title, loop=hass.loop, address=IP_ADDRESS, port=DEFAULT_PORT, persist_file=path, advertised_address=None, async_zeroconf_instance=zeroconf_mock, zeroconf_server=f'{uuid}-hap.local.', loader=ANY, iid_storage=ANY)
    assert homekit.driver.safe_mode is False

async def test_homekit_setup_ip_address(hass: HomeAssistant, hk_driver: Any, mock_async_zeroconf: Any) -> None:
    """Test setup with given IP address."""
    entry = MockConfigEntry(domain=DOMAIN, data={CONF_NAME: 'mock_name', CONF_PORT: 12345}, source=SOURCE_IMPORT)
    homekit = HomeKit(hass, BRIDGE_NAME, DEFAULT_PORT, '172.0.0.0', True, {}, {}, HOMEKIT_MODE_BRIDGE, None, entry_id=entry.entry_id, entry_title=entry.title)
    path = get_persist_fullpath_for_entry_id(hass, entry.entry_id)
    uuid = await instance_id.async_get(hass)
    with patch(f'{PATH_HOMEKIT}.HomeDriver', return_value=hk_driver) as mock_driver:
        homekit.iid_storage = MagicMock()
        await hass.async_add_executor_job(homekit.setup, mock_async_zeroconf, uuid)
    mock_driver.assert_called_with(hass, entry.entry_id, BRIDGE_NAME, entry.title, loop=hass.loop, address='172.0.0.0', port=DEFAULT_PORT, persist_file=path, advertised_address=None, async_zeroconf_instance=mock_async_zeroconf, zeroconf_server=f'{uuid}-hap.local.', loader=ANY, iid_storage=ANY)

async def test_homekit_with_single_advertise_ips(hass: HomeAssistant, hk_driver: Any, mock_async_zeroconf: Any, hass_storage: Any) -> None:
    """Test setup with a single advertise ips."""
    entry = MockConfigEntry(domain=DOMAIN, data={CONF_NAME: 'mock_name', CONF_PORT: 12345, CONF_ADVERTISE_IP: '1.3.4.4'}, source=SOURCE_IMPORT)
    entry.add_to_hass(hass)
    with patch(f'{PATH_HOMEKIT}.HomeDriver', return_value=hk_driver) as mock_driver:
        hk_driver.async_start = AsyncMock()
        await hass.config_entries.async_setup(entry.entry_id)
        await hass.async_block_till_done()
    mock_driver.assert_called_with(hass, entry.entry_id, ANY, entry.title, loop=hass.loop, address=DEFAULT_LISTEN, port=ANY, persist_file=ANY, advertised_address='1.3.4.4', async_zeroconf_instance=mock_async_zeroconf, zeroconf_server=ANY, loader=ANY, iid_storage=ANY)

async def test_homekit_with_many_advertise_ips(hass: HomeAssistant, hk_driver: Any, mock_async_zeroconf: Any, hass_storage: Any) -> None:
    """Test setup with many advertise ips."""
    entry = MockConfigEntry(domain=DOMAIN, data={CONF_NAME: 'mock_name', CONF_PORT: 12345, CONF_ADVERTISE_IP: ['1.3.4.4', '4.3.2.2']}, source=SOURCE_IMPORT)
    entry.add_to_hass(hass)
    with patch(f'{PATH_HOMEKIT}.HomeDriver', return_value=hk_driver) as mock_driver:
        hk_driver.async_start = AsyncMock()
        await hass.config_entries.async_setup(entry.entry_id)
        await hass.async_block_till_done()
    mock_driver.assert_called_with(hass, entry.entry_id, ANY, entry.title, loop=hass.loop, address=DEFAULT_LISTEN, port=ANY, persist_file=ANY, advertised_address=['1.3.4.4', '4.3.2.2'], async_zeroconf_instance=mock_async_zeroconf, zeroconf_server=ANY, loader=ANY, iid_storage=ANY)

@pytest.mark.usefixtures('mock_async_zeroconf')
async def test_homekit_setup_advertise_ips(hass: HomeAssistant, hk_driver: Any) -> None:
    """Test setup with given IP address to advertise."""
    entry = MockConfigEntry(domain=DOMAIN, data={CONF_NAME: 'mock_name', CONF_PORT: 12345}, source=SOURCE_IMPORT)
    homekit = HomeKit(hass, BRIDGE_NAME, DEFAULT_PORT, '0.0.0.0', True, {}, {}, HOMEKIT_MODE_BRIDGE, '192.168.1.100', entry_id=entry.entry_id, entry_title=entry.title)
    async_zeroconf_instance = MagicMock()
    path = get_persist_fullpath_for_entry_id(hass, entry.entry_id)
    uuid = await instance_id.async_get(hass)
    with patch(f'{PATH_HOMEKIT}.HomeDriver', return_value=hk_driver) as mock_driver:
        homekit.iid_storage = MagicMock()
        await hass.async_add_executor_job(homekit.setup, async_zeroconf_instance, uuid)
    mock_driver.assert_called_with(hass, entry.entry_id, BRIDGE_NAME, entry.title, loop=hass.loop, address='0.0.0.0', port=DEFAULT_PORT, persist_file=path, advertised_address='192.168.1.100', async_zeroconf_instance=async_zeroconf_instance, zeroconf_server=f'{uuid}-hap.local.', loader=ANY, iid_storage=ANY)

@pytest.mark.usefixtures('mock_async_zeroconf')
async def test_homekit_add_accessory(hass: HomeAssistant, mock_hap: Any) -> None:
    """Add accessory if config exists and get_acc returns an accessory."""
    entry = MockConfigEntry(domain=DOMAIN, data={CONF_NAME: 'mock_name', CONF_PORT: 12345})
    entry.add_to_hass(hass)
    homekit = _mock_homekit_bridge(hass, entry)
    mock_acc = Mock(category='any')
    with patch(f'{PATH_HOMEKIT}.HomeKit', return_value=homekit):
        assert await hass.config_entries.async_setup(entry.entry_id)
        await hass.async_block_till_done()
    homekit.bridge = _mock_pyhap_bridge()
    with patch(f'{PATH_HOMEKIT}.get_accessory') as mock_get_acc:
        mock_get_acc.side_effect = [None, mock_acc, None]
        state = State('light.demo', 'on')
        homekit.add_bridge_accessory(state)
        mock_get_acc.assert_called_with(hass, ANY, ANY, 1403373688, {})
        assert not homekit.bridge.add_accessory.called
        state = State('demo.test', 'on')
        homekit.add_bridge_accessory(state)
        mock_get_acc.assert_called_with(hass, ANY, ANY, 600325356, {})
        assert homekit.bridge.add_accessory.called
        state = State('demo.test_2', 'on')
        homekit.add_bridge_accessory(state)
        mock_get_acc.assert_called_with(hass, ANY, ANY, 1467253281, {})
        assert homekit.bridge.add_accessory.called
        await homekit.async_stop()

@pytest.mark.parametrize('acc_category', [CATEGORY_TELEVISION, CATEGORY_CAMERA])
@pytest.mark.usefixtures('mock_async_zeroconf')
async def test_homekit_warn_add_accessory_bridge(hass: HomeAssistant, acc_category: str, mock_hap: Any, caplog: Any) -> None:
    """Test we warn when adding cameras or tvs to a bridge."""
    entry = MockConfigEntry(domain=DOMAIN, data={CONF_NAME: 'mock_name', CONF_PORT: 12345})
    entry.add_to_hass(hass)
    homekit = _mock_homekit_bridge(hass, entry)
    with patch(f'{PATH_HOMEKIT}.HomeKit', return_value=homekit):
        assert await hass.config_entries.async_setup(entry.entry_id)
        await hass.async