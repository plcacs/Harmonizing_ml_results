"""Tests for the Entity Registry."""
from datetime import datetime, timedelta
from functools import partial
from typing import Any, Dict, Optional, Set, Tuple, Union
from unittest.mock import patch
import attr
from freezegun.api import FrozenDateTimeFactory
import pytest
import voluptuous as vol
from homeassistant import config_entries
from homeassistant.const import EVENT_HOMEASSISTANT_START, STATE_UNAVAILABLE, EntityCategory
from homeassistant.core import CoreState, HomeAssistant, callback
from homeassistant.exceptions import MaxLengthExceeded
from homeassistant.helpers import device_registry as dr, entity_registry as er
from homeassistant.util.dt import utc_from_timestamp
from tests.common import ANY, MockConfigEntry, async_capture_events, async_fire_time_changed, flush_store

YAML__OPEN_PATH = 'homeassistant.util.yaml.loader.open'

async def test_get(entity_registry: er.EntityRegistry) -> None:
    """Test we can get an item."""
    entry = entity_registry.async_get_or_create('light', 'hue', '1234')
    assert entity_registry.async_get(entry.entity_id) is entry
    assert entity_registry.async_get(entry.id) is entry
    assert entity_registry.async_get('blah') is None
    assert entity_registry.async_get('blah.blah') is None

async def test_get_or_create_returns_same_entry(hass: HomeAssistant, entity_registry: er.EntityRegistry) -> None:
    """Make sure we do not duplicate entries."""
    update_events = async_capture_events(hass, er.EVENT_ENTITY_REGISTRY_UPDATED)
    entry = entity_registry.async_get_or_create('light', 'hue', '1234')
    entry2 = entity_registry.async_get_or_create('light', 'hue', '1234')
    await hass.async_block_till_done()
    assert len(entity_registry.entities) == 1
    assert entry is entry2
    assert entry.entity_id == 'light.hue_1234'
    assert len(update_events) == 1
    assert update_events[0].data == {'action': 'create', 'entity_id': entry.entity_id}

def test_get_or_create_suggested_object_id(entity_registry: er.EntityRegistry) -> None:
    """Test that suggested_object_id works."""
    entry = entity_registry.async_get_or_create('light', 'hue', '1234', suggested_object_id='beer')
    assert entry.entity_id == 'light.beer'

def test_get_or_create_updates_data(hass: HomeAssistant, device_registry: dr.DeviceRegistry, entity_registry: er.EntityRegistry, freezer: FrozenDateTimeFactory) -> None:
    """Test that we update data in get_or_create."""
    config_subentry_id = 'blabla'
    orig_config_entry = MockConfigEntry(domain='light', subentries_data=[config_entries.ConfigSubentryData(data={}, subentry_id=config_subentry_id, subentry_type='test', title='Mock title', unique_id='test')])
    orig_config_entry.add_to_hass(hass)
    orig_device_entry = device_registry.async_get_or_create(config_entry_id=orig_config_entry.entry_id, connections={(dr.CONNECTION_NETWORK_MAC, '12:34:56:AB:CD:EF')})
    created = datetime.fromisoformat('2024-02-14T12:00:00.0+00:00')
    freezer.move_to(created)
    orig_entry = entity_registry.async_get_or_create('light', 'hue', '5678', capabilities={'max': 100}, config_entry=orig_config_entry, config_subentry_id=config_subentry_id, device_id=orig_device_entry.id, disabled_by=er.RegistryEntryDisabler.HASS, entity_category=EntityCategory.CONFIG, has_entity_name=True, hidden_by=er.RegistryEntryHider.INTEGRATION, original_device_class='mock-device-class', original_icon='initial-original_icon', original_name='initial-original_name', supported_features=5, translation_key='initial-translation_key', unit_of_measurement='initial-unit_of_measurement')
    assert set(entity_registry.async_device_ids()) == {orig_device_entry.id}
    assert orig_entry == er.RegistryEntry('light.hue_5678', '5678', 'hue', capabilities={'max': 100}, config_entry_id=orig_config_entry.entry_id, config_subentry_id=config_subentry_id, created_at=created, device_class=None, device_id=orig_device_entry.id, disabled_by=er.RegistryEntryDisabler.HASS, entity_category=EntityCategory.CONFIG, has_entity_name=True, hidden_by=er.RegistryEntryHider.INTEGRATION, icon=None, id=orig_entry.id, modified_at=created, name=None, original_device_class='mock-device-class', original_icon='initial-original_icon', original_name='initial-original_name', supported_features=5, translation_key='initial-translation_key', unit_of_measurement='initial-unit_of_measurement')
    new_config_entry = MockConfigEntry(domain='light')
    new_config_entry.add_to_hass(hass)
    new_device_entry = device_registry.async_get_or_create(config_entry_id=new_config_entry.entry_id, connections={(dr.CONNECTION_NETWORK_MAC, '34:56:AB:CD:EF:12')})
    modified = created + timedelta(minutes=5)
    freezer.move_to(modified)
    new_entry = entity_registry.async_get_or_create('light', 'hue', '5678', capabilities={'new-max': 150}, config_entry=new_config_entry, config_subentry_id=None, device_id=new_device_entry.id, disabled_by=er.RegistryEntryDisabler.USER, entity_category=EntityCategory.DIAGNOSTIC, has_entity_name=False, hidden_by=er.RegistryEntryHider.USER, original_device_class='new-mock-device-class', original_icon='updated-original_icon', original_name='updated-original_name', supported_features=10, translation_key='updated-translation_key', unit_of_measurement='updated-unit_of_measurement')
    assert new_entry == er.RegistryEntry('light.hue_5678', '5678', 'hue', aliases=set(), area_id=None, capabilities={'new-max': 150}, config_entry_id=new_config_entry.entry_id, config_subentry_id=None, created_at=created, device_class=None, device_id=new_device_entry.id, disabled_by=er.RegistryEntryDisabler.HASS, entity_category=EntityCategory.DIAGNOSTIC, has_entity_name=False, hidden_by=er.RegistryEntryHider.INTEGRATION, icon=None, id=orig_entry.id, modified_at=modified, name=None, original_device_class='new-mock-device-class', original_icon='updated-original_icon', original_name='updated-original_name', supported_features=10, translation_key='updated-translation_key', unit_of_measurement='updated-unit_of_measurement')
    assert set(entity_registry.async_device_ids()) == {new_device_entry.id}
    modified = created + timedelta(minutes=5)
    freezer.move_to(modified)
    new_entry = entity_registry.async_get_or_create('light', 'hue', '5678', capabilities=None, config_entry=None, device_id=None, disabled_by=None, entity_category=None, has_entity_name=None, hidden_by=None, original_device_class=None, original_icon=None, original_name=None, supported_features=None, translation_key=None, unit_of_measurement=None)
    assert new_entry == er.RegistryEntry('light.hue_5678', '5678', 'hue', aliases=set(), area_id=None, capabilities=None, config_entry_id=None, created_at=created, device_class=None, device_id=None, disabled_by=er.RegistryEntryDisabler.HASS, entity_category=None, has_entity_name=None, hidden_by=er.RegistryEntryHider.INTEGRATION, icon=None, id=orig_entry.id, modified_at=modified, name=None, original_device_class=None, original_icon=None, original_name=None, supported_features=0, translation_key=None, unit_of_measurement=None)
    assert set(entity_registry.async_device_ids()) == set()

def test_get_or_create_suggested_object_id_conflict_register(entity_registry: er.EntityRegistry) -> None:
    """Test that we don't generate an entity id that is already registered."""
    entry = entity_registry.async_get_or_create('light', 'hue', '1234', suggested_object_id='beer')
    entry2 = entity_registry.async_get_or_create('light', 'hue', '5678', suggested_object_id='beer')
    assert entry.entity_id == 'light.beer'
    assert entry2.entity_id == 'light.beer_2'

def test_get_or_create_suggested_object_id_conflict_existing(hass: HomeAssistant, entity_registry: er.EntityRegistry) -> None:
    """Test that we don't generate an entity id that currently exists."""
    hass.states.async_set('light.hue_1234', 'on')
    entry = entity_registry.async_get_or_create('light', 'hue', '1234')
    assert entry.entity_id == 'light.hue_1234_2'

def test_create_triggers_save(entity_registry: er.EntityRegistry) -> None:
    """Test that registering entry triggers a save."""
    with patch.object(entity_registry, 'async_schedule_save') as mock_schedule_save:
        entity_registry.async_get_or_create('light', 'hue', '1234')
    assert len(mock_schedule_save.mock_calls) == 1

async def test_loading_saving_data(hass: HomeAssistant, device_registry: dr.DeviceRegistry, entity_registry: er.EntityRegistry) -> None:
    """Test that we load/save data correctly."""
    mock_config = MockConfigEntry(domain='light')
    mock_config.add_to_hass(hass)
    device_entry = device_registry.async_get_or_create(config_entry_id=mock_config.entry_id, connections={(dr.CONNECTION_NETWORK_MAC, '12:34:56:AB:CD:EF')})
    orig_entry1 = entity_registry.async_get_or_create('light', 'hue', '1234')
    orig_entry2 = entity_registry.async_get_or_create('light', 'hue', '5678', capabilities={'max': 100}, config_entry=mock_config, device_id=device_entry.id, disabled_by=er.RegistryEntryDisabler.HASS, entity_category=EntityCategory.CONFIG, hidden_by=er.RegistryEntryHider.INTEGRATION, has_entity_name=True, original_device_class='mock-device-class', original_icon='hass:original-icon', original_name='Original Name', supported_features=5, translation_key='initial-translation_key', unit_of_measurement='initial-unit_of_measurement')
    entity_registry.async_update_entity(orig_entry2.entity_id, aliases={'initial_alias_1', 'initial_alias_2'}, area_id='mock-area-id', device_class='user-class', name='User Name', icon='hass:user-icon')
    entity_registry.async_update_entity_options(orig_entry2.entity_id, 'light', {'minimum_brightness': 20})
    entity_registry.async_update_entity(orig_entry2.entity_id, categories={'scope', 'id'}, labels={'label1', 'label2'})
    orig_entry2 = entity_registry.async_get(orig_entry2.entity_id)
    orig_entry3 = entity_registry.async_get_or_create('light', 'hue', 'ABCD')
    orig_entry4 = entity_registry.async_get_or_create('light', 'hue', 'EFGH')
    entity_registry.async_remove(orig_entry3.entity_id)
    entity_registry.async_remove(orig_entry4.entity_id)
    assert len(entity_registry.entities) == 2
    assert len(entity_registry.deleted_entities) == 2
    registry2 = er.EntityRegistry(hass)
    await flush_store(entity_registry._store)
    await registry2.async_load()
    assert list(entity_registry.entities) == list(registry2.entities)
    assert list(entity_registry.deleted_entities) == list(registry2.deleted_entities)
    new_entry1 = entity_registry.async_get_or_create('light', 'hue', '1234')
    new_entry2 = entity_registry.async_get_or_create('light', 'hue', '5678')
    new_entry3 = entity_registry.async_get_or_create('light', 'hue', 'ABCD')
    new_entry4 = entity_registry.async_get_or_create('light', 'hue', 'EFGH')
    assert orig_entry1 == new_entry1
    assert orig_entry2 == new_entry2
    assert orig_entry3.modified_at < new_entry3.modified_at
    assert attr.evolve(orig_entry3, modified_at=new_entry3.modified_at) == new_entry3
    assert orig_entry4.modified_at < new_entry4.modified_at
    assert attr.evolve(orig_entry4, modified_at=new_entry4.modified_at) == new_entry4
    assert new_entry2.area_id == 'mock-area-id'
    assert new_entry2.categories == {'scope', 'id'}
    assert new_entry2.capabilities == {'max': 100}
    assert new_entry2.config_entry_id == mock_config.entry_id
    assert new_entry2.device_class == 'user-class'
    assert new_entry2.device_id == device_entry.id
    assert new_entry2.disabled_by is er.RegistryEntryDisabler.HASS
    assert new_entry2.entity_category == 'config'
    assert new_entry2.icon == 'hass:user-icon'
    assert new_entry2.hidden_by == er.RegistryEntryHider.INTEGRATION
    assert new_entry2.has_entity_name is True
    assert new_entry2.labels == {'label1', 'label2'}
    assert new_entry2.name == 'User Name'
    assert new_entry2.options == {'light': {'minimum_brightness': 20}}
    assert new_entry2.original_device_class == 'mock-device-class'
    assert new_entry2.original_icon == 'hass:original-icon'
    assert new_entry2.original_name == 'Original Name'
    assert new_entry2.supported_features == 5
    assert new_entry2.translation_key == 'initial-translation_key'
    assert new_entry2.unit_of_measurement == 'initial-unit_of_measurement'

def test_generate_entity_considers_registered_entities(entity_registry: er.EntityRegistry) -> None:
    """Test that we don't create entity id that are already registered."""
    entry = entity_registry.async_get_or_create('light', 'hue', '1234')
    assert entry.entity_id == 'light.hue_1234'
    assert entity_registry.async_generate_entity_id('light', 'hue_1234') == 'light.hue_1234_2'

def test_generate_entity_considers_existing_entities(hass: HomeAssistant, entity_registry: er.EntityRegistry) -> None:
    """Test that we don't create entity id that currently exists."""
    hass.states.async_set('light.kitchen', 'on')
    assert entity_registry.async_generate_entity_id('light', 'kitchen') == 'light.kitchen_2'

def test_is_registered(entity_registry: er.EntityRegistry) -> None:
    """Test that is_registered works."""
    entry = entity_registry.async_get_or_create('light', 'hue', '1234')
    assert entity_registry.async_is_registered(entry.entity_id)
    assert not entity_registry.async_is_registered('light.non_existing')

@pytest.mark.parametrize('load_registries', [False])
async def test_filter_on_load(hass: HomeAssistant, hass_storage: Dict[str, Any]) -> None:
    """Test we transform some data when loading from storage."""
    hass_storage[er.STORAGE_KEY] = {'version': er.STORAGE_VERSION_MAJOR, 'minor_version': 1, 'data': {'entities': [{'entity_id': 'test.named', 'platform': 'super_platform', 'unique_id': 'with-name', 'name': 'registry override'}, {'entity_id': 'test.no_name', 'platform': 'super_platform', 'unique_id': 'without-name'}, {'entity_id': 'test.disabled_user', 'platform': 'super_platform', 'unique_id': 'disabled-user', 'disabled_by': 'user'}, {'entity_id': 'test.disabled_hass', 'platform': 'super_platform', 'unique_id': 'disabled-hass', 'disabled_by': 'hass'}]}}
    await er.async_load(hass)
    registry = er.async_get(hass)
    assert len(registry.entities) == 4
    assert set(registry.entities.keys()) == {'test.disabled_hass', 'test.disabled_user', 'test.named', 'test.no_name'}
    entry_with_name = registry.async_get_or_create('test', 'super_platform', 'with-name')
    entry_without_name = registry.async_get_or_create('test', 'super_platform', 'without-name')
    assert entry_with_name.name == 'registry override'
    assert entry_without_name.name is None
    assert not entry_with_name.disabled
    assert entry_with_name.created_at == utc_from_timestamp(0)
    assert entry_with_name.modified_at == utc_from_timestamp(0)
    entry_disabled_hass = registry.async_get_or_create('test', 'super_platform', 'disabled-hass')
    entry_disabled_user = registry.async_get_or_create('test', 'super_platform', 'disabled-user')
    assert entry_disabled_hass.disabled
    assert entry_disabled_hass.disabled_by is er.RegistryEntryDisabler.HASS
    assert entry_disabled_user.disabled
    assert entry_disabled_user.disabled_by is er.RegistryEntryDisabler.USER

@pytest.mark.parametrize('load_registries', [False])
async def test_load_bad_data(hass: HomeAssistant, hass_storage: Dict[str, Any], caplog: pytest.LogCaptureFixture) -> None:
    """Test loading invalid data."""
    hass_storage[er.STORAGE_KEY] = {'version': er.STORAGE_VERSION_MAJOR, 'minor_version': er.STORAGE_VERSION_MINOR, 'data': {'entities': [{'aliases': [], 'area_id': None, 'capabilities': None, 'categories': {}, 'config_entry_id': None, 'config_subentry_id': None, 'created_at': '2024-02-14T12:00:00.900075+00:00', 'device_class': None, 'device_id': None, 'disabled_by': None, 'entity_category': None, 'entity_id': 'test.test1', 'has_entity_name': False, 'hidden_by': None, 'icon': None, 'id': '00001', 'labels': [], 'modified_at': '2024-02-14T12:00:00.900075+00:00', 'name': None, 'options': None, 'original_device_class': None, 'original_icon': None, 'original_name': None, 'platform': 'super_platform', 'previous_unique_id': None, 'supported_features': 0, 'translation_key': None, 'unique_id': 123, 'unit_of_measurement': None}, {'aliases': [], 'area_id': None, 'capabilities': None, 'categories': {}, 'config_entry_id': None, 'config_subentry_id': None, 'created_at': '2024-02-14T12:00:00.900075+00:00', 'device_class': None, 'device_id': None, 'disabled_by': None, 'entity_category': None, 'entity_id': 'test.test2', 'has_entity_name': False, 'hidden_by': None, 'icon': None, 'id': '00002', 'labels': [], 'modified_at': '2024-02-14T12:00:00.900075+00:00', 'name': None, 'options': None, 'original_device_class': None, 'original_icon': None, 'original_name': None, 'platform': 'super_platform', 'previous_unique_id': None, 'supported_features': 0, 'translation_key': None, 'unique_id': ['not', 'valid'], 'unit_of_measurement': None}], 'deleted_entities': [{'config_entry_id': None, 'config_subentry_id': None, 'created_at': '2024-02-14T12:00:00.900075+00:00', 'entity_id': 'test.test3', 'id': '00003', 'modified_at': '2024-02-14T12:00:00.900075+00:00', 'orphaned_timestamp': None, 'platform': 'super_platform', 'unique_id': 234}, {'config_entry_id': None, 'config_subentry_id': None, 'created_at': '2024-02-14T12:00:00.900075+00:00', 'entity_id': 'test.test4', 'id': '00004', 'modified_at': '2024-02-14T12:00:00.900075+00:00', 'orphaned_timestamp': None, 'platform': 'super_platform', 'unique_id': ['also', 'not', 'valid']}]}}
    await er.async_load(hass)
    registry = er.async_get(hass)
    assert len(registry.entities) == 1
    assert set(registry.entities.keys()) == {'test.test1'}
    assert len(registry.deleted_entities) == 1
    assert set(registry.deleted_entities.keys()) == {('test', 'super_platform', 234)}
    assert "'test' from integration super_platform has a non string unique_id '123', please create a bug report" not in caplog.text
    assert "'test' from integration super_platform has a non string unique_id '234', please create a bug report" not in caplog.text
    assert "Entity registry entry 'test.test2' from integration super_platform could not be loaded: 'unique_id must be a string, got ['not', 'valid']', please create a bug report" in caplog.text

def test_async_get_entity_id(entity_registry: er.EntityRegistry) -> Optional[str]:
    """Test that entity_id is returned."""
    entry = entity_registry.async_get_or_create('light', 'hue', '1234')
    assert entry.entity_id == 'light.hue_1234'
    return entity_registry.async_get_entity_id('light', 'hue', '1234')

async def test_updating_config_entry_id(hass: HomeAssistant, entity_registry: er.EntityRegistry) -> None:
    """Test that we update config entry id in registry."""
    update_events = async_capture_events(hass, er.EVENT_ENTITY_REGISTRY_UPDATED)
    mock_config_1 = MockConfigEntry(domain='light', entry_id='mock-id-1')
    mock_config_1.add_to_hass(hass)
    entry = entity_registry.async_get_or_create('light', 'hue', '5678', config_entry=mock_config_1)
    mock_config_2 = MockConfigEntry(domain='light', entry_id='mock-id-2')
    mock_config_2.add_to_hass(hass)
    entry2 = entity_registry.async_get_or_create('light', 'hue', '5678', config_entry=mock_config_2)
    assert entry.entity_id == entry2.entity_id
    assert entry2.config_entry_id == 'mock-id-2'
    await hass.async_block_till_done()
    assert len(update_events) == 2
    assert update_events[0].data == {'action': 'create', 'entity_id': entry.entity_id}
    assert update_events[1].data == {'action': 'update', 'entity_id': entry.entity_id, 'changes': {'config_entry_id': 'mock-id-1'}}

async def test_removing_config_entry_id(hass: HomeAssistant, entity_registry: er.EntityRegistry) -> None:
    """Test that we update config entry id in registry."""
    update_events = async_capture_events(hass, er.EVENT_ENTITY_REGISTRY_UPDATED)
    mock_config = MockConfigEntry(domain='light', entry_id='mock-id-1')
    mock_config.add_to_hass(hass)
    entry = entity_registry.async_get_or_create('light', 'hue', '5678', config_entry=mock_config)
    assert entry.config_entry_id == 'mock-id-1'
    entity_registry.async_clear_config_entry('mock-id-1')
    assert not entity_registry.entities
    await hass.async_block_till_done()
    assert len(update_events) == 2
    assert update_events[0].data == {'action': 'create', 'entity_id': entry.entity_id}
    assert update_events[1].data == {'action': 'remove', 'entity_id': entry.entity_id}

async def test_deleted_entity_removing_config_entry_id(hass: HomeAssistant, entity_registry: er.EntityRegistry) -> None:
    """Test that we update config entry id in registry on deleted entity."""
    mock_config1 = MockConfigEntry(domain='light', entry_id='mock-id-1')
    mock_config2 = MockConfigEntry(domain='light', entry_id='mock-id-2')
    mock_config1.add_to_hass(hass)
    mock_config2.add_to_hass(hass)
    entry1 = entity_registry.async_get_or_create('light', 'hue', '5678', config_entry=mock_config1)
    assert entry1.config_entry_id == 'mock-id-1'
    entry2 = entity_registry.async_get_or_create('light', 'hue', '1234', config_entry=mock_config2)
    assert entry2.config_entry_id == 'mock-id-2'
    entity_registry.async_remove(entry1.entity_id)
    entity_registry.async_remove(entry2.entity_id)
    assert len(entity_registry.entities) == 0
    assert len(entity_registry.deleted_entities) == 2
    deleted_entry1 = entity_registry.deleted_entities['light', 'hue', '5678']
    assert deleted_entry1.config_entry_id == 'mock-id-1'
    assert deleted_entry1.orphaned_timestamp is None
    deleted_entry2 = entity_registry.deleted_entities['light', 'hue', '1234']
    assert deleted_entry2.config_entry_id == 'mock-id-2'
    assert deleted_entry2.orphaned_timestamp is None
    entity_registry.async_clear_config_entry('mock-id-1')
    assert len(entity_registry.entities) == 0
    assert len(entity_registry.deleted_entities) == 2
    deleted_entry1 = entity_registry.deleted_entities['light', 'hue', '5678']
    assert deleted_entry1.config_entry_id is None
    assert deleted_entry1.orphaned_timestamp is not None
    assert entity_registry.deleted_entities['light', 'hue', '1234'] == deleted_entry2

async def test_removing_config_subentry_id(hass: HomeAssistant, entity_registry: er.EntityRegistry) -> None:
    """Test that we update config subentry id in registry."""
    update_events = async_capture_events(hass, er.EVENT_ENTITY_REGISTRY_UPDATED)
    mock_config = MockConfigEntry(domain='light', entry_id='mock-id-1', subentries_data=[config_entries.ConfigSubentryData(data={}, subentry_id='mock-subentry-id-1', subentry_type='test', title='Mock title', unique_id='test')])
    mock_config.add_to_hass(hass)
    entry = entity_registry.async_get_or_create('light', 'hue', '5678', config_entry=mock_config, config_subentry_id='mock-subentry-id-1')
    assert entry.config_subentry_id == 'mock-subentry-id-1'
    hass.config_entries.async_remove_subentry(mock_config, 'mock-subentry-id-1')
    assert not entity_registry.entities
    await hass.async_block_till_done()
    assert len(update_events) == 2
    assert update_events[0].data == {'action': 'create', 'entity_id': entry.entity_id}
    assert update_events[1].data == {'action': 'remove', 'entity_id': entry.entity_id}

async def test_deleted_entity_removing_config_subentry_id(hass: HomeAssistant, entity_registry: er.EntityRegistry) -> None:
    """Test that we update config subentry id in registry on deleted entity."""
    mock_config = MockConfigEntry(domain='light', entry_id='mock-id-1', subentries_data=[config_entries.ConfigSubentryData(data={}, subentry_id='mock-subentry-id-1', subentry_type='test', title='Mock title', unique_id='test'), config_entries.ConfigSubentryData(data={}, subentry_id='mock-subentry-id-2', subentry_type='test', title='Mock title', unique_id='test')])
    mock_config.add_to_hass(hass)
    entry1 = entity_registry.async_get_or_create('light', 'hue', '5678', config_entry=mock_config, config_subentry_id='mock-subentry-id-1')
    assert entry1.config_subentry_id == 'mock-subentry-id-1'
    entry2 = entity_registry.async_get_or_create('light', 'hue', '1234', config_entry=mock_config, config_subentry_id='mock-subentry-id-2')
    assert entry2.config_subentry_id == 'mock-subentry-id-2'
    entity_registry.async_remove(entry1.entity_id)
    entity_registry.async_remove(entry2.entity_id)
    assert len(entity_registry.entities) == 0
    assert len(entity_registry.deleted_entities) == 2
    deleted_entry1 = entity_registry.deleted_entities['light', 'hue', '5678']
    assert deleted_entry1.config_entry_id == 'mock-id-1'
    assert deleted_entry1.config_subentry_id == 'mock-subentry-id-1'
    assert deleted_entry1.orphaned_timestamp is None
    deleted_entry2 = entity_registry.deleted_entities['light', 'hue', '1234']
    assert deleted_entry2.config_entry_id == 'mock-id-1'
    assert deleted_entry2.config_subentry_id == 'mock-subentry-id-2'
    assert deleted_entry2.orphaned_timestamp is None
    hass.config_entries.async_remove_subentry(mock_config, 'mock-subentry-id-1')
    assert len(entity_registry.entities) == 0
    assert len(entity_registry.deleted_entities) == 2
    deleted_entry1 = entity_registry.deleted_entities['light', 'hue', '5678']
    assert deleted_entry1.config_entry_id is None
    assert deleted_entry1.config_subentry_id is None
    assert deleted_entry1.orphaned_timestamp is not None
    assert entity_registry.deleted_entities['light', 'hue', '1234'] == deleted_entry2

async def test_removing_area_id(entity_registry: er.EntityRegistry) -> None:
    """Make sure we can clear area id."""
    entry = entity_registry.async_get_or_create('light', 'hue', '5678')
    entry_w_area = entity_registry.async_update_entity(entry.entity_id, area_id='12345A')
    entity_registry.async_clear_area_id('12345A')
    entry_wo_area = entity_registry.async_get(entry.entity_id)
    assert not entry_wo_area.area_id
    assert entry_w_area != entry_wo_area

@pytest.mark.parametrize('load_registries', [False])
async def test_migration_1_1(hass: HomeAssistant, hass_storage: Dict[str, Any]) -> None:
    """Test migration from version 1.1."""
    hass_storage[er.STORAGE_KEY] = {'version': 1, 'minor_version': 1, 'data': {'entities': [{'device_class': 'best_class', 'entity_id': 'test.entity', 'platform': 'super_platform', 'unique_id': 'very_unique'}]}}
    await er.async_load(hass)
    registry = er.async_get(hass)
    entry = registry.async_get_or_create('test', 'super_platform', 'very_unique')
    assert entry.device_class is None
    assert entry.original_device_class == 'best_class'
    await flush_store(registry._store)
    assert hass_storage[er.STORAGE_KEY] == {'version': er.STORAGE_VERSION_MAJOR, 'minor_version': er.STORAGE_VERSION_MINOR, 'key': er.STORAGE_KEY, 'data': {'entities': [{'aliases': [], 'area_id': None, 'capabilities': {}, 'categories': {}, 'config_entry_id': None, 'config_subentry_id': None, 'created_at': '1970-01-01T00:00:00+00:00', 'device_id': None, 'disabled_by': None, 'entity_category': None, 'entity_id': 'test.entity', 'has_entity_name': False, 'hidden_by': None, 'icon': None, 'id': ANY, 'labels': [], 'modified_at': '1970-01-01T00:00:00+00:00', 'name': None, 'options': {}, 'original_device_class': 'best_class', 'original_icon': None, 'original_name': None, 'platform': 'super_platform', 'previous_unique_id': None, 'supported_features': 0, 'translation_key': None, 'unique_id': 'very_unique', 'unit_of_measurement': None, 'device_class': None}], 'deleted_entities': []}}

@pytest.mark.parametrize('load_registries', [False])
async def test_migration_1_7(hass: HomeAssistant, hass_storage: Dict[str, Any]) -> None:
    """Test migration from version 1.7.

    This tests cleanup after frontend bug which incorrectly updated device_class
    """
    entity_dict = {'area_id': None, 'capabilities': {}, 'config_entry_id': None, 'device_id': None, 'disabled_by': None, 'entity_category': None, 'has_entity_name': False, 'hidden_by': None, 'icon': None, 'id': '12345', 'name': None, 'options': None, 'original_icon': None, 'original_name': None, 'platform': 'super_platform', 'supported_features': 0, 'unique_id': 'very_unique', 'unit_of_measurement': None}
    hass_storage[er.STORAGE_KEY] = {'version': 1, 'minor_version': 7, 'data': {'entities': [{**entity_dict, 'device_class': 'original_class_by_integration', 'entity_id': 'test.entity', 'original_device_class': 'new_class_by_integration'}, {**entity_dict, 'device_class': 'class_by_user', 'entity_id': 'binary_sensor.entity', 'original_device_class': 'class_by_integration'}, {**entity_dict, 'device_class': 'class_by_user', 'entity_id': 'cover.entity', 'original_device_class': 'class_by_integration'}]}}
    await er.async_load(hass)
    registry = er.async_get(hass)
    entry = registry.async_get_or_create('test', 'super_platform', 'very_unique')
    assert entry.device_class is None
    assert entry.original_device_class == 'new_class_by_integration'
    entry = registry.async_get_or_create('binary_sensor', 'super_platform', 'very_unique')
    assert entry.device_class == 'class_by_user'
    assert entry.original_device_class == 'class_by_integration'
    entry = registry.async_get_or_create('cover', 'super_platform', 'very_unique')
    assert entry.device_class == 'class_by_user'
    assert entry.original_device_class == 'class_by_integration'

@pytest.mark.parametrize('load_registries', [False])
async def test_migration_1_11(hass: HomeAssistant, hass_storage: Dict[str, Any]) -> None:
    """Test migration from version 1.11.

    This is the first version which has deleted entities, make sure deleted entities
    are updated.
    """
    hass_storage[er.STORAGE_KEY] = {'version': 1, 'minor_version': 11, 'data': {'entities': [{'aliases': [], 'area_id': None, 'capabilities': {}, 'config_entry_id': None, 'device_id': None, 'disabled_by': None, 'entity_category': None, 'entity_id': 'test.entity', 'has_entity_name': False, 'hidden_by': None, 'icon': None, 'id': '12345', 'modified_at': '1970-01-01T00:00:00+00:00', 'name': None, 'options': {}, 'original_device_class': 'best_class', 'original_icon': None, 'original_name': None, 'platform': 'super_platform', 'supported_features': 0, 'translation_key': None, 'unique_id': 'very_unique', 'unit_of_measurement': None, 'device_class': None}], 'deleted_entities': [{'config_entry_id': None, 'entity_id': 'test.deleted_entity', 'id': '23456', 'orphaned_timestamp': None, 'platform': 'super_duper_platform', 'unique_id': 'very_very_unique'}]}}
    await er.async_load(hass)
    registry = er.async_get(hass)
    entry = registry.async_get_or_create('test', 'super_platform', 'very_unique')
    assert entry.device_class is None
    assert entry.original_device_class == 'best_class'
    await flush_store(registry._store)
    assert hass_storage[er.STORAGE_KEY] == {'version': er.STORAGE_VERSION_MAJOR, 'minor_version': er.STORAGE_VERSION_MINOR, 'key': er.STORAGE_KEY, 'data': {'entities': [{'aliases': [], 'area_id': None, 'capabilities': {}, 'categories': {}, 'config_entry_id': None, 'config_subentry_id': None, 'created_at': '1970-01-01T00:00:00+00:00', 'device_id': None, 'disabled_by': None, 'entity_category': None, 'entity_id': 'test.entity', 'has_entity_name': False, 'hidden_by': None, 'icon': None, 'id': ANY, 'labels': [], 'modified_at': '1970-01-01T00:00:00+00:00', 'name': None, 'options': {}, 'original_device_class': 'best_class', 'original_icon': None, 'original_name': None, 'platform': 'super_platform', 'previous_unique_id': None, 'supported_features': 0, 'translation_key': None, 'unique_id': 'very_unique', 'unit_of_measurement': None, 'device_class': None}], 'deleted_entities': [{'config_entry_id': None, 'config_subentry_id': None, 'created_at': '1970-01-01T00:00:00+00:00', 'entity_id': 'test.deleted_entity', 'id': '23456', 'modified_at': '1970-01-01T00:00:00+00:00', 'orphaned_timestamp': None, 'platform': 'super_duper_platform', 'unique_id': 'very_very_unique'}]}}

async def test_update_entity_unique_id(hass: HomeAssistant, entity_registry: er.EntityRegistry) -> None:
    """Test entity's unique_id is updated."""
    mock_config = MockConfigEntry(domain='light', entry_id='mock-id-1')
    mock_config.add_to_hass(hass)
    entry = entity_registry.async_get_or_create('light', 'hue', '5678', config_entry=mock_config)
    assert entity_registry.async_get_entity_id('light', 'hue', '5678') == entry.entity_id
    new_unique_id = '1234'
    with patch.object(entity_registry, 'async_schedule_save') as mock_schedule_save:
        updated_entry = entity_registry.async_update_entity(entry.entity_id, new_unique_id=new_unique_id)
    assert updated_entry != entry
    assert updated_entry.unique_id == new_unique_id
    assert updated_entry.previous_unique_id == '5678'
    assert mock_schedule_save.call_count == 1
    assert entity_registry.async_get_entity_id('light', 'hue', '5678') is None
    assert entity_registry.async_get_entity_id('light', 'hue', '1234') == entry.entity_id

async def test_update_entity_unique_id_conflict(hass: HomeAssistant, entity_registry: er.EntityRegistry) -> None:
    """Test migration raises when unique_id already in use."""
    mock_config = MockConfigEntry(domain='light', entry_id='mock-id-1')
    mock_config.add_to_hass(hass)
    entry = entity_registry.async_get_or_create('light', 'hue', '5678', config_entry=mock_config)
    entry2 = entity_registry.async_get_or_create('light', 'hue', '1234', config_entry=mock_config)
    with patch.object(entity_registry, 'async_schedule_save') as mock_schedule_save, pytest.raises(ValueError):
        entity_registry.async_update_entity(entry.entity_id, new_unique_id=entry2.unique_id)
    assert mock_schedule_save.call_count == 0
    assert entity_registry.async_get_entity_id('light', 'hue', '5678') == entry.entity_id
    assert entity_registry.async_get_entity_id('light', 'hue', '1234') == entry2.entity_id

async def test_update_entity_entity_id(entity_registry: er.EntityRegistry) -> None:
    """Test entity's entity_id is updated."""
    entry = entity_registry.async_get_or_create('light', 'hue', '5678')
    assert entity_registry.async_get_entity_id('light', 'hue', '5678') == entry.entity_id
    new_entity_id = 'light.blah'
    assert new_entity_id != entry.entity_id
    with patch.object(entity_registry, 'async_schedule_save') as mock_schedule_save:
        updated_entry = entity_registry.async_update_entity(entry.entity_id, new_entity_id=new_entity_id)
    assert updated_entry != entry
    assert updated_entry.entity_id == new_entity_id
    assert mock_schedule_save.call_count == 1
    assert entity_registry.async_get(entry.entity_id) is None
    assert entity_registry.async_get(new_entity_id) is not None

async def test_update_entity_entity_id_entity_id(hass: HomeAssistant, entity_registry: er.EntityRegistry) -> None:
    """Test update raises when entity_id already in use."""
    entry = entity_registry.async_get_or_create('light', 'hue', '5678')
    entry2 = entity_registry.async_get_or_create('light', 'hue', '1234')
    state_entity_id = 'light.blah'
    hass.states.async_set(state_entity_id, 'on')
    assert entry.entity_id != state_entity_id
    assert entry2.entity_id != state_entity_id
    with patch.object(entity_registry, 'async_schedule_save') as mock_schedule_save, pytest.raises(ValueError):
        entity_registry.async_update_entity(entry.entity_id, new_entity_id=entry2.entity_id)
    assert mock_schedule_save.call_count == 0
    assert entity_registry.async_get_entity_id('light', 'hue', '5678') == entry.entity_id
    assert entity_registry.async_get(entry.entity_id) is entry
    assert entity_registry.async_get_entity_id('light', 'hue', '1234') == entry2.entity_id
    assert entity_registry.async_get(entry2.entity_id) is entry2
    with patch.object(entity_registry, 'async_schedule_save') as mock_schedule_save, pytest.raises(ValueError):
        entity_registry.async_update_entity(entry.entity_id, new_entity_id=state_entity_id)
    assert mock_schedule_save.call_count == 0
    assert entity_registry.async_get_entity_id('light', 'hue', '5678') == entry.entity_id
    assert entity_registry.async_get(entry.entity_id) is entry
    assert entity_registry.async_get(state_entity_id) is None

async def test_update_entity(hass: HomeAssistant, entity_registry: er.EntityRegistry) -> None:
    """Test updating entity."""
    mock_config = MockConfigEntry(domain='light', entry_id='mock-id-1')
    mock_config.add_to_hass(hass)
    entry = entity_registry.async_get_or_create('light', 'hue', '5678', config_entry=mock_config)
    for attr_name, new_value in (('aliases', {'alias_1', 'alias_2'}), ('disabled_by', er.RegistryEntryDisabler.USER), ('icon', 'new icon'), ('name', 'new name')):
        changes = {attr_name: new_value}
        updated_entry = entity_registry.async_update_entity(entry.entity_id, **changes)
        assert updated_entry != entry
        assert getattr(updated_entry, attr_name) == new_value
        assert getattr(updated_entry, attr_name) != getattr(entry, attr_name)
        assert entity_registry.async_get_entity_id('light', 'hue', '5678') == updated_entry.entity_id
        entry = updated_entry

async def test_update_entity_options(hass: HomeAssistant, entity_registry: er.EntityRegistry) -> None:
    """Test updating entity."""
    mock_config = MockConfigEntry(domain='light', entry_id='mock-id-1')
    mock_config.add_to_hass(hass)
    entry = entity_registry.async_get_or_create('light', 'hue', '5678', config_entry=mock_config)
    entity_registry.async_update_entity_options(entry.entity_id, 'light', {'minimum_brightness': 20})
    new_entry_1 = entity_registry.async_get(entry.entity_id)
    assert entry.options == {}
    assert new_entry_1.options == {'light': {'minimum_brightness': 20}}
    with pytest.raises(RuntimeError):
        new_entry_1.options['blah'] = {}
    with pytest.raises(RuntimeError):
        new_entry_1.options['light'] = {}
    with pytest.raises(RuntimeError):
        new_entry_1.options['light']['blah'] = 123
    with pytest.raises(RuntimeError):
        new_entry_1.options['light']['minimum_brightness'] = 123
    entity_registry.async_update_entity_options(entry.entity_id, 'light', {'minimum_brightness': 30})
    new_entry_2 = entity_registry.async_get(entry.entity_id)
    assert entry.options == {}
    assert new_entry_1.options == {'light': {'minimum_brightness': 20}}
    assert new_entry_2.options == {'light': {'minimum_brightness': 30}}

async def test_disabled_by(entity_registry: er.EntityRegistry) -> None:
    """Test that we can disable an entry when we create it."""
    entry = entity_registry.async_get_or_create('light', 'hue', '5678', disabled_by=er.RegistryEntryDisabler.HASS)
    assert entry.disabled_by is er.RegistryEntryDisabler.HASS
    assert entry.disabled is True
    entry = entity_registry.async_get_or_create('light', 'hue', '5678', disabled_by=er.RegistryEntryDisabler.INTEGRATION)
    assert entry.disabled_by is er.RegistryEntryDisabler.HASS
    assert entry.disabled is True
    entry2 = entity_registry.async_get_or_create('light', 'hue', '1234')
    assert entry2.disabled_by is None
    assert entry2.disabled is False

async def test_disabled_by_config_entry_pref(hass: HomeAssistant, entity_registry: er.EntityRegistry) -> None:
    """Test config entry preference setting disabled_by."""
    mock_config = MockConfigEntry(domain='light', entry_id='mock-id-1', pref_disable_new_entities=True)
    mock_config.add_to_hass(hass)
    entry = entity_registry.async_get_or_create('light', 'hue', 'AAAA', config_entry=mock_config)
    assert entry.disabled_by is er.RegistryEntryDisabler.INTEGRATION
    entry2 = entity_registry.async_get_or_create('light', 'hue', 'BBBB', config_entry=mock_config, disabled_by=er.RegistryEntryDisabler.USER)
    assert entry2.disabled_by is er.RegistryEntryDisabler.USER

async def test_hidden_by(entity_registry: er.EntityRegistry) -> None:
    """Test that we can hide an entry when we create it."""
    entry = entity_registry.async_get_or_create('light', 'hue', '5678', hidden_by=er.RegistryEntryHider.USER)
    assert entry.hidden_by is er.RegistryEntryHider.USER
    assert entry.hidden is True
    entry = entity_registry.async_get_or_create('light', 'hue', '5678', disabled_by=er.RegistryEntryHider.INTEGRATION)
    assert entry.hidden_by is er.RegistryEntryHider.USER
    assert entry.hidden is True
    entry2 = entity_registry.async_get_or_create('light', 'hue', '1234')
    assert entry2.hidden_by is None
    assert entry2.hidden is False

async def test_restore_states(hass: HomeAssistant, entity_registry: er.EntityRegistry) -> None:
    """Test restoring states."""
    hass.set_state(CoreState.not_running)
    entity_registry.async_get_or_create('light', 'hue', '1234', suggested_object_id='simple')
    entity_registry.async_get_or_create('light', 'hue', '5678', suggested_object_id='disabled', disabled_by=er.RegistryEntryDisabler.HASS)
    entity_registry.async_get_or_create('light', 'hue', '9012', suggested_object_id='all_info_set', capabilities={'max': 100}, supported_features=5, original_device_class='mock-device-class', original_name='Mock Original Name', original_icon='hass:original-icon')
    hass.bus.async_fire(EVENT_HOMEASSISTANT_START, {})
    await hass.async_block_till_done()
    simple = hass.states.get('light.simple')
    assert simple is not None
    assert simple.state == STATE_UNAVAILABLE
    assert simple.attributes == {'restored': True, 'supported_features': 0}
    disabled = hass.states.get('light.disabled')
    assert disabled is None
    all_info_set = hass.states.get('light.all_info_set')
    assert all_info_set is not None
    assert all_info_set.state == STATE_UNAVAILABLE
    assert all_info_set.attributes == {'max': 100, 'supported_features': 5, 'device_class': 'mock-device-class', 'restored': True, 'friendly_name': 'Mock Original Name', 'icon': 'hass:original-icon'}
    entity_registry.async_remove('light.disabled')
    entity_registry.async_remove('light.simple')
    entity_registry.async_remove('light.all_info_set')
    await hass.async_block_till_done()
    assert hass.states.get('light.simple') is None
    assert hass.states.get('light.disabled') is None
    assert hass.states.get('light.all_info_set') is None

async def test_remove_device_removes_entities(hass: HomeAssistant, entity_registry: er.EntityRegistry, device_registry: dr.DeviceRegistry) -> None:
    """Test that we remove entities tied to a device."""
    config_entry = MockConfigEntry(domain='light')
    config_entry.add_to_hass(hass)
    device_entry = device_registry.async_get_or_create(config_entry_id=config_entry.entry_id, connections={(dr.CONNECTION_NETWORK_MAC, '12:34:56:AB:CD:EF')})
    entry = entity_registry.async_get_or_create('light', 'hue', '5678', config_entry=config_entry, device_id=device_entry.id)
    assert entity_registry.async_is_registered(entry.entity_id)
    device_registry.async_remove_device(device_entry.id)
    await hass.async_block_till_done()
    assert not entity_registry.async_is_registered(entry.entity_id)

async def test_remove_config_entry_from_device_removes_entities(hass: HomeAssistant, device_registry: dr.DeviceRegistry, entity_registry: er.EntityRegistry) -> None:
    """Test that we remove entities tied to a