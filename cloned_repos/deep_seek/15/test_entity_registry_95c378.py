"""Tests for the Entity Registry."""
from datetime import datetime, timedelta
from functools import partial
from typing import Any, Dict, List, Optional, Set, Tuple, Union
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

YAML__OPEN_PATH: str = 'homeassistant.util.yaml.loader.open'

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

def test_get_or_create_updates_data(
    hass: HomeAssistant,
    device_registry: dr.DeviceRegistry,
    entity_registry: er.EntityRegistry,
    freezer: FrozenDateTimeFactory
) -> None:
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

async def test_loading_saving_data(
    hass: HomeAssistant,
    device_registry: dr.DeviceRegistry,
    entity_registry: er.EntityRegistry
) -> None:
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
    hass_storage[er.STOR