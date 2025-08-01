"""Functions used to migrate unique IDs for Z-Wave JS entities."""
from __future__ import annotations
from dataclasses import dataclass
import logging
from typing import Optional, Set, List, Mapping
from zwave_js_server.model.driver import Driver
from zwave_js_server.model.node import Node
from zwave_js_server.model.value import Value as ZwaveValue
from homeassistant.const import STATE_UNAVAILABLE, Platform
from homeassistant.core import HomeAssistant, callback
from homeassistant.helpers import device_registry as dr, entity_registry as er
from .const import DOMAIN
from .discovery import ZwaveDiscoveryInfo
from .helpers import get_unique_id, get_valueless_base_unique_id

_LOGGER = logging.getLogger(__name__)


@dataclass
class ValueID:
    command_class: str
    endpoint: str
    property_: str
    property_key: Optional[str] = None

    @staticmethod
    def from_unique_id(unique_id: str) -> ValueID:
        """Get a ValueID from a unique ID.

        This also works for Notification CC Binary Sensors which have their
        own unique ID format.
        """
        return ValueID.from_string_id(unique_id.split('.')[1])

    @staticmethod
    def from_string_id(value_id_str: str) -> ValueID:
        """Get a ValueID from a string representation of the value ID."""
        parts = value_id_str.split('-')
        property_key = parts[4] if len(parts) > 4 else None
        return ValueID(parts[1], parts[2], parts[3], property_key=property_key)

    def is_same_value_different_endpoints(self, other: ValueID) -> bool:
        """Return whether two value IDs are the same excluding endpoint."""
        return (
            self.command_class == other.command_class
            and self.property_ == other.property_
            and (self.property_key == other.property_key)
            and (self.endpoint != other.endpoint)
        )


@callback
def async_migrate_old_entity(
    hass: HomeAssistant,
    ent_reg: er.EntityRegistry,
    registered_unique_ids: Set[str],
    platform: Platform,
    device: dr.DeviceEntry,
    unique_id: str,
) -> None:
    """Migrate existing entity if current one can't be found and an old one exists."""
    if ent_reg.async_get_entity_id(platform, DOMAIN, unique_id):
        return
    value_id: ValueID = ValueID.from_unique_id(unique_id)
    existing_entity_entries = []
    for entry in er.async_entries_for_device(ent_reg, device.id):
        if entry.domain != platform or entry.unique_id in registered_unique_ids:
            continue
        try:
            old_ent_value_id = ValueID.from_unique_id(entry.unique_id)
        except IndexError:
            continue
        if value_id.is_same_value_different_endpoints(old_ent_value_id):
            existing_entity_entries.append(entry)
            if len(existing_entity_entries) > 1:
                return
    if not existing_entity_entries:
        return
    entry = existing_entity_entries[0]
    state = hass.states.get(entry.entity_id)
    if not state or state.state == STATE_UNAVAILABLE:
        async_migrate_unique_id(ent_reg, platform, entry.unique_id, unique_id)


@callback
def async_migrate_unique_id(
    ent_reg: er.EntityRegistry,
    platform: Platform,
    old_unique_id: str,
    new_unique_id: str,
) -> None:
    """Check if entity with old unique ID exists, and if so migrate it to new ID."""
    if not (entity_id := ent_reg.async_get_entity_id(platform, DOMAIN, old_unique_id)):
        return
    _LOGGER.debug(
        "Migrating entity %s from old unique ID '%s' to new unique ID '%s'",
        entity_id,
        old_unique_id,
        new_unique_id,
    )
    try:
        ent_reg.async_update_entity(entity_id, new_unique_id=new_unique_id)
    except ValueError:
        _LOGGER.debug(
            "Entity %s can't be migrated because the unique ID is taken; Cleaning it up since it is likely no longer valid",
            entity_id,
        )
        ent_reg.async_remove(entity_id)


@callback
def async_migrate_discovered_value(
    hass: HomeAssistant,
    ent_reg: er.EntityRegistry,
    registered_unique_ids: Set[str],
    device: dr.DeviceEntry,
    driver: Driver,
    disc_info: ZwaveDiscoveryInfo,
) -> None:
    """Migrate unique ID for entity/entities tied to discovered value."""
    new_unique_id = get_unique_id(driver, disc_info.primary_value.value_id)
    if new_unique_id in registered_unique_ids:
        return
    old_unique_ids = [
        get_unique_id(driver, value_id) for value_id in get_old_value_ids(disc_info.primary_value)
    ]
    if disc_info.platform == Platform.BINARY_SENSOR and disc_info.platform_hint == 'notification':
        for state_key in disc_info.primary_value.metadata.states:
            if state_key == '0':
                continue
            new_bin_sensor_unique_id = f'{new_unique_id}.{state_key}'
            if new_bin_sensor_unique_id in registered_unique_ids:
                continue
            for old_unique_id in old_unique_ids:
                async_migrate_unique_id(
                    ent_reg,
                    disc_info.platform,
                    f'{old_unique_id}.{state_key}',
                    new_bin_sensor_unique_id,
                )
            async_migrate_old_entity(hass, ent_reg, registered_unique_ids, disc_info.platform, device, new_bin_sensor_unique_id)
            registered_unique_ids.add(new_bin_sensor_unique_id)
        return
    for old_unique_id in old_unique_ids:
        async_migrate_unique_id(ent_reg, disc_info.platform, old_unique_id, new_unique_id)
    async_migrate_old_entity(hass, ent_reg, registered_unique_ids, disc_info.platform, device, new_unique_id)
    registered_unique_ids.add(new_unique_id)


@callback
def async_migrate_statistics_sensors(
    hass: HomeAssistant,
    driver: Driver,
    node: Node,
    key_map: Mapping[str, str],
) -> None:
    """Migrate statistics sensors to new unique IDs.

    - Migrate camel case keys in unique IDs to snake keys.
    """
    ent_reg = er.async_get(hass)
    base_unique_id = f'{get_valueless_base_unique_id(driver, node)}.statistics'
    for new_key, old_key in key_map.items():
        if new_key == old_key:
            continue
        old_unique_id = f'{base_unique_id}_{old_key}'
        new_unique_id = f'{base_unique_id}_{new_key}'
        async_migrate_unique_id(ent_reg, Platform.SENSOR, old_unique_id, new_unique_id)


@callback
def get_old_value_ids(value: ZwaveValue) -> List[str]:
    """Get old value IDs so we can migrate entity unique ID."""
    value_ids: List[str] = []
    command_class = value.command_class
    endpoint = value.endpoint or '00'
    property_ = value.property_
    property_key_name = value.property_key_name or '00'
    value_ids.append(f'{value.node.node_id}.{value.node.node_id}-{command_class}-{endpoint}-{property_}-{property_key_name}')
    endpoint = '00' if value.endpoint is None else value.endpoint
    property_key = '00' if value.property_key is None else value.property_key
    property_key_name = value.property_key_name or '00'
    value_id = f'{value.node.node_id}-{command_class}-{endpoint}-{property_}-{property_key}-{property_key_name}'
    value_ids.extend([f'{value.node.node_id}.{value_id}', value_id])
    return value_ids