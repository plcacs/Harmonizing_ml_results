"""Support for ISY binary sensors."""
from __future__ import annotations
from datetime import datetime, timedelta
from typing import Any, Optional, Dict, List, Tuple, Set, cast
from pyisy.constants import CMD_OFF, CMD_ON, ISY_VALUE_UNKNOWN, PROTO_INSTEON, PROTO_ZWAVE
from pyisy.helpers import NodeProperty
from pyisy.nodes import Group, Node
from homeassistant.components.binary_sensor import BinarySensorDeviceClass, BinarySensorEntity
from homeassistant.config_entries import ConfigEntry
from homeassistant.const import STATE_ON, Platform
from homeassistant.core import CALLBACK_TYPE, HomeAssistant, callback
from homeassistant.helpers.device_registry import DeviceInfo
from homeassistant.helpers.entity_platform import AddConfigEntryEntitiesCallback
from homeassistant.helpers.event import async_call_later
from homeassistant.helpers.restore_state import RestoreEntity
from .const import _LOGGER, BINARY_SENSOR_DEVICE_TYPES_ISY, BINARY_SENSOR_DEVICE_TYPES_ZWAVE, DOMAIN, SUBNODE_CLIMATE_COOL, SUBNODE_CLIMATE_HEAT, SUBNODE_DUSK_DAWN, SUBNODE_HEARTBEAT, SUBNODE_LOW_BATTERY, SUBNODE_MOTION_DISABLED, SUBNODE_NEGATIVE, SUBNODE_TAMPER, TYPE_CATEGORY_CLIMATE, TYPE_INSTEON_MOTION
from .entity import ISYNodeEntity, ISYProgramEntity
from .models import IsyData

DEVICE_PARENT_REQUIRED: List[BinarySensorDeviceClass] = [BinarySensorDeviceClass.OPENING, BinarySensorDeviceClass.MOISTURE, BinarySensorDeviceClass.MOTION]

async def async_setup_entry(
    hass: HomeAssistant,
    entry: ConfigEntry,
    async_add_entities: AddConfigEntryEntitiesCallback,
) -> None:
    """Set up the ISY binary sensor platform."""
    entities: List[BinarySensorEntity] = []
    entities_by_address: Dict[str, ISYInsteonBinarySensorEntity] = {}
    child_nodes: List[Tuple[Node, Optional[BinarySensorDeviceClass], Optional[str], Optional[DeviceInfo]] = []
    isy_data: IsyData = hass.data[DOMAIN][entry.entry_id]
    devices: Dict[str, DeviceInfo] = isy_data.devices
    for node in isy_data.nodes[Platform.BINARY_SENSOR]:
        assert isinstance(node, Node)
        device_info = devices.get(node.primary_node)
        device_class, device_type = _detect_device_type_and_class(node)
        if node.protocol == PROTO_INSTEON:
            if node.parent_node is not None:
                child_nodes.append((node, device_class, device_type, device_info))
                continue
            entity = ISYInsteonBinarySensorEntity(node, device_class, device_info=device_info)
        else:
            entity = ISYBinarySensorEntity(node, device_class, device_info=device_info)
        entities.append(entity)
        if isinstance(entity, ISYInsteonBinarySensorEntity):
            entities_by_address[node.address] = entity
    for node, device_class, device_type, device_info in child_nodes:
        subnode_id = int(node.address.split(' ')[-1], 16)
        if device_type is not None and device_type.startswith(TYPE_CATEGORY_CLIMATE):
            if subnode_id == SUBNODE_CLIMATE_COOL:
                entity = ISYInsteonBinarySensorEntity(node, BinarySensorDeviceClass.COLD, False, device_info=device_info)
                entities.append(entity)
            elif subnode_id == SUBNODE_CLIMATE_HEAT:
                entity = ISYInsteonBinarySensorEntity(node, BinarySensorDeviceClass.HEAT, False, device_info=device_info)
                entities.append(entity)
            continue
        if device_class in DEVICE_PARENT_REQUIRED:
            parent_entity = entities_by_address.get(node.parent_node.address)
            if not parent_entity:
                _LOGGER.error('Node %s has a parent node %s, but no device was created for the parent. Skipping', node.address, node.parent_node)
                continue
        if device_class in (BinarySensorDeviceClass.OPENING, BinarySensorDeviceClass.MOISTURE):
            if subnode_id == SUBNODE_NEGATIVE:
                assert isinstance(parent_entity, ISYInsteonBinarySensorEntity)
                parent_entity.add_negative_node(node)
            elif subnode_id == SUBNODE_HEARTBEAT:
                assert isinstance(parent_entity, ISYInsteonBinarySensorEntity)
                entity = ISYBinarySensorHeartbeat(node, parent_entity, device_info=device_info)
                parent_entity.add_heartbeat_device(entity)
                entities.append(entity)
            continue
        if device_class == BinarySensorDeviceClass.MOTION and device_type is not None and any((device_type.startswith(t) for t in TYPE_INSTEON_MOTION)):
            assert isinstance(parent_entity, ISYInsteonBinarySensorEntity)
            initial_state = None if parent_entity.state is None else False
            if subnode_id == SUBNODE_DUSK_DAWN:
                entity = ISYInsteonBinarySensorEntity(node, BinarySensorDeviceClass.LIGHT, device_info=device_info)
                entities.append(entity)
                continue
            if subnode_id == SUBNODE_LOW_BATTERY:
                entity = ISYInsteonBinarySensorEntity(node, BinarySensorDeviceClass.BATTERY, initial_state, device_info=device_info)
                entities.append(entity)
                continue
            if subnode_id in SUBNODE_TAMPER:
                entity = ISYInsteonBinarySensorEntity(node, BinarySensorDeviceClass.PROBLEM, initial_state, device_info=device_info)
                entities.append(entity)
                continue
            if subnode_id in SUBNODE_MOTION_DISABLED:
                entity = ISYInsteonBinarySensorEntity(node, device_info=device_info)
                entities.append(entity)
                continue
        entity = ISYBinarySensorEntity(node, force_device_class=device_class, device_info=device_info)
        entities.append(entity)
    for name, status, _ in isy_data.programs[Platform.BINARY_SENSOR]:
        entities.append(ISYBinarySensorProgramEntity(name, status))
    async_add_entities(entities)

def _detect_device_type_and_class(node: Node) -> Tuple[Optional[BinarySensorDeviceClass], Optional[str]]:
    try:
        device_type = node.type
    except AttributeError:
        return (None, None)
    if node.protocol == PROTO_ZWAVE:
        device_type = f'Z{node.zwave_props.category}'
        for device_class, values in BINARY_SENSOR_DEVICE_TYPES_ZWAVE.items():
            if node.zwave_props.category in values:
                return (device_class, device_type)
        return (None, device_type)
    for device_class, values in BINARY_SENSOR_DEVICE_TYPES_ISY.items():
        if any((device_type.startswith(t) for t in values)):
            return (device_class, device_type)
    return (None, device_type)

class ISYBinarySensorEntity(ISYNodeEntity, BinarySensorEntity):
    """Representation of a basic ISY binary sensor device."""

    def __init__(
        self,
        node: Node,
        force_device_class: Optional[BinarySensorDeviceClass] = None,
        unknown_state: Optional[bool] = None,
        device_info: Optional[DeviceInfo] = None,
    ) -> None:
        """Initialize the ISY binary sensor device."""
        super().__init__(node, device_info=device_info)
        self._attr_device_class = force_device_class

    @property
    def is_on(self) -> Optional[bool]:
        """Get whether the ISY binary sensor device is on."""
        if self._node.status == ISY_VALUE_UNKNOWN:
            return None
        return bool(self._node.status)

class ISYInsteonBinarySensorEntity(ISYBinarySensorEntity):
    """Representation of an ISY Insteon binary sensor device."""

    def __init__(
        self,
        node: Node,
        force_device_class: Optional[BinarySensorDeviceClass] = None,
        unknown_state: Optional[bool] = None,
        device_info: Optional[DeviceInfo] = None,
    ) -> None:
        """Initialize the ISY binary sensor device."""
        super().__init__(node, force_device_class, device_info=device_info)
        self._negative_node: Optional[Node] = None
        self._heartbeat_device: Optional[ISYBinarySensorHeartbeat] = None
        self._computed_state: Optional[bool] = None
        self._status_was_unknown: bool = False
        if self._node.status == ISY_VALUE_UNKNOWN:
            self._computed_state = unknown_state
            self._status_was_unknown = True
        else:
            self._computed_state = bool(self._node.status)
            self._status_was_unknown = False

    async def async_added_to_hass(self) -> None:
        """Subscribe to the node and subnode event emitters."""
        await super().async_added_to_hass()
        self._node.control_events.subscribe(self._async_positive_node_control_handler)
        if self._negative_node is not None:
            self._negative_node.control_events.subscribe(self._async_negative_node_control_handler)

    def add_heartbeat_device(self, entity: ISYBinarySensorHeartbeat) -> None:
        """Register a heartbeat device for this sensor."""
        self._heartbeat_device = entity

    def _async_heartbeat(self) -> None:
        """Send a heartbeat to our heartbeat device, if we have one."""
        if self._heartbeat_device is not None:
            self._heartbeat_device.async_heartbeat()

    def add_negative_node(self, child: Node) -> None:
        """Add a negative node to this binary sensor device."""
        self._negative_node = child
        if self._negative_node.status != ISY_VALUE_UNKNOWN and self._negative_node.status == self._node.status:
            self._computed_state = None

    @callback
    def _async_negative_node_control_handler(self, event: Any) -> None:
        """Handle an "On" control event from the "negative" node."""
        if event.control == CMD_ON:
            _LOGGER.debug('Sensor %s turning Off via the Negative node sending a DON command', self.name)
            self._computed_state = False
            self.async_write_ha_state()
            self._async_heartbeat()

    @callback
    def _async_positive_node_control_handler(self, event: Any) -> None:
        """Handle On and Off control event coming from the primary node."""
        if event.control == CMD_ON:
            _LOGGER.debug('Sensor %s turning On via the Primary node sending a DON command', self.name)
            self._computed_state = True
            self.async_write_ha_state()
            self._async_heartbeat()
        if event.control == CMD_OFF:
            _LOGGER.debug('Sensor %s turning Off via the Primary node sending a DOF command', self.name)
            self._computed_state = False
            self.async_write_ha_state()
            self._async_heartbeat()

    @callback
    def async_on_update(self, event: Any) -> None:
        """Primary node status updates."""
        if self._status_was_unknown and self._computed_state is None:
            self._computed_state = bool(self._node.status)
            self._status_was_unknown = False
            self.async_write_ha_state()
            self._async_heartbeat()

    @property
    def is_on(self) -> Optional[bool]:
        """Get whether the ISY binary sensor device is on."""
        if self._computed_state is None:
            return None
        if self.device_class in (BinarySensorDeviceClass.LIGHT, BinarySensorDeviceClass.MOISTURE):
            return not self._computed_state
        return self._computed_state

class ISYBinarySensorHeartbeat(ISYNodeEntity, BinarySensorEntity, RestoreEntity):
    """Representation of the battery state of an ISY sensor."""
    _attr_device_class = BinarySensorDeviceClass.BATTERY

    def __init__(
        self,
        node: Node,
        parent_device: ISYInsteonBinarySensorEntity,
        device_info: Optional[DeviceInfo] = None,
    ) -> None:
        """Initialize the ISY binary sensor device."""
        super().__init__(node, device_info=device_info)
        self._parent_device: ISYInsteonBinarySensorEntity = parent_device
        self._heartbeat_timer: Optional[CALLBACK_TYPE] = None
        self._computed_state: Optional[bool] = None
        if self.state is None:
            self._computed_state = False

    async def async_added_to_hass(self) -> None:
        """Subscribe to the node and subnode event emitters."""
        await super().async_added_to_hass()
        self._node.control_events.subscribe(self._heartbeat_node_control_handler)
        self._restart_timer()
        if (last_state := (await self.async_get_last_state())) is not None:
            if last_state.state == STATE_ON:
                self._computed_state = True

    def _heartbeat_node_control_handler(self, event: Any) -> None:
        """Update the heartbeat timestamp when any ON/OFF event is sent."""
        if event.control in (CMD_ON, CMD_OFF):
            self.async_heartbeat()

    @callback
    def async_heartbeat(self) -> None:
        """Mark the device as online, and restart the 25 hour timer."""
        self._computed_state = False
        self._restart_timer()
        self.async_write_ha_state()

    def _restart_timer(self) -> None:
        """Restart the 25 hour timer."""
        if self._heartbeat_timer is not None:
            self._heartbeat_timer()
            self._heartbeat_timer = None

        @callback
        def timer_elapsed(now: datetime) -> None:
            """Heartbeat missed; set state to ON to indicate dead battery."""
            self._computed_state = True
            self._heartbeat_timer = None
            self.async_write_ha_state()
        self._heartbeat_timer = async_call_later(self.hass, timedelta(hours=25), timer_elapsed)

    @callback
    def async_on_update(self, event: Any) -> None:
        """Ignore node status updates."""

    @property
    def is_on(self) -> bool:
        """Get whether the ISY binary sensor device is on."""
        return bool(self._computed_state)

    @property
    def extra_state_attributes(self) -> Dict[str, Any]:
        """Get the state attributes for the device."""
        attr = super().extra_state_attributes
        attr['parent_entity_id'] = self._parent_device.entity_id
        return attr

class ISYBinarySensorProgramEntity(ISYProgramEntity, BinarySensorEntity):
    """Representation of an ISY binary sensor program."""

    @property
    def is_on(self) -> bool:
        """Get whether the ISY binary sensor device is on."""
        return bool(self._node.status)
