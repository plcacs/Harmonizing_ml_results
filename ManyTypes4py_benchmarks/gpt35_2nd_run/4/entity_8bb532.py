from __future__ import annotations
from typing import Any, cast
from pyisy.constants import ATTR_ACTION, ATTR_CONTROL, COMMAND_FRIENDLY_NAME, EMPTY_TIME, EVENT_PROPS_IGNORED, NC_NODE_ENABLED, PROTO_INSTEON, PROTO_ZWAVE, TAG_ADDRESS, TAG_ENABLED
from pyisy.helpers import EventListener, NodeProperty
from pyisy.nodes import Group, Node, NodeChangedEvent
from pyisy.programs import Program
from pyisy.variables import Variable
from homeassistant.const import STATE_OFF, STATE_ON
from homeassistant.core import callback
from homeassistant.exceptions import HomeAssistantError
from homeassistant.helpers.device_registry import DeviceInfo
from homeassistant.helpers.entity import Entity, EntityDescription
from .const import DOMAIN

class ISYEntity(Entity):
    _attr_has_entity_name: bool = False
    _attr_should_poll: bool = False

    def __init__(self, node: Node, device_info: DeviceInfo = None) -> None:
        self._node: Node = node
        self._attr_name: str = node.name
        if device_info is None:
            device_info = DeviceInfo(identifiers={(DOMAIN, node.isy.uuid)})
        self._attr_device_info: DeviceInfo = device_info
        self._attr_unique_id: str = f'{node.isy.uuid}_{node.address}'
        self._attrs: dict = {}
        self._change_handler: Any = None
        self._control_handler: Any = None

    async def async_added_to_hass(self) -> None:
        self._change_handler = self._node.status_events.subscribe(self.async_on_update)
        if hasattr(self._node, 'control_events'):
            self._control_handler = self._node.control_events.subscribe(self.async_on_control)

    @callback
    def async_on_update(self, event: NodeChangedEvent) -> None:
        self.async_write_ha_state()

    @callback
    def async_on_control(self, event: NodeChangedEvent) -> None:
        event_data: dict = {'entity_id': self.entity_id, 'control': event.control, 'value': event.value, 'formatted': event.formatted, 'uom': event.uom, 'precision': event.prec}
        if event.control not in EVENT_PROPS_IGNORED:
            self.async_write_ha_state()
        self.hass.bus.async_fire('isy994_control', event_data)

class ISYNodeEntity(ISYEntity):
    def __init__(self, node: Node, device_info: DeviceInfo = None) -> None:
        super().__init__(node, device_info=device_info)
        if hasattr(node, 'parent_node') and node.parent_node is None:
            self._attr_has_entity_name: bool = True
            self._attr_name: str = None

    @property
    def available(self) -> bool:
        return getattr(self._node, TAG_ENABLED, True)

    @property
    def extra_state_attributes(self) -> dict:
        attrs: dict = self._attrs
        node: Node = self._node
        if node.protocol != PROTO_INSTEON and hasattr(node, 'aux_properties'):
            for name, value in self._node.aux_properties.items():
                attr_name: str = COMMAND_FRIENDLY_NAME.get(name, name)
                attrs[attr_name] = str(value.formatted).lower()
        if hasattr(node, 'group_all_on'):
            attrs['group_all_on'] = STATE_ON if node.group_all_on else STATE_OFF
        return self._attrs

    async def async_send_node_command(self, command: str) -> None:
        if not hasattr(self._node, command):
            raise HomeAssistantError(f'Invalid service call: {command} for device {self.entity_id}')
        await getattr(self._node, command)()

    async def async_send_raw_node_command(self, command: str, value: Any = None, unit_of_measurement: Any = None, parameters: Any = None) -> None:
        if not hasattr(self._node, 'send_cmd'):
            raise HomeAssistantError(f'Invalid service call: {command} for device {self.entity_id}')
        await self._node.send_cmd(command, value, unit_of_measurement, parameters)

    async def async_get_zwave_parameter(self, parameter: Any) -> None:
        if self._node.protocol != PROTO_ZWAVE:
            raise HomeAssistantError(f'Invalid service call: cannot request Z-Wave Parameter for non-Z-Wave device {self.entity_id}')
        await self._node.get_zwave_parameter(parameter)

    async def async_set_zwave_parameter(self, parameter: Any, value: Any, size: Any) -> None:
        if self._node.protocol != PROTO_ZWAVE:
            raise HomeAssistantError(f'Invalid service call: cannot set Z-Wave Parameter for non-Z-Wave device {self.entity_id}')
        await self._node.set_zwave_parameter(parameter, value, size)
        await self._node.get_zwave_parameter(parameter)

    async def async_rename_node(self, name: str) -> None:
        await self._node.rename(name)

class ISYProgramEntity(ISYEntity):
    def __init__(self, name: str, status: Any, actions: Any = None) -> None:
        super().__init__(status)
        self._attr_name: str = name
        self._actions: Any = actions

    @property
    def extra_state_attributes(self) -> dict:
        attr: dict = {}
        if self._actions:
            attr['actions_enabled'] = self._actions.enabled
            if self._actions.last_finished != EMPTY_TIME:
                attr['actions_last_finished'] = self._actions.last_finished
            if self._actions.last_run != EMPTY_TIME:
                attr['actions_last_run'] = self._actions.last_run
            if self._actions.last_update != EMPTY_TIME:
                attr['actions_last_update'] = self._actions.last_update
            attr['ran_else'] = self._actions.ran_else
            attr['ran_then'] = self._actions.ran_then
            attr['run_at_startup'] = self._actions.run_at_startup
            attr['running'] = self._actions.running
        attr['status_enabled'] = self._node.enabled
        if self._node.last_finished != EMPTY_TIME:
            attr['status_last_finished'] = self._node.last_finished
        if self._node.last_run != EMPTY_TIME:
            attr['status_last_run'] = self._node.last_run
        if self._node.last_update != EMPTY_TIME:
            attr['status_last_update'] = self._node.last_update
        return attr

class ISYAuxControlEntity(Entity):
    _attr_should_poll: bool = False

    def __init__(self, node: Node, control: str, unique_id: str, description: EntityDescription, device_info: DeviceInfo) -> None:
        self._node: Node = node
        self._control: str = control
        name: str = COMMAND_FRIENDLY_NAME.get(control, control).replace('_', ' ').title()
        if node.address != node.primary_node:
            name = f'{node.name} {name}'
        self._attr_name: str = name
        self.entity_description: EntityDescription = description
        self._attr_has_entity_name: bool = node.address == node.primary_node
        self._attr_unique_id: str = unique_id
        self._attr_device_info: DeviceInfo = device_info
        self._change_handler: Any = None
        self._availability_handler: Any = None

    async def async_added_to_hass(self) -> None:
        self._change_handler = self._node.control_events.subscribe(self.async_on_update, event_filter={ATTR_CONTROL: self._control}, key=self.unique_id)
        self._availability_handler = self._node.isy.nodes.status_events.subscribe(self.async_on_update, event_filter={TAG_ADDRESS: self._node.address, ATTR_ACTION: NC_NODE_ENABLED}, key=self.unique_id)

    @callback
    def async_on_update(self, event: NodeChangedEvent, key: str) -> None:
        self.async_write_ha_state()

    @property
    def available(self) -> bool:
        return cast(bool, self._node.enabled)
