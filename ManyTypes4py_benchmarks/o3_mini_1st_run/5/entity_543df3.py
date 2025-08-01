from __future__ import annotations
import asyncio
import logging
from typing import Any, Optional, List, Dict, Union
from rflink.protocol import ProtocolBase
from homeassistant.const import ATTR_ENTITY_ID, ATTR_STATE, STATE_ON
from homeassistant.core import callback
from homeassistant.exceptions import HomeAssistantError
from homeassistant.helpers.dispatcher import async_dispatcher_connect
from homeassistant.helpers.entity import Entity
from homeassistant.helpers.restore_state import RestoreEntity
from .const import (
    DATA_ENTITY_GROUP_LOOKUP,
    DATA_ENTITY_LOOKUP,
    DEFAULT_SIGNAL_REPETITIONS,
    EVENT_KEY_COMMAND,
    SIGNAL_AVAILABILITY,
    SIGNAL_HANDLE_EVENT,
    TMP_ENTITY,
)
from .utils import brightness_to_rflink, identify_event_type

_LOGGER: logging.Logger = logging.getLogger(__name__)

EVENT_BUTTON_PRESSED: str = 'button_pressed'


class RflinkDevice(Entity):
    _state: Optional[bool] = None
    _available: bool = True
    _attr_should_poll: bool = False

    def __init__(
        self,
        device_id: str,
        initial_event: Optional[Dict[str, Any]] = None,
        name: Optional[str] = None,
        aliases: Optional[List[str]] = None,
        group: bool = True,
        group_aliases: Optional[List[str]] = None,
        nogroup_aliases: Optional[List[str]] = None,
        fire_event: bool = False,
        signal_repetitions: int = DEFAULT_SIGNAL_REPETITIONS,
    ) -> None:
        self._initial_event: Optional[Dict[str, Any]] = initial_event
        self._device_id: str = device_id
        self._attr_unique_id: str = device_id
        if name:
            self._name: str = name
        else:
            self._name = device_id
        self._aliases: Optional[List[str]] = aliases
        self._group: bool = group
        self._group_aliases: Optional[List[str]] = group_aliases
        self._nogroup_aliases: Optional[List[str]] = nogroup_aliases
        self._should_fire_event: bool = fire_event
        self._signal_repetitions: int = signal_repetitions

    @callback
    def handle_event_callback(self, event: Dict[str, Any]) -> None:
        self._handle_event(event)
        self.async_write_ha_state()
        if self._should_fire_event and identify_event_type(event) == EVENT_KEY_COMMAND:
            self.hass.bus.async_fire(
                EVENT_BUTTON_PRESSED,
                {ATTR_ENTITY_ID: self.entity_id, ATTR_STATE: event[EVENT_KEY_COMMAND]},
            )
            _LOGGER.debug('Fired bus event for %s: %s', self.entity_id, event[EVENT_KEY_COMMAND])

    def _handle_event(self, event: Dict[str, Any]) -> None:
        raise NotImplementedError

    @property
    def name(self) -> str:
        return self._name

    @property
    def is_on(self) -> bool:
        if self.assumed_state:
            return False
        return self._state is True

    @property
    def assumed_state(self) -> bool:
        return self._state is None

    @property
    def available(self) -> bool:
        return self._available

    @callback
    def _availability_callback(self, availability: bool) -> None:
        self._available = availability
        self.async_write_ha_state()

    async def async_added_to_hass(self) -> None:
        await super().async_added_to_hass()
        tmp_entity: str = TMP_ENTITY.format(self._device_id)
        if tmp_entity in self.hass.data[DATA_ENTITY_LOOKUP][EVENT_KEY_COMMAND][self._device_id]:
            self.hass.data[DATA_ENTITY_LOOKUP][EVENT_KEY_COMMAND][self._device_id].remove(tmp_entity)
        self.hass.data[DATA_ENTITY_LOOKUP][EVENT_KEY_COMMAND][self._device_id].append(self.entity_id)
        if self._group:
            self.hass.data[DATA_ENTITY_GROUP_LOOKUP][EVENT_KEY_COMMAND][self._device_id].append(self.entity_id)
        if self._aliases:
            for _id in self._aliases:
                self.hass.data[DATA_ENTITY_LOOKUP][EVENT_KEY_COMMAND][_id].append(self.entity_id)
                self.hass.data[DATA_ENTITY_GROUP_LOOKUP][EVENT_KEY_COMMAND][_id].append(self.entity_id)
        if self._group_aliases:
            for _id in self._group_aliases:
                self.hass.data[DATA_ENTITY_GROUP_LOOKUP][EVENT_KEY_COMMAND][_id].append(self.entity_id)
        if self._nogroup_aliases:
            for _id in self._nogroup_aliases:
                self.hass.data[DATA_ENTITY_LOOKUP][EVENT_KEY_COMMAND][_id].append(self.entity_id)
        self.async_on_remove(
            async_dispatcher_connect(self.hass, SIGNAL_AVAILABILITY, self._availability_callback)
        )
        self.async_on_remove(
            async_dispatcher_connect(
                self.hass, SIGNAL_HANDLE_EVENT.format(self.entity_id), self.handle_event_callback
            )
        )
        if self._initial_event:
            self.handle_event_callback(self._initial_event)


class RflinkCommand(RflinkDevice):
    _repetition_task: Optional[asyncio.Task[Any]] = None
    _protocol: Optional[ProtocolBase] = None
    _wait_ack: Optional[bool] = None

    @classmethod
    def set_rflink_protocol(cls, protocol: ProtocolBase, wait_ack: Optional[bool] = None) -> None:
        cls._protocol = protocol
        if wait_ack is not None:
            cls._wait_ack = wait_ack

    @classmethod
    def is_connected(cls) -> bool:
        return bool(cls._protocol)

    @classmethod
    async def send_command(cls, device_id: str, action: str) -> Any:
        assert cls._protocol is not None
        return await cls._protocol.send_command_ack(device_id, action)

    async def _async_handle_command(self, command: str, *args: Any) -> None:
        self.cancel_queued_send_commands()
        if command == 'turn_on':
            cmd: str = 'on'
            self._state = True
        elif command == 'turn_off':
            cmd = 'off'
            self._state = False
        elif command == 'dim':
            cmd = str(brightness_to_rflink(args[0]))
            self._state = True
        elif command == 'toggle':
            cmd = 'on'
            self._state = self._state in [None, False]
        elif command == 'close_cover':
            cmd = 'DOWN'
            self._state = False
        elif command == 'open_cover':
            cmd = 'UP'
            self._state = True
        elif command == 'stop_cover':
            cmd = 'STOP'
            self._state = True
        else:
            raise ValueError(f"Unknown command: {command}")
        await self._async_send_command(cmd, self._signal_repetitions)
        self.async_write_ha_state()

    def cancel_queued_send_commands(self) -> None:
        if self._repetition_task:
            self._repetition_task.cancel()

    async def _async_send_command(self, cmd: str, repetitions: int) -> None:
        _LOGGER.debug('Sending command: %s to Rflink device: %s', cmd, self._device_id)
        if not self.is_connected():
            raise HomeAssistantError('Cannot send command, not connected!')
        assert self._protocol is not None
        if self._wait_ack:
            await self._protocol.send_command_ack(self._device_id, cmd)
        else:
            self._protocol.send_command(self._device_id, cmd)
        if repetitions > 1:
            self._repetition_task = self.hass.async_create_task(
                self._async_send_command(cmd, repetitions - 1),  # type: ignore
                eager_start=False,
            )


class SwitchableRflinkDevice(RflinkCommand, RestoreEntity):
    async def async_added_to_hass(self) -> None:
        await super().async_added_to_hass()
        old_state = await self.async_get_last_state()
        if old_state is not None:
            self._state = old_state.state == STATE_ON

    def _handle_event(self, event: Dict[str, Any]) -> None:
        self.cancel_queued_send_commands()
        command: Union[str, Any] = event['command']
        if command in ['on', 'allon']:
            self._state = True
        elif command in ['off', 'alloff']:
            self._state = False

    async def async_turn_on(self, **kwargs: Any) -> None:
        await self._async_handle_command('turn_on')

    async def async_turn_off(self, **kwargs: Any) -> None:
        await self._async_handle_command('turn_off')