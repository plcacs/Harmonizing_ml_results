from __future__ import annotations
import asyncio
import logging
from rflink.protocol import ProtocolBase
from homeassistant.const import ATTR_ENTITY_ID, ATTR_STATE, STATE_ON
from homeassistant.core import callback
from homeassistant.exceptions import HomeAssistantError
from homeassistant.helpers.dispatcher import async_dispatcher_connect
from homeassistant.helpers.entity import Entity
from homeassistant.helpers.restore_state import RestoreEntity
from .const import DATA_ENTITY_GROUP_LOOKUP, DATA_ENTITY_LOOKUP, DEFAULT_SIGNAL_REPETITIONS, EVENT_KEY_COMMAND, SIGNAL_AVAILABILITY, SIGNAL_HANDLE_EVENT, TMP_ENTITY
from .utils import brightness_to_rflink, identify_event_type

_LOGGER: logging.Logger = logging.getLogger(__name__)

EVENT_BUTTON_PRESSED: str = 'button_pressed'

class RflinkDevice(Entity):
    _state: bool | None = None
    _available: bool = True
    _attr_should_poll: bool = False

    def __init__(self, device_id: str, initial_event: dict | None = None, name: str | None = None, aliases: list[str] | None = None, group: bool = True, group_aliases: list[str] | None = None, nogroup_aliases: list[str] | None = None, fire_event: bool = False, signal_repetitions: int = DEFAULT_SIGNAL_REPETITIONS) -> None:
        ...

    @callback
    def handle_event_callback(self, event: dict) -> None:
        ...

    def _handle_event(self, event: dict) -> None:
        ...

    @property
    def name(self) -> str:
        ...

    @property
    def is_on(self) -> bool:
        ...

    @property
    def assumed_state(self) -> bool:
        ...

    @property
    def available(self) -> bool:
        ...

    @callback
    def _availability_callback(self, availability: bool) -> None:
        ...

    async def async_added_to_hass(self) -> None:
        ...

class RflinkCommand(RflinkDevice):
    _repetition_task: asyncio.Task | None = None
    _protocol: ProtocolBase | None = None
    _wait_ack: bool | None = None

    @classmethod
    def set_rflink_protocol(cls, protocol: ProtocolBase, wait_ack: bool | None) -> None:
        ...

    @classmethod
    def is_connected(cls) -> bool:
        ...

    @classmethod
    async def send_command(cls, device_id: str, action: str) -> None:
        ...

    async def _async_handle_command(self, command: str, *args) -> None:
        ...

    def cancel_queued_send_commands(self) -> None:
        ...

    async def _async_send_command(self, cmd: str, repetitions: int) -> None:
        ...

class SwitchableRflinkDevice(RflinkCommand, RestoreEntity):
    async def async_added_to_hass(self) -> None:
        ...

    def _handle_event(self, event: dict) -> None:
        ...

    async def async_turn_on(self, **kwargs) -> None:
        ...

    async def async_turn_off(self, **kwargs) -> None:
        ...
