"""Support for Rflink devices."""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Dict, List, Optional

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

_LOGGER = logging.getLogger(__name__)

EVENT_BUTTON_PRESSED: str = "button_pressed"


class RflinkDevice(Entity):
    """Representation of a Rflink device.

    Contains the common logic for Rflink entities.
    """

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
        """Initialize the device."""
        # Rflink specific attributes for every component type
        self._initial_event: Optional[Dict[str, Any]] = initial_event
        self._device_id: str = device_id
        self._attr_unique_id: str = device_id
        if name:
            self._name: str = name
        else:
            self._name: str = device_id

        self._aliases: Optional[List[str]] = aliases
        self._group: bool = group
        self._group_aliases: Optional[List[str]] = group_aliases
        self._nogroup_aliases: Optional[List[str]] = nogroup_aliases
        self._should_fire_event: bool = fire_event
        self._signal_repetitions: int = signal_repetitions

    @callback
    def handle_event_callback(self, event: Dict[str, Any]) -> None:
        """Handle incoming event for device type."""
        # Call platform specific event handler
        self._handle_event(event)

        # Propagate changes through ha
        self.async_write_ha_state()

        # Put command onto bus for user to subscribe to
        if self._should_fire_event and identify_event_type(event) == EVENT_KEY_COMMAND:
            self.hass.bus.async_fire(
                EVENT_BUTTON_PRESSED,
                {ATTR_ENTITY_ID: self.entity_id, ATTR_STATE: event[EVENT_KEY_COMMAND]},
            )
            _LOGGER.debug(
                "Fired bus event for %s: %s", self.entity_id, event[EVENT_KEY_COMMAND]
            )

    def _handle_event(self, event: Dict[str, Any]) -> None:
        """Platform specific event handler."""
        raise NotImplementedError

    @property
    def name(self) -> str:
        """Return a name for the device."""
        return self._name

    @property
    def is_on(self) -> bool:
        """Return true if device is on."""
        if self.assumed_state:
            return False
        return self._state  # type: ignore

    @property
    def assumed_state(self) -> bool:
        """Assume device state until first device event sets state."""
        return self._state is None

    @property
    def available(self) -> bool:
        """Return True if entity is available."""
        return self._available

    @callback
    def _availability_callback(self, availability: bool) -> None:
        """Update availability state."""
        self._available = availability
        self.async_write_ha_state()

    async def async_added_to_hass(self) -> None:
        """Register update callback."""
        await super().async_added_to_hass()
        # Remove temporary bogus entity_id if added
        tmp_entity: str = TMP_ENTITY.format(self._device_id)
        if (
            tmp_entity
            in self.hass.data[DATA_ENTITY_LOOKUP][EVENT_KEY_COMMAND][self._device_id]
        ):
            self.hass.data[DATA_ENTITY_LOOKUP][EVENT_KEY_COMMAND][
                self._device_id
            ].remove(tmp_entity)

        # Register id and aliases
        self.hass.data[DATA_ENTITY_LOOKUP][EVENT_KEY_COMMAND][self._device_id].append(
            self.entity_id
        )
        if self._group:
            self.hass.data[DATA_ENTITY_GROUP_LOOKUP][EVENT_KEY_COMMAND][
                self._device_id
            ].append(self.entity_id)
        # aliases respond to both normal and group commands (allon/alloff)
        if self._aliases:
            for _id in self._aliases:
                self.hass.data[DATA_ENTITY_LOOKUP][EVENT_KEY_COMMAND][_id].append(
                    self.entity_id
                )
                self.hass.data[DATA_ENTITY_GROUP_LOOKUP][EVENT_KEY_COMMAND][_id].append(
                    self.entity_id
                )
        # group_aliases only respond to group commands (allon/alloff)
        if self._group_aliases:
            for _id in self._group_aliases:
                self.hass.data[DATA_ENTITY_GROUP_LOOKUP][EVENT_KEY_COMMAND][_id].append(
                    self.entity_id
                )
        # nogroup_aliases only respond to normal commands
        if self._nogroup_aliases:
            for _id in self._nogroup_aliases:
                self.hass.data[DATA_ENTITY_LOOKUP][EVENT_KEY_COMMAND][_id].append(
                    self.entity_id
                )
        self.async_on_remove(
            async_dispatcher_connect(
                self.hass, SIGNAL_AVAILABILITY, self._availability_callback
            )
        )
        self.async_on_remove(
            async_dispatcher_connect(
                self.hass,
                SIGNAL_HANDLE_EVENT.format(self.entity_id),
                self.handle_event_callback,
            )
        )

        # Process the initial event now that the entity is created
        if self._initial_event:
            self.handle_event_callback(self._initial_event)


class RflinkCommand(RflinkDevice):
    """Singleton class to make Rflink command interface available to entities.

    This class is to be inherited by every Entity class that is actionable
    (switches/lights). It exposes the Rflink command interface for these
    entities.

    The Rflink interface is managed as a class level and set during setup (and
    reset on reconnect).
    """

    # Keep repetition tasks to cancel if state is changed before repetitions
    # are sent
    _repetition_task: Optional[asyncio.Task[None]] = None

    _protocol: Optional[ProtocolBase] = None

    _wait_ack: Optional[bool] = None

    @classmethod
    def set_rflink_protocol(
        cls, protocol: Optional[ProtocolBase], wait_ack: Optional[bool] = None
    ) -> None:
        """Set the Rflink asyncio protocol as a class variable."""
        cls._protocol = protocol
        if wait_ack is not None:
            cls._wait_ack = wait_ack

    @classmethod
    def is_connected(cls) -> bool:
        """Return connection status."""
        return bool(cls._protocol)

    @classmethod
    async def send_command(cls, device_id: str, action: str) -> Any:
        """Send device command to Rflink and wait for acknowledgement."""
        return await cls._protocol.send_command_ack(device_id, action)

    async def _async_handle_command(self, command: str, *args: Any) -> None:
        """Do bookkeeping for command, send it to rflink and update state."""
        self.cancel_queued_send_commands()

        if command == "turn_on":
            cmd = "on"
            self._state = True

        elif command == "turn_off":
            cmd = "off"
            self._state = False

        elif command == "dim":
            # convert brightness to rflink dim level
            cmd = str(brightness_to_rflink(args[0]))
            self._state = True

        elif command == "toggle":
            cmd = "on"
            # if the state is unknown or false, it gets set as true
            # if the state is true, it gets set as false
            self._state = self._state in [None, False]

        # Cover options for RFlink
        elif command == "close_cover":
            cmd = "DOWN"
            self._state = False

        elif command == "open_cover":
            cmd = "UP"
            self._state = True

        elif command == "stop_cover":
            cmd = "STOP"
            self._state = True

        # Send initial command and queue repetitions.
        # This allows the entity state to be updated quickly and not having to
        # wait for all repetitions to be sent
        await self._async_send_command(cmd, self._signal_repetitions)

        # Update state of entity
        self.async_write_ha_state()

    def cancel_queued_send_commands(self) -> None:
        """Cancel queued signal repetition commands.

        For example when user changed state while repetitions are still
        queued for broadcast. Or when an incoming Rflink command (remote
        switch) changes the state.
        """
        # cancel any outstanding tasks from the previous state change
        if self._repetition_task:
            self._repetition_task.cancel()

    async def _async_send_command(self, cmd: str, repetitions: int) -> None:
        """Send a command for device to Rflink gateway."""
        _LOGGER.debug("Sending command: %s to Rflink device: %s", cmd, self._device_id)

        if not self.is_connected():
            raise HomeAssistantError("Cannot send command, not connected!")

        if self._wait_ack:
            # Puts command on outgoing buffer then waits for Rflink to confirm
            # the command has been sent out.
            await self._protocol.send_command_ack(self._device_id, cmd)
        else:
            # Puts command on outgoing buffer and returns straight away.
            # Rflink protocol/transport handles asynchronous writing of buffer
            # to serial/tcp device. Does not wait for command send
            # confirmation.
            self._protocol.send_command(self._device_id, cmd)

        if repetitions > 1:
            self._repetition_task = self.hass.async_create_task(
                self._async_send_command(cmd, repetitions - 1), eager_start=False
            )


class SwitchableRflinkDevice(RflinkCommand, RestoreEntity):
    """Rflink entity which can switch on/off (eg: light, switch)."""

    async def async_added_to_hass(self) -> None:
        """Restore RFLink device state (ON/OFF)."""
        await super().async_added_to_hass()
        if (old_state := await self.async_get_last_state()) is not None:
            self._state: Optional[bool] = old_state.state == STATE_ON

    def _handle_event(self, event: Dict[str, Any]) -> None:
        """Adjust state if Rflink picks up a remote command for this device."""
        self.cancel_queued_send_commands()

        command: str = event["command"]
        if command in ["on", "allon"]:
            self._state = True
        elif command in ["off", "alloff"]:
            self._state = False

    async def async_turn_on(self, **kwargs: Any) -> None:
        """Turn the device on."""
        await self._async_handle_command("turn_on")

    async def async_turn_off(self, **kwargs: Any) -> None:
        """Turn the device off."""
        await self._async_handle_command("turn_off")
