"""Support for the for Danfoss Air HRV sswitches."""
from __future__ import annotations

import logging
from typing import Any, Protocol

from pydanfossair.commands import ReadCommand, UpdateCommand
from homeassistant.components.switch import SwitchEntity
from homeassistant.core import HomeAssistant
from homeassistant.helpers.entity_platform import AddEntitiesCallback
from homeassistant.helpers.typing import ConfigType, DiscoveryInfoType

from . import DOMAIN as DANFOSS_AIR_DOMAIN

_LOGGER = logging.getLogger(__name__)


class DanfossAirDataProtocol(Protocol):
    def update_state(self, update_command: UpdateCommand, state_command: ReadCommand) -> None: ...
    def update(self) -> None: ...
    def get_value(self, state_command: ReadCommand) -> bool | None: ...


def setup_platform(
    hass: HomeAssistant,
    config: ConfigType,
    add_entities: AddEntitiesCallback,
    discovery_info: DiscoveryInfoType | None = None,
) -> None:
    """Set up the Danfoss Air HRV switch platform."""
    data: DanfossAirDataProtocol = hass.data[DANFOSS_AIR_DOMAIN]
    switches: list[tuple[str, ReadCommand, UpdateCommand, UpdateCommand]] = [
        ("Danfoss Air Boost", ReadCommand.boost, UpdateCommand.boost_activate, UpdateCommand.boost_deactivate),
        ("Danfoss Air Bypass", ReadCommand.bypass, UpdateCommand.bypass_activate, UpdateCommand.bypass_deactivate),
        ("Danfoss Air Automatic Bypass", ReadCommand.automatic_bypass, UpdateCommand.bypass_activate, UpdateCommand.bypass_deactivate),
    ]
    add_entities((DanfossAir(data, switch[0], switch[1], switch[2], switch[3]) for switch in switches))


class DanfossAir(SwitchEntity):
    """Representation of a Danfoss Air HRV Switch."""

    def __init__(
        self,
        data: DanfossAirDataProtocol,
        name: str,
        state_command: ReadCommand,
        on_command: UpdateCommand,
        off_command: UpdateCommand,
    ) -> None:
        """Initialize the switch."""
        self._data: DanfossAirDataProtocol = data
        self._name: str = name
        self._state_command: ReadCommand = state_command
        self._on_command: UpdateCommand = on_command
        self._off_command: UpdateCommand = off_command
        self._state: bool | None = None

    @property
    def name(self) -> str:
        """Return the name of the switch."""
        return self._name

    @property
    def is_on(self) -> bool | None:
        """Return true if switch is on."""
        return self._state

    def turn_on(self, **kwargs: Any) -> None:
        """Turn the switch on."""
        _LOGGER.debug("Turning on switch with command %s", self._on_command)
        self._data.update_state(self._on_command, self._state_command)

    def turn_off(self, **kwargs: Any) -> None:
        """Turn the switch off."""
        _LOGGER.debug("Turning off switch with command %s", self._off_command)
        self._data.update_state(self._off_command, self._state_command)

    def update(self) -> None:
        """Update the switch's state."""
        self._data.update()
        self._state = self._data.get_value(self._state_command)
        if self._state is None:
            _LOGGER.debug("Could not get data for %s", self._state_command)