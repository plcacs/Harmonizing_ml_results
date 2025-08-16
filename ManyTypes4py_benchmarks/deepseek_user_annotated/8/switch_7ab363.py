"""Support for wake on lan."""

from __future__ import annotations

import logging
import subprocess as sp
from typing import Any, Optional

import voluptuous as vol
import wakeonlan

from homeassistant.components.switch import (
    PLATFORM_SCHEMA as SWITCH_PLATFORM_SCHEMA,
    SwitchEntity,
)
from homeassistant.const import (
    CONF_BROADCAST_ADDRESS,
    CONF_BROADCAST_PORT,
    CONF_HOST,
    CONF_MAC,
    CONF_NAME,
)
from homeassistant.core import HomeAssistant
from homeassistant.helpers import config_validation as cv, device_registry as dr
from homeassistant.helpers.entity_platform import AddEntitiesCallback
from homeassistant.helpers.script import Script
from homeassistant.helpers.typing import ConfigType, DiscoveryInfoType

from .const import CONF_OFF_ACTION, DEFAULT_NAME, DEFAULT_PING_TIMEOUT, DOMAIN

_LOGGER: logging.Logger = logging.getLogger(__name__)

PLATFORM_SCHEMA = SWITCH_PLATFORM_SCHEMA.extend(
    {
        vol.Required(CONF_MAC): cv.string,
        vol.Optional(CONF_BROADCAST_ADDRESS): cv.string,
        vol.Optional(CONF_BROADCAST_PORT): cv.port,
        vol.Optional(CONF_HOST): cv.string,
        vol.Optional(CONF_NAME, default=DEFAULT_NAME): cv.string,
        vol.Optional(CONF_OFF_ACTION): cv.SCRIPT_SCHEMA,
    }
)


async def async_setup_platform(
    hass: HomeAssistant,
    config: ConfigType,
    async_add_entities: AddEntitiesCallback,
    discovery_info: DiscoveryInfoType | None = None,
) -> None:
    """Set up a wake on lan switch."""
    broadcast_address: Optional[str] = config.get(CONF_BROADCAST_ADDRESS)
    broadcast_port: Optional[int] = config.get(CONF_BROADCAST_PORT)
    host: Optional[str] = config.get(CONF_HOST)
    mac_address: str = config[CONF_MAC]
    name: str = config[CONF_NAME]
    off_action: Optional[list[Any]] = config.get(CONF_OFF_ACTION)

    async_add_entities(
        [
            WolSwitch(
                hass,
                name,
                host,
                mac_address,
                off_action,
                broadcast_address,
                broadcast_port,
            )
        ],
        host is not None,
    )


class WolSwitch(SwitchEntity):
    """Representation of a wake on lan switch."""

    def __init__(
        self,
        hass: HomeAssistant,
        name: str,
        host: Optional[str],
        mac_address: str,
        off_action: Optional[list[Any]],
        broadcast_address: Optional[str],
        broadcast_port: Optional[int],
    ) -> None:
        """Initialize the WOL switch."""
        self._attr_name: str = name
        self._host: Optional[str] = host
        self._mac_address: str = mac_address
        self._broadcast_address: Optional[str] = broadcast_address
        self._broadcast_port: Optional[int] = broadcast_port
        self._off_script: Optional[Script] = (
            Script(hass, off_action, name, DOMAIN) if off_action else None
        )
        self._state: bool = False
        self._attr_assumed_state: bool = host is None
        self._attr_should_poll: bool = bool(not self._attr_assumed_state)
        self._attr_unique_id: str = dr.format_mac(mac_address)

    @property
    def is_on(self) -> bool:
        """Return true if switch is on."""
        return self._state

    def turn_on(self, **kwargs: Any) -> None:
        """Turn the device on."""
        service_kwargs: dict[str, Any] = {}
        if self._broadcast_address is not None:
            service_kwargs["ip_address"] = self._broadcast_address
        if self._broadcast_port is not None:
            service_kwargs["port"] = self._broadcast_port

        _LOGGER.debug(
            "Send magic packet to mac %s (broadcast: %s, port: %s)",
            self._mac_address,
            self._broadcast_address,
            self._broadcast_port,
        )

        wakeonlan.send_magic_packet(self._mac_address, **service_kwargs)

        if self._attr_assumed_state:
            self._state = True
            self.schedule_update_ha_state()

    def turn_off(self, **kwargs: Any) -> None:
        """Turn the device off if an off action is present."""
        if self._off_script is not None:
            self._off_script.run(context=self._context)

        if self._attr_assumed_state:
            self._state = False
            self.schedule_update_ha_state()

    def update(self) -> None:
        """Check if device is on and update the state. Only called if assumed state is false."""
        ping_cmd: list[str] = [
            "ping",
            "-c",
            "1",
            "-W",
            str(DEFAULT_PING_TIMEOUT),
            str(self._host),
        ]

        status: int = sp.call(ping_cmd, stdout=sp.DEVNULL, stderr=sp.DEVNULL)
        self._state = not bool(status)
