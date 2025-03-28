"""Support for Xiaomi Yeelight WiFi color bulb."""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from yeelight import BulbException
from yeelight.aio import KEY_CONNECTED, AsyncBulb

from homeassistant.const import CONF_ID, CONF_NAME
from homeassistant.core import HomeAssistant, callback
from homeassistant.helpers.dispatcher import async_dispatcher_send
from homeassistant.helpers.event import async_call_later

from .const import (
    ACTIVE_COLOR_FLOWING,
    ACTIVE_MODE_NIGHTLIGHT,
    DATA_UPDATED,
    STATE_CHANGE_TIME,
    UPDATE_REQUEST_PROPERTIES,
)
from .scanner import YeelightScanner

_LOGGER = logging.getLogger(__name__)


@callback
def async_format_model(model: str) -> str:
    """Generate a more human readable model."""
    return model.replace("_", " ").title()


@callback
def async_format_id(id_: Optional[str]) -> str:
    """Generate a more human readable id."""
    return hex(int(id_, 16)) if id_ else "None"


@callback
def async_format_model_id(model: str, id_: Optional[str]) -> str:
    """Generate a more human readable name."""
    return f"{async_format_model(model)} {async_format_id(id_)}"


@callback
def _async_unique_name(capabilities: Dict[str, Any]) -> str:
    """Generate name from capabilities."""
    model_id = async_format_model_id(capabilities["model"], capabilities["id"])
    return f"Yeelight {model_id}"


def update_needs_bg_power_workaround(data: Dict[str, Any]) -> bool:
    """Check if a push update needs the bg_power workaround.

    Some devices will push the incorrect state for bg_power.

    To work around this any time we are pushed an update
    with bg_power, we force poll state which will be correct.
    """
    return "bg_power" in data


class YeelightDevice:
    """Represents single Yeelight device."""

    def __init__(
        self, hass: HomeAssistant, host: str, config: Dict[str, Any], bulb: AsyncBulb
    ) -> None:
        """Initialize device."""
        self._hass: HomeAssistant = hass
        self._config: Dict[str, Any] = config
        self._host: str = host
        self._bulb_device: AsyncBulb = bulb
        self.capabilities: Dict[str, Any] = {}
        self._device_type: Optional[str] = None
        self._available: bool = True
        self._initialized: bool = False
        self._name: Optional[str] = None

    @property
    def bulb(self) -> AsyncBulb:
        """Return bulb device."""
        return self._bulb_device

    @property
    def name(self) -> Optional[str]:
        """Return the name of the device if any."""
        return self._name

    @property
    def config(self) -> Dict[str, Any]:
        """Return device config."""
        return self._config

    @property
    def host(self) -> str:
        """Return hostname."""
        return self._host

    @property
    def available(self) -> bool:
        """Return true is device is available."""
        return self._available

    @callback
    def async_mark_unavailable(self) -> None:
        """Set unavailable on api call failure due to a network issue."""
        self._available = False

    @property
    def model(self) -> Optional[str]:
        """Return configured/autodetected device model."""
        return self._bulb_device.model or self.capabilities.get("model")

    @property
    def fw_version(self) -> Optional[str]:
        """Return the firmware version."""
        return self.capabilities.get("fw_ver")

    @property
    def unique_id(self) -> Optional[str]:
        """Return the unique ID of the device."""
        return self.capabilities.get("id")

    @property
    def is_nightlight_supported(self) -> bool:
        """Return true / false if nightlight is supported.

        Uses brightness as it appears to be supported in both ceiling and other lights.
        """
        return self._nightlight_brightness is not None

    @property
    def is_nightlight_enabled(self) -> bool:
        """Return true / false if nightlight is currently enabled."""
        # Only ceiling lights have active_mode, from SDK docs:
        # active_mode 0: daylight mode / 1: moonlight mode (ceiling light only)
        if self._active_mode is not None:
            return int(self._active_mode) == ACTIVE_MODE_NIGHTLIGHT

        if self._nightlight_brightness is not None:
            return int(self._nightlight_brightness) > 0

        return False

    @property
    def is_color_flow_enabled(self) -> bool:
        """Return true / false if color flow is currently running."""
        return self._color_flow is not None and int(self._color_flow) == ACTIVE_COLOR_FLOWING

    @property
    def _active_mode(self) -> Optional[str]:
        return self.bulb.last_properties.get("active_mode")

    @property
    def _color_flow(self) -> Optional[str]:
        return self.bulb.last_properties.get("flowing")

    @property
    def _nightlight_brightness(self) -> Optional[str]:
        return self.bulb.last_properties.get("nl_br")

    @property
    def type(self) -> Optional[str]:
        """Return bulb type."""
        if not self._device_type:
            self._device_type = self.bulb.bulb_type

        return self._device_type

    async def _async_update_properties(self) -> None:
        """Read new properties from the device."""
        try:
            await self.bulb.async_get_properties(UPDATE_REQUEST_PROPERTIES)
            self._available = True
            if not self._initialized:
                self._initialized = True
        except TimeoutError as ex:
            _LOGGER.debug(
                "timed out while trying to update device %s, %s: %s",
                self._host,
                self.name,
                ex,
            )
        except OSError as ex:
            if self._available:  # just inform once
                _LOGGER.error(
                    "Unable to update device %s, %s: %s", self._host, self.name, ex
                )
            self._available = False
        except BulbException as ex:
            _LOGGER.debug(
                "Unable to update device %s, %s: %s", self._host, self.name, ex
            )

    async def async_setup(self) -> None:
        """Fetch capabilities and setup name if available."""
        scanner: YeelightScanner = YeelightScanner.async_get(self._hass)
        self.capabilities = await scanner.async_get_capabilities(self._host) or {}
        if self.capabilities:
            self._bulb_device.set_capabilities(self.capabilities)
        if name := self._config.get(CONF_NAME):
            # Override default name when name is set in config
            self._name = name
        elif self.capabilities:
            # Generate name from model and id when capabilities is available
            self._name = _async_unique_name(self.capabilities)
        elif self.model and (id_ := self._config.get(CONF_ID)):
            self._name = f"Yeelight {async_format_model_id(self.model, id_)}"
        else:
            self._name = self._host  # Default name is host

    async def async_update(self, force: bool = False) -> None:
        """Update device properties and send data updated signal."""
        if not force and self._initialized and self._available:
            # No need to poll unless force, already connected
            return
        await self._async_update_properties()
        async_dispatcher_send(self._hass, DATA_UPDATED.format(self._host))

    async def _async_forced_update(self, _now: Any) -> None:
        """Call a forced update."""
        await self.async_update(True)

    @callback
    def async_update_callback(self, data: Dict[str, Any]) -> None:
        """Update push from device."""
        _LOGGER.debug("Received callback: %s", data)
        was_available: bool = self._available
        self._available = data.get(KEY_CONNECTED, True)
        if update_needs_bg_power_workaround(data) or (
            not was_available and self._available
        ):
            # On reconnect the properties may be out of sync
            #
            # If the device drops the connection right away, we do not want to
            # do a property resync via async_update since its about
            # to be called when async_setup_entry reaches the end of the
            # function
            #
            async_call_later(self._hass, STATE_CHANGE_TIME, self._async_forced_update)
        async_dispatcher_send(self._hass, DATA_UPDATED.format(self._host))
