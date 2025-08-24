"""Support for Concord232 alarm control panels."""
from __future__ import annotations

import datetime
import logging
from typing import Any

from concord232 import client as concord232_client
import requests
import voluptuous as vol

from homeassistant.components.alarm_control_panel import (
    PLATFORM_SCHEMA as ALARM_CONTROL_PANEL_PLATFORM_SCHEMA,
    AlarmControlPanelEntity,
    AlarmControlPanelEntityFeature,
    AlarmControlPanelState,
    CodeFormat,
)
from homeassistant.const import CONF_CODE, CONF_HOST, CONF_MODE, CONF_NAME, CONF_PORT
from homeassistant.core import HomeAssistant
from homeassistant.helpers import config_validation as cv
from homeassistant.helpers.entity_platform import AddEntitiesCallback
from homeassistant.helpers.typing import ConfigType, DiscoveryInfoType

_LOGGER: logging.Logger = logging.getLogger(__name__)

DEFAULT_HOST: str = "localhost"
DEFAULT_NAME: str = "CONCORD232"
DEFAULT_PORT: int = 5007
DEFAULT_MODE: str = "audible"

SCAN_INTERVAL: datetime.timedelta = datetime.timedelta(seconds=10)

PLATFORM_SCHEMA: vol.Schema = ALARM_CONTROL_PANEL_PLATFORM_SCHEMA.extend(
    {
        vol.Optional(CONF_HOST, default=DEFAULT_HOST): cv.string,
        vol.Optional(CONF_NAME, default=DEFAULT_NAME): cv.string,
        vol.Optional(CONF_CODE): cv.string,
        vol.Optional(CONF_MODE, default=DEFAULT_MODE): cv.string,
        vol.Optional(CONF_PORT, default=DEFAULT_PORT): cv.port,
    }
)


def setup_platform(
    hass: HomeAssistant,
    config: ConfigType,
    add_entities: AddEntitiesCallback,
    discovery_info: DiscoveryInfoType | None = None,
) -> None:
    """Set up the Concord232 alarm control panel platform."""
    name: str = config[CONF_NAME]
    code: str | None = config.get(CONF_CODE)
    mode: str = config[CONF_MODE]
    host: str = config[CONF_HOST]
    port: int = config[CONF_PORT]

    url: str = f"http://{host}:{port}"

    try:
        add_entities([Concord232Alarm(url, name, code, mode)], True)
    except requests.exceptions.ConnectionError as ex:
        _LOGGER.error("Unable to connect to Concord232: %s", str(ex))


class Concord232Alarm(AlarmControlPanelEntity):
    """Representation of the Concord232-based alarm panel."""

    _attr_code_format: CodeFormat = CodeFormat.NUMBER
    _attr_supported_features: AlarmControlPanelEntityFeature = (
        AlarmControlPanelEntityFeature.ARM_HOME | AlarmControlPanelEntityFeature.ARM_AWAY
    )

    def __init__(self, url: str, name: str, code: str | None, mode: str) -> None:
        """Initialize the Concord232 alarm panel."""
        self._attr_name: str = name
        self._code: str | None = code
        self._alarm_control_panel_option_default_code: str | None = code
        self._mode: str = mode
        self._url: str = url
        self._alarm: concord232_client.Client = concord232_client.Client(self._url)
        self._alarm.partitions = self._alarm.list_partitions()
        self._attr_alarm_state: AlarmControlPanelState | None = None

    def update(self) -> None:
        """Update values from API."""
        try:
            part: dict[str, Any] = self._alarm.list_partitions()[0]
        except requests.exceptions.ConnectionError as ex:
            _LOGGER.error(
                "Unable to connect to %(host)s: %(reason)s",
                {"host": self._url, "reason": ex},
            )
            return
        except IndexError:
            _LOGGER.error("Concord232 reports no partitions")
            return

        if part["arming_level"] == "Off":
            self._attr_alarm_state = AlarmControlPanelState.DISARMED
        elif "Home" in part["arming_level"]:
            self._attr_alarm_state = AlarmControlPanelState.ARMED_HOME
        else:
            self._attr_alarm_state = AlarmControlPanelState.ARMED_AWAY

    def alarm_disarm(self, code: str | None = None) -> None:
        """Send disarm command."""
        if not self._validate_code(code, AlarmControlPanelState.DISARMED):
            return
        self._alarm.disarm(code)

    def alarm_arm_home(self, code: str | None = None) -> None:
        """Send arm home command."""
        if not self._validate_code(code, AlarmControlPanelState.ARMED_HOME):
            return
        if self._mode == "silent":
            self._alarm.arm("stay", "silent")
        else:
            self._alarm.arm("stay")

    def alarm_arm_away(self, code: str | None = None) -> None:
        """Send arm away command."""
        if not self._validate_code(code, AlarmControlPanelState.ARMED_AWAY):
            return
        self._alarm.arm("away")

    def _validate_code(self, code: str | None, state: AlarmControlPanelState) -> bool:
        """Validate given code."""
        if self._code is None:
            return True
        alarm_code: str | None = self._code
        check: bool = not alarm_code or code == alarm_code
        if not check:
            _LOGGER.warning("Invalid code given for %s", state)
        return check