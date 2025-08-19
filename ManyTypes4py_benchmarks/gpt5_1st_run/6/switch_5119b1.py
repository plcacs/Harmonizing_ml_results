"""Support for SCSGate switches."""
from __future__ import annotations

import logging
from typing import Any, Final

import voluptuous as vol
from homeassistant.components.switch import (
    PLATFORM_SCHEMA as SWITCH_PLATFORM_SCHEMA,
    SwitchEntity,
)
from homeassistant.const import ATTR_ENTITY_ID, ATTR_STATE, CONF_DEVICES, CONF_NAME
from homeassistant.core import HomeAssistant
from homeassistant.helpers import config_validation as cv
from homeassistant.helpers.entity_platform import AddEntitiesCallback
from homeassistant.helpers.typing import ConfigType, DiscoveryInfoType
from scsgate.messages import ScenarioTriggeredMessage, StateMessage
from scsgate.tasks import ToggleStatusTask

from . import CONF_SCS_ID, DOMAIN, SCSGATE_SCHEMA

ATTR_SCENARIO_ID: Final[str] = "scenario_id"
CONF_TRADITIONAL: Final[str] = "traditional"
CONF_SCENARIO: Final[str] = "scenario"

PLATFORM_SCHEMA: vol.Schema = SWITCH_PLATFORM_SCHEMA.extend(
    {vol.Required(CONF_DEVICES): cv.schema_with_slug_keys(SCSGATE_SCHEMA)}
)


def setup_platform(
    hass: HomeAssistant,
    config: ConfigType,
    add_entities: AddEntitiesCallback,
    discovery_info: DiscoveryInfoType | None = None,
) -> None:
    """Set up the SCSGate switches."""
    logger = logging.getLogger(__name__)
    scsgate: Any = hass.data[DOMAIN]
    _setup_traditional_switches(
        logger=logger,
        config=config,
        scsgate=scsgate,
        add_entities_callback=add_entities,
    )
    _setup_scenario_switches(logger=logger, config=config, scsgate=scsgate, hass=hass)


def _setup_traditional_switches(
    logger: logging.Logger,
    config: ConfigType,
    scsgate: Any,
    add_entities_callback: AddEntitiesCallback,
) -> None:
    """Add traditional SCSGate switches."""
    traditional: dict[str, dict[str, Any]] | None = config.get(CONF_TRADITIONAL)
    switches: list[SCSGateSwitch] = []
    if traditional:
        for entity_info in traditional.values():
            if entity_info[CONF_SCS_ID] in scsgate.devices:
                continue
            name: str = entity_info[CONF_NAME]
            scs_id: str = entity_info[CONF_SCS_ID]
            logger.info("Adding %s scsgate.traditional_switch", name)
            switch = SCSGateSwitch(name=name, scs_id=scs_id, logger=logger, scsgate=scsgate)
            switches.append(switch)
    add_entities_callback(switches)
    scsgate.add_devices_to_register(switches)


def _setup_scenario_switches(
    logger: logging.Logger, config: ConfigType, scsgate: Any, hass: HomeAssistant
) -> None:
    """Add only SCSGate scenario switches."""
    scenario: dict[str, dict[str, Any]] | None = config.get(CONF_SCENARIO)
    if scenario:
        for entity_info in scenario.values():
            if entity_info[CONF_SCS_ID] in scsgate.devices:
                continue
            name: str = entity_info[CONF_NAME]
            scs_id: str = entity_info[CONF_SCS_ID]
            logger.info("Adding %s scsgate.scenario_switch", name)
            switch = SCSGateScenarioSwitch(name=name, scs_id=scs_id, logger=logger, hass=hass)
            scsgate.add_device(switch)


class SCSGateSwitch(SwitchEntity):
    """Representation of a SCSGate switch."""

    _attr_should_poll: bool = False

    def __init__(self, scs_id: str, name: str, logger: logging.Logger, scsgate: Any) -> None:
        """Initialize the switch."""
        self._name: str = name
        self._scs_id: str = scs_id
        self._toggled: bool = False
        self._logger: logging.Logger = logger
        self._scsgate: Any = scsgate

    @property
    def scs_id(self) -> str:
        """Return the SCS ID."""
        return self._scs_id

    @property
    def name(self) -> str:
        """Return the name of the device if any."""
        return self._name

    @property
    def is_on(self) -> bool:
        """Return true if switch is on."""
        return self._toggled

    def turn_on(self, **kwargs: Any) -> None:
        """Turn the device on."""
        self._scsgate.append_task(ToggleStatusTask(target=self._scs_id, toggled=True))
        self._toggled = True
        self.schedule_update_ha_state()

    def turn_off(self, **kwargs: Any) -> None:
        """Turn the device off."""
        self._scsgate.append_task(ToggleStatusTask(target=self._scs_id, toggled=False))
        self._toggled = False
        self.schedule_update_ha_state()

    def process_event(self, message: Any) -> None:
        """Handle a SCSGate message related with this switch."""
        if getattr(message, "toggled", None) is None:
            self._logger.warning("Switch %s received unknown message %s", self._scs_id, message)
            return
        if self._toggled == message.toggled:
            self._logger.info(
                "Switch %s, ignoring message %s because state already active", self._scs_id, message
            )
            return
        self._toggled = bool(message.toggled)
        self.schedule_update_ha_state()
        command: str = "off"
        if self._toggled:
            command = "on"
        assert self.hass is not None
        self.hass.bus.fire("button_pressed", {ATTR_ENTITY_ID: self._scs_id, ATTR_STATE: command})


class SCSGateScenarioSwitch:
    """Provides a SCSGate scenario switch.

    This switch is always in an 'off" state, when toggled it's used to trigger
    events.
    """

    def __init__(self, scs_id: str, name: str, logger: logging.Logger, hass: HomeAssistant) -> None:
        """Initialize the scenario."""
        self._name: str = name
        self._scs_id: str = scs_id
        self._logger: logging.Logger = logger
        self._hass: HomeAssistant = hass

    @property
    def scs_id(self) -> str:
        """Return the SCS ID."""
        return self._scs_id

    @property
    def name(self) -> str:
        """Return the name of the device if any."""
        return self._name

    def process_event(self, message: Any) -> None:
        """Handle a SCSGate message related with this switch."""
        scenario_id: Any
        if isinstance(message, StateMessage):
            scenario_id = message.bytes[4]
        elif isinstance(message, ScenarioTriggeredMessage):
            scenario_id = message.scenario
        else:
            self._logger.warning("Scenario switch: received unknown message %s", message)
            return
        self._hass.bus.fire(
            "scenario_switch_triggered",
            {ATTR_ENTITY_ID: int(self._scs_id), ATTR_SCENARIO_ID: int(scenario_id, 16)},
        )