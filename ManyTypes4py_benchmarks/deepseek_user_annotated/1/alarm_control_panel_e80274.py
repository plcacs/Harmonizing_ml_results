"""Support for manual alarms controllable via MQTT."""

from __future__ import annotations

import datetime
import logging
from typing import Any, Final, Optional, Union

import voluptuous as vol

from homeassistant.components import mqtt
from homeassistant.components.alarm_control_panel import (
    AlarmControlPanelEntity,
    AlarmControlPanelEntityFeature,
    AlarmControlPanelState,
    CodeFormat,
)
from homeassistant.const import (
    CONF_CODE,
    CONF_DELAY_TIME,
    CONF_DISARM_AFTER_TRIGGER,
    CONF_NAME,
    CONF_PENDING_TIME,
    CONF_PLATFORM,
    CONF_TRIGGER_TIME,
)
from homeassistant.core import Event, EventStateChangedData, HomeAssistant, callback
from homeassistant.exceptions import HomeAssistantError
from homeassistant.helpers import config_validation as cv
from homeassistant.helpers.entity_platform import AddEntitiesCallback
from homeassistant.helpers.event import (
    async_track_point_in_time,
    async_track_state_change_event,
)
from homeassistant.helpers.typing import ConfigType, DiscoveryInfoType
from homeassistant.util import dt as dt_util

_LOGGER: Final = logging.getLogger(__name__)

CONF_CODE_TEMPLATE: Final = "code_template"
CONF_CODE_ARM_REQUIRED: Final = "code_arm_required"

CONF_PAYLOAD_DISARM: Final = "payload_disarm"
CONF_PAYLOAD_ARM_HOME: Final = "payload_arm_home"
CONF_PAYLOAD_ARM_AWAY: Final = "payload_arm_away"
CONF_PAYLOAD_ARM_NIGHT: Final = "payload_arm_night"
CONF_PAYLOAD_ARM_VACATION: Final = "payload_arm_vacation"
CONF_PAYLOAD_ARM_CUSTOM_BYPASS: Final = "payload_arm_custom_bypass"

CONF_ALARM_ARMED_AWAY: Final = "armed_away"
CONF_ALARM_ARMED_CUSTOM_BYPASS: Final = "armed_custom_bypass"
CONF_ALARM_ARMED_HOME: Final = "armed_home"
CONF_ALARM_ARMED_NIGHT: Final = "armed_night"
CONF_ALARM_ARMED_VACATION: Final = "armed_vacation"
CONF_ALARM_DISARMED: Final = "disarmed"
CONF_ALARM_PENDING: Final = "pending"
CONF_ALARM_TRIGGERED: Final = "triggered"

DEFAULT_ALARM_NAME: Final = "HA Alarm"
DEFAULT_DELAY_TIME: Final = datetime.timedelta(seconds=0)
DEFAULT_PENDING_TIME: Final = datetime.timedelta(seconds=60)
DEFAULT_TRIGGER_TIME: Final = datetime.timedelta(seconds=120)
DEFAULT_DISARM_AFTER_TRIGGER: Final = False
DEFAULT_ARM_AWAY: Final = "ARM_AWAY"
DEFAULT_ARM_HOME: Final = "ARM_HOME"
DEFAULT_ARM_NIGHT: Final = "ARM_NIGHT"
DEFAULT_ARM_VACATION: Final = "ARM_VACATION"
DEFAULT_ARM_CUSTOM_BYPASS: Final = "ARM_CUSTOM_BYPASS"
DEFAULT_DISARM: Final = "DISARM"

SUPPORTED_STATES: Final = [
    AlarmControlPanelState.DISARMED,
    AlarmControlPanelState.ARMED_AWAY,
    AlarmControlPanelState.ARMED_HOME,
    AlarmControlPanelState.ARMED_NIGHT,
    AlarmControlPanelState.ARMED_VACATION,
    AlarmControlPanelState.ARMED_CUSTOM_BYPASS,
    AlarmControlPanelState.TRIGGERED,
]

SUPPORTED_PRETRIGGER_STATES: Final = [
    state for state in SUPPORTED_STATES if state != AlarmControlPanelState.TRIGGERED
]

SUPPORTED_PENDING_STATES: Final = [
    state for state in SUPPORTED_STATES if state != AlarmControlPanelState.DISARMED
]

ATTR_PRE_PENDING_STATE: Final = "pre_pending_state"
ATTR_POST_PENDING_STATE: Final = "post_pending_state"


def _state_validator(config: dict[str, Any]) -> dict[str, Any]:
    """Validate the state."""
    for state in SUPPORTED_PRETRIGGER_STATES:
        if CONF_DELAY_TIME not in config[state]:
            config[state] = config[state] | {CONF_DELAY_TIME: config[CONF_DELAY_TIME]}
        if CONF_TRIGGER_TIME not in config[state]:
            config[state] = config[state] | {
                CONF_TRIGGER_TIME: config[CONF_TRIGGER_TIME]
            }
    for state in SUPPORTED_PENDING_STATES:
        if CONF_PENDING_TIME not in config[state]:
            config[state] = config[state] | {
                CONF_PENDING_TIME: config[CONF_PENDING_TIME]
            }

    return config


def _state_schema(state: str) -> vol.Schema:
    """Validate the state."""
    schema: dict[str, Any] = {}
    if state in SUPPORTED_PRETRIGGER_STATES:
        schema[vol.Optional(CONF_DELAY_TIME)] = vol.All(
            cv.time_period, cv.positive_timedelta
        )
        schema[vol.Optional(CONF_TRIGGER_TIME)] = vol.All(
            cv.time_period, cv.positive_timedelta
        )
    if state in SUPPORTED_PENDING_STATES:
        schema[vol.Optional(CONF_PENDING_TIME)] = vol.All(
            cv.time_period, cv.positive_timedelta
        )
    return vol.Schema(schema)


PLATFORM_SCHEMA: Final = vol.Schema(
    vol.All(
        mqtt.config.MQTT_BASE_SCHEMA.extend(
            {
                vol.Required(CONF_PLATFORM): "manual_mqtt",
                vol.Optional(CONF_NAME, default=DEFAULT_ALARM_NAME): cv.string,
                vol.Exclusive(CONF_CODE, "code validation"): cv.string,
                vol.Exclusive(CONF_CODE_TEMPLATE, "code validation"): cv.template,
                vol.Optional(CONF_DELAY_TIME, default=DEFAULT_DELAY_TIME): vol.All(
                    cv.time_period, cv.positive_timedelta
                ),
                vol.Optional(CONF_PENDING_TIME, default=DEFAULT_PENDING_TIME): vol.All(
                    cv.time_period, cv.positive_timedelta
                ),
                vol.Optional(CONF_TRIGGER_TIME, default=DEFAULT_TRIGGER_TIME): vol.All(
                    cv.time_period, cv.positive_timedelta
                ),
                vol.Optional(
                    CONF_DISARM_AFTER_TRIGGER, default=DEFAULT_DISARM_AFTER_TRIGGER
                ): cv.boolean,
                vol.Optional(CONF_ALARM_ARMED_AWAY, default={}): _state_schema(
                    AlarmControlPanelState.ARMED_AWAY
                ),
                vol.Optional(CONF_ALARM_ARMED_HOME, default={}): _state_schema(
                    AlarmControlPanelState.ARMED_HOME
                ),
                vol.Optional(CONF_ALARM_ARMED_NIGHT, default={}): _state_schema(
                    AlarmControlPanelState.ARMED_NIGHT
                ),
                vol.Optional(CONF_ALARM_ARMED_VACATION, default={}): _state_schema(
                    AlarmControlPanelState.ARMED_VACATION
                ),
                vol.Optional(CONF_ALARM_ARMED_CUSTOM_BYPASS, default={}): _state_schema(
                    AlarmControlPanelState.ARMED_CUSTOM_BYPASS
                ),
                vol.Optional(CONF_ALARM_DISARMED, default={}): _state_schema(
                    AlarmControlPanelState.DISARMED
                ),
                vol.Optional(CONF_ALARM_TRIGGERED, default={}): _state_schema(
                    AlarmControlPanelState.TRIGGERED
                ),
                vol.Required(mqtt.CONF_COMMAND_TOPIC): mqtt.valid_publish_topic,
                vol.Required(mqtt.CONF_STATE_TOPIC): mqtt.valid_subscribe_topic,
                vol.Optional(CONF_CODE_ARM_REQUIRED, default=True): cv.boolean,
                vol.Optional(
                    CONF_PAYLOAD_ARM_AWAY, default=DEFAULT_ARM_AWAY
                ): cv.string,
                vol.Optional(
                    CONF_PAYLOAD_ARM_HOME, default=DEFAULT_ARM_HOME
                ): cv.string,
                vol.Optional(
                    CONF_PAYLOAD_ARM_NIGHT, default=DEFAULT_ARM_NIGHT
                ): cv.string,
                vol.Optional(
                    CONF_PAYLOAD_ARM_VACATION, default=DEFAULT_ARM_VACATION
                ): cv.string,
                vol.Optional(
                    CONF_PAYLOAD_ARM_CUSTOM_BYPASS, default=DEFAULT_ARM_CUSTOM_BYPASS
                ): cv.string,
                vol.Optional(CONF_PAYLOAD_DISARM, default=DEFAULT_DISARM): cv.string,
            }
        ),
        _state_validator,
    )
)


async def async_setup_platform(
    hass: HomeAssistant,
    config: ConfigType,
    add_entities: AddEntitiesCallback,
    discovery_info: DiscoveryInfoType | None = None,
) -> None:
    """Set up the manual MQTT alarm platform."""
    if not await mqtt.async_wait_for_mqtt_client(hass):
        _LOGGER.error("MQTT integration is not available")
        return
    add_entities(
        [
            ManualMQTTAlarm(
                hass,
                config[CONF_NAME],
                config.get(CONF_CODE),
                config.get(CONF_CODE_TEMPLATE),
                config.get(CONF_DISARM_AFTER_TRIGGER, DEFAULT_DISARM_AFTER_TRIGGER),
                config.get(mqtt.CONF_STATE_TOPIC),
                config.get(mqtt.CONF_COMMAND_TOPIC),
                config.get(mqtt.CONF_QOS),
                config.get(CONF_CODE_ARM_REQUIRED),
                config.get(CONF_PAYLOAD_DISARM),
                config.get(CONF_PAYLOAD_ARM_HOME),
                config.get(CONF_PAYLOAD_ARM_AWAY),
                config.get(CONF_PAYLOAD_ARM_NIGHT),
                config.get(CONF_PAYLOAD_ARM_VACATION),
                config.get(CONF_PAYLOAD_ARM_CUSTOM_BYPASS),
                config,
            )
        ]
    )


class ManualMQTTAlarm(AlarmControlPanelEntity):
    """Representation of an alarm status."""

    _attr_should_poll: bool = False
    _attr_supported_features: AlarmControlPanelEntityFeature = (
        AlarmControlPanelEntityFeature.ARM_HOME
        | AlarmControlPanelEntityFeature.ARM_AWAY
        | AlarmControlPanelEntityFeature.ARM_NIGHT
        | AlarmControlPanelEntityFeature.ARM_VACATION
        | AlarmControlPanelEntityFeature.TRIGGER
        | AlarmControlPanelEntityFeature.ARM_CUSTOM_BYPASS
    )

    def __init__(
        self,
        hass: HomeAssistant,
        name: str,
        code: str | None,
        code_template: Any | None,
        disarm_after_trigger: bool,
        state_topic: str,
        command_topic: str,
        qos: int,
        code_arm_required: bool,
        payload_disarm: str,
        payload_arm_home: str,
        payload_arm_away: str,
        payload_arm_night: str,
        payload_arm_vacation: str,
        payload_arm_custom_bypass: str,
        config: ConfigType,
    ) -> None:
        """Init the manual MQTT alarm panel."""
        self._state: str = AlarmControlPanelState.DISARMED
        self._hass: HomeAssistant = hass
        self._attr_name: str = name
        if code_template:
            self._code: str | None = code_template
        else:
            self._code = code or None
        self._disarm_after_trigger: bool = disarm_after_trigger
        self._previous_state: str = self._state
        self._state_ts: datetime.datetime | None = None

        self._delay_time_by_state: dict[str, datetime.timedelta] = {
            state: config[state][CONF_DELAY_TIME]
            for state in SUPPORTED_PRETRIGGER_STATES
        }
        self._trigger_time_by_state: dict[str, datetime.timedelta] = {
            state: config[state][CONF_TRIGGER_TIME]
            for state in SUPPORTED_PRETRIGGER_STATES
        }
        self._pending_time_by_state: dict[str, datetime.timedelta] = {
            state: config[state][CONF_PENDING_TIME]
            for state in SUPPORTED_PENDING_STATES
        }

        self._state_topic: str = state_topic
        self._command_topic: str = command_topic
        self._qos: int = qos
        self._attr_code_arm_required: bool = code_arm_required
        self._payload_disarm: str = payload_disarm
        self._payload_arm_home: str = payload_arm_home
        self._payload_arm_away: str = payload_arm_away
        self._payload_arm_night: str = payload_arm_night
        self._payload_arm_vacation: str = payload_arm_vacation
        self._payload_arm_custom_bypass: str = payload_arm_custom_bypass

    @property
    def alarm_state(self) -> AlarmControlPanelState:
        """Return the state of the device."""
        if self._state == AlarmControlPanelState.TRIGGERED:
            if self._within_pending_time(self._state):
                return AlarmControlPanelState.PENDING
            trigger_time = self._trigger_time_by_state[self._previous_state]
            if (
                self._state_ts + self._pending_time(self._state) + trigger_time
            ) < dt_util.utcnow():
                if self._disarm_after_trigger:
                    return AlarmControlPanelState.DISARMED
                self._state = self._previous_state
                return self._state

        if self._state in SUPPORTED_PENDING_STATES and self._within_pending_time(
            self._state
        ):
            return AlarmControlPanelState.PENDING

        return self._state

    @property
    def _active_state(self) -> str:
        """Get the current state."""
        if self.state == AlarmControlPanelState.PENDING:
            return self._previous_state
        return self._state

    def _pending_time(self, state: str) -> datetime.timedelta:
        """Get the pending time."""
        pending_time = self._pending_time_by_state[state]
        if state == AlarmControlPanelState.TRIGGERED:
            pending_time += self._delay_time_by_state[self._previous_state]
        return pending_time

    def _within_pending_time(self, state: str) -> bool:
        """Get if the action is in the pending time window."""
        return self._state_ts + self._pending_time(state) > dt_util.utcnow()

    @property
    def code_format(self) -> CodeFormat | None:
        """Return one or more digits/characters."""
        if self._code is None:
            return None
        if isinstance(self._code, str) and self._code.isdigit():
            return CodeFormat.NUMBER
        return CodeFormat.TEXT

    async def async_alarm_disarm(self, code: str | None = None) -> None:
        """Send disarm command."""
        self._async_validate_code(code, AlarmControlPanelState.DISARMED)
        self._state = AlarmControlPanelState.DISARMED
        self._state_ts = dt_util.utcnow()
        self.async_write_ha_state()

    async def async_alarm_arm_home(self, code: str | None = None) -> None:
        """Send arm home command."""
        self._async_validate_code(code, AlarmControlPanelState.ARMED_HOME)
        self._async_update_state(AlarmControlPanelState.ARMED_HOME)

    async def async_alarm_arm_away(self, code: str | None = None) -> None:
        """Send arm away command."""
        self._async_validate_code(code, AlarmControlPanelState.ARMED_AWAY)
        self._async_update_state(AlarmControlPanelState.ARMED_AWAY)

    async def async_alarm_arm_night(self, code: str | None = None) -> None:
        """Send arm night command."""
        self._async_validate_code(code, AlarmControlPanelState.ARMED_NIGHT)
        self._async_update_state(AlarmControlPanelState.ARMED_NIGHT)

    async def async_alarm_arm_vacation(self, code: str | None = None) -> None:
        """Send arm vacation command."""
        self._async_validate_code(code, AlarmControlPanelState.ARMED_VACATION)
        self._async_update_state(AlarmControlPanelState.ARMED_VACATION)

    async def async_alarm_arm_custom_bypass(self, code: str | None = None) -> None:
        """Send arm custom bypass command."""
        self._async_validate_code(code, AlarmControlPanelState.ARMED_CUSTOM_BYPASS)
        self._async_update_state(AlarmControlPanelState.ARMED_CUSTOM_BYPASS)

    async def async_alarm_trigger(self, code: str | None = None) -> None:
        """Send alarm trigger command."""
        if not self._trigger_time_by_state[self._active_state]:
            return
        self._async_update_state(AlarmControlPanelState.TRIGGERED)

    def _async_update_state(self, state: str) -> None:
        """Update the state."""
        if self._state == state:
            return

        self._previous_state = self._state
        self._state = state
        self._state_ts = dt_util.utcnow()
        self.async_write_ha_state()

        pending_time = self._pending_time(state)
        if state == AlarmControlPanelState.TRIGGERED:
            async_t