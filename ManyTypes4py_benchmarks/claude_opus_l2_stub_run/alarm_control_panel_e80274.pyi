from __future__ import annotations

import datetime
import logging
from typing import Any

import voluptuous as vol

from homeassistant.components.alarm_control_panel import (
    AlarmControlPanelEntity,
    AlarmControlPanelEntityFeature,
    AlarmControlPanelState,
    CodeFormat,
)
from homeassistant.core import Event, EventStateChangedData, HomeAssistant, callback
from homeassistant.helpers.entity_platform import AddEntitiesCallback
from homeassistant.helpers.typing import ConfigType, DiscoveryInfoType

_LOGGER: logging.Logger

CONF_CODE_TEMPLATE: str
CONF_CODE_ARM_REQUIRED: str
CONF_PAYLOAD_DISARM: str
CONF_PAYLOAD_ARM_HOME: str
CONF_PAYLOAD_ARM_AWAY: str
CONF_PAYLOAD_ARM_NIGHT: str
CONF_PAYLOAD_ARM_VACATION: str
CONF_PAYLOAD_ARM_CUSTOM_BYPASS: str
CONF_ALARM_ARMED_AWAY: str
CONF_ALARM_ARMED_CUSTOM_BYPASS: str
CONF_ALARM_ARMED_HOME: str
CONF_ALARM_ARMED_NIGHT: str
CONF_ALARM_ARMED_VACATION: str
CONF_ALARM_DISARMED: str
CONF_ALARM_PENDING: str
CONF_ALARM_TRIGGERED: str

DEFAULT_ALARM_NAME: str
DEFAULT_DELAY_TIME: datetime.timedelta
DEFAULT_PENDING_TIME: datetime.timedelta
DEFAULT_TRIGGER_TIME: datetime.timedelta
DEFAULT_DISARM_AFTER_TRIGGER: bool
DEFAULT_ARM_AWAY: str
DEFAULT_ARM_HOME: str
DEFAULT_ARM_NIGHT: str
DEFAULT_ARM_VACATION: str
DEFAULT_ARM_CUSTOM_BYPASS: str
DEFAULT_DISARM: str

SUPPORTED_STATES: list[AlarmControlPanelState]
SUPPORTED_PRETRIGGER_STATES: list[AlarmControlPanelState]
SUPPORTED_PENDING_STATES: list[AlarmControlPanelState]

ATTR_PRE_PENDING_STATE: str
ATTR_POST_PENDING_STATE: str

def _state_validator(config: dict[str, Any]) -> dict[str, Any]: ...
def _state_schema(state: AlarmControlPanelState) -> vol.Schema: ...

PLATFORM_SCHEMA: vol.Schema

async def async_setup_platform(
    hass: HomeAssistant,
    config: ConfigType,
    add_entities: AddEntitiesCallback,
    discovery_info: DiscoveryInfoType | None = ...,
) -> None: ...

class ManualMQTTAlarm(AlarmControlPanelEntity):
    _attr_should_poll: bool
    _attr_supported_features: AlarmControlPanelEntityFeature
    _state: AlarmControlPanelState
    _hass: HomeAssistant
    _attr_name: str
    _code: str | Any | None
    _disarm_after_trigger: bool
    _previous_state: AlarmControlPanelState
    _state_ts: datetime.datetime | None
    _delay_time_by_state: dict[AlarmControlPanelState, datetime.timedelta]
    _trigger_time_by_state: dict[AlarmControlPanelState, datetime.timedelta]
    _pending_time_by_state: dict[AlarmControlPanelState, datetime.timedelta]
    _state_topic: str | None
    _command_topic: str | None
    _qos: int | None
    _attr_code_arm_required: bool
    _payload_disarm: str | None
    _payload_arm_home: str | None
    _payload_arm_away: str | None
    _payload_arm_night: str | None
    _payload_arm_vacation: str | None
    _payload_arm_custom_bypass: str | None

    def __init__(
        self,
        hass: HomeAssistant,
        name: str,
        code: str | None,
        code_template: Any | None,
        disarm_after_trigger: bool,
        state_topic: str | None,
        command_topic: str | None,
        qos: int | None,
        code_arm_required: bool,
        payload_disarm: str | None,
        payload_arm_home: str | None,
        payload_arm_away: str | None,
        payload_arm_night: str | None,
        payload_arm_vacation: str | None,
        payload_arm_custom_bypass: str | None,
        config: dict[str, Any],
    ) -> None: ...

    @property
    def alarm_state(self) -> AlarmControlPanelState: ...
    @property
    def _active_state(self) -> AlarmControlPanelState: ...
    def _pending_time(self, state: AlarmControlPanelState) -> datetime.timedelta: ...
    def _within_pending_time(self, state: AlarmControlPanelState) -> bool: ...
    @property
    def code_format(self) -> CodeFormat | None: ...
    async def async_alarm_disarm(self, code: str | None = ...) -> None: ...
    async def async_alarm_arm_home(self, code: str | None = ...) -> None: ...
    async def async_alarm_arm_away(self, code: str | None = ...) -> None: ...
    async def async_alarm_arm_night(self, code: str | None = ...) -> None: ...
    async def async_alarm_arm_vacation(self, code: str | None = ...) -> None: ...
    async def async_alarm_arm_custom_bypass(self, code: str | None = ...) -> None: ...
    async def async_alarm_trigger(self, code: str | None = ...) -> None: ...
    def _async_update_state(self, state: AlarmControlPanelState) -> None: ...
    def _async_validate_code(self, code: str | None, state: AlarmControlPanelState) -> None: ...
    @property
    def extra_state_attributes(self) -> dict[str, Any]: ...
    @callback
    def async_scheduled_update(self, now: datetime.datetime) -> None: ...
    async def async_added_to_hass(self) -> None: ...
    async def _async_state_changed_listener(self, event: Event[EventStateChangedData]) -> None: ...