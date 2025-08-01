"""Support for manual alarms."""
from __future__ import annotations
import datetime
from typing import Any, cast, Dict, Optional, Union
import voluptuous as vol
from homeassistant.components.alarm_control_panel import (
    PLATFORM_SCHEMA as ALARM_CONTROL_PANEL_PLATFORM_SCHEMA,
    AlarmControlPanelEntity,
    AlarmControlPanelEntityFeature,
    AlarmControlPanelState,
    CodeFormat,
)
from homeassistant.const import (
    CONF_ARMING_TIME,
    CONF_CODE,
    CONF_DELAY_TIME,
    CONF_DISARM_AFTER_TRIGGER,
    CONF_NAME,
    CONF_TRIGGER_TIME,
    CONF_UNIQUE_ID,
)
from homeassistant.core import HomeAssistant, callback
from homeassistant.exceptions import ServiceValidationError
from homeassistant.helpers import config_validation as cv
from homeassistant.helpers.entity_platform import AddEntitiesCallback
from homeassistant.helpers.event import async_track_point_in_time
from homeassistant.helpers.restore_state import RestoreEntity
from homeassistant.helpers.template import Template
from homeassistant.helpers.typing import ConfigType, DiscoveryInfoType
from homeassistant.util import dt as dt_util

DOMAIN = 'manual'
CONF_ARMING_STATES = 'arming_states'
CONF_CODE_TEMPLATE = 'code_template'
CONF_CODE_ARM_REQUIRED = 'code_arm_required'
CONF_ALARM_ARMED_AWAY = 'armed_away'
CONF_ALARM_ARMED_CUSTOM_BYPASS = 'armed_custom_bypass'
CONF_ALARM_ARMED_HOME = 'armed_home'
CONF_ALARM_ARMED_NIGHT = 'armed_night'
CONF_ALARM_ARMED_VACATION = 'armed_vacation'
CONF_ALARM_ARMING = 'arming'
CONF_ALARM_DISARMED = 'disarmed'
CONF_ALARM_PENDING = 'pending'
CONF_ALARM_TRIGGERED = 'triggered'
DEFAULT_ALARM_NAME = 'HA Alarm'
DEFAULT_DELAY_TIME = datetime.timedelta(seconds=60)
DEFAULT_ARMING_TIME = datetime.timedelta(seconds=60)
DEFAULT_TRIGGER_TIME = datetime.timedelta(seconds=120)
DEFAULT_DISARM_AFTER_TRIGGER = False
SUPPORTED_STATES = [
    AlarmControlPanelState.DISARMED,
    AlarmControlPanelState.ARMED_AWAY,
    AlarmControlPanelState.ARMED_HOME,
    AlarmControlPanelState.ARMED_NIGHT,
    AlarmControlPanelState.ARMED_VACATION,
    AlarmControlPanelState.ARMED_CUSTOM_BYPASS,
    AlarmControlPanelState.TRIGGERED,
]
SUPPORTED_PRETRIGGER_STATES = [state for state in SUPPORTED_STATES if state != AlarmControlPanelState.TRIGGERED]
SUPPORTED_ARMING_STATES = [state for state in SUPPORTED_STATES if state not in (AlarmControlPanelState.DISARMED, AlarmControlPanelState.TRIGGERED)]
SUPPORTED_ARMING_STATE_TO_FEATURE = {
    AlarmControlPanelState.ARMED_AWAY: AlarmControlPanelEntityFeature.ARM_AWAY,
    AlarmControlPanelState.ARMED_HOME: AlarmControlPanelEntityFeature.ARM_HOME,
    AlarmControlPanelState.ARMED_NIGHT: AlarmControlPanelEntityFeature.ARM_NIGHT,
    AlarmControlPanelState.ARMED_VACATION: AlarmControlPanelEntityFeature.ARM_VACATION,
    AlarmControlPanelState.ARMED_CUSTOM_BYPASS: AlarmControlPanelEntityFeature.ARM_CUSTOM_BYPASS,
}
ATTR_PREVIOUS_STATE = 'previous_state'
ATTR_NEXT_STATE = 'next_state'

def _state_validator(config: Dict[str, Any]) -> Dict[str, Any]:
    """Validate the state."""
    for state in SUPPORTED_PRETRIGGER_STATES:
        if CONF_DELAY_TIME not in config[state]:
            config[state] = config[state] | {CONF_DELAY_TIME: config[CONF_DELAY_TIME]}
        if CONF_TRIGGER_TIME not in config[state]:
            config[state] = config[state] | {CONF_TRIGGER_TIME: config[CONF_TRIGGER_TIME]}
    for state in SUPPORTED_ARMING_STATES:
        if CONF_ARMING_TIME not in config[state]:
            config[state] = config[state] | {CONF_ARMING_TIME: config[CONF_ARMING_TIME]}
    return config

def _state_schema(state: AlarmControlPanelState) -> vol.Schema:
    """Validate the state."""
    schema = {}
    if state in SUPPORTED_PRETRIGGER_STATES:
        schema[vol.Optional(CONF_DELAY_TIME)] = vol.All(cv.time_period, cv.positive_timedelta)
        schema[vol.Optional(CONF_TRIGGER_TIME)] = vol.All(cv.time_period, cv.positive_timedelta)
    if state in SUPPORTED_ARMING_STATES:
        schema[vol.Optional(CONF_ARMING_TIME)] = vol.All(cv.time_period, cv.positive_timedelta)
    return vol.Schema(schema)

PLATFORM_SCHEMA = vol.Schema(
    vol.All(
        ALARM_CONTROL_PANEL_PLATFORM_SCHEMA.extend({
            vol.Optional(CONF_NAME, default=DEFAULT_ALARM_NAME): cv.string,
            vol.Optional(CONF_UNIQUE_ID): cv.string,
            vol.Exclusive(CONF_CODE, 'code validation'): cv.string,
            vol.Exclusive(CONF_CODE_TEMPLATE, 'code validation'): cv.template,
            vol.Optional(CONF_CODE_ARM_REQUIRED, default=True): cv.boolean,
            vol.Optional(CONF_DELAY_TIME, default=DEFAULT_DELAY_TIME): vol.All(cv.time_period, cv.positive_timedelta),
            vol.Optional(CONF_ARMING_TIME, default=DEFAULT_ARMING_TIME): vol.All(cv.time_period, cv.positive_timedelta),
            vol.Optional(CONF_TRIGGER_TIME, default=DEFAULT_TRIGGER_TIME): vol.All(cv.time_period, cv.positive_timedelta),
            vol.Optional(CONF_DISARM_AFTER_TRIGGER, default=DEFAULT_DISARM_AFTER_TRIGGER): cv.boolean,
            vol.Optional(CONF_ARMING_STATES, default=SUPPORTED_ARMING_STATES): vol.All(cv.ensure_list, [vol.In(SUPPORTED_ARMING_STATES)]),
            vol.Optional(CONF_ALARM_ARMED_AWAY, default={}): _state_schema(AlarmControlPanelState.ARMED_AWAY),
            vol.Optional(CONF_ALARM_ARMED_HOME, default={}): _state_schema(AlarmControlPanelState.ARMED_HOME),
            vol.Optional(CONF_ALARM_ARMED_NIGHT, default={}): _state_schema(AlarmControlPanelState.ARMED_NIGHT),
            vol.Optional(CONF_ALARM_ARMED_VACATION, default={}): _state_schema(AlarmControlPanelState.ARMED_VACATION),
            vol.Optional(CONF_ALARM_ARMED_CUSTOM_BYPASS, default={}): _state_schema(AlarmControlPanelState.ARMED_CUSTOM_BYPASS),
            vol.Optional(CONF_ALARM_DISARMED, default={}): _state_schema(AlarmControlPanelState.DISARMED),
            vol.Optional(CONF_ALARM_TRIGGERED, default={}): _state_schema(AlarmControlPanelState.TRIGGERED),
        }),
        _state_validator,
    )
)

async def async_setup_platform(
    hass: HomeAssistant,
    config: ConfigType,
    async_add_entities: AddEntitiesCallback,
    discovery_info: Optional[DiscoveryInfoType] = None,
) -> None:
    """Set up the manual alarm platform."""
    async_add_entities([
        ManualAlarm(
            hass,
            config[CONF_NAME],
            config.get(CONF_UNIQUE_ID),
            config.get(CONF_CODE),
            config.get(CONF_CODE_TEMPLATE),
            config[CONF_CODE_ARM_REQUIRED],
            config[CONF_DISARM_AFTER_TRIGGER],
            config,
        )
    ])

class ManualAlarm(AlarmControlPanelEntity, RestoreEntity):
    """Representation of an alarm status."""
    _attr_should_poll = False

    def __init__(
        self,
        hass: HomeAssistant,
        name: str,
        unique_id: Optional[str],
        code: Optional[str],
        code_template: Optional[Template],
        code_arm_required: bool,
        disarm_after_trigger: bool,
        config: ConfigType,
    ) -> None:
        """Init the manual alarm panel."""
        self._state: AlarmControlPanelState = AlarmControlPanelState.DISARMED
        self._hass = hass
        self._attr_name = name
        self._attr_unique_id = unique_id
        self._code: Union[str, Template, None] = code_template or code or None
        self._attr_code_arm_required = code_arm_required
        self._disarm_after_trigger = disarm_after_trigger
        self._previous_state: AlarmControlPanelState = self._state
        self._state_ts: datetime.datetime = dt_util.utcnow()
        self._delay_time_by_state: Dict[AlarmControlPanelState, datetime.timedelta] = {
            state: config[state][CONF_DELAY_TIME] for state in SUPPORTED_PRETRIGGER_STATES
        }
        self._trigger_time_by_state: Dict[AlarmControlPanelState, datetime.timedelta] = {
            state: config[state][CONF_TRIGGER_TIME] for state in SUPPORTED_PRETRIGGER_STATES
        }
        self._arming_time_by_state: Dict[AlarmControlPanelState, datetime.timedelta] = {
            state: config[state][CONF_ARMING_TIME] for state in SUPPORTED_ARMING_STATES
        }
        self._attr_supported_features = AlarmControlPanelEntityFeature.TRIGGER
        for arming_state in config.get(CONF_ARMING_STATES, SUPPORTED_ARMING_STATES):
            self._attr_supported_features |= SUPPORTED_ARMING_STATE_TO_FEATURE[arming_state]

    @property
    def alarm_state(self) -> AlarmControlPanelState:
        """Return the state of the device."""
        if self._state == AlarmControlPanelState.TRIGGERED:
            if self._within_pending_time(self._state):
                return AlarmControlPanelState.PENDING
            trigger_time = self._trigger_time_by_state[self._previous_state]
            if self._state_ts + self._pending_time(self._state) + trigger_time < dt_util.utcnow():
                if self._disarm_after_trigger:
                    return AlarmControlPanelState.DISARMED
                self._state = self._previous_state
                return self._state
        if self._state in SUPPORTED_ARMING_STATES and self._within_arming_time(self._state):
            return AlarmControlPanelState.ARMING
        return self._state

    @property
    def _active_state(self) -> AlarmControlPanelState:
        """Get the current state."""
        if self.state in (AlarmControlPanelState.PENDING, AlarmControlPanelState.ARMING):
            return self._previous_state
        return self._state

    def _arming_time(self, state: AlarmControlPanelState) -> datetime.timedelta:
        """Get the arming time."""
        return self._arming_time_by_state[state]

    def _pending_time(self, state: AlarmControlPanelState) -> datetime.timedelta:
        """Get the pending time."""
        return self._delay_time_by_state[self._previous_state]

    def _within_arming_time(self, state: AlarmControlPanelState) -> bool:
        """Get if the action is in the arming time window."""
        return self._state_ts + self._arming_time(state) > dt_util.utcnow()

    def _within_pending_time(self, state: AlarmControlPanelState) -> bool:
        """Get if the action is in the pending time window."""
        return self._state_ts + self._pending_time(state) > dt_util.utcnow()

    @property
    def code_format(self) -> Optional[CodeFormat]:
        """Return one or more digits/characters."""
        if self._code is None:
            return None
        if isinstance(self._code, str) and self._code.isdigit():
            return CodeFormat.NUMBER
        return CodeFormat.TEXT

    async def async_alarm_disarm(self, code: Optional[str] = None) -> None:
        """Send disarm command."""
        self._async_validate_code(code, AlarmControlPanelState.DISARMED)
        self._state = AlarmControlPanelState.DISARMED
        self._state_ts = dt_util.utcnow()
        self.async_write_ha_state()

    async def async_alarm_arm_home(self, code: Optional[str] = None) -> None:
        """Send arm home command."""
        self._async_validate_code(code, AlarmControlPanelState.ARMED_HOME)
        self._async_update_state(AlarmControlPanelState.ARMED_HOME)

    async def async_alarm_arm_away(self, code: Optional[str] = None) -> None:
        """Send arm away command."""
        self._async_validate_code(code, AlarmControlPanelState.ARMED_AWAY)
        self._async_update_state(AlarmControlPanelState.ARMED_AWAY)

    async def async_alarm_arm_night(self, code: Optional[str] = None) -> None:
        """Send arm night command."""
        self._async_validate_code(code, AlarmControlPanelState.ARMED_NIGHT)
        self._async_update_state(AlarmControlPanelState.ARMED_NIGHT)

    async def async_alarm_arm_vacation(self, code: Optional[str] = None) -> None:
        """Send arm vacation command."""
        self._async_validate_code(code, AlarmControlPanelState.ARMED_VACATION)
        self._async_update_state(AlarmControlPanelState.ARMED_VACATION)

    async def async_alarm_arm_custom_bypass(self, code: Optional[str] = None) -> None:
        """Send arm custom bypass command."""
        self._async_validate_code(code, AlarmControlPanelState.ARMED_CUSTOM_BYPASS)
        self._async_update_state(AlarmControlPanelState.ARMED_CUSTOM_BYPASS)

    async def async_alarm_trigger(self, code: Optional[str] = None) -> None:
        """Send alarm trigger command."""
        if not self._trigger_time_by_state[self._active_state]:
            return
        self._async_update_state(AlarmControlPanelState.TRIGGERED)

    def _async_update_state(self, state: AlarmControlPanelState) -> None:
        """Update the state."""
        if self._state == state:
            return
        self._previous_state = self._state
        self._state = state
        self._state_ts = dt_util.utcnow()
        self.async_write_ha_state()
        self._async_set_state_update_events()

    def _async_set_state_update_events(self) -> None:
        """Set up state update events."""
        state = self._state
        if state == AlarmControlPanelState.TRIGGERED:
            pending_time = self._pending_time(state)
            async_track_point_in_time(
                self._hass,
                self.async_scheduled_update,
                self._state_ts + pending_time,
            )
            trigger_time = self._trigger_time_by_state[self._previous_state]
            async_track_point_in_time(
                self._hass,
                self.async_scheduled_update,
                self._state_ts + pending_time + trigger_time,
            )
        elif state in SUPPORTED_ARMING_STATES:
            arming_time = self._arming_time(state)
            if arming_time:
                async_track_point_in_time(
                    self._hass,
                    self.async_scheduled_update,
                    self._state_ts + arming_time,
                )

    def _async_validate_code(self, code: Optional[str], state: AlarmControlPanelState) -> None:
        """Validate given code."""
        if state != AlarmControlPanelState.DISARMED and (not self.code_arm_required) or self._code is None:
            return
        alarm_code: str
        if isinstance(self._code, str):
            alarm_code = self._code
        else:
            alarm_code = cast(str, self._code.async_render(parse_result=False, from_state=self._state, to_state=state))
        if not alarm_code or code == alarm_code:
            return
        raise ServiceValidationError(
            'Invalid alarm code provided',
            translation_domain=DOMAIN,
            translation_key='invalid_code',
        )

    @property
    def extra_state_attributes(self) -> Dict[str, Optional[AlarmControlPanelState]]:
        """Return the state attributes."""
        if self.state in (AlarmControlPanelState.PENDING, AlarmControlPanelState.ARMING):
            prev_state = self._previous_state
            state = self._state
        elif self.state == AlarmControlPanelState.TRIGGERED:
            prev_state = self._previous_state
            state = None
        else:
            prev_state = None
            state = None
        return {ATTR_PREVIOUS_STATE: prev_state, ATTR_NEXT_STATE: state}

    @callback
    def async_scheduled_update(self, now: datetime.datetime) -> None:
        """Update state at a scheduled point in time."""
        self.async_write_ha_state()

    async def async_added_to_hass(self) -> None:
        """Run when entity about to be added to hass."""
        await super().async_added_to_hass()
        if (state := (await self.async_get_last_state())):
            self._state_ts = state.last_updated
            if (next_state := state.attributes.get(ATTR_NEXT_STATE)):
                self._state = AlarmControlPanelState(next_state)
            else:
                self._state = AlarmControlPanelState(state.state