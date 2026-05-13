from __future__ import annotations
import datetime
from typing import Any, Dict, Optional
import voluptuous as vol
from homeassistant.components.alarm_control_panel import (
    AlarmControlPanelEntity,
    AlarmControlPanelEntityFeature,
    AlarmControlPanelState,
    CodeFormat,
)
from homeassistant.core import Event, HomeAssistant, callback
from homeassistant.exceptions import HomeAssistantError
from homeassistant.helpers.typing import ConfigType, DiscoveryInfoType

CONF_CODE: str
CONF_DELAY_TIME: str
CONF_DISARM_AFTER_TRIGGER: str
CONF_NAME: str
CONF_PENDING_TIME: str
CONF_PLATFORM: str
CONF_TRIGGER_TIME: str
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

def _state_validator(config: Dict[str, Any]) -> Dict[str, Any]: ...
def _state_schema(state: AlarmControlPanelState) -> vol.Schema: ...
PLATFORM_SCHEMA: vol.Schema

async def async_setup_platform(
    hass: HomeAssistant,
    config: ConfigType,
    add_entities: AddEntitiesCallback,
    discovery_info: Optional[DiscoveryInfoType] = None,
) -> None: ...

class ManualMQTTAlarm(AlarmControlPanelEntity):
    _attr_should_poll: bool
    _attr_supported_features: AlarmControlPanelEntityFeature
    _state: AlarmControlPanelState
    _hass: HomeAssistant
    _code: Any
    _disarm_after_trigger: bool
    _previous_state: AlarmControlPanelState
    _state_ts: datetime.datetime
    _delay_time_by_state: Dict[AlarmControlPanelState, datetime.timedelta]
    _trigger_time_by_state: Dict[AlarmControlPanelState, datetime.timedelta]
    _pending_time_by_state: Dict[AlarmControlPanelState, datetime.timedelta]
    _state_topic: str
    _command_topic: str
    _qos: int
    _attr_code_arm_required: bool
    _payload_disarm: str
    _payload_arm_home: str
    _payload_arm_away: str
    _payload_arm_night: str
    _payload_arm_vacation: str
    _payload_arm_custom_bypass: str

    def __init__(
        self,
        hass: HomeAssistant,
        name: str,
        code: Optional[str],
        code_template: Optional[Any],
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
        config: Dict[str, Any],
    ) -> None: ...

    @property
    def alarm_state(self) -> AlarmControlPanelState: ...

    @property
    def _active_state(self) -> AlarmControlPanelState: ...

    def _pending_time(self, state: AlarmControlPanelState) -> datetime.timedelta: ...
    def _within_pending_time(self, state: AlarmControlPanelState) -> bool: ...

    @property
    def code_format(self) -> Optional[CodeFormat]: ...

    async def async_alarm_disarm(self, code: Optional[str] = None) -> None: ...
    async def async_alarm_arm_home(self, code: Optional[str] = None) -> None: ...
    async def async_alarm_arm_away(self, code: Optional[str] = None) -> None: ...
    async def async_alarm_arm_night(self, code: Optional[str] = None) -> None: ...
    async def async_alarm_arm_vacation(self, code: Optional[str] = None) -> None: ...
    async def async_alarm_arm_custom_bypass(self, code: Optional[str] = None) -> None: ...
    async def async_alarm_trigger(self, code: Optional[str] = None) -> None: ...

    def _async_update_state(self, state: AlarmControlPanelState) -> None: ...
    def _async_validate_code(self, code: Optional[str], state: AlarmControlPanelState) -> None: ...

    @property
    def extra_state_attributes(self) -> Dict[str, AlarmControlPanelState]: ...

    @callback
    def async_scheduled_update(self, now: datetime.datetime) -> None: ...

    async def async_added_to_hass(self) -> None: ...
    async def _async_state_changed_listener(self, event: Event) -> None: ...