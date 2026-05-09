"""Support for manual alarms controllable via MQTT."""
from __future__ import annotations
import datetime
import logging
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
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
from homeassistant.helpers.config_validation import time_period
from homeassistant.helpers.typing import ConfigType

_LOGGER = logging.getLogger(__name__)
CONF_CODE_TEMPLATE = 'code_template'
CONF_CODE_ARM_REQUIRED = 'code_arm_required'
CONF_PAYLOAD_DISARM = 'payload_disarm'
CONF_PAYLOAD_ARM_HOME = 'payload_arm_home'
CONF_PAYLOAD_ARM_AWAY = 'payload_arm_away'
CONF_PAYLOAD_ARM_NIGHT = 'payload_arm_night'
CONF_PAYLOAD_ARM_VACATION = 'payload_arm_vacation'
CONF_PAYLOAD_ARM_CUSTOM_BYPASS = 'payload_arm_custom_bypass'
CONF_ALARM_ARMED_AWAY = 'armed_away'
CONF_ALARM_ARMED_CUSTOM_BYPASS = 'armed_custom_bypass'
CONF_ALARM_ARMED_HOME = 'armed_home'
CONF_ALARM_ARMED_NIGHT = 'armed_night'
CONF_ALARM_ARMED_VACATION = 'armed_vacation'
CONF_ALARM_DISARMED = 'disarmed'
CONF_ALARM_PENDING = 'pending'
CONF_ALARM_TRIGGERED = 'triggered'
DEFAULT_ALARM_NAME = 'HA Alarm'
DEFAULT_DELAY_TIME = datetime.timedelta(seconds=0)
DEFAULT_PENDING_TIME = datetime.timedelta(seconds=60)
DEFAULT_TRIGGER_TIME = datetime.timedelta(seconds=120)
DEFAULT_DISARM_AFTER_TRIGGER = False
DEFAULT_ARM_AWAY = 'ARM_AWAY'
DEFAULT_ARM_HOME = 'ARM_HOME'
DEFAULT_ARM_NIGHT = 'ARM_NIGHT'
DEFAULT_ARM_VACATION = 'ARM_VACATION'
DEFAULT_ARM_CUSTOM_BYPASS = 'ARM_CUSTOM_BYPASS'
DEFAULT_DISARM = 'DISARM'
SUPPORTED_STATES = [
    AlarmControlPanelState.DISARMED,
    AlarmControlPanelState.ARMED_AWAY,
    AlarmControlPanelState.ARMED_HOME,
    AlarmControlPanelState.ARMED_NIGHT,
    AlarmControlPanelState.ARMED_VACATION,
    AlarmControlPanelState.ARMED_CUSTOM_BYPASS,
    AlarmControlPanelState.TRIGGERED,
]
SUPPORTED_PRETRIGGER_STATES = [
    state for state in SUPPORTED_STATES if state != AlarmControlPanelState.TRIGGERED
]
SUPPORTED_PENDING_STATES = [
    state for state in SUPPORTED_STATES if state != AlarmControlPanelState.DISARMED
]
ATTR_PRE_PENDING_STATE = 'pre_pending_state'
ATTR_POST_PENDING_STATE = 'post_pending_state'

PLATFORM_SCHEMA: vol.Schema = vol.Schema(...)

async def async_setup_platform(
    hass: HomeAssistant,
    config: ConfigType,
    add_entities: AddEntitiesCallback,
    discovery_info: Optional[DiscoveryInfoType] = None,
) -> None:
    ...

class ManualMQTTAlarm(AlarmControlPanelEntity):
    _attr_should_poll = False
    _attr_supported_features = (
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
        code: Optional[str],
        code_template: Optional[str],
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
        ...

    @property
    def alarm_state(self) -> AlarmControlPanelState:
        ...

    @property
    def _active_state(self) -> AlarmControlPanelState:
        ...

    def _pending_time(self, state: AlarmControlPanelState) -> datetime.timedelta:
        ...

    def _within_pending_time(self, state: AlarmControlPanelState) -> bool:
        ...

    @property
    def code_format(self) -> Optional[CodeFormat]:
        ...

    async def async_alarm_disarm(self, code: Optional[str] = None) -> None:
        ...

    async def async_alarm_arm_home(self, code: Optional[str] = None) -> None:
        ...

    async def async_alarm_arm_away(self, code: Optional[str] = None) -> None:
        ...

    async def async_alarm_arm_night(self, code: Optional[str] = None) -> None:
        ...

    async def async_alarm_arm_vacation(self, code: Optional[str] = None) -> None:
        ...

    async def async_alarm_arm_custom_bypass(self, code: Optional[str] = None) -> None:
        ...

    async def async_alarm_trigger(self, code: Optional[str] = None) -> None:
        ...

    def _async_update_state(self, state: AlarmControlPanelState) -> None:
        ...

    def _async_validate_code(self, code: Optional[str], state: AlarmControlPanelState) -> None:
        ...

    @property
    def extra_state_attributes(self) -> Dict[str, AlarmControlPanelState]:
        ...

    @callback
    def async_scheduled_update(self, now: datetime.datetime) -> None:
        ...

    async def async_added_to_hass(self) -> None:
        ...

    async def _async_state_changed_listener(self, event: Event) -> None:
        ...