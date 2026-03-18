from typing import Any, Awaitable, Dict, List, Optional
import datetime
from homeassistant.components.alarm_control_panel import (
    AlarmControlPanelEntity,
    AlarmControlPanelEntityFeature,
    AlarmControlPanelState,
    CodeFormat,
)
from homeassistant.core import Event, HomeAssistant
from homeassistant.helpers.entity_platform import AddEntitiesCallback
from homeassistant.helpers.typing import ConfigType, DiscoveryInfoType

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
SUPPORTED_STATES: List[AlarmControlPanelState]
SUPPORTED_PRETRIGGER_STATES: List[AlarmControlPanelState]
SUPPORTED_PENDING_STATES: List[AlarmControlPanelState]
ATTR_PRE_PENDING_STATE: str
ATTR_POST_PENDING_STATE: str
PLATFORM_SCHEMA: Any

def _state_validator(config: Any) -> Any: ...
def _state_schema(state: Any) -> Any: ...

async def async_setup_platform(
    hass: HomeAssistant,
    config: ConfigType,
    add_entities: AddEntitiesCallback,
    discovery_info: Optional[DiscoveryInfoType] = ...,
) -> None: ...

class ManualMQTTAlarm(AlarmControlPanelEntity):
    _attr_should_poll: bool
    _attr_supported_features: AlarmControlPanelEntityFeature

    def __init__(
        self,
        hass: HomeAssistant,
        name: str,
        code: Optional[str],
        code_template: Any,
        disarm_after_trigger: bool,
        state_topic: str,
        command_topic: str,
        qos: Optional[int],
        code_arm_required: bool,
        payload_disarm: str,
        payload_arm_home: str,
        payload_arm_away: str,
        payload_arm_night: str,
        payload_arm_vacation: str,
        payload_arm_custom_bypass: str,
        config: Any,
    ) -> None: ...
    @property
    def alarm_state(self) -> AlarmControlPanelState: ...
    @property
    def _active_state(self) -> AlarmControlPanelState: ...
    def _pending_time(self, state: AlarmControlPanelState) -> datetime.timedelta: ...
    def _within_pending_time(self, state: AlarmControlPanelState) -> bool: ...
    @property
    def code_format(self) -> Optional[CodeFormat]: ...
    async def async_alarm_disarm(self, code: Optional[Any] = ...) -> None: ...
    async def async_alarm_arm_home(self, code: Optional[Any] = ...) -> None: ...
    async def async_alarm_arm_away(self, code: Optional[Any] = ...) -> None: ...
    async def async_alarm_arm_night(self, code: Optional[Any] = ...) -> None: ...
    async def async_alarm_arm_vacation(self, code: Optional[Any] = ...) -> None: ...
    async def async_alarm_arm_custom_bypass(self, code: Optional[Any] = ...) -> None: ...
    async def async_alarm_trigger(self, code: Optional[Any] = ...) -> None: ...
    def _async_update_state(self, state: AlarmControlPanelState) -> None: ...
    def _async_validate_code(self, code: Any, state: AlarmControlPanelState) -> None: ...
    @property
    def extra_state_attributes(self) -> Dict[str, Any]: ...
    def async_scheduled_update(self, now: Any) -> None: ...
    async def async_added_to_hass(self) -> None: ...
    async def _async_state_changed_listener(self, event: Event) -> None: ...