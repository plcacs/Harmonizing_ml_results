from typing import Any

# === Internal dependency: homeassistant.components.alarm_control_panel ===
class AlarmControlPanelEntity(Entity):
    def state_attributes(self) -> dict[str, Any] | None: ...
# re-export: from .const import AlarmControlPanelEntityFeature
# re-export: from .const import CodeFormat

# === Internal dependency: homeassistant.components.mqtt ===
# re-export: from .client import async_publish
# re-export: from .client import async_subscribe
# re-export: from .const import CONF_COMMAND_TOPIC
# re-export: from .const import CONF_QOS
# re-export: from .const import CONF_STATE_TOPIC
# re-export: from .util import async_wait_for_mqtt_client
# re-export: from .util import valid_publish_topic
# re-export: from .util import valid_subscribe_topic

# === Internal dependency: homeassistant.const ===
CONF_CODE: Final
CONF_DELAY_TIME: Final
CONF_DISARM_AFTER_TRIGGER: Final
CONF_NAME: Final
CONF_PENDING_TIME: Final
CONF_PLATFORM: Final
CONF_TRIGGER_TIME: Final
STATE_ALARM_DISARMED: Final
STATE_ALARM_ARMED_HOME: Final
STATE_ALARM_ARMED_AWAY: Final
STATE_ALARM_ARMED_NIGHT: Final
STATE_ALARM_ARMED_VACATION: Final
STATE_ALARM_ARMED_CUSTOM_BYPASS: Final
STATE_ALARM_PENDING: Final
STATE_ALARM_TRIGGERED: Final

# === Internal dependency: homeassistant.core ===
def callback(func: _CallableT) -> _CallableT: ...

# === Internal dependency: homeassistant.exceptions ===
class HomeAssistantError(Exception): ...

# === Internal dependency: homeassistant.helpers.config_validation ===
def boolean(value: Any) -> bool: ...
def positive_timedelta(value: timedelta) -> timedelta: ...
def string(value: Any) -> str: ...
def template(value: Any | None) -> template_helper.Template: ...
time_period: Any

# === Internal dependency: homeassistant.helpers.event ===
def async_track_state_change_event(hass: HomeAssistant, entity_ids: str | Iterable[str], action: Callable[[Event[EventStateChangedData]], Any], job_type: HassJobType | None = ...) -> CALLBACK_TYPE: ...
def async_track_point_in_time(hass: HomeAssistant, action: HassJob[[datetime], Coroutine[Any, Any, None] | None] | Callable[[datetime], Coroutine[Any, Any, None] | None], point_in_time: datetime) -> CALLBACK_TYPE: ...

# === Internal dependency: homeassistant.util.dt ===
utcnow: partial

# === Third-party dependency: voluptuous ===
# Used symbols: All, Any, Exclusive, Optional, Required, Schema