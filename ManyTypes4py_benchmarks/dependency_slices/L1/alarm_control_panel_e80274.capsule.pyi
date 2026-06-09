from typing import Any

# === Internal dependency: homeassistant.components.alarm_control_panel ===
class AlarmControlPanelEntity(Entity):
    def state_attributes(self): ...
from .const import AlarmControlPanelEntityFeature
from .const import CodeFormat

# === Internal dependency: homeassistant.components.mqtt ===
from .client import async_publish
from .client import async_subscribe
from .const import CONF_COMMAND_TOPIC
from .const import CONF_QOS
from .const import CONF_STATE_TOPIC
from .util import async_wait_for_mqtt_client
from .util import valid_publish_topic
from .util import valid_subscribe_topic

# === Internal dependency: homeassistant.const ===
CONF_CODE = 'code'
CONF_DELAY_TIME = 'delay_time'
CONF_DISARM_AFTER_TRIGGER = 'disarm_after_trigger'
CONF_NAME = 'name'
CONF_PENDING_TIME = 'pending_time'
CONF_PLATFORM = 'platform'
CONF_TRIGGER_TIME = 'trigger_time'
STATE_ALARM_DISARMED = 'disarmed'
STATE_ALARM_ARMED_HOME = 'armed_home'
STATE_ALARM_ARMED_AWAY = 'armed_away'
STATE_ALARM_ARMED_NIGHT = 'armed_night'
STATE_ALARM_ARMED_VACATION = 'armed_vacation'
STATE_ALARM_ARMED_CUSTOM_BYPASS = 'armed_custom_bypass'
STATE_ALARM_PENDING = 'pending'
STATE_ALARM_TRIGGERED = 'triggered'

# === Internal dependency: homeassistant.core ===
def callback(func): ...

# === Internal dependency: homeassistant.exceptions ===
class HomeAssistantError(Exception): ...

# === Internal dependency: homeassistant.helpers.config_validation ===
def has_at_least_one_key(*keys): ...
def boolean(value): ...
def time_period_str(value): ...
def time_period_seconds(value): ...
def positive_timedelta(value): ...
def string(value): ...
def template(value): ...
_TIME_PERIOD_DICT_KEYS = ('days', 'hours', 'minutes', 'seconds', 'milliseconds')
time_period_dict = vol.All(...)
time_period = vol.Any(...)

# === Internal dependency: homeassistant.helpers.event ===
def async_track_state_change_event(hass, entity_ids, action, job_type=...): ...
def async_track_point_in_time(hass, action, point_in_time): ...

# === Internal dependency: homeassistant.util.dt ===
UTC = dt.UTC
utcnow = partial(...)

# === Third-party dependency: voluptuous ===
# Used symbols: All, Any, Exclusive, Optional, Required, Schema