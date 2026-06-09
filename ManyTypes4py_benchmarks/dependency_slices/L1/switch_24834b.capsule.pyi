# === Internal dependency: homeassistant.components.rachio.const ===
DOMAIN = 'rachio'
CONF_MANUAL_RUN_MINS = 'manual_run_mins'
DEFAULT_MANUAL_RUN_MINS = 10
SLOPE_FLAT = 'ZERO_THREE'
SLOPE_SLIGHT = 'FOUR_SIX'
SLOPE_MODERATE = 'SEVEN_TWELVE'
SLOPE_STEEP = 'OVER_TWELVE'
KEY_DEVICE_ID = 'deviceId'
KEY_IMAGE_URL = 'imageUrl'
KEY_ENABLED = 'enabled'
KEY_ID = 'id'
KEY_NAME = 'name'
KEY_ON = 'on'
KEY_DURATION = 'totalDuration'
KEY_RAIN_DELAY = 'rainDelayExpirationDate'
KEY_RAIN_DELAY_END = 'endTime'
KEY_SUBTYPE = 'subType'
KEY_SUMMARY = 'summary'
KEY_TYPE = 'type'
KEY_ZONE_ID = 'zoneId'
KEY_ZONE_NUMBER = 'zoneNumber'
KEY_SCHEDULE_ID = 'scheduleId'
KEY_CUSTOM_SHADE = 'customShade'
KEY_CUSTOM_CROP = 'customCrop'
KEY_CUSTOM_SLOPE = 'customSlope'
KEY_REPORTED_STATE = 'reportedState'
KEY_STATE = 'state'
KEY_CURRENT_STATUS = 'lastWateringAction'
SCHEDULE_TYPE_FIXED = 'FIXED'
SCHEDULE_TYPE_FLEX = 'FLEX'
SERVICE_SET_ZONE_MOISTURE = 'set_zone_moisture_percent'
SERVICE_START_WATERING = 'start_watering'
SERVICE_START_MULTIPLE_ZONES = 'start_multiple_zone_schedule'
SIGNAL_RACHIO_UPDATE = f'{DOMAIN}_update'
SIGNAL_RACHIO_CONTROLLER_UPDATE = f'{SIGNAL_RACHIO_UPDATE}_controller'
SIGNAL_RACHIO_RAIN_DELAY_UPDATE = f'{SIGNAL_RACHIO_UPDATE}_rain_delay'
SIGNAL_RACHIO_ZONE_UPDATE = f'{SIGNAL_RACHIO_UPDATE}_zone'
SIGNAL_RACHIO_SCHEDULE_UPDATE = f'{SIGNAL_RACHIO_UPDATE}_schedule'

# === Internal dependency: homeassistant.components.rachio.device ===
class RachioPerson: ...

# === Internal dependency: homeassistant.components.rachio.entity ===
class RachioDevice(Entity):
    ...
class RachioHoseTimerEntity(CoordinatorEntity[RachioUpdateCoordinator]):
    def available(self): ...
    def _update_attr(self): ...

# === Internal dependency: homeassistant.components.rachio.webhooks ===
SUBTYPE_SLEEP_MODE_ON = 'SLEEP_MODE_ON'
SUBTYPE_SLEEP_MODE_OFF = 'SLEEP_MODE_OFF'
SUBTYPE_RAIN_DELAY_ON = 'RAIN_DELAY_ON'
SUBTYPE_RAIN_DELAY_OFF = 'RAIN_DELAY_OFF'
SUBTYPE_SCHEDULE_STARTED = 'SCHEDULE_STARTED'
SUBTYPE_SCHEDULE_STOPPED = 'SCHEDULE_STOPPED'
SUBTYPE_SCHEDULE_COMPLETED = 'SCHEDULE_COMPLETED'
SUBTYPE_ZONE_STARTED = 'ZONE_STARTED'
SUBTYPE_ZONE_STOPPED = 'ZONE_STOPPED'
SUBTYPE_ZONE_COMPLETED = 'ZONE_COMPLETED'
SUBTYPE_ZONE_PAUSED = 'ZONE_PAUSED'

# === Internal dependency: homeassistant.components.switch ===
class SwitchEntity(ToggleEntity):
    ...

# === Internal dependency: homeassistant.const ===
ATTR_ID = 'id'
ATTR_ENTITY_ID = 'entity_id'

# === Internal dependency: homeassistant.core ===
def callback(func): ...

# === Internal dependency: homeassistant.exceptions ===
class HomeAssistantError(Exception): ...

# === Internal dependency: homeassistant.helpers.config_validation ===
def entity_ids(value): ...
def ensure_list_csv(value): ...
positive_int = vol.All(...)

# === Internal dependency: homeassistant.helpers.dispatcher ===
def async_dispatcher_connect(hass, signal, target): ...

# === Internal dependency: homeassistant.helpers.entity ===
class Entity: ...

# === Internal dependency: homeassistant.helpers.entity_platform ===
def async_get_current_platform(): ...

# === Internal dependency: homeassistant.helpers.event ===
def async_track_point_in_utc_time(hass, action, point_in_time): ...

# === Internal dependency: homeassistant.util.dt ===
def now(time_zone=...): ...
def as_timestamp(dt_value): ...
def parse_datetime(dt_str): ...
def parse_datetime(dt_str, *, raise_on_error): ...
def parse_datetime(dt_str, *, raise_on_error=...): ...
UTC = dt.UTC
utc_from_timestamp = partial(...)

# === Third-party dependency: voluptuous ===
# Used symbols: All, Optional, Required, Schema