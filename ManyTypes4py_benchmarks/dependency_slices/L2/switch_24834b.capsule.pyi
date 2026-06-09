from typing import Any

# === Internal dependency: homeassistant.components.rachio.const ===
DOMAIN: str
CONF_MANUAL_RUN_MINS: str
DEFAULT_MANUAL_RUN_MINS: int
SLOPE_FLAT: str
SLOPE_SLIGHT: str
SLOPE_MODERATE: str
SLOPE_STEEP: str
KEY_DEVICE_ID: str
KEY_IMAGE_URL: str
KEY_ENABLED: str
KEY_ID: str
KEY_NAME: str
KEY_ON: str
KEY_DURATION: str
KEY_RAIN_DELAY: str
KEY_RAIN_DELAY_END: str
KEY_SUBTYPE: str
KEY_SUMMARY: str
KEY_TYPE: str
KEY_ZONE_ID: str
KEY_ZONE_NUMBER: str
KEY_SCHEDULE_ID: str
KEY_CUSTOM_SHADE: str
KEY_CUSTOM_CROP: str
KEY_CUSTOM_SLOPE: str
KEY_REPORTED_STATE: str
KEY_STATE: str
KEY_CURRENT_STATUS: str
SCHEDULE_TYPE_FIXED: str
SCHEDULE_TYPE_FLEX: str
SERVICE_SET_ZONE_MOISTURE: str
SERVICE_START_WATERING: str
SERVICE_START_MULTIPLE_ZONES: str
SIGNAL_RACHIO_CONTROLLER_UPDATE: Any
SIGNAL_RACHIO_RAIN_DELAY_UPDATE: Any
SIGNAL_RACHIO_ZONE_UPDATE: Any
SIGNAL_RACHIO_SCHEDULE_UPDATE: Any

# === Internal dependency: homeassistant.components.rachio.device ===
class RachioPerson: ...

# === Internal dependency: homeassistant.components.rachio.entity ===
class RachioDevice(Entity):
    ...
class RachioHoseTimerEntity(CoordinatorEntity[RachioUpdateCoordinator]):
    def available(self) -> bool: ...
    def _update_attr(self) -> None: ...

# === Internal dependency: homeassistant.components.rachio.webhooks ===
SUBTYPE_SLEEP_MODE_ON: str
SUBTYPE_SLEEP_MODE_OFF: str
SUBTYPE_RAIN_DELAY_ON: str
SUBTYPE_RAIN_DELAY_OFF: str
SUBTYPE_SCHEDULE_STARTED: str
SUBTYPE_SCHEDULE_STOPPED: str
SUBTYPE_SCHEDULE_COMPLETED: str
SUBTYPE_ZONE_STARTED: str
SUBTYPE_ZONE_STOPPED: str
SUBTYPE_ZONE_COMPLETED: str
SUBTYPE_ZONE_PAUSED: str

# === Internal dependency: homeassistant.components.switch ===
class SwitchEntity(ToggleEntity):
    ...

# === Internal dependency: homeassistant.const ===
ATTR_ID: Final
ATTR_ENTITY_ID: Final

# === Internal dependency: homeassistant.core ===
def callback(func: _CallableT) -> _CallableT: ...

# === Internal dependency: homeassistant.exceptions ===
class HomeAssistantError(Exception): ...

# === Internal dependency: homeassistant.helpers.config_validation ===
def entity_ids(value: str | list) -> list[str]: ...
def ensure_list_csv(value: Any) -> list: ...
positive_int: All

# === Internal dependency: homeassistant.helpers.dispatcher ===
def async_dispatcher_connect(hass: HomeAssistant, signal: SignalType[*_Ts,], target: Callable[[*_Ts], Any]) -> Callable[[], None]: ...
def async_dispatcher_connect(hass: HomeAssistant, signal: str, target: Callable[..., Any]) -> Callable[[], None]: ...
def async_dispatcher_connect(hass: HomeAssistant, signal: SignalType[*_Ts,] | str, target: Callable[[*_Ts], Any] | Callable[..., Any]) -> Callable[[], None]: ...

# === Internal dependency: homeassistant.helpers.entity ===
class Entity: ...

# === Internal dependency: homeassistant.helpers.entity_platform ===
def async_get_current_platform() -> EntityPlatform: ...

# === Internal dependency: homeassistant.helpers.event ===
def async_track_point_in_utc_time(hass: HomeAssistant, action: HassJob[[datetime], Coroutine[Any, Any, None] | None] | Callable[[datetime], Coroutine[Any, Any, None] | None], point_in_time: datetime) -> CALLBACK_TYPE: ...

# === Internal dependency: homeassistant.util.dt ===
def now(time_zone: dt.tzinfo | None = ...) -> dt.datetime: ...
def as_timestamp(dt_value: dt.datetime | str) -> float: ...
def parse_datetime(dt_str: str) -> dt.datetime | None: ...
def parse_datetime(dt_str: str, *, raise_on_error: Literal[True]) -> dt.datetime: ...
def parse_datetime(dt_str: str, *, raise_on_error: Literal[False]) -> dt.datetime | None: ...
def parse_datetime(dt_str: str, *, raise_on_error: bool = ...) -> dt.datetime | None: ...
utc_from_timestamp: partial

# === Third-party dependency: voluptuous ===
# Used symbols: All, Optional, Required, Schema