from typing import Any

# === Internal dependency: homeassistant.components.rachio.const ===
DOMAIN = 'rachio'
KEY_DEVICES = 'devices'
KEY_ENABLED = 'enabled'
KEY_EXTERNAL_ID = 'externalId'
KEY_ID = 'id'
KEY_NAME = 'name'
KEY_MODEL = 'model'
KEY_STATUS = 'status'
KEY_SERIAL_NUMBER = 'serialNumber'
KEY_MAC_ADDRESS = 'macAddress'
KEY_USERNAME = 'username'
KEY_ZONES = 'zones'
KEY_SCHEDULES = 'scheduleRules'
KEY_FLEX_SCHEDULES = 'flexScheduleRules'
KEY_BASE_STATIONS = 'baseStations'
MODEL_GENERATION_1 = 'GENERATION1'
SERVICE_PAUSE_WATERING = 'pause_watering'
SERVICE_RESUME_WATERING = 'resume_watering'
SERVICE_STOP_WATERING = 'stop_watering'
LISTEN_EVENT_TYPES = ['DEVICE_STATUS_EVENT', 'ZONE_STATUS_EVENT', 'RAIN_DELAY_EVENT', 'RAIN_SENSOR_DETECTION_EVENT', 'SCHEDULE_STATUS_EVENT']
WEBHOOK_CONST_ID = 'homeassistant.rachio:'

# === Internal dependency: homeassistant.components.rachio.coordinator ===
class RachioUpdateCoordinator(DataUpdateCoordinator[dict[str, Any]]): ...

# === Internal dependency: homeassistant.const ===
EVENT_HOMEASSISTANT_STOP = EventType(...)

# === Internal dependency: homeassistant.exceptions ===
class ConfigEntryNotReady(IntegrationError): ...
class ConfigEntryAuthFailed(IntegrationError): ...

# === Internal dependency: homeassistant.helpers.config_validation ===
def string(value): ...
positive_int = vol.All(...)

# === Internal dependency: homeassistant.util.event_type ===
class EventType(Generic[_DataT]): ...

# === Third-party dependency: voluptuous ===
# Used symbols: All, Optional, Schema