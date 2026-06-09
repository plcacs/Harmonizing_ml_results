from typing import Any

# === Internal dependency: homeassistant.components.rachio.const ===
DOMAIN: str
KEY_DEVICES: str
KEY_ENABLED: str
KEY_EXTERNAL_ID: str
KEY_ID: str
KEY_NAME: str
KEY_MODEL: str
KEY_STATUS: str
KEY_SERIAL_NUMBER: str
KEY_MAC_ADDRESS: str
KEY_USERNAME: str
KEY_ZONES: str
KEY_SCHEDULES: str
KEY_FLEX_SCHEDULES: str
KEY_BASE_STATIONS: str
MODEL_GENERATION_1: str
SERVICE_PAUSE_WATERING: str
SERVICE_RESUME_WATERING: str
SERVICE_STOP_WATERING: str
LISTEN_EVENT_TYPES: Any
WEBHOOK_CONST_ID: str

# === Internal dependency: homeassistant.components.rachio.coordinator ===
class RachioUpdateCoordinator(DataUpdateCoordinator[dict[str, Any]]): ...

# === Internal dependency: homeassistant.const ===
EVENT_HOMEASSISTANT_STOP: EventType[NoEventData]

# === Internal dependency: homeassistant.exceptions ===
class ConfigEntryNotReady(IntegrationError): ...
class ConfigEntryAuthFailed(IntegrationError): ...

# === Internal dependency: homeassistant.helpers.config_validation ===
def string(value: Any) -> str: ...
positive_int: All

# === Third-party dependency: voluptuous ===
# Used symbols: All, Optional, Schema