from typing import Any

# === Third-party dependency: certifi ===
# Used symbols: where

# === Internal dependency: homeassistant.components.mqtt ===
# re-export: from homeassistant.const import CONF_DISCOVERY
# re-export: from .client import async_publish
# re-export: from .client import async_subscribe
# re-export: from .client import publish
# re-export: from .client import subscribe
# re-export: from .const import ATTR_PAYLOAD
# re-export: from .const import ATTR_QOS
# re-export: from .const import ATTR_RETAIN
# re-export: from .const import ATTR_TOPIC
# re-export: from .const import CONF_BIRTH_MESSAGE
# re-export: from .const import CONF_BROKER
# re-export: from .const import CONF_CERTIFICATE
# re-export: from .const import CONF_WILL_MESSAGE
# re-export: from .const import DOMAIN
# re-export: from .models import MqttCommandTemplate

# === Internal dependency: homeassistant.components.mqtt.client ===
RECONNECT_INTERVAL_SECONDS: int

# === Internal dependency: homeassistant.components.mqtt.models ===
class ReceiveMessage:
    ...

# === Internal dependency: homeassistant.config_entries ===
class ConfigEntryState(Enum): ...
class ConfigEntryDisabler(StrEnum): ...

# === Internal dependency: homeassistant.const ===
CONF_PROTOCOL: Final
EVENT_HOMEASSISTANT_STARTED: EventType[NoEventData]
EVENT_HOMEASSISTANT_STOP: EventType[NoEventData]
class UnitOfTemperature(StrEnum): ...

# === Internal dependency: homeassistant.core ===
def callback(func: _CallableT) -> _CallableT: ...
class CoreState(enum.Enum): ...

# === Internal dependency: homeassistant.exceptions ===
class HomeAssistantError(Exception): ...

# === Internal dependency: homeassistant.util.dt ===
utcnow: partial

# === Unresolved dependency: paho.mqtt.client ===
# Used unresolved symbols: MQTT_ERR_CONN_LOST, MQTT_ERR_SUCCESS, WebsocketWrapper

# === Third-party dependency: pytest ===
# Used symbols: fixture, mark, raises

# === Internal dependency: tests.common ===
def async_fire_mqtt_message(hass: HomeAssistant, topic: str, payload: bytes | str, qos: int = ..., retain: bool = ...) -> None: ...
def async_fire_time_changed(hass: HomeAssistant, datetime_: datetime | None = ..., fire_all: bool = ...) -> None: ...
class MockConfigEntry(config_entries.ConfigEntry):
    def __init__(self, *, data = ..., disabled_by = ..., domain = ..., entry_id = ..., minor_version = ..., options = ..., pref_disable_new_entities = ..., pref_disable_polling = ..., reason = ..., source = ..., state = ..., title = ..., unique_id = ..., version = ...) -> None: ...
    def add_to_hass(self, hass: HomeAssistant) -> None: ...

# === Internal dependency: tests.components.mqtt.conftest ===
ENTRY_DEFAULT_BIRTH_MESSAGE: Any

# === Internal dependency: tests.components.mqtt.test_common ===
def help_all_subscribe_calls(mqtt_client_mock: MqttMockPahoClient) -> list[Any]: ...