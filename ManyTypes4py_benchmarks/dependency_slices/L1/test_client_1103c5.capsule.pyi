# === Third-party dependency: certifi ===
# Used symbols: where

# === Internal dependency: homeassistant.components.mqtt ===
from homeassistant.const import CONF_DISCOVERY
from .client import async_publish
from .client import async_subscribe
from .client import publish
from .client import subscribe
from .const import ATTR_PAYLOAD
from .const import ATTR_QOS
from .const import ATTR_RETAIN
from .const import ATTR_TOPIC
from .const import CONF_BIRTH_MESSAGE
from .const import CONF_BROKER
from .const import CONF_CERTIFICATE
from .const import CONF_WILL_MESSAGE
from .const import DOMAIN
from .models import MqttCommandTemplate

# === Internal dependency: homeassistant.components.mqtt.client ===
RECONNECT_INTERVAL_SECONDS = 10

# === Internal dependency: homeassistant.components.mqtt.models ===
class ReceiveMessage:
    ...

# === Internal dependency: homeassistant.config_entries ===
class ConfigEntryState(Enum): ...
class ConfigEntryDisabler(StrEnum): ...

# === Internal dependency: homeassistant.const ===
class UnitOfTemperature(StrEnum): ...
CONF_PROTOCOL = 'protocol'
EVENT_HOMEASSISTANT_STARTED = EventType(...)
EVENT_HOMEASSISTANT_STOP = EventType(...)

# === Internal dependency: homeassistant.core ===
def callback(func): ...
class CoreState(enum.Enum): ...

# === Internal dependency: homeassistant.exceptions ===
class HomeAssistantError(Exception): ...

# === Internal dependency: homeassistant.util.dt ===
UTC = dt.UTC
utcnow = partial(...)

# === Internal dependency: homeassistant.util.event_type ===
class EventType(Generic[_DataT]): ...

# === Unresolved dependency: paho.mqtt.client ===
# Used unresolved symbols: MQTT_ERR_CONN_LOST, MQTT_ERR_SUCCESS, WebsocketWrapper

# === Third-party dependency: pytest ===
# Used symbols: fixture, mark, raises

# === Internal dependency: tests.common ===
def async_fire_mqtt_message(hass, topic, payload, qos=..., retain=...): ...
def async_fire_time_changed(hass, datetime_=..., fire_all=...): ...
class MockConfigEntry(config_entries.ConfigEntry):
    def __init__(self, *, data=..., disabled_by=..., domain=..., entry_id=..., minor_version=..., options=..., pref_disable_new_entities=..., pref_disable_polling=..., reason=..., source=..., state=..., title=..., unique_id=..., version=...): ...
    def add_to_hass(self, hass): ...

# === Internal dependency: tests.components.mqtt.conftest ===
ENTRY_DEFAULT_BIRTH_MESSAGE = {mqtt.CONF_BROKER: 'mock-broker', mqtt.CONF_BIRTH_MESSAGE: {mqtt.ATTR_TOPIC: 'homeassistant/status', mqtt.ATTR_PAYLOAD: 'online', mqtt.ATTR_QOS: 0, mqtt.ATTR_RETAIN: False}}

# === Internal dependency: tests.components.mqtt.test_common ===
def help_all_subscribe_calls(mqtt_client_mock): ...