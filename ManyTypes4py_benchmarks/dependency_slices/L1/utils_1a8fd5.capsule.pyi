from typing import Any

# === Third-party dependency: aiohttp.web ===
# Used symbols: WebSocketResponse

# === Third-party dependency: aioshelly.block_device ===
# Used symbols: COAP

# === Third-party dependency: aioshelly.const ===
MODEL_1L: str
MODEL_DIMMER: str
MODEL_DIMMER_2: str
MODEL_EM3: str
MODEL_I3: str
MODEL_NAMES: Any
DEFAULT_COAP_PORT: int
BLOCK_GENERATIONS: Any
RPC_GENERATIONS: Any
DEFAULT_HTTP_PORT: int

# === Third-party dependency: aioshelly.rpc_device ===
# Used symbols: WsServer

# === Internal dependency: homeassistant.components.http ===
from homeassistant.helpers.http import HomeAssistantView

# === Internal dependency: homeassistant.components.network ===
async def async_get_adapters(hass): ...
async def async_get_enabled_source_ips(hass): ...
def async_only_default_interface_enabled(adapters): ...

# === Internal dependency: homeassistant.components.shelly.const ===
DOMAIN = 'shelly'
LOGGER = getLogger(...)
CONF_COAP_PORT = 'coap_port'
BASIC_INPUTS_EVENTS_TYPES = {'single', 'long'}
SHBTN_INPUTS_EVENTS_TYPES = {'single', 'double', 'triple', 'long'}
RPC_INPUTS_EVENTS_TYPES = {'btn_down', 'btn_up', 'single_push', 'double_push', 'triple_push', 'long_push'}
BLOCK_INPUTS_EVENTS_TYPES = {'single', 'double', 'triple', 'long', 'single_long', 'long_single'}
SHIX3_1_INPUTS_EVENTS_TYPES = BLOCK_INPUTS_EVENTS_TYPES
SHBTN_MODELS = [MODEL_BUTTON1, MODEL_BUTTON1_V2]
UPTIME_DEVIATION = 5
FIRMWARE_UNSUPPORTED_ISSUE_ID = 'firmware_unsupported_{unique}'
GEN1_RELEASE_URL = 'https://shelly-api-docs.shelly.cloud/gen1/#changelog'
GEN2_RELEASE_URL = 'https://shelly-api-docs.shelly.cloud/gen2/changelog/'
DEVICES_WITHOUT_FIRMWARE_CHANGELOG = (MODEL_WALL_DISPLAY, MODEL_MOTION, MODEL_MOTION_2, MODEL_VALVE)
CONF_GEN = 'gen'

# === Internal dependency: homeassistant.const ===
CONF_PORT = 'port'
EVENT_HOMEASSISTANT_STOP = EventType(...)

# === Internal dependency: homeassistant.core ===
def callback(func): ...

# === Internal dependency: homeassistant.helpers.device_registry ===
def format_mac(mac): ...
def async_get(hass): ...
CONNECTION_NETWORK_MAC = 'mac'

# === Internal dependency: homeassistant.helpers.entity_registry ===
def async_get(hass): ...

# === Internal dependency: homeassistant.helpers.issue_registry ===
class IssueSeverity(StrEnum):
    ERROR = 'error'
def async_create_issue(hass, domain, issue_id, *, breaks_in_ha_version=..., data=..., is_fixable, is_persistent=..., issue_domain=..., learn_more_url=..., severity, translation_key, translation_placeholders=...): ...

# === Internal dependency: homeassistant.helpers.singleton ===
def singleton(data_key): ...

# === Internal dependency: homeassistant.util.dt ===
UTC = dt.UTC
utcnow = partial(...)

# === Internal dependency: homeassistant.util.event_type ===
class EventType(Generic[_DataT]): ...