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
# re-export: from homeassistant.helpers.http import HomeAssistantView

# === Internal dependency: homeassistant.components.network ===
async def async_get_adapters(hass: HomeAssistant) -> list[Adapter]: ...
async def async_get_enabled_source_ips(hass: HomeAssistant) -> list[IPv4Address | IPv6Address]: ...
def async_only_default_interface_enabled(adapters: list[Adapter]) -> bool: ...

# === Internal dependency: homeassistant.components.shelly.const ===
DOMAIN: Final
LOGGER: Logger
CONF_COAP_PORT: Final
BASIC_INPUTS_EVENTS_TYPES: Final
SHBTN_INPUTS_EVENTS_TYPES: Final
RPC_INPUTS_EVENTS_TYPES: Final
BLOCK_INPUTS_EVENTS_TYPES: Final
SHBTN_MODELS: Final
UPTIME_DEVIATION: Final
SHIX3_1_INPUTS_EVENTS_TYPES = BLOCK_INPUTS_EVENTS_TYPES
FIRMWARE_UNSUPPORTED_ISSUE_ID: str
GEN1_RELEASE_URL: str
GEN2_RELEASE_URL: str
DEVICES_WITHOUT_FIRMWARE_CHANGELOG: Any
CONF_GEN: str

# === Internal dependency: homeassistant.const ===
CONF_PORT: Final
EVENT_HOMEASSISTANT_STOP: EventType[NoEventData]

# === Internal dependency: homeassistant.core ===
def callback(func: _CallableT) -> _CallableT: ...

# === Internal dependency: homeassistant.helpers.device_registry ===
def format_mac(mac: str) -> str: ...
def async_get(hass: HomeAssistant) -> DeviceRegistry: ...
CONNECTION_NETWORK_MAC: str

# === Internal dependency: homeassistant.helpers.entity_registry ===
def async_get(hass: HomeAssistant) -> EntityRegistry: ...

# === Internal dependency: homeassistant.helpers.issue_registry ===
class IssueSeverity(StrEnum):
    ERROR: str
def async_create_issue(hass: HomeAssistant, domain: str, issue_id: str, *, breaks_in_ha_version: str | None = ..., data: dict[str, str | int | float | None] | None = ..., is_fixable: bool, is_persistent: bool = ..., issue_domain: str | None = ..., learn_more_url: str | None = ..., severity: IssueSeverity, translation_key: str, translation_placeholders: dict[str, str] | None = ...) -> None: ...

# === Internal dependency: homeassistant.helpers.singleton ===
def singleton(data_key: HassKey[_T]) -> Callable[[_FuncType[_T]], _FuncType[_T]]: ...
def singleton(data_key: str) -> Callable[[_FuncType[_T]], _FuncType[_T]]: ...
def singleton(data_key: Any) -> Callable[[_FuncType[_T]], _FuncType[_T]]: ...

# === Internal dependency: homeassistant.util.dt ===
utcnow: partial