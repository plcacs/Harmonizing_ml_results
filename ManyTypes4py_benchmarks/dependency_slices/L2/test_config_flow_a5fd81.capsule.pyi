# === Third-party dependency: google_nest_sdm.exceptions ===
class SubscriberException(GoogleNestException):
    ...
class AuthException(ApiException):
class ConfigurationException(GoogleNestException): ...

# === Third-party dependency: google_nest_sdm.structure ===
class Structure(TraitDataClass): ...

# === Internal dependency: homeassistant.components.dhcp ===
class DhcpServiceInfo(BaseServiceInfo):
    ...

# === Internal dependency: homeassistant.components.nest.const ===
DOMAIN: str
OAUTH2_AUTHORIZE: str
OAUTH2_TOKEN: str

# === Internal dependency: homeassistant.config_entries ===
class ConfigEntry(Generic[_DataT]): ...
SOURCE_DHCP: str
SOURCE_USER: str

# === Internal dependency: homeassistant.data_entry_flow ===
class FlowResultType(StrEnum): ...
class FlowResult(TypedDict, Generic[_HandlerT]): ...

# === Internal dependency: homeassistant.helpers.config_entry_oauth2_flow ===
def _encode_jwt(hass: HomeAssistant, data: dict) -> str: ...

# === Third-party dependency: pytest ===
# Used symbols: fixture, mark

# === Internal dependency: tests.common ===
class MockConfigEntry(config_entries.ConfigEntry):
    def __init__(self, *, data = ..., disabled_by = ..., domain = ..., entry_id = ..., minor_version = ..., options = ..., pref_disable_new_entities = ..., pref_disable_polling = ..., reason = ..., source = ..., state = ..., title = ..., unique_id = ..., version = ...) -> None: ...
    def add_to_hass(self, hass: HomeAssistant) -> None: ...

# === Internal dependency: tests.components.nest.common ===
class NestTestConfig: ...
PROJECT_ID: str
CLIENT_ID: str
CLOUD_PROJECT_ID: str
SUBSCRIBER_ID: str
TEST_CONFIG_APP_CREDS: NestTestConfig
TEST_CONFIGFLOW_APP_CREDS: NestTestConfig