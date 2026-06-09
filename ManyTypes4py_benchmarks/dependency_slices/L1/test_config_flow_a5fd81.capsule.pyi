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
DOMAIN = 'nest'
OAUTH2_AUTHORIZE = 'https://nestservices.google.com/partnerconnections/{project_id}/auth'
OAUTH2_TOKEN = 'https://www.googleapis.com/oauth2/v4/token'

# === Internal dependency: homeassistant.config_entries ===
class ConfigEntry(Generic[_DataT]): ...
SOURCE_DHCP = 'dhcp'
SOURCE_USER = 'user'

# === Internal dependency: homeassistant.data_entry_flow ===
class FlowResultType(StrEnum): ...
class FlowResult(TypedDict, Generic[_HandlerT]): ...

# === Internal dependency: homeassistant.helpers.config_entry_oauth2_flow ===
def _encode_jwt(hass, data): ...

# === Third-party dependency: pytest ===
# Used symbols: fixture, mark

# === Internal dependency: tests.common ===
class MockConfigEntry(config_entries.ConfigEntry):
    def __init__(self, *, data=..., disabled_by=..., domain=..., entry_id=..., minor_version=..., options=..., pref_disable_new_entities=..., pref_disable_polling=..., reason=..., source=..., state=..., title=..., unique_id=..., version=...): ...
    def add_to_hass(self, hass): ...

# === Internal dependency: tests.components.nest.common ===
class NestTestConfig:
    ...
PROJECT_ID = 'some-project-id'
CLIENT_ID = 'some-client-id'
CLIENT_SECRET = 'some-client-secret'
CLOUD_PROJECT_ID = 'cloud-id-9876'
SUBSCRIBER_ID = 'projects/cloud-id-9876/subscriptions/subscriber-id-9876'
TEST_CONFIG_APP_CREDS = NestTestConfig(...)
TEST_CONFIGFLOW_APP_CREDS = NestTestConfig(...)