from typing import Any

# === Internal dependency: homeassistant.components.media_player ===
# re-export: from .const import DOMAIN

# === Internal dependency: homeassistant.components.sonos ===
# re-export: from .const import DOMAIN

# === Internal dependency: homeassistant.components.ssdp ===
class SsdpServiceInfo(BaseServiceInfo): ...
ATTR_UPNP_UDN: str
SsdpChange: Enum

# === Internal dependency: homeassistant.components.zeroconf ===
class ZeroconfServiceInfo(BaseServiceInfo): ...

# === Internal dependency: homeassistant.const ===
CONF_HOSTS: Final

# === Third-party dependency: pytest ===
# Used symbols: fixture

# === Third-party dependency: soco ===
# Used symbols: SoCo

# === Third-party dependency: soco.alarms ===
class Alarms(_SocoSingletonBase):
    def __init__(self) -> Any: ...

# === Third-party dependency: soco.data_structures ===
class DidlFavorite(DidlItem): ...
class SearchResult(ListOfMusicInfoItems): ...

# === Third-party dependency: soco.events_base ===
class Event: ...

# === Internal dependency: tests.common ===
def load_fixture(filename: str, integration: str | None = ...) -> str: ...
def load_json_value_fixture(filename: str, integration: str | None = ...) -> JsonValueType: ...
class MockConfigEntry(config_entries.ConfigEntry): ...