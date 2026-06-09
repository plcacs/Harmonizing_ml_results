from typing import Any

# === Internal dependency: homeassistant.components.media_player ===
from .const import DOMAIN

# === Internal dependency: homeassistant.components.sonos ===
from .const import DOMAIN

# === Internal dependency: homeassistant.components.ssdp ===
class SsdpServiceInfo(BaseServiceInfo): ...
ATTR_UPNP_UDN = 'UDN'
SsdpChange = Enum(...)

# === Internal dependency: homeassistant.components.zeroconf ===
class ZeroconfServiceInfo(BaseServiceInfo): ...

# === Internal dependency: homeassistant.const ===
CONF_HOSTS = 'hosts'

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
def load_fixture(filename, integration=...): ...
def load_json_value_fixture(filename, integration=...): ...
class MockConfigEntry(config_entries.ConfigEntry): ...