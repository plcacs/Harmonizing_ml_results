from typing import Any

# === Internal dependency: homeassistant.components.plex.const ===
SERVERS: Final
DOMAIN: str
PLEX_SERVER_CONFIG: str

# === Internal dependency: homeassistant.const ===
CONF_URL: Final

# === Third-party dependency: pytest ===
# Used symbols: fixture

# === Internal dependency: tests.common ===
def load_fixture(filename: str, integration: str | None = ...) -> str: ...
class MockConfigEntry(config_entries.ConfigEntry): ...

# === Internal dependency: tests.components.plex.const ===
DEFAULT_DATA: Any
DEFAULT_OPTIONS: Any
PLEX_DIRECT_URL: str

# === Internal dependency: tests.components.plex.helpers ===
def websocket_connected(mock_websocket) -> Any: ...

# === Internal dependency: tests.components.plex.mock_classes ===
class MockGDM: ...