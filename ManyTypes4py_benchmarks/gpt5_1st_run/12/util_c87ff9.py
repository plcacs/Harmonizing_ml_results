from __future__ import annotations

from typing import Any, TypedDict

from ipaddress import ip_address
from homeassistant.components.kodi.const import DEFAULT_SSL
from homeassistant.helpers.service_info.zeroconf import ZeroconfServiceInfo


class HostConfig(TypedDict):
    host: str
    port: int
    ssl: bool


class Credentials(TypedDict):
    username: str
    password: str


class WSPortConfig(TypedDict):
    ws_port: int


class ImportConfig(TypedDict):
    name: str
    host: str
    port: int
    ws_port: int
    username: str
    password: str
    ssl: bool
    timeout: int


TEST_HOST: HostConfig = {"host": "1.1.1.1", "port": 8080, "ssl": DEFAULT_SSL}
TEST_CREDENTIALS: Credentials = {"username": "username", "password": "password"}
TEST_WS_PORT: WSPortConfig = {"ws_port": 9090}
UUID: str = "11111111-1111-1111-1111-111111111111"
TEST_DISCOVERY: ZeroconfServiceInfo = ZeroconfServiceInfo(
    ip_address=ip_address("1.1.1.1"),
    ip_addresses=[ip_address("1.1.1.1")],
    port=8080,
    hostname="hostname.local.",
    type="_xbmc-jsonrpc-h._tcp.local.",
    name="hostname._xbmc-jsonrpc-h._tcp.local.",
    properties={"uuid": UUID},
)
TEST_DISCOVERY_WO_UUID: ZeroconfServiceInfo = ZeroconfServiceInfo(
    ip_address=ip_address("1.1.1.1"),
    ip_addresses=[ip_address("1.1.1.1")],
    port=8080,
    hostname="hostname.local.",
    type="_xbmc-jsonrpc-h._tcp.local.",
    name="hostname._xbmc-jsonrpc-h._tcp.local.",
    properties={},
)
TEST_IMPORT: ImportConfig = {
    "name": "name",
    "host": "1.1.1.1",
    "port": 8080,
    "ws_port": 9090,
    "username": "username",
    "password": "password",
    "ssl": True,
    "timeout": 7,
}


def get_kodi_connection(
    host: str,
    port: int,
    ws_port: int | None,
    username: str,
    password: str,
    ssl: bool = False,
    timeout: int = 5,
    session: Any | None = None,
) -> MockConnection | MockWSConnection:
    """Get Kodi connection."""
    if ws_port is None:
        return MockConnection()
    return MockWSConnection()


class MockConnection:
    """A mock kodi connection."""

    def __init__(self, connected: bool = True) -> None:
        """Mock the Kodi connection."""
        self._connected: bool = connected

    async def connect(self) -> None:
        """Mock connect."""
        return None

    @property
    def connected(self) -> bool:
        """Mock connected."""
        return self._connected

    @property
    def can_subscribe(self) -> bool:
        """Mock can_subscribe."""
        return False

    async def close(self) -> None:
        """Mock close."""
        return None

    @property
    def server(self) -> None:
        """Mock server."""
        return None


class MockWSConnection:
    """A mock kodi websocket connection."""

    def __init__(self, connected: bool = True) -> None:
        """Mock the websocket connection."""
        self._connected: bool = connected

    async def connect(self) -> None:
        """Mock connect."""
        return None

    @property
    def connected(self) -> bool:
        """Mock connected."""
        return self._connected

    @property
    def can_subscribe(self) -> bool:
        """Mock can_subscribe."""
        return False

    async def close(self) -> None:
        """Mock close."""
        return None

    @property
    def server(self) -> None:
        """Mock server."""
        return None