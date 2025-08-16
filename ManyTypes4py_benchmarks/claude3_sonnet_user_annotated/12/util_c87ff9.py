"""Test the Kodi config flow."""

from ipaddress import ip_address
from typing import Any, Dict, Optional, Union

from homeassistant.components.kodi.const import DEFAULT_SSL
from homeassistant.helpers.service_info.zeroconf import ZeroconfServiceInfo

TEST_HOST: Dict[str, Union[str, int, bool]] = {
    "host": "1.1.1.1",
    "port": 8080,
    "ssl": DEFAULT_SSL,
}

TEST_CREDENTIALS: Dict[str, str] = {"username": "username", "password": "password"}


TEST_WS_PORT: Dict[str, int] = {"ws_port": 9090}

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


TEST_IMPORT: Dict[str, Union[str, int, bool]] = {
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
    ws_port: Optional[int],
    username: Optional[str],
    password: Optional[str],
    ssl: bool = False,
    timeout: int = 5,
    session: Optional[Any] = None,
) -> Union["MockConnection", "MockWSConnection"]:
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

    @property
    def server(self) -> None:
        """Mock server."""
        return None
