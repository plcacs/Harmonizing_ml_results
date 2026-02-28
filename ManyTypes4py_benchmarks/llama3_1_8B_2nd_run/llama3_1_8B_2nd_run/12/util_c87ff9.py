"""Test the Kodi config flow."""
from ipaddress import IPv4Address
from homeassistant.components.kodi.const import DEFAULT_SSL
from homeassistant.helpers.service_info.zeroconf import ZeroconfServiceInfo

TEST_HOST = {'host': IPv4Address('1.1.1.1'), 'port': 8080, 'ssl': DEFAULT_SSL}
TEST_CREDENTIALS = {'username': str, 'password': str}
TEST_WS_PORT = {'ws_port': int}
UUID = str
TEST_DISCOVERY = ZeroconfServiceInfo[
    IPv4Address('1.1.1.1'), 
    list[IPv4Address], 
    int, 
    str, 
    str, 
    dict[str, str]
](ip_address=IPv4Address('1.1.1.1'), ip_addresses=[IPv4Address('1.1.1.1')], port=8080, hostname='hostname.local.', type='_xbmc-jsonrpc-h._tcp.local.', name='hostname._xbmc-jsonrpc-h._tcp.local.', properties={'uuid': UUID})
TEST_DISCOVERY_WO_UUID = ZeroconfServiceInfo[
    IPv4Address('1.1.1.1'), 
    list[IPv4Address], 
    int, 
    str, 
    str, 
    dict[str, str]
](ip_address=IPv4Address('1.1.1.1'), ip_addresses=[IPv4Address('1.1.1.1')], port=8080, hostname='hostname.local.', type='_xbmc-jsonrpc-h._tcp.local.', name='hostname._xbmc-jsonrpc-h._tcp.local.', properties={})
TEST_IMPORT = {'name': str, 'host': IPv4Address('1.1.1.1'), 'port': int, 'ws_port': int, 'username': str, 'password': str, 'ssl': bool, 'timeout': int}

async def get_kodi_connection(
    host: IPv4Address, 
    port: int, 
    ws_port: int | None, 
    username: str, 
    password: str, 
    ssl: bool = False, 
    timeout: int = 5, 
    session: None | object = None
) -> MockConnection | MockWSConnection:
    """Get Kodi connection."""
    if ws_port is None:
        return MockConnection()
    return MockWSConnection()

class MockConnection:
    """A mock kodi connection."""

    def __init__(self, connected: bool = True) -> None:
        """Mock the Kodi connection."""
        self._connected = connected

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
        self._connected = connected

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
