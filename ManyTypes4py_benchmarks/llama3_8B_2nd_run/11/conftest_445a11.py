from typing import Any, Callable, Coroutine, Generator, List
import asyncio
import copy
import ipaddress
import unittest.mock
from collections.abc import Callable, Coroutine, Generator

class SonosMockEventListener:
    """Mock the event listener."""

    def __init__(self, ip_address: ipaddress.IPAddress) -> None:
        """Initialize the mock event listener."""
        self.address: List[ipaddress.IPAddress] = [ip_address, '8080']

class SonosMockSubscribe:
    """Mock the subscription."""

    def __init__(self, ip_address: ipaddress.IPAddress, *args: Any, **kwargs: Any) -> None:
        """Initialize the mock subscriber."""
        self.event_listener: SonosMockEventListener = SonosMockEventListener(ip_address)
        self.service: Any = Mock()
        self.callback_future: asyncio.Future[Callable[[], Any]] = None
        self._callback: Callable[[], Any] = None

    @property
    def callback(self) -> Callable[[], Any]:
        """Return the callback."""
        return self._callback

    @callback.setter
    def callback(self, callback: Callable[[], Any]) -> None:
        """Set the callback."""
        self._callback = callback
        future = self._get_callback_future()
        if not future.done():
            future.set_result(callback)

    def _get_callback_future(self) -> asyncio.Future[Callable[[], Any]]:
        """Get the callback future."""
        if not self.callback_future:
            self.callback_future = asyncio.get_running_loop().create_future()
        return self.callback_future

    async def wait_for_callback_to_be_set(self) -> None:
        """Wait for the callback to be set."""
        return await self._get_callback_future()

    async def unsubscribe(self) -> None:
        """Unsubscribe mock."""
        pass

class SonosMockService:
    """Mock a Sonos Service used in callbacks."""

    def __init__(self, service_type: str, ip_address: ipaddress.IPAddress = '192.168.42.2') -> None:
        """Initialize the instance."""
        self.service_type: str = service_type
        self.subscribe: Callable[[], SonosMockSubscribe] = AsyncMock(return_value=SonosMockSubscribe(ip_address))

class SonosMockEvent:
    """Mock a sonos Event used in callbacks."""

    def __init__(self, soco: Any, service: Any, variables: Any) -> None:
        """Initialize the instance."""
        self.sid: str = f'{soco.uid}_sub0000000001'
        self.seq: str = '0'
        self.timestamp: float = 1621000000.0
        self.service: Any = service
        self.variables: Any = variables

    def increment_variable(self, var_name: str) -> Any:
        """Increment the value of the var_name key in variables dict attribute.

        Assumes value has a format of <str>:<int>.
        """
        self.variables = copy.deepcopy(self.variables)
        base, count = self.variables[var_name].split(':')
        newcount = int(count) + 1
        self.variables[var_name] = ':'.join([base, str(newcount)])
        return self.variables[var_name]

@pytest.fixture
def zeroconf_payload() -> ZeroconfServiceInfo:
    """Return a default zeroconf payload."""
    return ZeroconfServiceInfo(ip_address=ipaddress.IPAddress('192.168.4.2'), ip_addresses=[ipaddress.IPAddress('192.168.4.2')], hostname='Sonos-aaa', name='Sonos-aaa@Living Room._sonos._tcp.local.', port=None, properties={'bootseq': '1234'}, type='mock_type')

@pytest.fixture
async def async_autosetup_sonos(async_setup_sonos) -> Coroutine[Any, Any, None]:
    """Set up a Sonos integration instance on test run."""
    await async_setup_sonos()

@pytest.fixture
def async_setup_sonos(hass: HomeAssistant, config_entry: Any, fire_zgs_event: Any) -> Coroutine[Any, Any, None]:
    """Return a coroutine to set up a Sonos integration instance on demand."""

    async def _wrapper() -> None:
        config_entry.add_to_hass(hass)
        sonos_alarms = Alarms()
        sonos_alarms.last_alarm_list_version = 'RINCON_test:0'
        assert await hass.config_entries.async_setup(config_entry.entry_id)
        await hass.async_block_till_done(wait_background_tasks=True)
        await fire_zgs_event()
        await hass.async_block_till_done(wait_background_tasks=True)
    return _wrapper

# ... and so on
