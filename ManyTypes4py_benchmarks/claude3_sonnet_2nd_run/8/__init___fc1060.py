"""Support for KEBA charging stations."""
import asyncio
import logging
from typing import Any, Callable, Dict, List, Optional, Union
from keba_kecontact.connection import KebaKeContact
import voluptuous as vol
from homeassistant.const import CONF_HOST, Platform
from homeassistant.core import HomeAssistant, ServiceCall
from homeassistant.helpers import config_validation as cv, discovery
from homeassistant.helpers.typing import ConfigType

_LOGGER = logging.getLogger(__name__)
DOMAIN = 'keba'
PLATFORMS = (Platform.BINARY_SENSOR, Platform.SENSOR, Platform.LOCK, Platform.NOTIFY)
CONF_RFID = 'rfid'
CONF_FS = 'failsafe'
CONF_FS_TIMEOUT = 'failsafe_timeout'
CONF_FS_FALLBACK = 'failsafe_fallback'
CONF_FS_PERSIST = 'failsafe_persist'
CONF_FS_INTERVAL = 'refresh_interval'
MAX_POLLING_INTERVAL = 5
MAX_FAST_POLLING_COUNT = 4
CONFIG_SCHEMA = vol.Schema({DOMAIN: vol.Schema({vol.Required(CONF_HOST): cv.string, vol.Optional(CONF_RFID, default='00845500'): cv.string, vol.Optional(CONF_FS, default=False): cv.boolean, vol.Optional(CONF_FS_TIMEOUT, default=30): cv.positive_int, vol.Optional(CONF_FS_FALLBACK, default=6): cv.positive_int, vol.Optional(CONF_FS_PERSIST, default=0): cv.positive_int, vol.Optional(CONF_FS_INTERVAL, default=5): cv.positive_int})}, extra=vol.ALLOW_EXTRA)
_SERVICE_MAP: Dict[str, str] = {'request_data': 'async_request_data', 'set_energy': 'async_set_energy', 'set_current': 'async_set_current', 'authorize': 'async_start', 'deauthorize': 'async_stop', 'enable': 'async_enable_ev', 'disable': 'async_disable_ev', 'set_failsafe': 'async_set_failsafe'}

async def async_setup(hass: HomeAssistant, config: ConfigType) -> bool:
    """Check connectivity and version of KEBA charging station."""
    host = config[DOMAIN][CONF_HOST]
    rfid = config[DOMAIN][CONF_RFID]
    refresh_interval = config[DOMAIN][CONF_FS_INTERVAL]
    keba = KebaHandler(hass, host, rfid, refresh_interval)
    hass.data[DOMAIN] = keba
    if not await keba.setup():
        _LOGGER.error('Could not find a charging station at %s', host)
        return False
    failsafe = config[DOMAIN][CONF_FS]
    timeout = config[DOMAIN][CONF_FS_TIMEOUT] if failsafe else 0
    fallback = config[DOMAIN][CONF_FS_FALLBACK] if failsafe else 0
    persist = config[DOMAIN][CONF_FS_PERSIST] if failsafe else 0
    try:
        hass.loop.create_task(keba.set_failsafe(timeout, fallback, persist))
    except ValueError as ex:
        _LOGGER.warning('Could not set failsafe mode %s', ex)

    async def execute_service(call: ServiceCall) -> None:
        """Execute a service to KEBA charging station.

        This must be a member function as we need access to the keba
        object here.
        """
        function_name = _SERVICE_MAP[call.service]
        function_call = getattr(keba, function_name)
        await function_call(call.data)
    for service in _SERVICE_MAP:
        hass.services.async_register(DOMAIN, service, execute_service)
    for platform in PLATFORMS:
        hass.async_create_task(discovery.async_load_platform(hass, platform, DOMAIN, {}, config))
    keba.start_periodic_request()
    return True

class KebaHandler(KebaKeContact):
    """Representation of a KEBA charging station connection."""

    def __init__(self, hass: HomeAssistant, host: str, rfid: str, refresh_interval: int) -> None:
        """Initialize charging station connection."""
        super().__init__(host, self.hass_callback)
        self._update_listeners: List[Callable[[], None]] = []
        self._hass = hass
        self.rfid = rfid
        self.device_name: str = 'keba'
        self.device_id: str = 'keba_wallbox_'
        self._refresh_interval: int = max(MAX_POLLING_INTERVAL, refresh_interval)
        self._fast_polling_count: int = MAX_FAST_POLLING_COUNT
        self._polling_task: Optional[asyncio.Task] = None

    def start_periodic_request(self) -> None:
        """Start periodic data polling."""
        self._polling_task = self._hass.loop.create_task(self._periodic_request())

    async def _periodic_request(self) -> None:
        """Send  periodic update requests."""
        await self.request_data()
        if self._fast_polling_count < MAX_FAST_POLLING_COUNT:
            self._fast_polling_count += 1
            _LOGGER.debug('Periodic data request executed, now wait for 2 seconds')
            await asyncio.sleep(2)
        else:
            _LOGGER.debug('Periodic data request executed, now wait for %s seconds', self._refresh_interval)
            await asyncio.sleep(self._refresh_interval)
        _LOGGER.debug('Periodic data request rescheduled')
        self._polling_task = self._hass.loop.create_task(self._periodic_request())

    async def setup(self, loop: Optional[asyncio.AbstractEventLoop] = None) -> bool:
        """Initialize KebaHandler object."""
        await super().setup(loop)
        await self.request_data()
        if self.get_value('Serial') is not None and self.get_value('Product') is not None:
            self.device_id = f'keba_wallbox_{self.get_value("Serial")}'
            self.device_name = self.get_value('Product')
            return True
        return False

    def hass_callback(self, data: Dict[str, Any]) -> None:
        """Handle component notification via callback."""
        for listener in self._update_listeners:
            listener()
        _LOGGER.debug('Notifying %d listeners', len(self._update_listeners))

    def _set_fast_polling(self) -> None:
        _LOGGER.debug('Fast polling enabled')
        self._fast_polling_count = 0
        if self._polling_task:
            self._polling_task.cancel()
        self._polling_task = self._hass.loop.create_task(self._periodic_request())

    def add_update_listener(self, listener: Callable[[], None]) -> None:
        """Add a listener for update notifications."""
        self._update_listeners.append(listener)
        listener()

    async def async_request_data(self, param: Dict[str, Any]) -> None:
        """Request new data in async way."""
        await self.request_data()
        _LOGGER.debug('New data from KEBA wallbox requested')

    async def async_set_energy(self, param: Dict[str, Any]) -> None:
        """Set energy target in async way."""
        try:
            energy = param['energy']
            await self.set_energy(float(energy))
            self._set_fast_polling()
        except (KeyError, ValueError) as ex:
            _LOGGER.warning('Energy value is not correct. %s', ex)

    async def async_set_current(self, param: Dict[str, Any]) -> None:
        """Set current maximum in async way."""
        try:
            current = param['current']
            await self.set_current(float(current))
        except (KeyError, ValueError) as ex:
            _LOGGER.warning('Current value is not correct. %s', ex)

    async def async_start(self, param: Optional[Dict[str, Any]] = None) -> None:
        """Authorize EV in async way."""
        await self.start(self.rfid)
        self._set_fast_polling()

    async def async_stop(self, param: Optional[Dict[str, Any]] = None) -> None:
        """De-authorize EV in async way."""
        await self.stop(self.rfid)
        self._set_fast_polling()

    async def async_enable_ev(self, param: Optional[Dict[str, Any]] = None) -> None:
        """Enable EV in async way."""
        await self.enable(True)
        self._set_fast_polling()

    async def async_disable_ev(self, param: Optional[Dict[str, Any]] = None) -> None:
        """Disable EV in async way."""
        await self.enable(False)
        self._set_fast_polling()

    async def async_set_failsafe(self, param: Optional[Dict[str, Any]] = None) -> None:
        """Set failsafe mode in async way."""
        try:
            if param is None:
                param = {}
            timeout = param.get(CONF_FS_TIMEOUT, 0)
            fallback = param.get(CONF_FS_FALLBACK, 0)
            persist = param.get(CONF_FS_PERSIST, 0)
            await self.set_failsafe(int(timeout), float(fallback), bool(persist))
            self._set_fast_polling()
        except (KeyError, ValueError) as ex:
            _LOGGER.warning('Values are not correct for: failsafe_timeout, failsafe_fallback and/or failsafe_persist: %s', ex)
