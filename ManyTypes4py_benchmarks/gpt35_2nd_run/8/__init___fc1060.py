import asyncio
import logging
from keba_kecontact.connection import KebaKeContact
import voluptuous as vol
from homeassistant.const import CONF_HOST, Platform
from homeassistant.core import HomeAssistant, ServiceCall
from homeassistant.helpers import config_validation as cv, discovery
from homeassistant.helpers.typing import ConfigType
_LOGGER: logging.Logger = logging.getLogger(__name__)
DOMAIN: str = 'keba'
PLATFORMS: tuple = (Platform.BINARY_SENSOR, Platform.SENSOR, Platform.LOCK, Platform.NOTIFY)
CONF_RFID: str = 'rfid'
CONF_FS: str = 'failsafe'
CONF_FS_TIMEOUT: str = 'failsafe_timeout'
CONF_FS_FALLBACK: str = 'failsafe_fallback'
CONF_FS_PERSIST: str = 'failsafe_persist'
CONF_FS_INTERVAL: str = 'refresh_interval'
MAX_POLLING_INTERVAL: int = 5
MAX_FAST_POLLING_COUNT: int = 4
CONFIG_SCHEMA: vol.Schema = vol.Schema({DOMAIN: vol.Schema({vol.Required(CONF_HOST): cv.string, vol.Optional(CONF_RFID, default='00845500'): cv.string, vol.Optional(CONF_FS, default=False): cv.boolean, vol.Optional(CONF_FS_TIMEOUT, default=30): cv.positive_int, vol.Optional(CONF_FS_FALLBACK, default=6): cv.positive_int, vol.Optional(CONF_FS_PERSIST, default=0): cv.positive_int, vol.Optional(CONF_FS_INTERVAL, default=5): cv.positive_int})}, extra=vol.ALLOW_EXTRA)
_SERVICE_MAP: dict = {'request_data': 'async_request_data', 'set_energy': 'async_set_energy', 'set_current': 'async_set_current', 'authorize': 'async_start', 'deauthorize': 'async_stop', 'enable': 'async_enable_ev', 'disable': 'async_disable_ev', 'set_failsafe': 'async_set_failsafe'}

async def async_setup(hass: HomeAssistant, config: ConfigType) -> bool:
    host: str = config[DOMAIN][CONF_HOST]
    rfid: str = config[DOMAIN][CONF_RFID]
    refresh_interval: int = config[DOMAIN][CONF_FS_INTERVAL]
    keba: KebaHandler = KebaHandler(hass, host, rfid, refresh_interval)
    hass.data[DOMAIN] = keba
    if not await keba.setup():
        _LOGGER.error('Could not find a charging station at %s', host)
        return False
    failsafe: bool = config[DOMAIN][CONF_FS]
    timeout: int = config[DOMAIN][CONF_FS_TIMEOUT] if failsafe else 0
    fallback: int = config[DOMAIN][CONF_FS_FALLBACK] if failsafe else 0
    persist: int = config[DOMAIN][CONF_FS_PERSIST] if failsafe else 0
    try:
        hass.loop.create_task(keba.set_failsafe(timeout, fallback, persist))
    except ValueError as ex:
        _LOGGER.warning('Could not set failsafe mode %s', ex)

    async def execute_service(call: ServiceCall) -> None:
        function_name: str = _SERVICE_MAP[call.service]
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
        super().__init__(host, self.hass_callback)
        self._update_listeners: list = []
        self._hass: HomeAssistant = hass
        self.rfid: str = rfid
        self.device_name: str = 'keba'
        self.device_id: str = 'keba_wallbox_'
        self._refresh_interval: int = max(MAX_POLLING_INTERVAL, refresh_interval)
        self._fast_polling_count: int = MAX_FAST_POLLING_COUNT
        self._polling_task: asyncio.Task = None

    def start_periodic_request(self) -> None:
        self._polling_task = self._hass.loop.create_task(self._periodic_request())

    async def _periodic_request(self) -> None:
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

    async def setup(self, loop: asyncio.AbstractEventLoop = None) -> bool:
        await super().setup(loop)
        await self.request_data()
        if self.get_value('Serial') is not None and self.get_value('Product') is not None:
            self.device_id = f'keba_wallbox_{self.get_value('Serial')}'
            self.device_name = self.get_value('Product')
            return True
        return False

    def hass_callback(self, data) -> None:
        for listener in self._update_listeners:
            listener()
        _LOGGER.debug('Notifying %d listeners', len(self._update_listeners))

    def _set_fast_polling(self) -> None:
        _LOGGER.debug('Fast polling enabled')
        self._fast_polling_count = 0
        self._polling_task.cancel()
        self._polling_task = self._hass.loop.create_task(self._periodic_request())

    def add_update_listener(self, listener) -> None:
        self._update_listeners.append(listener)
        listener()

    async def async_request_data(self, param) -> None:
        await self.request_data()
        _LOGGER.debug('New data from KEBA wallbox requested')

    async def async_set_energy(self, param) -> None:
        try:
            energy: float = param['energy']
            await self.set_energy(float(energy))
            self._set_fast_polling()
        except (KeyError, ValueError) as ex:
            _LOGGER.warning('Energy value is not correct. %s', ex)

    async def async_set_current(self, param) -> None:
        try:
            current: float = param['current']
            await self.set_current(float(current))
        except (KeyError, ValueError) as ex:
            _LOGGER.warning('Current value is not correct. %s', ex)

    async def async_start(self, param=None) -> None:
        await self.start(self.rfid)
        self._set_fast_polling()

    async def async_stop(self, param=None) -> None:
        await self.stop(self.rfid)
        self._set_fast_polling()

    async def async_enable_ev(self, param=None) -> None:
        await self.enable(True)
        self._set_fast_polling()

    async def async_disable_ev(self, param=None) -> None:
        await self.enable(False)
        self._set_fast_polling()

    async def async_set_failsafe(self, param=None) -> None:
        try:
            timeout: int = param[CONF_FS_TIMEOUT]
            fallback: float = param[CONF_FS_FALLBACK]
            persist: bool = param[CONF_FS_PERSIST]
            await self.set_failsafe(int(timeout), float(fallback), bool(persist))
            self._set_fast_polling()
        except (KeyError, ValueError) as ex:
            _LOGGER.warning('Values are not correct for: failsafe_timeout, failsafe_fallback and/or failsafe_persist: %s', ex)
