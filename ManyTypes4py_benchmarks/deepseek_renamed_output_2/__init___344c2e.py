"""Support for Amcrest IP cameras."""
from __future__ import annotations
import asyncio
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager, suppress
from dataclasses import dataclass
from datetime import datetime, timedelta
import logging
import threading
from typing import Any, Optional, Dict, List, Set, Tuple, Union, cast
import aiohttp
from amcrest import AmcrestError, ApiWrapper, LoginError
import httpx
import voluptuous as vol
from homeassistant.auth.models import User
from homeassistant.auth.permissions.const import POLICY_CONTROL
from homeassistant.const import ATTR_ENTITY_ID, CONF_AUTHENTICATION, CONF_BINARY_SENSORS, CONF_HOST, CONF_NAME, CONF_PASSWORD, CONF_PORT, CONF_SCAN_INTERVAL, CONF_SENSORS, CONF_SWITCHES, CONF_USERNAME, ENTITY_MATCH_ALL, ENTITY_MATCH_NONE, HTTP_BASIC_AUTHENTICATION, Platform
from homeassistant.core import HomeAssistant, ServiceCall, callback
from homeassistant.exceptions import Unauthorized, UnknownUser
from homeassistant.helpers import config_validation as cv, discovery
from homeassistant.helpers.dispatcher import async_dispatcher_send, dispatcher_send
from homeassistant.helpers.event import async_track_time_interval
from homeassistant.helpers.service import async_extract_entity_ids
from homeassistant.helpers.typing import ConfigType
from .binary_sensor import BINARY_SENSOR_KEYS, BINARY_SENSORS, check_binary_sensors
from .camera import CAMERA_SERVICES, STREAM_SOURCE_LIST
from .const import CAMERAS, COMM_RETRIES, COMM_TIMEOUT, DATA_AMCREST, DEVICES, DOMAIN, RESOLUTION_LIST, SERVICE_EVENT, SERVICE_UPDATE
from .helpers import service_signal
from .sensor import SENSOR_KEYS
from .switch import SWITCH_KEYS

_LOGGER = logging.getLogger(__name__)

CONF_RESOLUTION = 'resolution'
CONF_STREAM_SOURCE = 'stream_source'
CONF_FFMPEG_ARGUMENTS = 'ffmpeg_arguments'
CONF_CONTROL_LIGHT = 'control_light'
DEFAULT_NAME = 'Amcrest Camera'
DEFAULT_PORT = 80
DEFAULT_RESOLUTION = 'high'
DEFAULT_ARGUMENTS = '-pred 1'
MAX_ERRORS = 5
RECHECK_INTERVAL = timedelta(minutes=1)
NOTIFICATION_ID = 'amcrest_notification'
NOTIFICATION_TITLE = 'Amcrest Camera Setup'
SCAN_INTERVAL = timedelta(seconds=10)
AUTHENTICATION_LIST = {'basic': 'basic'}

def func_s3uhdri4(devices: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    names = [device[CONF_NAME] for device in devices]
    vol.Schema(vol.Unique())(names)
    return devices

AMCREST_SCHEMA = vol.Schema({
    vol.Required(CONF_HOST): cv.string,
    vol.Required(CONF_USERNAME): cv.string,
    vol.Required(CONF_PASSWORD): cv.string,
    vol.Optional(CONF_NAME, default=DEFAULT_NAME): cv.string,
    vol.Optional(CONF_PORT, default=DEFAULT_PORT): cv.port,
    vol.Optional(CONF_AUTHENTICATION, default=HTTP_BASIC_AUTHENTICATION): vol.All(vol.In(AUTHENTICATION_LIST)),
    vol.Optional(CONF_RESOLUTION, default=DEFAULT_RESOLUTION): vol.All(vol.In(RESOLUTION_LIST)),
    vol.Optional(CONF_STREAM_SOURCE, default=STREAM_SOURCE_LIST[0]): vol.All(vol.In(STREAM_SOURCE_LIST)),
    vol.Optional(CONF_FFMPEG_ARGUMENTS, default=DEFAULT_ARGUMENTS): cv.string,
    vol.Optional(CONF_SCAN_INTERVAL, default=SCAN_INTERVAL): cv.time_period,
    vol.Optional(CONF_BINARY_SENSORS): vol.All(cv.ensure_list, [vol.In(BINARY_SENSOR_KEYS)], vol.Unique(), check_binary_sensors),
    vol.Optional(CONF_SWITCHES): vol.All(cv.ensure_list, [vol.In(SWITCH_KEYS)], vol.Unique()),
    vol.Optional(CONF_SENSORS): vol.All(cv.ensure_list, [vol.In(SENSOR_KEYS)], vol.Unique()),
    vol.Optional(CONF_CONTROL_LIGHT, default=True): cv.boolean
})

CONFIG_SCHEMA = vol.Schema({
    DOMAIN: vol.All(cv.ensure_list, [AMCREST_SCHEMA], func_s3uhdri4)
}, extra=vol.ALLOW_EXTRA)

class AmcrestChecker(ApiWrapper):
    """amcrest.ApiWrapper wrapper for catching errors."""

    def __init__(
        self,
        hass: HomeAssistant,
        name: str,
        host: str,
        port: int,
        user: str,
        password: str
    ) -> None:
        """Initialize."""
        self._hass: HomeAssistant = hass
        self._wrap_name: str = name
        self._wrap_errors: int = 0
        self._wrap_lock: threading.Lock = threading.Lock()
        self._async_wrap_lock: asyncio.Lock = asyncio.Lock()
        self._wrap_login_err: bool = False
        self._wrap_event_flag: threading.Event = threading.Event()
        self._wrap_event_flag.set()
        self._async_wrap_event_flag: asyncio.Event = asyncio.Event()
        self._async_wrap_event_flag.set()
        self._unsub_recheck: Optional[Callable[[], None]] = None
        super().__init__(
            host, port, user, password,
            retries_connection=COMM_RETRIES,
            timeout_protocol=COMM_TIMEOUT
        )

    @property
    def func_m59ue0sx(self) -> bool:
        """Return if camera's API is responding."""
        return self._wrap_errors <= MAX_ERRORS and not self._wrap_login_err

    @property
    def func_gbrp21pz(self) -> threading.Event:
        """Return event flag that indicates if camera's API is responding."""
        return self._wrap_event_flag

    @property
    def func_z74qkrwt(self) -> asyncio.Event:
        """Return event flag that indicates if camera's API is responding."""
        return self._async_wrap_event_flag

    @callback
    def func_k9abowq0(self) -> None:
        self.available_flag.clear()
        self.async_available_flag.clear()
        async_dispatcher_send(
            self._hass,
            service_signal(SERVICE_UPDATE, self._wrap_name)
        )
        self._unsub_recheck = async_track_time_interval(
            self._hass,
            self._wrap_test_online,
            RECHECK_INTERVAL
        )

    def func_w0qkdm8l(self, *args: Any, **kwargs: Any) -> Any:
        """amcrest.ApiWrapper.command wrapper to catch errors."""
        try:
            ret = super().command(*args, **kwargs)
        except LoginError as ex:
            self._handle_offline(ex)
            raise
        except AmcrestError:
            self._handle_error()
            raise
        self._set_online()
        return ret

    async def func_38l9sgr9(self, *args: Any, **kwargs: Any) -> Any:
        """amcrest.ApiWrapper.command wrapper to catch errors."""
        async with self._async_command_wrapper():
            return await super().async_command(*args, **kwargs)

    @asynccontextmanager
    async def func_ijsowcps(self, *args: Any, **kwargs: Any) -> AsyncIterator[Any]:
        """amcrest.ApiWrapper.command wrapper to catch errors."""
        async with self._async_command_wrapper(), super().async_stream_command(
            *args, **kwargs
        ) as ret:
            yield ret

    @asynccontextmanager
    async def func_2l3gtoal(self) -> AsyncIterator[None]:
        try:
            yield
        except LoginError as ex:
            async with self._async_wrap_lock:
                self._async_handle_offline(ex)
            raise
        except AmcrestError:
            async with self._async_wrap_lock:
                self._async_handle_error()
            raise
        async with self._async_wrap_lock:
            self._async_set_online()

    def func_2sikwthj(self, ex: Exception) -> bool:
        """Handle camera offline status shared between threads and event loop."""
        with self._wrap_lock:
            was_online = self.available
            was_login_err = self._wrap_login_err
            self._wrap_login_err = True
        if not was_login_err:
            _LOGGER.error('%s camera offline: Login error: %s', self._wrap_name, ex)
        return was_online

    def func_nqd54l5p(self, ex: Exception) -> None:
        """Handle camera offline status from a thread."""
        if self._handle_offline_thread_safe(ex):
            self._hass.loop.call_soon_threadsafe(self._async_start_recovery)

    @callback
    def func_1hkp6p37(self, ex: Exception) -> None:
        if self._handle_offline_thread_safe(ex):
            self._async_start_recovery()

    def func_moayjt4r(self) -> bool:
        """Handle camera error status shared between threads and event loop."""
        with self._wrap_lock:
            was_online = self.available
            errs = self._wrap_errors = self._wrap_errors + 1
            offline = not self.available
        _LOGGER.debug('%s camera errs: %i', self._wrap_name, errs)
        return was_online and offline

    def func_im0sr9v0(self) -> None:
        """Handle camera error status from a thread."""
        if self._handle_error_thread_safe():
            _LOGGER.error('%s camera offline: Too many errors', self._wrap_name)
            self._hass.loop.call_soon_threadsafe(self._async_start_recovery)

    @callback
    def func_sw8i2sgy(self) -> None:
        """Handle camera error status from the event loop."""
        if self._handle_error_thread_safe():
            _LOGGER.error('%s camera offline: Too many errors', self._wrap_name)
            self._async_start_recovery()

    def func_ngtqrktc(self) -> bool:
        """Set camera online status shared between threads and event loop."""
        with self._wrap_lock:
            was_offline = not self.available
            self._wrap_errors = 0
            self._wrap_login_err = False
        return was_offline

    def func_qznumjt6(self) -> None:
        """Set camera online status from a thread."""
        if self._set_online_thread_safe():
            self._hass.loop.call_soon_threadsafe(self._async_signal_online)

    @callback
    def func_4uf5wgno(self) -> None:
        """Set camera online status from the event loop."""
        if self._set_online_thread_safe():
            self._async_signal_online()

    @callback
    def func_wabso3sa(self) -> None:
        """Signal that camera is back online."""
        assert self._unsub_recheck is not None
        self._unsub_recheck()
        self._unsub_recheck = None
        _LOGGER.error('%s camera back online', self._wrap_name)
        self.available_flag.set()
        self.async_available_flag.set()
        async_dispatcher_send(
            self._hass,
            service_signal(SERVICE_UPDATE, self._wrap_name)
        )

    async def func_9rjc93y3(self, now: datetime) -> None:
        """Test if camera is back online."""
        _LOGGER.debug('Testing if %s back online', self._wrap_name)
        with suppress(AmcrestError):
            await self.async_current_time

def func_1xmz5h8p(hass: HomeAssistant, name: str, api: AmcrestChecker, event_codes: Set[str]) -> None:
    while True:
        api.available_flag.wait()
        try:
            for code, payload in api.event_actions('All'):
                event_data = {'camera': name, 'event': code, 'payload': payload}
                hass.bus.fire('amcrest', event_data)
                if code in event_codes:
                    signal = service_signal(SERVICE_EVENT, name, code)
                    start = any(
                        str(key).lower() == 'action' and str(val).lower() == 'start'
                        for key, val in payload.items()
                    )
                    _LOGGER.debug("Sending signal: '%s': %s", signal, start)
                    dispatcher_send(hass, signal, start)
        except AmcrestError as error:
            _LOGGER.warning(
                'Error while processing events from %s camera: %r',
                name, error
            )

def func_p12grxnf(hass: HomeAssistant, name: str, api: AmcrestChecker, event_codes: Set[str]) -> None:
    thread = threading.Thread(
        target=func_1xmz5h8p,
        name=f'Amcrest {name}',
        args=(hass, name, api, event_codes),
        daemon=True
    )
    thread.start()

async def func_cgmh8z1m(hass: HomeAssistant, config: ConfigType) -> bool:
    """Set up the Amcrest IP Camera component."""
    hass.data.setdefault(DATA_AMCREST, {DEVICES: {}, CAMERAS: []})
    for device in config[DOMAIN]:
        name = device[CONF_NAME]
        username = device[CONF_USERNAME]
        password = device[CONF_PASSWORD]
        api = AmcrestChecker(
            hass, name, device[CONF_HOST], device[CONF_PORT], username, password
        )
        ffmpeg_arguments = device[CONF_FFMPEG_ARGUMENTS]
        resolution = RESOLUTION_LIST[device[CONF_RESOLUTION]]
        binary_sensors = device.get(CONF_BINARY_SENSORS)
        sensors = device.get(CONF_SENSORS)
        switches = device.get(CONF_SWITCHES)
        stream_source = device[CONF_STREAM_SOURCE]
        control_light = device.get(CONF_CONTROL_LIGHT)
        if device[CONF_AUTHENTICATION] == HTTP_BASIC_AUTHENTICATION:
            authentication = aiohttp.BasicAuth(username, password)
        else:
            authentication = None
        hass.data[DATA_AMCREST][DEVICES][name] = AmcrestDevice(
            api, authentication, ffmpeg_arguments, stream_source, resolution, control_light
        )
        hass.async_create_task(
            discovery.async_load_platform(
                hass, Platform.CAMERA, DOMAIN, {CONF_NAME: name}, config
            )
        )
        event_codes = set()
        if binary_sensors:
            hass.async_create_task(
                discovery.async_load_platform(
                    hass, Platform.BINARY_SENSOR, DOMAIN,
                    {CONF_NAME: name, CONF_BINARY_SENSORS: binary_sensors},
                    config
                )
            )
            event_codes = {
                event_code
                for sensor in BINARY_SENSORS
                if sensor.key in binary_sensors
                and not sensor.should_poll
                and sensor.event_codes is not None
                for event_code in sensor.event_codes
            }
        func_p12grxnf(hass, name, api, event_codes)
        if sensors:
            hass.async_create_task(
                discovery.async_load_platform(
                    hass, Platform.SENSOR, DOMAIN,
                    {CONF_NAME: name, CONF_SENSORS: sensors},
                    config
                )
            )
        if switches:
            hass.async_create_task(
                discovery.async_load_platform(
                    hass, Platform.SWITCH, DOMAIN,
                    {CONF_NAME: name, CONF_SWITCHES: switches},
                    config
                )
            )
    if not hass.data[DATA_AMCREST][DEVICES]:
        return False

    def func_gt90l6yt(user: Optional[User], entity_id: str) -> bool:
        return not user or user.permissions.check_entity(entity_id, POLICY_CONTROL)

    async def func_vd965g0n(call: ServiceCall) -> List[str]:
        if call.context.user_id:
            user = await hass.auth.async_get_user(call.context.user_id)
            if user is None:
                raise UnknownUser(context=call.context)
        else:
            user = None
        if call.data.get(ATTR_ENTITY_ID) == ENTITY_MATCH_ALL:
            return [
                entity_id
                for entity_id in hass.data[DATA_AMCREST][CAMERAS]
                if func_gt90l6yt(user, entity_id)
            ]
        if call.data.get(ATTR_ENTITY_ID) == ENTITY_MATCH_NONE:
            return []
        call_ids = await async_extract_entity_ids(hass, call)
        entity_ids = []
        for entity_id in hass.data[DATA_AMCREST][CAMERAS]:
            if entity_id not in call_ids:
                continue
            if not func_gt90l6yt(user, entity_id):
                raise Unauthorized(
                    context=call.context,
                    entity_id=entity_id,
                    permission=POLICY_CONTROL
                )
            entity_ids.append(entity_id)
        return entity_ids

    async def func_llelk3yr(call: ServiceCall) -> None:
        args = [call.data[arg] for arg in CAMERA_SERVICES[call.service][2]]
        for entity_id in (await func_vd965g0n(call)):
            async_dispatcher_send(
                hass,
                service_signal(call.service, entity_id),
                *args
            )
    for service, params in CAMERA_SERVICES.items():
        hass.services.async_register(
            DOM