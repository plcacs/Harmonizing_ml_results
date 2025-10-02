"""Support for Telldus Live."""
import asyncio
from functools import partial
import logging
from tellduslive import DIM, TURNON, UP, Session
import voluptuous as vol
from homeassistant import config_entries
from homeassistant.config_entries import ConfigEntry
from homeassistant.const import CONF_HOST, CONF_SCAN_INTERVAL
from homeassistant.core import HomeAssistant
from homeassistant.helpers import config_validation as cv, device_registry as dr
from homeassistant.helpers.dispatcher import async_dispatcher_send
from homeassistant.helpers.event import async_call_later
from homeassistant.helpers.typing import ConfigType
from .const import (
    DOMAIN,
    KEY_SCAN_INTERVAL,
    KEY_SESSION,
    MIN_UPDATE_INTERVAL,
    NOT_SO_PRIVATE_KEY,
    PUBLIC_KEY,
    SCAN_INTERVAL,
    SIGNAL_UPDATE_ENTITY,
    TELLDUS_DISCOVERY_NEW,
)
from typing import Any, Callable, Dict, Set, Optional

APPLICATION_NAME: str = 'Home Assistant'
_LOGGER: logging.Logger = logging.getLogger(__name__)

CONFIG_SCHEMA = vol.Schema(
    {
        DOMAIN: vol.Schema(
            {
                vol.Optional(CONF_HOST, default=DOMAIN): cv.string,
                vol.Optional(
                    CONF_SCAN_INTERVAL, default=SCAN_INTERVAL
                ): vol.All(cv.time_period, vol.Clamp(min=MIN_UPDATE_INTERVAL)),
            }
        )
    },
    extra=vol.ALLOW_EXTRA,
)

DATA_CONFIG_ENTRY_LOCK: str = 'tellduslive_config_entry_lock'
CONFIG_ENTRY_IS_SETUP: str = 'telldus_config_entry_is_setup'
NEW_CLIENT_TASK: str = 'telldus_new_client_task'
INTERVAL_TRACKER: str = f'{DOMAIN}_INTERVAL'


async def async_setup_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    """Create a tellduslive session."""
    conf: Dict[str, Any] = entry.data[KEY_SESSION]
    if CONF_HOST in conf:
        session: Session = await hass.async_add_executor_job(partial(Session, **conf))
    else:
        session = Session(
            PUBLIC_KEY,
            NOT_SO_PRIVATE_KEY,
            application=APPLICATION_NAME,
            **conf,
        )
    if not session.is_authorized:
        _LOGGER.error('Authentication Error')
        return False
    hass.data[DATA_CONFIG_ENTRY_LOCK] = asyncio.Lock()
    hass.data[CONFIG_ENTRY_IS_SETUP] = set()
    hass.data[NEW_CLIENT_TASK] = hass.loop.create_task(
        async_new_client(hass, session, entry)
    )
    return True


async def async_new_client(
    hass: HomeAssistant, session: Session, entry: ConfigEntry
) -> None:
    """Add the hubs associated with the current client to device_registry."""
    interval: int = entry.data[KEY_SCAN_INTERVAL]
    _LOGGER.debug('Update interval %s seconds', interval)
    client: 'TelldusLiveClient' = TelldusLiveClient(
        hass, entry, session, interval
    )
    hass.data[DOMAIN] = client
    dev_reg = dr.async_get(hass)
    hubs = await client.async_get_hubs()
    for hub in hubs:
        _LOGGER.debug('Connected hub %s', hub['name'])
        dev_reg.async_get_or_create(
            config_entry_id=entry.entry_id,
            identifiers={(DOMAIN, hub['id'])},
            manufacturer='Telldus',
            name=hub['name'],
            model=hub['type'],
            sw_version=hub['version'],
        )
    await client.update()


async def async_setup(hass: HomeAssistant, config: ConfigType) -> bool:
    """Set up the Telldus Live component."""
    if DOMAIN not in config:
        return True
    host: Optional[str] = config[DOMAIN].get(CONF_HOST)
    scan_interval: int = config[DOMAIN][CONF_SCAN_INTERVAL]
    hass.async_create_task(
        hass.config_entries.flow.async_init(
            DOMAIN,
            context={'source': config_entries.SOURCE_IMPORT},
            data={
                CONF_HOST: host,
                KEY_SCAN_INTERVAL: scan_interval,
            },
        )
    )
    return True


async def async_unload_entry(hass: HomeAssistant, config_entry: ConfigEntry) -> bool:
    """Unload a config entry."""
    task: asyncio.Task = hass.data.get(NEW_CLIENT_TASK)
    if task and not task.done():
        task.cancel()
    interval_tracker: Callable[[], None] = hass.data.pop(INTERVAL_TRACKER)
    interval_tracker()
    unload_ok: bool = await hass.config_entries.async_unload_platforms(
        config_entry, hass.data.get(CONFIG_ENTRY_IS_SETUP, set())
    )
    hass.data.pop(DOMAIN, None)
    hass.data.pop(DATA_CONFIG_ENTRY_LOCK, None)
    hass.data.pop(CONFIG_ENTRY_IS_SETUP, None)
    return unload_ok


class TelldusLiveClient:
    """Get the latest data and update the states."""

    def __init__(
        self,
        hass: HomeAssistant,
        config_entry: ConfigEntry,
        session: Session,
        interval: int,
    ) -> None:
        """Initialize the TelldusLive data object."""
        self._known_devices: Set[str] = set()
        self._device_infos: Dict[str, Any] = {}
        self._hass: HomeAssistant = hass
        self._config_entry: ConfigEntry = config_entry
        self._client: Session = session
        self._interval: int = interval

    async def async_get_hubs(self) -> list:
        """Return hubs registered for the user."""
        clients = await self._hass.async_add_executor_job(self._client.get_clients)
        return clients or []

    def device_info(self, device_id: str) -> Optional[Any]:
        """Return device info."""
        return self._device_infos.get(device_id)

    @staticmethod
    def identify_device(device: Any) -> str:
        """Find out what type of HA component to create."""
        if device.is_sensor:
            return 'sensor'
        if device.methods & DIM:
            return 'light'
        if device.methods & UP:
            return 'cover'
        if device.methods & TURNON:
            return 'switch'
        if device.methods == 0:
            return 'binary_sensor'
        _LOGGER.warning('Unidentified device type (methods: %d)', device.methods)
        return 'switch'

    async def _discover(self, device_id: str) -> None:
        """Discover the component."""
        device = self._client.device(device_id)
        component: str = self.identify_device(device)
        device_info = await self._hass.async_add_executor_job(device.info)
        self._device_infos[device_id] = device_info
        async with self._hass.data[DATA_CONFIG_ENTRY_LOCK]:
            if component not in self._hass.data[CONFIG_ENTRY_IS_SETUP]:
                await self._hass.config_entries.async_forward_entry_setups(
                    self._config_entry, [component]
                )
                self._hass.data[CONFIG_ENTRY_IS_SETUP].add(component)
        device_ids: list = []
        if device.is_sensor:
            device_ids.extend(
                [
                    (device.device_id, item.name, item.scale) for item in device.items
                ]
            )
        else:
            device_ids.append(device_id)
        for _id in device_ids:
            async_dispatcher_send(
                self._hass,
                TELLDUS_DISCOVERY_NEW.format(component, DOMAIN),
                _id,
            )

    async def update(self, *args: Any) -> None:
        """Periodically poll the servers for current state."""
        try:
            success = await self._hass.async_add_executor_job(self._client.update)
            if not success:
                _LOGGER.warning('Failed request')
                return
            dev_ids: Set[str] = {dev.device_id for dev in self._client.devices}
            new_devices: Set[str] = dev_ids - self._known_devices
            for d_id in new_devices:
                await self._discover(d_id)
            self._known_devices |= new_devices
            async_dispatcher_send(self._hass, SIGNAL_UPDATE_ENTITY)
        finally:
            self._hass.data[INTERVAL_TRACKER] = async_call_later(
                self._hass, self._interval, self.update
            )

    def device(self, device_id: str) -> Any:
        """Return device representation."""
        return self._client.device(device_id)

    def is_available(self, device_id: str) -> bool:
        """Return device availability."""
        return device_id in self._client.device_ids
