"""Support for Apple HomeKit."""

from __future__ import annotations

import asyncio
from collections import defaultdict
from collections.abc import Iterable
from copy import deepcopy
import ipaddress
import logging
import os
import socket
from typing import Any, cast, Dict, List, Optional, Set, Tuple, Union

from aiohttp import web
from pyhap import util as pyhap_util
from pyhap.characteristic import Characteristic
from pyhap.const import STANDALONE_AID
from pyhap.loader import get_loader
from pyhap.service import Service
import voluptuous as vol
from zeroconf.asyncio import AsyncZeroconf

from homeassistant.components import device_automation, network, zeroconf
from homeassistant.components.binary_sensor import (
    DOMAIN as BINARY_SENSOR_DOMAIN,
    BinarySensorDeviceClass,
)
from homeassistant.components.camera import DOMAIN as CAMERA_DOMAIN
from homeassistant.components.device_automation.trigger import (
    async_validate_trigger_config,
)
from homeassistant.components.event import DOMAIN as EVENT_DOMAIN, EventDeviceClass
from homeassistant.components.http import KEY_HASS, HomeAssistantView
from homeassistant.components.humidifier import DOMAIN as HUMIDIFIER_DOMAIN
from homeassistant.components.lock import DOMAIN as LOCK_DOMAIN
from homeassistant.components.sensor import DOMAIN as SENSOR_DOMAIN, SensorDeviceClass
from homeassistant.config_entries import SOURCE_IMPORT, ConfigEntry
from homeassistant.const import (
    ATTR_BATTERY_CHARGING,
    ATTR_BATTERY_LEVEL,
    ATTR_DEVICE_ID,
    ATTR_ENTITY_ID,
    ATTR_HW_VERSION,
    ATTR_MANUFACTURER,
    ATTR_MODEL,
    ATTR_SW_VERSION,
    CONF_DEVICES,
    CONF_IP_ADDRESS,
    CONF_NAME,
    CONF_PORT,
    EVENT_HOMEASSISTANT_STOP,
    SERVICE_RELOAD,
)
from homeassistant.core import (
    CALLBACK_TYPE,
    HomeAssistant,
    ServiceCall,
    State,
    callback,
)
from homeassistant.exceptions import HomeAssistantError, Unauthorized
from homeassistant.helpers import (
    config_validation as cv,
    device_registry as dr,
    entity_registry as er,
    instance_id,
)
from homeassistant.helpers.dispatcher import async_dispatcher_connect
from homeassistant.helpers.entityfilter import (
    BASE_FILTER_SCHEMA,
    FILTER_SCHEMA,
    EntityFilter,
)
from homeassistant.helpers.reload import async_integration_yaml_config
from homeassistant.helpers.service import (
    async_extract_referenced_entity_ids,
    async_register_admin_service,
)
from homeassistant.helpers.start import async_at_started
from homeassistant.helpers.typing import ConfigType
from homeassistant.loader import IntegrationNotFound, async_get_integration
from homeassistant.util.async_ import create_eager_task

from . import (  # noqa: F401
    type_cameras,
    type_covers,
    type_fans,
    type_humidifiers,
    type_lights,
    type_locks,
    type_media_players,
    type_remotes,
    type_security_systems,
    type_sensors,
    type_switches,
    type_thermostats,
)
from .accessories import HomeAccessory, HomeBridge, HomeDriver, get_accessory
from .aidmanager import AccessoryAidStorage
from .const import (
    ATTR_INTEGRATION,
    BRIDGE_NAME,
    BRIDGE_SERIAL_NUMBER,
    CONF_ADVERTISE_IP,
    CONF_ENTITY_CONFIG,
    CONF_ENTRY_INDEX,
    CONF_EXCLUDE_ACCESSORY_MODE,
    CONF_FILTER,
    CONF_HOMEKIT_MODE,
    CONF_LINKED_BATTERY_CHARGING_SENSOR,
    CONF_LINKED_BATTERY_SENSOR,
    CONF_LINKED_DOORBELL_SENSOR,
    CONF_LINKED_HUMIDITY_SENSOR,
    CONF_LINKED_MOTION_SENSOR,
    CONFIG_OPTIONS,
    DEFAULT_EXCLUDE_ACCESSORY_MODE,
    DEFAULT_HOMEKIT_MODE,
    DEFAULT_PORT,
    DOMAIN,
    HOMEKIT_MODE_ACCESSORY,
    HOMEKIT_MODES,
    MANUFACTURER,
    PERSIST_LOCK_DATA,
    SERVICE_HOMEKIT_RESET_ACCESSORY,
    SERVICE_HOMEKIT_UNPAIR,
    SHUTDOWN_TIMEOUT,
    SIGNAL_RELOAD_ENTITIES,
)
from .iidmanager import AccessoryIIDStorage
from .models import HomeKitConfigEntry, HomeKitEntryData
from .type_triggers import DeviceTriggerAccessory
from .util import (
    accessory_friendly_name,
    async_dismiss_setup_message,
    async_port_is_available,
    async_show_setup_message,
    get_persist_fullpath_for_entry_id,
    remove_state_files_for_entry_id,
    state_needs_accessory_mode,
    validate_entity_config,
)

_LOGGER: logging.Logger = logging.getLogger(__name__)

MAX_DEVICES: int = 150  # includes the bridge

# #### Driver Status ####
STATUS_READY: int = 0
STATUS_RUNNING: int = 1
STATUS_STOPPED: int = 2
STATUS_WAIT: int = 3

PORT_CLEANUP_CHECK_INTERVAL_SECS: int = 1

_HOMEKIT_CONFIG_UPDATE_TIME: int = (
    10  # number of seconds to wait for homekit to see the c# change
)
_HAS_IPV6: bool = hasattr(socket, "AF_INET6")
_DEFAULT_BIND: List[str] = ["0.0.0.0", "::"] if _HAS_IPV6 else ["0.0.0.0"]


BATTERY_CHARGING_SENSOR: Tuple[str, BinarySensorDeviceClass] = (
    BINARY_SENSOR_DOMAIN,
    BinarySensorDeviceClass.BATTERY_CHARGING,
)
BATTERY_SENSOR: Tuple[str, SensorDeviceClass] = (SENSOR_DOMAIN, SensorDeviceClass.BATTERY)
MOTION_EVENT_SENSOR: Tuple[str, EventDeviceClass] = (EVENT_DOMAIN, EventDeviceClass.MOTION)
MOTION_SENSOR: Tuple[str, BinarySensorDeviceClass] = (BINARY_SENSOR_DOMAIN, BinarySensorDeviceClass.MOTION)
DOORBELL_EVENT_SENSOR: Tuple[str, EventDeviceClass] = (EVENT_DOMAIN, EventDeviceClass.DOORBELL)
HUMIDITY_SENSOR: Tuple[str, SensorDeviceClass] = (SENSOR_DOMAIN, SensorDeviceClass.HUMIDITY)


def _has_all_unique_names_and_ports(
    bridges: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Validate that each homekit bridge configured has a unique name."""
    names: List[str] = [bridge[CONF_NAME] for bridge in bridges]
    ports: List[int] = [bridge[CONF_PORT] for bridge in bridges]
    vol.Schema(vol.Unique())(names)
    vol.Schema(vol.Unique())(ports)
    return bridges


BRIDGE_SCHEMA: vol.Schema = vol.All(
    vol.Schema(
        {
            vol.Optional(CONF_HOMEKIT_MODE, default=DEFAULT_HOMEKIT_MODE): vol.In(
                HOMEKIT_MODES
            ),
            vol.Optional(CONF_NAME, default=BRIDGE_NAME): vol.All(
                cv.string, vol.Length(min=3, max=25)
            ),
            vol.Optional(CONF_PORT, default=DEFAULT_PORT): cv.port,
            vol.Optional(CONF_IP_ADDRESS): vol.All(ipaddress.ip_address, cv.string),
            vol.Optional(CONF_ADVERTISE_IP): vol.All(
                cv.ensure_list, [ipaddress.ip_address], [cv.string]
            ),
            vol.Optional(CONF_FILTER, default={}): BASE_FILTER_SCHEMA,
            vol.Optional(CONF_ENTITY_CONFIG, default={}): validate_entity_config,
            vol.Optional(CONF_DEVICES): cv.ensure_list,
        },
        extra=vol.ALLOW_EXTRA,
    ),
)

CONFIG_SCHEMA: vol.Schema = vol.Schema(
    {DOMAIN: vol.All(cv.ensure_list, [BRIDGE_SCHEMA], _has_all_unique_names_and_ports)},
    extra=vol.ALLOW_EXTRA,
)


RESET_ACCESSORY_SERVICE_SCHEMA: vol.Schema = vol.Schema(
    {vol.Required(ATTR_ENTITY_ID): cv.entity_ids}
)


UNPAIR_SERVICE_SCHEMA: vol.Schema = vol.All(
    vol.Schema(cv.ENTITY_SERVICE_FIELDS),
    cv.has_at_least_one_key(ATTR_DEVICE_ID),
)


def _async_all_homekit_instances(hass: HomeAssistant) -> List[HomeKit]:
    """All active HomeKit instances."""
    hk_data: Optional[HomeKitEntryData] = None
    return [
        hk_data.homekit
        for entry in hass.config_entries.async_entries(DOMAIN)
        if (hk_data := getattr(entry, "runtime_data", None))
    ]


def _async_get_imported_entries_indices(
    current_entries: List[ConfigEntry],
) -> Tuple[Dict[str, ConfigEntry], Dict[int, ConfigEntry]]:
    """Return a dicts of the entries by name and port."""

    # For backwards compat, its possible the first bridge is using the default
    # name.
    entries_by_name: Dict[str, ConfigEntry] = {}
    entries_by_port: Dict[int, ConfigEntry] = {}
    for entry in current_entries:
        if entry.source != SOURCE_IMPORT:
            continue
        entries_by_name[entry.data.get(CONF_NAME, BRIDGE_NAME)] = entry
        entries_by_port[entry.data.get(CONF_PORT, DEFAULT_PORT)] = entry
    return entries_by_name, entries_by_port


async def async_setup(hass: HomeAssistant, config: ConfigType) -> bool:
    """Set up the HomeKit from yaml."""
    hass.data[PERSIST_LOCK_DATA] = asyncio.Lock()

    # Initialize the loader before loading entries to ensure
    # there is no race where multiple entries try to load it
    # at the same time.
    await hass.async_add_executor_job(get_loader)

    _async_register_events_and_services(hass)

    if DOMAIN not in config:
        return True

    current_entries: List[ConfigEntry] = hass.config_entries.async_entries(DOMAIN)
    entries_by_name, entries_by_port = _async_get_imported_entries_indices(
        current_entries
    )

    for index, conf in enumerate(config[DOMAIN]):
        if _async_update_config_entry_from_yaml(
            hass, entries_by_name, entries_by_port, conf
        ):
            continue

        conf[CONF_ENTRY_INDEX] = index
        hass.async_create_task(
            hass.config_entries.flow.async_init(
                DOMAIN,
                context={"source": SOURCE_IMPORT},
                data=conf,
            ),
            eager_start=True,
        )

    return True


@callback
def _async_update_config_entry_from_yaml(
    hass: HomeAssistant,
    entries_by_name: Dict[str, ConfigEntry],
    entries_by_port: Dict[int, ConfigEntry],
    conf: ConfigType,
) -> bool:
    """Update a config entry with the latest yaml.

    Returns True if a matching config entry was found

    Returns False if there is no matching config entry
    """
    if not (
        matching_entry := entries_by_name.get(conf.get(CONF_NAME, BRIDGE_NAME))
        or entries_by_port.get(conf.get(CONF_PORT, DEFAULT_PORT))
    ):
        return False

    # If they alter the yaml config we import the changes
    # since there currently is no practical way to support
    # all the options in the UI at this time.
    data: Dict[str, Any] = conf.copy()
    options: Dict[str, Any] = {}
    for key in CONFIG_OPTIONS:
        if key in data:
            options[key] = data[key]
            del data[key]

    hass.config_entries.async_update_entry(matching_entry, data=data, options=options)
    return True


async def async_setup_entry(hass: HomeAssistant, entry: HomeKitConfigEntry) -> bool:
    """Set up HomeKit from a config entry."""
    _async_import_options_from_data_if_missing(hass, entry)

    conf: Dict[str, Any] = entry.data
    options: Dict[str, Any] = entry.options

    name: str = conf[CONF_NAME]
    port: int = conf[CONF_PORT]
    _LOGGER.debug("Begin setup HomeKit for %s", name)

    # ip_address and advertise_ip are yaml only
    ip_address: Union[str, List[str]] = conf.get(CONF_IP_ADDRESS, _DEFAULT_BIND)
    advertise_ips: List[str] = conf.get(
        CONF_ADVERTISE_IP
    ) or await network.async_get_announce_addresses(hass)

    # exclude_accessory_mode is only used for config flow
    # to indicate that the config entry was setup after
    # we started creating config entries for entities that
    # to run in accessory mode and that we should never include
    # these entities on the bridge. For backwards compatibility
    # with users who have not migrated yet we do not do exclude
    # these entities by default as we cannot migrate automatically
    # since it requires a re-pairing.
    exclude_accessory_mode: bool = conf.get(
        CONF_EXCLUDE_ACCESSORY_MODE, DEFAULT_EXCLUDE_ACCESSORY_MODE
    )
    homekit_mode: str = options.get(CONF_HOMEKIT_MODE, DEFAULT_HOMEKIT_MODE)
    entity_config: Dict[str, Any] = options.get(CONF_ENTITY_CONFIG, {}).copy()
    entity_filter: EntityFilter = FILTER_SCHEMA(options.get(CONF_FILTER, {}))
    devices: List[str] = options.get(CONF_DEVICES, [])

    homekit: HomeKit = HomeKit(
        hass,
        name,
        port,
        ip_address,
        entity_filter,
        exclude_accessory_mode,
        entity_config,
        homekit_mode,
        advertise_ips,
        entry.entry_id,
        entry.title,
        devices=devices,
    )

    entry.async_on_unload(entry.add_update_listener(_async_update_listener))
    entry.async_on_unload(
        hass.bus.async_listen_once(EVENT_HOMEASSISTANT_STOP, homekit.async_stop)
    )

    entry_data: HomeKitEntryData = HomeKitEntryData(
        homekit=homekit, pairing_qr=None, pairing_qr_secret=None
    )
    entry.runtime_data = entry_data

    async def _async_start_homekit(hass: HomeAssistant) -> None:
        await homekit.async_start()

    entry.async_on_unload(async_at_started(hass, _async_start_homekit))

    return True


async def _async_update_listener(
    hass: HomeAssistant, entry: HomeKitConfigEntry
) -> None:
    """Handle options update."""
    if entry.source == SOURCE_IMPORT:
        return
    await hass.config_entries.async_reload(entry.entry_id)


async def async_unload_entry(hass: HomeAssistant, entry: HomeKitConfigEntry) -> bool:
    """Unload a config entry."""
    async_dismiss_setup_message(hass, entry.entry_id)
    entry_data: HomeKitEntryData = entry.runtime_data
    homekit: HomeKit = entry_data.homekit

    if homekit.status == STATUS_RUNNING:
        await homekit.async_stop()

    logged_shutdown_wait: bool = False
    for _ in range(SHUTDOWN_TIMEOUT):
        if async_port_is_available(entry.data[CONF_PORT]):
            break

        if not logged_shutdown_wait:
            _LOGGER.debug("Waiting for the HomeKit server to shutdown")
            logged_shutdown_wait = True

        await asyncio.sleep(PORT_CLEANUP_CHECK_INTERVAL_SECS)

    return True


async def async_remove_entry(hass: HomeAssistant, entry: HomeKitConfigEntry) -> None:
    """Remove a config entry."""
    await hass.async_add_executor_job(
        remove_state_files_for_entry_id, hass, entry.entry_id
    )


@callback
def _async_import_options_from_data_if_missing(
    hass: HomeAssistant, entry: HomeKitConfigEntry
) -> None:
    options: Dict[str, Any] = deepcopy(dict(entry.options))
    data: Dict[str, Any] = deepcopy(dict(entry.data))
    modified: bool = False
    for importable_option in CONFIG_OPTIONS:
        if importable_option not in entry.options and importable_option in entry.data:
            options[importable_option] = entry.data[importable_option]
            del data[importable_option]
            modified = True

    if modified:
        hass.config_entries.async_update_entry(entry, data=data, options=options)


@callback
def _async_register_events_and_services(hass: HomeAssistant) -> None:
    """Register events and services for HomeKit."""
    hass.http.register_view(HomeKitPairingQRView)

    async def async_handle_homekit_reset_accessory(service: ServiceCall) -> None:
        """Handle reset accessory HomeKit service call."""
        for homekit in _async_all_homekit_instances(hass):
            if homekit.status != STATUS_RUNNING:
                _LOGGER.warning(
                    "HomeKit is not running. Either it is waiting to be "
                    "started or has been stopped"
                )
                continue

            entity_ids: List[str] = cast(List[str], service.data.get("entity_id"))
            await homekit.async_reset_accessories(entity_ids)

    hass.services.async_register(
        DOMAIN,
        SERVICE_HOMEKIT_RESET_ACCESSORY,
        async_handle_homekit_reset_accessory,
        schema=RESET_ACCESSORY_SERVICE_SCHEMA,
    )

    async def async_handle_homekit_unpair(service: ServiceCall) -> None:
        """Handle unpair HomeKit service call."""
        referenced: Dict[str, Set[str]] = async_extract_referenced_entity_ids(hass, service)
        dev_reg: dr.DeviceRegistry = dr.async_get(hass)
        for device_id in referenced.referenced_devices:
            if not (dev_reg_ent := dev_reg.async_get(device_id)):
                raise HomeAssistantError(f"No device found for device id: {device_id}")
            macs: List[str] = [
                cval
                for ctype, cval in dev_reg_ent.connections
                if ctype == dr.CONNECTION_NETWORK_MAC
            ]
            matching_instances: List[HomeKit] = [
                homekit
                for homekit in _async_all_homekit_instances(hass)
                if homekit.driver and dr.format_mac(homekit.driver.state.mac) in macs
            ]
            if not matching_instances:
                raise HomeAssistantError(
                    f"No homekit accessory found for device id: {device_id}"
                )
            for homekit in matching_instances:
                homekit.async_unpair()

    hass.services.async_register(
        DOMAIN,
        SERVICE_HOMEKIT_UNPAIR,
        async_handle_homekit_unpair,
        schema=UNPAIR_SERVICE_SCHEMA,
    )

    async def _handle_homekit_reload(service: ServiceCall) -> None:
        """Handle start HomeKit service call."""
        config: Optional[ConfigType] = await async_integration_yaml_config(hass, DOMAIN)

        if not config or DOMAIN not in config:
            return

        current_entries: List[ConfigEntry] = hass.config_entries.async_entries(DOMAIN)
        entries_by_name, entries_by_port = _async_get_imported_entries_indices(
            current_entries
        )

        for conf in config[DOMAIN]:
            _async_update_config_entry_from_yaml(
                hass, entries_by_name, entries_by_port, conf
            )

        reload_tasks: List[asyncio.Task] = [
            create_eager_task(hass.config_entries.async_reload(entry.entry_id))
            for entry in current_entries
        ]

        await asyncio.gather(*reload_tasks)

    async_register_admin_service(
        hass,
        DOMAIN,
        SERVICE_RELOAD,
        _handle_homekit_reload,
    )


class HomeKit:
    """Class to handle all actions between HomeKit and Home Assistant."""

    def __init__(
        self,
        hass: HomeAssistant,
        name: str,
        port: int,
        ip_address: Optional[str],
        entity_filter: EntityFilter,
        exclude_accessory_mode: bool,
        entity_config: Dict[str, Any],
        homekit_mode: str,
        advertise_ips: List[str],
        entry_id: str,
        entry_title: str,
        devices: Optional[List[str]] = None,
    ) -> None:
        """Initialize a HomeKit object."""
        self.hass: HomeAssistant = hass
        self._name: str = name
        self._port: int = port
        self._ip_address: Optional[str] = ip_address
        self._filter: EntityFilter = entity_filter
        self._config: defaultdict[str, Dict[str, Any]] = defaultdict(
            dict, entity_config
        )
        self._exclude_accessory_mode: bool = exclude_accessory_mode
        self._advertise_ips: List[str] = advertise_ips
        self._entry_id: str = entry_id
        self._entry_title: str = entry_title
        self._homekit_mode: str = homekit_mode
        self._devices: List[str] = devices or []
        self.aid_storage: Optional[AccessoryAidStorage] = None
        self.iid_storage: Optional[AccessoryIIDStorage] = None
        self.status: int = STATUS_READY
        self.driver: Optional[HomeDriver] = None
        self.bridge: Optional[HomeBridge] = None
        self._reset_lock: asyncio.Lock = asyncio.Lock()
        self._cancel_reload_dispatcher: Optional[CALLBACK_TYPE] = None

    def setup(self, async_zeroconf_instance: AsyncZeroconf, uuid: str) -> bool:
        """Set up bridge and accessory driver.

        Returns True if data was loaded from disk

        Returns False if the persistent data was not loaded
        """
        assert self.iid_storage is not None
        persist_file: str = get_persist_fullpath_for_entry_id(self.hass, self._entry_id)
        self.driver: HomeDriver = HomeDriver(
            self.hass,
            self._entry_id,
            self._name,
            self._entry_title,
            loop=self.hass.loop,
            address=self._ip_address,
            port=self._port,
            persist_file=persist_file,
            advertised_address=self._advertise_ips,
            async_zeroconf_instance=async_zeroconf_instance,
            zeroconf_server=f"{uuid}-hap.local.",
            loader=get_loader(),
            iid_storage=self.iid_storage,
        )
        # If we do not load the mac address will be wrong
        # as pyhap uses a random one until state is restored
        if os.path.exists(persist_file):
            self.driver.load()
            return True

        # If there is no persist file, we need to generate a mac
        self.driver.state.mac: str = pyhap_util.generate_mac()
        return False

    async def async_reset_accessories(self, entity_ids: Iterable[str]) -> None:
        """Reset the accessory to load the latest configuration."""
        _LOGGER.debug("Resetting accessories: %s", entity_ids)
        async with self._reset_lock:
            if not self.bridge:
                # For accessory mode reset and reload are the same
                await self._async_reload_accessories_in_accessory_mode(entity_ids)
                return
            await self._async_reset_accessories_in_bridge_mode(entity_ids)

    async def async_reload_accessories(self, entity_ids: Iterable[str]) -> None:
        """Reload the accessory to load the latest configuration."""
        _LOGGER.debug("Reloading accessories: %s", entity_ids)
        async with self._reset_lock:
            if not self.bridge:
                await self._async_reload_accessories_in_accessory_mode(entity_ids)
                return
            await self._async_reload_accessories_in_bridge_mode(entity_ids)

    @callback
    def _async_shutdown_accessory(self, accessory: HomeAccessory) -> None:
        """Shutdown an accessory."""
        assert self.driver is not None
        accessory.async_stop()
        # Deallocate the IIDs for the accessory
        iid_manager = accessory.iid_manager
        services: List[Service] = accessory.services
        for service in services:
            iid_manager.remove_obj(service)
            characteristics: List[Characteristic] = service.characteristics
            for char in characteristics:
                iid_manager.remove_obj(char)

    async def _async_reload_accessories_in_accessory_mode(
        self, entity_ids: Iterable[str]
    ) -> None:
        """Reset accessories in accessory mode."""
        assert self.driver is not None

        acc: HomeAccessory = cast(HomeAccessory, self.driver.accessory)
        if acc.entity_id not in entity_ids:
            return
        if not (state := self.hass.states.get(acc.entity_id)):
            _LOGGER.warning(
                "The underlying entity %s disappeared during reload", acc.entity_id
            )
            return
        self._async_shutdown_accessory(acc)
        if new_acc := self._async_create_single_accessory([state])):
            self.driver.accessory = new_acc
            new_acc.run()
            self._async_update_accessories_hash()

    def _async_remove_accessories_by_entity_id(
        self, entity_ids: Iterable[str]
    ) -> List[str]:
        """Remove accessories by entity id."""
        assert self.aid_storage is not None
        assert self.bridge is not None
        removed: List[str] = []
        acc: Optional[HomeAccessory] = None
        for entity_id in entity_ids:
            aid: int = self.aid_storage.get_or_allocate_aid_for_entity_id(entity_id)
            if aid not in self.bridge.accessories:
                continue
            if acc := self.async_remove_bridge_accessory(aid):
                self._async_shutdown_accessory(acc)
                removed.append(entity_id)
        return removed

    async def _async_reset_accessories_in_bridge_mode(
        self, entity_ids: Iterable[str]
    ) -> None:
        """Reset accessories in bridge mode."""
        if not (removed := self._async_remove_accessories_by_entity_id(entity_ids)):
            _LOGGER.debug("No accessories to reset in bridge mode for: %s", entity_ids)
            return
        # With a reset, we need to remove the accessories,
        # and force config change so iCloud deletes them from
        # the database.
        assert self.driver is not None
        self._async_update_accessories_hash()
        await asyncio.sleep(_HOMEKIT_CONFIG_UPDATE_TIME)
        await self._async_recreate_removed_accessories_in_bridge_mode(removed)

    async def _async_reload_accessories_in_bridge_mode(
        self, entity_ids: Iterable[str]
    ) -> None:
        """Reload accessories in bridge mode."""
        removed: List[str] = self._async_remove_accessories_by_entity_id(entity_ids)
        await self._async_recreate_removed_accessories_in_bridge_mode(removed)

    async def _async_recreate_removed_accessories_in_bridge_mode(
        self, removed: List[str]
    ) -> None:
        """Recreate removed accessories in bridge mode."""
        for entity_id in removed:
            if not (state := self.hass.states.get(entity_id)):
                _LOGGER.warning(
                    "The underlying entity %s disappeared during reload", entity_id
                )
                continue
            if acc := self.add_bridge_accessory(state):
                acc.run()
        self._async_update_accessories_hash()

    @callback
    def _async_update_accessories_hash(self) -> bool:
        """Update the accessories hash."""
        assert self.driver is not None
        driver: HomeDriver = self.driver
        old_hash: int = driver.state.accessories_hash
        new_hash: int = driver.accessories_hash
        if driver.state.set_accessories_hash(new_hash):
            _LOGGER.debug(
                "Updating HomeKit accessories hash from %s -> %s", old_hash, new_hash
            )
            driver.async_persist()
            driver.async_update_advertisement()
            return True
        _LOGGER.debug("HomeKit accessories hash is unchanged: %s", new_hash)
        return False

    def add_bridge_accessory(self, state: State) -> Optional[HomeAccessory]:
        """Try adding accessory to bridge if configured beforehand."""
        assert self.driver is not None

        if self._would_exceed_max_devices(state.entity_id):
            return None

        if state_needs_accessory_mode(state):
            if self._exclude_accessory_mode:
                return None
            _LOGGER.warning(
                (
                    "The bridge %s has entity %s. For best performance, "
                    "and to prevent unexpected unavailability, create and "
                    "pair a separate HomeKit instance in accessory mode for "
                    "this entity"
                ),
                self._name,
                state.entity_id,
            )

        assert self.aid_storage is not None
        assert self.bridge is not None
        aid: int = self.aid_storage.get_or_allocate_aid_for_entity_id(state.entity_id)
        conf: Dict[str, Any] = self._config.get(state.entity_id, {}).copy()
        # If an accessory cannot be created or added due to an exception
        # of any kind (usually in pyhap) it should not prevent
        # the rest of the accessories from being created
        try:
            acc: Optional[HomeAccessory] = get_accessory(self.hass, self.driver, state, aid, conf)
            if acc is not None:
                self.bridge.add_accessory(acc)
                return acc
        except Exception:
            _LOGGER.exception(
                "Failed to create a HomeKit accessory for %s", state.entity_id
            )
        return None

    def _would_exceed_max_devices(self, name: Optional[str]) -> bool:
        """Check if adding another devices would reach the limit and log."""
        # The bridge itself counts as an accessory
        assert self.bridge is not None
        if len(self.bridge.accessories) + 1 >= MAX_DEVICES:
            _LOGGER.warning(
                (
                    "Cannot add %s as this would exceed the %d device limit. Consider"
                    " using the filter option"
                ),
                name,
                MAX_DEVICES,
            )
            return True
        return False

    async def add_bridge_triggers_accessory(
        self, device: dr.DeviceEntry, device_triggers: List[Dict[str, Any]]
    ) -> None:
        """Add device automation triggers to the bridge."""
        if self._would_exceed_max_devices(device.name):
            return

        assert self.aid_storage is not None
        assert self.bridge is not None
        aid: int = self.aid_storage.get_or_allocate_aid(device.id, device.id)
        # If an accessory cannot be created or added due to an exception
        # of any kind (usually in pyhap) it should not prevent
        # the rest of the accessories from being created
        config: Dict[str, Any] = {}
        self._fill_config_from_device_registry_entry(device, config)
        trigger_accessory: DeviceTriggerAccessory = DeviceTriggerAccessory(
            self.hass,
            self.driver,
            device.name,
            None,
            aid,
            config,
            device_id=device.id,
            device_triggers=device_triggers,
        )
        await trigger_accessory.async_attach()
        self.bridge.add_accessory(trigger_accessory)

    @callback
    def async_remove_bridge_accessory(self, aid: int) -> Optional[HomeAccessory]:
        """Try adding accessory to bridge if configured beforehand."""
        assert self.bridge is not None
        if acc := self.bridge.accessories.pop(aid, None):
            return cast(HomeAccessory, acc)
        return None

    async def async_configure_accessories(self) -> List[State]:
        """Configure accessories for the included states."""
        dev_reg: dr.DeviceRegistry = dr.async_get(self.hass)
        ent_reg: er.EntityRegistry = er.async_get(self.hass)
        device_lookup: Dict[str, Dict[Tuple[str, Optional[str]], str]] = {}
        entity_states: List[State] = []
        entity_filter = self._filter.get_filter()
        entries = ent_reg.entities
        for state in self.hass.states.async_all():
            entity_id: str = state.entity_id
            if not entity_filter(entity_id):
                continue

            if ent_reg_ent := ent_reg.async_get(entity_id):
                if (
                    ent_reg_ent.entity_category is not None
                    or ent_reg_ent.hidden_by is not None
                ) and not self._filter.explicitly_included(entity_id):
                    continue

                await self._async_set_device_info_attributes(
                    ent_reg_ent, dev_reg, entity_id
                )
                if device_id := ent_reg_ent.device_id:
                    if device_id not in device_lookup:
                        device_lookup[device_id] = {
                            (
                                entry.domain,
                                entry.device_class or entry.original_device_class,
                            ): entry.entity_id
                            for entry in entries.get_entries_for_device_id(device_id)
                        }
                    self._async_configure_linked_sensors(
                        ent_reg_ent, device_lookup[device_id], state
                    )

            entity_states.append(state)

        return entity_states

    async def async_start(self, *args: Any) -> None:
        """Load storage and start."""
        if self.status != STATUS_READY:
            return
        self.status = STATUS_WAIT
        self._cancel_reload_dispatcher = async_dispatcher_connect(
            self.hass,
            SIGNAL_RELOAD_ENTITIES.format(self._entry_id),
            self.async_reload_accessories,
        )
        async_zc_instance: AsyncZeroconf = await zeroconf.async_get_async_instance(self.hass)
        uuid: str = await instance_id.async_get(self.hass)
        self.aid_storage: AccessoryAidStorage = AccessoryAidStorage(self.hass, self._entry_id)
        self.iid_storage: AccessoryIIDStorage = AccessoryIIDStorage(self.hass, self._entry_id)
        # Avoid gather here since it will be I/O bound anyways
        await self.aid_storage.async_initialize()
        await self.iid_storage.async_initialize()
        loaded_from_disk: bool = await self.hass.async_add_executor_job(
            self.setup, async_zc_instance, uuid
        )
        assert self.driver is not None

        if not await self._async_create_accessories():
            return
        self._async_register_bridge()
        _LOGGER.debug("Driver start for %s", self