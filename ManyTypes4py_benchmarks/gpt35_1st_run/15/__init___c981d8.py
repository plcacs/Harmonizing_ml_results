from __future__ import annotations
import asyncio
from collections.abc import Iterable
from http import HTTPStatus
import importlib
import logging
from aiohttp.client_exceptions import ClientConnectionError, ClientResponseError
from pysmartapp.event import EVENT_TYPE_DEVICE
from pysmartthings import APIInvalidGrant, Attribute, Capability, SmartThings
from homeassistant.config_entries import SOURCE_IMPORT, ConfigEntry
from homeassistant.const import CONF_ACCESS_TOKEN, CONF_CLIENT_ID, CONF_CLIENT_SECRET
from homeassistant.core import HomeAssistant
from homeassistant.exceptions import ConfigEntryAuthFailed, ConfigEntryError, ConfigEntryNotReady
from homeassistant.helpers import config_validation as cv
from homeassistant.helpers.aiohttp_client import async_get_clientsession
from homeassistant.helpers.dispatcher import async_dispatcher_send
from homeassistant.helpers.event import async_track_time_interval
from homeassistant.helpers.typing import ConfigType
from homeassistant.loader import async_get_loaded_integration
from homeassistant.setup import SetupPhases, async_pause_setup
from .config_flow import SmartThingsFlowHandler
from .const import CONF_APP_ID, CONF_INSTALLED_APP_ID, CONF_LOCATION_ID, CONF_REFRESH_TOKEN, DATA_BROKERS, DATA_MANAGER, DOMAIN, EVENT_BUTTON, PLATFORMS, SIGNAL_SMARTTHINGS_UPDATE, TOKEN_REFRESH_INTERVAL
from .smartapp import format_unique_id, setup_smartapp, setup_smartapp_endpoint, smartapp_sync_subscriptions, unload_smartapp_endpoint, validate_installed_app, validate_webhook_requirements
_LOGGER: logging.Logger = logging.getLogger(__name__)
CONFIG_SCHEMA: ConfigType = cv.config_entry_only_config_schema(DOMAIN)

async def async_setup(hass: HomeAssistant, config: ConfigType) -> bool:
    await setup_smartapp_endpoint(hass, False)
    return True

async def async_migrate_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    hass.async_create_task(hass.config_entries.async_remove(entry.entry_id))
    if not hass.config_entries.flow.async_progress_by_handler(DOMAIN):
        hass.async_create_task(hass.config_entries.flow.async_init(DOMAIN, context={'source': SOURCE_IMPORT}))
    return False

async def async_setup_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    if entry.unique_id is None:
        hass.config_entries.async_update_entry(entry, unique_id=format_unique_id(entry.data[CONF_APP_ID], entry.data[CONF_LOCATION_ID]))
    if not validate_webhook_requirements(hass):
        _LOGGER.warning("The 'base_url' of the 'http' integration must be configured and start with 'https://'")
        return False
    api: SmartThings = SmartThings(async_get_clientsession(hass), entry.data[CONF_ACCESS_TOKEN])
    await async_get_loaded_integration(hass, DOMAIN).async_get_platforms(PLATFORMS)
    try:
        manager = hass.data[DOMAIN][DATA_MANAGER]
        smart_app = manager.smartapps.get(entry.data[CONF_APP_ID])
        if not smart_app:
            app = await api.app(entry.data[CONF_APP_ID])
            smart_app = setup_smartapp(hass, app)
        installed_app = await validate_installed_app(api, entry.data[CONF_INSTALLED_APP_ID])
        scenes = await async_get_entry_scenes(entry, api)
        token = await api.generate_tokens(entry.data[CONF_CLIENT_ID], entry.data[CONF_CLIENT_SECRET], entry.data[CONF_REFRESH_TOKEN])
        hass.config_entries.async_update_entry(entry, data={**entry.data, CONF_REFRESH_TOKEN: token.refresh_token})
        devices = await api.devices(location_ids=[installed_app.location_id])

        async def retrieve_device_status(device):
            try:
                await device.status.refresh()
            except ClientResponseError:
                _LOGGER.debug('Unable to update status for device: %s (%s), the device will be excluded', device.label, device.device_id, exc_info=True)
                devices.remove(device)
        await asyncio.gather(*(retrieve_device_status(d) for d in devices.copy()))
        await smartapp_sync_subscriptions(hass, token.access_token, installed_app.location_id, installed_app.installed_app_id, devices)
        with async_pause_setup(hass, SetupPhases.WAIT_IMPORT_PLATFORMS):
            broker = await hass.async_add_import_executor_job(DeviceBroker, hass, entry, token, smart_app, devices, scenes)
        broker.connect()
        hass.data[DOMAIN][DATA_BROKERS][entry.entry_id] = broker
    except APIInvalidGrant as ex:
        raise ConfigEntryAuthFailed from ex
    except ClientResponseError as ex:
        if ex.status in (HTTPStatus.UNAUTHORIZED, HTTPStatus.FORBIDDEN):
            raise ConfigEntryError('The access token is no longer valid. Please remove the integration and set up again.') from ex
        _LOGGER.debug(ex, exc_info=True)
        raise ConfigEntryNotReady from ex
    except (ClientConnectionError, RuntimeWarning) as ex:
        _LOGGER.debug(ex, exc_info=True)
        raise ConfigEntryNotReady from ex
    await hass.config_entries.async_forward_entry_setups(entry, PLATFORMS)
    return True

async def async_get_entry_scenes(entry: ConfigEntry, api: SmartThings) -> list:
    try:
        return await api.scenes(location_id=entry.data[CONF_LOCATION_ID])
    except ClientResponseError as ex:
        if ex.status == HTTPStatus.FORBIDDEN:
            _LOGGER.exception("Unable to load scenes for configuration entry '%s' because the access token does not have the required access", entry.title)
        else:
            raise
    return []

async def async_unload_entry(hass: HomeAssistant, entry: ConfigEntry) -> None:
    broker = hass.data[DOMAIN][DATA_BROKERS].pop(entry.entry_id, None)
    if broker:
        broker.disconnect()
    return await hass.config_entries.async_unload_platforms(entry, PLATFORMS)

async def async_remove_entry(hass: HomeAssistant, entry: ConfigEntry) -> None:
    api: SmartThings = SmartThings(async_get_clientsession(hass), entry.data[CONF_ACCESS_TOKEN])
    installed_app_id = entry.data[CONF_INSTALLED_APP_ID]
    try:
        await api.delete_installed_app(installed_app_id)
    except ClientResponseError as ex:
        if ex.status == HTTPStatus.FORBIDDEN:
            _LOGGER.debug('Installed app %s has already been removed', installed_app_id, exc_info=True)
        else:
            raise
    _LOGGER.debug('Removed installed app %s', installed_app_id)
    all_entries = hass.config_entries.async_entries(DOMAIN)
    app_id = entry.data[CONF_APP_ID]
    app_count = sum((1 for entry in all_entries if entry.data[CONF_APP_ID] == app_id))
    if app_count > 1:
        _LOGGER.debug('App %s was not removed because it is in use by other configuration entries', app_id)
        return
    try:
        await api.delete_app(app_id)
    except ClientResponseError as ex:
        if ex.status == HTTPStatus.FORBIDDEN:
            _LOGGER.debug('App %s has already been removed', app_id, exc_info=True)
        else:
            raise
    _LOGGER.debug('Removed app %s', app_id)
    if len(all_entries) == 1:
        await unload_smartapp_endpoint(hass)

class DeviceBroker:
    def __init__(self, hass: HomeAssistant, entry: ConfigEntry, token: Token, smart_app: SmartApp, devices: Iterable, scenes: Iterable):
        self._hass = hass
        self._entry = entry
        self._installed_app_id = entry.data[CONF_INSTALLED_APP_ID]
        self._smart_app = smart_app
        self._token = token
        self._event_disconnect = None
        self._regenerate_token_remove = None
        self._assignments = self._assign_capabilities(devices)
        self.devices = {device.device_id: device for device in devices}
        self.scenes = {scene.scene_id: scene for scene in scenes}

    def _assign_capabilities(self, devices: Iterable) -> dict:
        assignments = {}
        for device in devices:
            capabilities = device.capabilities.copy()
            slots = {}
            for platform in PLATFORMS:
                platform_module = importlib.import_module(f'.{platform}', self.__module__)
                if not hasattr(platform_module, 'get_capabilities'):
                    continue
                assigned = platform_module.get_capabilities(capabilities)
                if not assigned:
                    continue
                for capability in assigned:
                    if capability not in capabilities:
                        continue
                    capabilities.remove(capability)
                    slots[capability] = platform
            assignments[device.device_id] = slots
        return assignments

    def connect(self) -> None:
        async def regenerate_refresh_token(now):
            await self._token.refresh(self._entry.data[CONF_CLIENT_ID], self._entry.data[CONF_CLIENT_SECRET])
            self._hass.config_entries.async_update_entry(self._entry, data={**self._entry.data, CONF_REFRESH_TOKEN: self._token.refresh_token})
            _LOGGER.debug('Regenerated refresh token for installed app: %s', self._installed_app_id)
        self._regenerate_token_remove = async_track_time_interval(self._hass, regenerate_refresh_token, TOKEN_REFRESH_INTERVAL)
        self._event_disconnect = self._smart_app.connect_event(self._event_handler)

    def disconnect(self) -> None:
        if self._regenerate_token_remove:
            self._regenerate_token_remove()
        if self._event_disconnect:
            self._event_disconnect()

    def get_assigned(self, device_id: str, platform: str) -> list:
        slots = self._assignments.get(device_id, {})
        return [key for key, value in slots.items() if value == platform]

    def any_assigned(self, device_id: str, platform: str) -> bool:
        slots = self._assignments.get(device_id, {})
        return any((value for value in slots.values() if value == platform))

    async def _event_handler(self, req, resp, app) -> None:
        if req.installed_app_id != self._installed_app_id:
            return
        updated_devices = set()
        for evt in req.events:
            if evt.event_type != EVENT_TYPE_DEVICE:
                continue
            if not (device := self.devices.get(evt.device_id)):
                continue
            device.status.apply_attribute_update(evt.component_id, evt.capability, evt.attribute, evt.value, data=evt.data)
            if evt.capability == Capability.button and evt.attribute == Attribute.button:
                data = {'component_id': evt.component_id, 'device_id': evt.device_id, 'location_id': evt.location_id, 'value': evt.value, 'name': device.label, 'data': evt.data}
                self._hass.bus.async_fire(EVENT_BUTTON, data)
                _LOGGER.debug('Fired button event: %s', data)
            else:
                data = {'location_id': evt.location_id, 'device_id': evt.device_id, 'component_id': evt.component_id, 'capability': evt.capability, 'attribute': evt.attribute, 'value': evt.value, 'data': evt.data}
                _LOGGER.debug('Push update received: %s', data)
            updated_devices.add(device.device_id)
        async_dispatcher_send(self._hass, SIGNAL_SMARTTHINGS_UPDATE, updated_devices)
