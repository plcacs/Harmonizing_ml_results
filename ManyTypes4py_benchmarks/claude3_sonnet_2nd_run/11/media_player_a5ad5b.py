"""Vizio SmartCast Device support."""
from __future__ import annotations
from datetime import timedelta
import logging
from typing import Any, Dict, List, Optional, Set, Union, cast

from pyvizio import AppConfig, VizioAsync
from pyvizio.api.apps import find_app_name
from pyvizio.const import APP_HOME, INPUT_APPS, NO_APP_RUNNING, UNKNOWN_APP

from homeassistant.components.media_player import MediaPlayerDeviceClass, MediaPlayerEntity, MediaPlayerEntityFeature, MediaPlayerState
from homeassistant.config_entries import ConfigEntry
from homeassistant.const import CONF_ACCESS_TOKEN, CONF_DEVICE_CLASS, CONF_EXCLUDE, CONF_HOST, CONF_INCLUDE, CONF_NAME
from homeassistant.core import HomeAssistant, callback
from homeassistant.helpers import device_registry as dr, entity_platform
from homeassistant.helpers.aiohttp_client import async_get_clientsession
from homeassistant.helpers.device_registry import DeviceInfo
from homeassistant.helpers.dispatcher import async_dispatcher_connect, async_dispatcher_send
from homeassistant.helpers.entity_platform import AddConfigEntryEntitiesCallback

from .const import CONF_ADDITIONAL_CONFIGS, CONF_APPS, CONF_VOLUME_STEP, DEFAULT_TIMEOUT, DEFAULT_VOLUME_STEP, DEVICE_ID, DOMAIN, SERVICE_UPDATE_SETTING, SUPPORTED_COMMANDS, UPDATE_SETTING_SCHEMA, VIZIO_AUDIO_SETTINGS, VIZIO_DEVICE_CLASSES, VIZIO_MUTE, VIZIO_MUTE_ON, VIZIO_SOUND_MODE, VIZIO_VOLUME
from .coordinator import VizioAppsDataUpdateCoordinator

_LOGGER = logging.getLogger(__name__)
SCAN_INTERVAL = timedelta(seconds=30)
PARALLEL_UPDATES = 0

async def async_setup_entry(
    hass: HomeAssistant, 
    config_entry: ConfigEntry, 
    async_add_entities: AddConfigEntryEntitiesCallback
) -> None:
    """Set up a Vizio media player entry."""
    host: str = config_entry.data[CONF_HOST]
    token: Optional[str] = config_entry.data.get(CONF_ACCESS_TOKEN)
    name: str = config_entry.data[CONF_NAME]
    device_class: str = config_entry.data[CONF_DEVICE_CLASS]
    volume_step: int = config_entry.options.get(
        CONF_VOLUME_STEP, 
        config_entry.data.get(CONF_VOLUME_STEP, DEFAULT_VOLUME_STEP)
    )
    
    params: Dict[str, Any] = {}
    
    if not config_entry.options:
        params['options'] = {CONF_VOLUME_STEP: volume_step}
        include_or_exclude_key: Optional[str] = next(
            (key for key in config_entry.data.get(CONF_APPS, {}) if key in (CONF_INCLUDE, CONF_EXCLUDE)), 
            None
        )
        if include_or_exclude_key:
            params['options'][CONF_APPS] = {
                include_or_exclude_key: config_entry.data[CONF_APPS][include_or_exclude_key].copy()
            }
    
    if not config_entry.data.get(CONF_VOLUME_STEP):
        new_data = config_entry.data.copy()
        new_data.update({CONF_VOLUME_STEP: volume_step})
        params['data'] = new_data
    
    if params:
        hass.config_entries.async_update_entry(config_entry, **params)
    
    device: VizioAsync = VizioAsync(
        DEVICE_ID, 
        host, 
        name, 
        auth_token=token, 
        device_type=VIZIO_DEVICE_CLASSES[device_class], 
        session=async_get_clientsession(hass, False), 
        timeout=DEFAULT_TIMEOUT
    )
    
    apps_coordinator: Optional[VizioAppsDataUpdateCoordinator] = hass.data[DOMAIN].get(CONF_APPS)
    entity: VizioDevice = VizioDevice(config_entry, device, name, device_class, apps_coordinator)
    
    async_add_entities([entity], update_before_add=True)
    
    platform = entity_platform.async_get_current_platform()
    platform.async_register_entity_service(
        SERVICE_UPDATE_SETTING, 
        UPDATE_SETTING_SCHEMA, 
        "async_update_setting"
    )

class VizioDevice(MediaPlayerEntity):
    """Media Player implementation which performs REST requests to device."""
    
    _attr_has_entity_name: bool = True
    _attr_name: Optional[str] = None
    _received_device_info: bool = False

    def __init__(
        self, 
        config_entry: ConfigEntry, 
        device: VizioAsync, 
        name: str, 
        device_class: str, 
        apps_coordinator: Optional[VizioAppsDataUpdateCoordinator]
    ) -> None:
        """Initialize Vizio device."""
        self._config_entry: ConfigEntry = config_entry
        self._apps_coordinator: Optional[VizioAppsDataUpdateCoordinator] = apps_coordinator
        self._volume_step: int = config_entry.options[CONF_VOLUME_STEP]
        self._current_input: Optional[str] = None
        self._current_app_config: Optional[AppConfig] = None
        self._available_inputs: List[str] = []
        self._available_apps: List[str] = []
        self._all_apps: Optional[List[Dict[str, Any]]] = apps_coordinator.data if apps_coordinator else None
        self._conf_apps: Dict[str, Any] = config_entry.options.get(CONF_APPS, {})
        self._additional_app_configs: List[Dict[str, Any]] = config_entry.data.get(CONF_APPS, {}).get(CONF_ADDITIONAL_CONFIGS, [])
        self._device: VizioAsync = device
        self._max_volume: float = float(device.get_max_volume())
        self._attr_assumed_state: bool = True
        self._attr_sound_mode_list: List[str] = []
        self._attr_supported_features: MediaPlayerEntityFeature = SUPPORTED_COMMANDS[device_class]
        unique_id: Optional[str] = config_entry.unique_id
        assert unique_id
        self._attr_unique_id: str = unique_id
        self._attr_device_class: str = device_class
        self._attr_device_info: DeviceInfo = DeviceInfo(
            identifiers={(DOMAIN, unique_id)}, 
            manufacturer="VIZIO", 
            name=name
        )

    def _apps_list(self, apps: List[str]) -> List[str]:
        """Return process apps list based on configured filters."""
        if self._conf_apps.get(CONF_INCLUDE):
            return [app for app in apps if app in self._conf_apps[CONF_INCLUDE]]
        if self._conf_apps.get(CONF_EXCLUDE):
            return [app for app in apps if app not in self._conf_apps[CONF_EXCLUDE]]
        return apps

    async def async_update(self) -> None:
        """Retrieve latest state of the device."""
        is_on: Optional[bool] = await self._device.get_power_state(log_api_exception=False)
        
        if is_on is None:
            if self._attr_available:
                _LOGGER.warning("Lost connection to %s", self._config_entry.data[CONF_HOST])
                self._attr_available = False
            return
        
        if not self._attr_available:
            _LOGGER.warning("Restored connection to %s", self._config_entry.data[CONF_HOST])
            self._attr_available = True
        
        if not self._received_device_info:
            device_reg = dr.async_get(self.hass)
            assert self._config_entry.unique_id
            device = device_reg.async_get_device(identifiers={(DOMAIN, self._config_entry.unique_id)})
            if device:
                device_reg.async_update_device(
                    device.id, 
                    model=await self._device.get_model_name(log_api_exception=False), 
                    sw_version=await self._device.get_version(log_api_exception=False)
                )
                self._received_device_info = True
        
        if not is_on:
            self._attr_state = MediaPlayerState.OFF
            self._attr_volume_level = None
            self._attr_is_volume_muted = None
            self._current_input = None
            self._attr_app_name = None
            self._current_app_config = None
            self._attr_sound_mode = None
            return
        
        self._attr_state = MediaPlayerState.ON
        
        audio_settings: Optional[Dict[str, Any]] = await self._device.get_all_settings(VIZIO_AUDIO_SETTINGS, log_api_exception=False)
        if audio_settings:
            self._attr_volume_level = float(audio_settings[VIZIO_VOLUME]) / self._max_volume
            if VIZIO_MUTE in audio_settings:
                self._attr_is_volume_muted = audio_settings[VIZIO_MUTE].lower() == VIZIO_MUTE_ON
            else:
                self._attr_is_volume_muted = None
            
            if VIZIO_SOUND_MODE in audio_settings:
                self._attr_supported_features |= MediaPlayerEntityFeature.SELECT_SOUND_MODE
                self._attr_sound_mode = audio_settings[VIZIO_SOUND_MODE]
                if not self._attr_sound_mode_list:
                    self._attr_sound_mode_list = await self._device.get_setting_options(
                        VIZIO_AUDIO_SETTINGS, 
                        VIZIO_SOUND_MODE, 
                        log_api_exception=False
                    )
            else:
                self._attr_supported_features &= ~MediaPlayerEntityFeature.SELECT_SOUND_MODE
        
        input_: Optional[str] = await self._device.get_current_input(log_api_exception=False)
        if input_:
            self._current_input = input_
        
        inputs: Optional[List[Any]] = await self._device.get_inputs_list(log_api_exception=False)
        if not inputs:
            return
        
        self._available_inputs = [input_.name for input_ in inputs]
        
        if self._attr_device_class == MediaPlayerDeviceClass.SPEAKER or not any((app for app in INPUT_APPS if app in self._available_inputs)):
            return
        
        self._available_apps = self._apps_list([app["name"] for app in self._all_apps or ()])
        self._current_app_config = await self._device.get_current_app_config(log_api_exception=False)
        self._attr_app_name = find_app_name(
            self._current_app_config, 
            [APP_HOME, *(self._all_apps or ()), *self._additional_app_configs]
        )
        
        if self._attr_app_name == NO_APP_RUNNING:
            self._attr_app_name = None

    def _get_additional_app_names(self) -> List[str]:
        """Return list of additional apps that were included in configuration.yaml."""
        return [additional_app["name"] for additional_app in self._additional_app_configs]

    @staticmethod
    async def _async_send_update_options_signal(hass: HomeAssistant, config_entry: ConfigEntry) -> None:
        """Send update event when Vizio config entry is updated."""
        async_dispatcher_send(hass, config_entry.entry_id, config_entry)

    async def _async_update_options(self, config_entry: ConfigEntry) -> None:
        """Update options if the update signal comes from this entity."""
        self._volume_step = config_entry.options[CONF_VOLUME_STEP]
        self._conf_apps.update(config_entry.options.get(CONF_APPS, {}))

    async def async_update_setting(self, setting_type: str, setting_name: str, new_value: Any) -> None:
        """Update a setting when update_setting service is called."""
        await self._device.set_setting(setting_type, setting_name, new_value, log_api_exception=False)

    async def async_added_to_hass(self) -> None:
        """Register callbacks when entity is added."""
        self.async_on_remove(self._config_entry.add_update_listener(self._async_send_update_options_signal))
        self.async_on_remove(async_dispatcher_connect(self.hass, self._config_entry.entry_id, self._async_update_options))
        
        if not self._apps_coordinator:
            return

        @callback
        def apps_list_update() -> None:
            """Update list of all apps."""
            if not self._apps_coordinator:
                return
            self._all_apps = self._apps_coordinator.data
            self.async_write_ha_state()
            
        self.async_on_remove(self._apps_coordinator.async_add_listener(apps_list_update))

    @property
    def source(self) -> Optional[str]:
        """Return current input of the device."""
        if self._attr_app_name is not None and self._current_input in INPUT_APPS:
            return self._attr_app_name
        return self._current_input

    @property
    def source_list(self) -> List[str]:
        """Return list of available inputs of the device."""
        if self._available_apps:
            return [
                *(_input for _input in self._available_inputs if _input not in INPUT_APPS), 
                *self._available_apps, 
                *(app for app in self._get_additional_app_names() if app not in self._available_apps)
            ]
        return self._available_inputs

    @property
    def app_id(self) -> Optional[Dict[str, Any]]:
        """Return the ID of the current app if it is unknown by pyvizio."""
        if self._current_app_config and self.source == UNKNOWN_APP:
            return {
                "APP_ID": self._current_app_config.APP_ID, 
                "NAME_SPACE": self._current_app_config.NAME_SPACE, 
                "MESSAGE": self._current_app_config.MESSAGE
            }
        return None

    async def async_select_sound_mode(self, sound_mode: str) -> None:
        """Select sound mode."""
        if sound_mode in (self._attr_sound_mode_list or ()):
            await self._device.set_setting(VIZIO_AUDIO_SETTINGS, VIZIO_SOUND_MODE, sound_mode, log_api_exception=False)

    async def async_turn_on(self) -> None:
        """Turn the device on."""
        await self._device.pow_on(log_api_exception=False)

    async def async_turn_off(self) -> None:
        """Turn the device off."""
        await self._device.pow_off(log_api_exception=False)

    async def async_mute_volume(self, mute: bool) -> None:
        """Mute the volume."""
        if mute:
            await self._device.mute_on(log_api_exception=False)
            self._attr_is_volume_muted = True
        else:
            await self._device.mute_off(log_api_exception=False)
            self._attr_is_volume_muted = False

    async def async_media_previous_track(self) -> None:
        """Send previous channel command."""
        await self._device.ch_down(log_api_exception=False)

    async def async_media_next_track(self) -> None:
        """Send next channel command."""
        await self._device.ch_up(log_api_exception=False)

    async def async_select_source(self, source: str) -> None:
        """Select input source."""
        if source in self._available_inputs:
            await self._device.set_input(source, log_api_exception=False)
        elif source in self._get_additional_app_names():
            await self._device.launch_app_config(
                **next((app["config"] for app in self._additional_app_configs if app["name"] == source)), 
                log_api_exception=False
            )
        elif source in self._available_apps:
            await self._device.launch_app(source, self._all_apps, log_api_exception=False)

    async def async_volume_up(self) -> None:
        """Increase volume of the device."""
        await self._device.vol_up(num=self._volume_step, log_api_exception=False)
        if self._attr_volume_level is not None:
            self._attr_volume_level = min(1.0, self._attr_volume_level + self._volume_step / self._max_volume)

    async def async_volume_down(self) -> None:
        """Decrease volume of the device."""
        await self._device.vol_down(num=self._volume_step, log_api_exception=False)
        if self._attr_volume_level is not None:
            self._attr_volume_level = max(0.0, self._attr_volume_level - self._volume_step / self._max_volume)

    async def async_set_volume_level(self, volume: float) -> None:
        """Set volume level."""
        if self._attr_volume_level is not None:
            if volume > self._attr_volume_level:
                num = int(self._max_volume * (volume - self._attr_volume_level))
                await self._device.vol_up(num=num, log_api_exception=False)
                self._attr_volume_level = volume
            elif volume < self._attr_volume_level:
                num = int(self._max_volume * (self._attr_volume_level - volume))
                await self._device.vol_down(num=num, log_api_exception=False)
                self._attr_volume_level = volume

    async def async_media_play(self) -> None:
        """Play whatever media is currently active."""
        await self._device.play(log_api_exception=False)

    async def async_media_pause(self) -> None:
        """Pause whatever media is currently active."""
        await self._device.pause(log_api_exception=False)
