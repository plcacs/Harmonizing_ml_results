from __future__ import annotations
import asyncio
import datetime
from datetime import timedelta
from functools import partial
import logging
from math import ceil
from typing import Any
from miio import Ceil, Device as MiioDevice, DeviceException, PhilipsBulb, PhilipsEyecare, PhilipsMoonlight
from miio.gateway.gateway import GATEWAY_MODEL_AC_V1, GATEWAY_MODEL_AC_V2, GATEWAY_MODEL_AC_V3, GatewayException
import voluptuous as vol
from homeassistant.components.light import ATTR_BRIGHTNESS, ATTR_COLOR_TEMP_KELVIN, ATTR_HS_COLOR, ColorMode, LightEntity
from homeassistant.config_entries import ConfigEntry
from homeassistant.const import ATTR_ENTITY_ID, CONF_DEVICE, CONF_HOST, CONF_MODEL, CONF_TOKEN
from homeassistant.core import HomeAssistant, ServiceCall
from homeassistant.helpers import config_validation as cv
from homeassistant.helpers.device_registry import DeviceInfo
from homeassistant.helpers.entity_platform import AddConfigEntryEntitiesCallback
from homeassistant.util import color as color_util, dt as dt_util
from .const import CONF_FLOW_TYPE, CONF_GATEWAY, DOMAIN, KEY_COORDINATOR, MODELS_LIGHT_BULB, MODELS_LIGHT_CEILING, MODELS_LIGHT_EYECARE, MODELS_LIGHT_MONO, MODELS_LIGHT_MOON, SERVICE_EYECARE_MODE_OFF, SERVICE_EYECARE_MODE_ON, SERVICE_NIGHT_LIGHT_MODE_OFF, SERVICE_NIGHT_LIGHT_MODE_ON, SERVICE_REMINDER_OFF, SERVICE_REMINDER_ON, SERVICE_SET_DELAYED_TURN_OFF, SERVICE_SET_SCENE
from .entity import XiaomiGatewayDevice, XiaomiMiioEntity
from .typing import ServiceMethodDetails

_LOGGER: logging.Logger = logging.getLogger(__name__)
DEFAULT_NAME: str = 'Xiaomi Philips Light'
DATA_KEY: str = 'light.xiaomi_miio'
CCT_MIN: int = 1
CCT_MAX: int = 100
DELAYED_TURN_OFF_MAX_DEVIATION_SECONDS: int = 4
DELAYED_TURN_OFF_MAX_DEVIATION_MINUTES: int = 1
SUCCESS: list[str] = ['ok']
ATTR_SCENE: str = 'scene'
ATTR_DELAYED_TURN_OFF: str = 'delayed_turn_off'
ATTR_TIME_PERIOD: str = 'time_period'
ATTR_NIGHT_LIGHT_MODE: str = 'night_light_mode'
ATTR_AUTOMATIC_COLOR_TEMPERATURE: str = 'automatic_color_temperature'
ATTR_REMINDER: str = 'reminder'
ATTR_EYECARE_MODE: str = 'eyecare_mode'
ATTR_SLEEP_ASSISTANT: str = 'sleep_assistant'
ATTR_SLEEP_OFF_TIME: str = 'sleep_off_time'
ATTR_TOTAL_ASSISTANT_SLEEP_TIME: str = 'total_assistant_sleep_time'
ATTR_BAND_SLEEP: str = 'band_sleep'
ATTR_BAND: str = 'band'
XIAOMI_MIIO_SERVICE_SCHEMA: vol.Schema = vol.Schema({vol.Optional(ATTR_ENTITY_ID): cv.entity_ids})
SERVICE_SCHEMA_SET_SCENE: vol.Schema = XIAOMI_MIIO_SERVICE_SCHEMA.extend({vol.Required(ATTR_SCENE): vol.All(vol.Coerce(int), vol.Clamp(min=1, max=6)})
SERVICE_SCHEMA_SET_DELAYED_TURN_OFF: vol.Schema = XIAOMI_MIIO_SERVICE_SCHEMA.extend({vol.Required(ATTR_TIME_PERIOD): cv.positive_time_period})
SERVICE_TO_METHOD: dict[str, ServiceMethodDetails] = {
    SERVICE_SET_DELAYED_TURN_OFF: ServiceMethodDetails(method='async_set_delayed_turn_off', schema=SERVICE_SCHEMA_SET_DELAYED_TURN_OFF),
    SERVICE_SET_SCENE: ServiceMethodDetails(method='async_set_scene', schema=SERVICE_SCHEMA_SET_SCENE),
    SERVICE_REMINDER_ON: ServiceMethodDetails(method='async_reminder_on'),
    SERVICE_REMINDER_OFF: ServiceMethodDetails(method='async_reminder_off'),
    SERVICE_NIGHT_LIGHT_MODE_ON: ServiceMethodDetails(method='async_night_light_mode_on'),
    SERVICE_NIGHT_LIGHT_MODE_OFF: ServiceMethodDetails(method='async_night_light_mode_off'),
    SERVICE_EYECARE_MODE_ON: ServiceMethodDetails(method='async_eyecare_mode_on'),
    SERVICE_EYECARE_MODE_OFF: ServiceMethodDetails(method='async_eyecare_mode_off')
}

async def async_setup_entry(hass: HomeAssistant, config_entry: ConfigEntry, async_add_entities: AddConfigEntryEntitiesCallback) -> None:
    entities: list[LightEntity] = []
    if config_entry.data[CONF_FLOW_TYPE] == CONF_GATEWAY:
        gateway = hass.data[DOMAIN][config_entry.entry_id][CONF_GATEWAY]
        if gateway.model not in [GATEWAY_MODEL_AC_V1, GATEWAY_MODEL_AC_V2, GATEWAY_MODEL_AC_V3]:
            entities.append(XiaomiGatewayLight(gateway, config_entry.title, config_entry.unique_id))
        sub_devices = gateway.devices
        for sub_device in sub_devices.values():
            if sub_device.device_type == 'LightBulb':
                coordinator = hass.data[DOMAIN][config_entry.entry_id][KEY_COORDINATOR][sub_device.sid]
                entities.append(XiaomiGatewayBulb(coordinator, sub_device, config_entry))
    if config_entry.data[CONF_FLOW_TYPE] == CONF_DEVICE:
        if DATA_KEY not in hass.data:
            hass.data[DATA_KEY] = {}
        host: str = config_entry.data[CONF_HOST]
        token: str = config_entry.data[CONF_TOKEN]
        name: str = config_entry.title
        model: str = config_entry.data[CONF_MODEL]
        unique_id: str = config_entry.unique_id
        _LOGGER.debug('Initializing with host %s (token %s...)', host, token[:5])
        if model in MODELS_LIGHT_EYECARE:
            light = PhilipsEyecare(host, token)
            entity = XiaomiPhilipsEyecareLamp(name, light, config_entry, unique_id)
            entities.append(entity)
            hass.data[DATA_KEY][host] = entity
            entities.append(XiaomiPhilipsEyecareLampAmbientLight(name, light, config_entry, unique_id))
        elif model in MODELS_LIGHT_CEILING:
            light = Ceil(host, token)
            entity = XiaomiPhilipsCeilingLamp(name, light, config_entry, unique_id)
            entities.append(entity)
            hass.data[DATA_KEY][host] = entity
        elif model in MODELS_LIGHT_MOON:
            light = PhilipsMoonlight(host, token)
            entity = XiaomiPhilipsMoonlightLamp(name, light, config_entry, unique_id)
            entities.append(entity)
            hass.data[DATA_KEY][host] = entity
        elif model in MODELS_LIGHT_BULB:
            light = PhilipsBulb(host, token)
            entity = XiaomiPhilipsBulb(name, light, config_entry, unique_id)
            entities.append(entity)
            hass.data[DATA_KEY][host] = entity
        elif model in MODELS_LIGHT_MONO:
            light = PhilipsBulb(host, token)
            entity = XiaomiPhilipsGenericLight(name, light, config_entry, unique_id)
            entities.append(entity)
            hass.data[DATA_KEY][host] = entity
        else:
            _LOGGER.error('Unsupported device found! Please create an issue at https://github.com/syssi/philipslight/issues and provide the following data: %s', model)
            return

        async def async_service_handler(service: ServiceCall) -> None:
            method = SERVICE_TO_METHOD[service.service]
            params = {key: value for key, value in service.data.items() if key != ATTR_ENTITY_ID}
            if (entity_ids := service.data.get(ATTR_ENTITY_ID)):
                target_devices = [dev for dev in hass.data[DATA_KEY].values() if dev.entity_id in entity_ids]
            else:
                target_devices = hass.data[DATA_KEY].values()
            update_tasks = []
            for target_device in target_devices:
                if not hasattr(target_device, method.method):
                    continue
                await getattr(target_device, method.method)(**params)
                update_tasks.append(asyncio.create_task(target_device.async_update_ha_state(True)))
            if update_tasks:
                await asyncio.wait(update_tasks)
        for xiaomi_miio_service, method in SERVICE_TO_METHOD.items():
            schema = method.schema or XIAOMI_MIIO_SERVICE_SCHEMA
            hass.services.async_register(DOMAIN, xiaomi_miio_service, async_service_handler, schema=schema)
    async_add_entities(entities, update_before_add=True)

class XiaomiPhilipsAbstractLight(XiaomiMiioEntity, LightEntity):
    _attr_color_mode: ColorMode = ColorMode.BRIGHTNESS
    _attr_supported_color_modes: set[ColorMode] = {ColorMode.BRIGHTNESS}

    def __init__(self, name: str, device: Any, entry: ConfigEntry, unique_id: str) -> None:
        super().__init__(name, device, entry, unique_id)
        self._brightness: int | None = None
        self._available: bool = False
        self._state: bool | None = None
        self._state_attrs: dict[str, Any] = {}

    @property
    def available(self) -> bool:
        return self._available

    @property
    def extra_state_attributes(self) -> dict[str, Any]:
        return self._state_attrs

    @property
    def is_on(self) -> bool:
        return self._state

    @property
    def brightness(self) -> int:
        return self._brightness

    async def _try_command(self, mask_error: str, func: Any, *args, **kwargs) -> bool:
        try:
            result = await self.hass.async_add_executor_job(partial(func, *args, **kwargs))
        except DeviceException as exc:
            if self._available:
                _LOGGER.error(mask_error, exc)
                self._available = False
            return False
        _LOGGER.debug('Response received from light: %s', result)
        return result == SUCCESS

    async def async_turn_on(self, **kwargs: Any) -> None:
        if ATTR_BRIGHTNESS in kwargs:
            brightness = kwargs[ATTR_BRIGHTNESS]
            percent_brightness = ceil(100 * brightness / 255.0)
            _LOGGER.debug('Setting brightness: %s %s%%', brightness, percent_brightness)
            result = await self._try_command('Setting brightness failed: %s', self._device.set_brightness, percent_brightness)
            if result:
                self._brightness = brightness
        else:
            await self._try_command('Turning the light on failed.', self._device.on)

    async def async_turn_off(self, **kwargs: Any) -> None:
        await self._try_command('Turning the light off failed.', self._device.off)

    async def async_update(self) -> None:
        try:
            state = await self.hass.async_add_executor_job(self._device.status)
        except DeviceException as ex:
            if self._available:
                self._available = False
                _LOGGER.error('Got exception while fetching the state: %s', ex)
            return
        _LOGGER.debug('Got new state: %s', state)
        self._available = True
        self._state = state.is_on
        self._brightness = ceil(255 / 100.0 * state.brightness)

class XiaomiPhilipsGenericLight(XiaomiPhilipsAbstractLight):
    def __init__(self, name: str, device: Any, entry: ConfigEntry, unique_id: str) -> None:
        super().__init__(name, device, entry, unique_id)
        self._state_attrs.update({ATTR_SCENE: None, ATTR_DELAYED_TURN_OFF: None})

    async def async_update(self) -> None:
        try:
            state = await self.hass.async_add_executor_job(self._device.status)
        except DeviceException as ex:
            if self._available:
                self._available = False
                _LOGGER.error('Got exception while fetching the state: %s', ex)
            return
        _LOGGER.debug('Got new state: %s', state)
        self._available = True
        self._state = state.is_on
        self._brightness = ceil(255 / 100.0 * state.brightness)
        delayed_turn_off = self.delayed_turn_off_timestamp(state.delay_off_countdown, dt_util.utcnow(), self._state_attrs[ATTR_DELAYED_TURN_OFF])
        self._state_attrs.update({ATTR_SCENE: state.scene, ATTR_DELAYED_TURN_OFF: delayed_turn_off})

    async def async_set_scene(self, scene: int) -> None:
        await self._try_command('Setting a fixed scene failed.', self._device.set_scene, scene)

    async def async_set_delayed_turn_off(self, time_period: timedelta) -> None:
        await self._try_command('Setting the turn off delay failed.', self._device.delay_off, time_period.total_seconds())

    @staticmethod
    def delayed_turn_off_timestamp(countdown: int, current: datetime.datetime, previous: datetime.datetime) -> datetime.datetime | None:
        if countdown is not None and countdown > 0:
            new = current.replace(microsecond=0) + timedelta(seconds=countdown)
            if previous is None:
                return new
            lower = timedelta(seconds=-DELAYED_TURN_OFF_MAX_DEVIATION_SECONDS)
            upper = timedelta(seconds=DELAYED_TURN_OFF_MAX_DEVIATION_SECONDS)
            diff = previous - new
            if lower < diff < upper:
                return previous
            return new
        return None

class XiaomiPhilipsBulb(XiaomiPhilipsGenericLight):
    _attr_color_mode: ColorMode = ColorMode.COLOR_TEMP
    _attr_supported_color_modes: set[ColorMode] = {ColorMode.COLOR_TEMP}

    def __init__(self, name: str, device: Any, entry: ConfigEntry, unique_id: str) -> None:
        super().__init__(name, device, entry, unique_id)
        self._color_temp: int | None = None

    @property
    def _current_mireds(self) -> int:
        return self._color_temp

    @property
    def _min_mireds(self) -> int:
        return 175

    @property
    def _max_mireds(self) -> int:
        return 333

    @property
    def color_temp_kelvin(self) -> int | None:
        return color_util.color_temperature_mired_to_kelvin(self._color_temp) if self._color_temp else None

    @property
    def min_color_temp_kelvin(self) -> int:
        return color_util.color_temperature_mired_to_kelvin(self._max_mireds)

    @property
    def max_color_temp_kelvin(self) -> int:
        return color_util.color_temperature_mired_to_kelvin(self._min_mireds)

    async def async_turn_on(self, **kwargs: Any) -> None:
        if ATTR_COLOR_TEMP_KELVIN in kwargs:
            color_temp = color_util.color_temperature_kelvin_to_mired(kwargs[ATTR_COLOR_TEMP_KELVIN])
            percent_color_temp = self.translate(color_temp, self._max_mireds, self._min_mireds, CCT_MIN, CCT_MAX)
        if ATTR_BRIGHTNESS in kwargs:
            brightness = kwargs[ATTR_BRIGHTNESS]
            percent_brightness = ceil(100 * brightness / 255.0)
        if ATTR_BRIGHTNESS in kwargs and ATTR_COLOR_TEMP_KELVIN in kwargs:
            _LOGGER.debug('Setting brightness and color temperature: %s %s%%, %s mireds, %s%% cct', brightness, percent_brightness, color_temp, percent_color_temp)
            result = await self._try_command('Setting brightness and color temperature failed: %s bri, %s cct', self._device.set_brightness_and_color_temperature, percent_brightness, percent_color_temp)
            if result:
                self._color_temp = color_temp
                self._brightness = brightness
        elif ATTR_COLOR_TEMP_KELVIN in kwargs:
            _LOGGER.debug('Setting color temperature: %s mireds, %s%% cct', color_temp, percent_color_temp)
            result = await self._try_command('Setting color temperature failed: %s cct', self._device.set_color_temperature, percent_color_temp)
            if result:
                self._color_temp = color_temp
        elif ATTR_BRIGHTNESS in kwargs:
            brightness = kwargs[ATTR_BRIGHTNESS]
            percent_brightness = ceil(100 * brightness / 255.0)
            _LOGGER.debug('Setting brightness: %s %s%%', brightness, percent_brightness)
            result = await self._try_command('Setting brightness failed: %s', self._device.set_brightness, percent_brightness)
            if result:
                self._brightness = brightness
        else:
            await self._try_command('Turning the light on failed.', self._device.on)

    async def async_update(self) -> None:
        try:
            state = await self.hass.async_add_executor_job(self._device.status)
        except DeviceException as ex:
            if self._available:
                self._available = False
                _LOGGER.error('Got exception while fetching the state: %s', ex)
            return
        _LOGGER.debug('Got new state: %s', state)
        self._available = True
        self._state = state.is_on
        self._brightness = ceil(255 / 100.0 * state.brightness)
        self._color_temp = self.translate(state.color_temperature, CCT_MIN, CCT_MAX, self._max_mireds, self._min_mireds)
        delayed_turn_off = self.delayed_turn_off_timestamp(state.delay_off_countdown, dt_util.utcnow(), self._state_attrs[ATTR_DELAYED_TURN_OFF])
        self._state_attrs.update({ATTR_SCENE: state.scene, ATTR_DELAYED_TURN_OFF: delayed_turn_off})

    @staticmethod
    def translate(value: int, left_min: int, left_max: int, right_min: int, right_max: int) -> int:
        left_span = left_max - left_min
        right_span = right_max - right_min
        value_scaled = float(value - left_min) / float(left_span)
        return int(right_min + value_scaled * right_span)

class XiaomiPhilipsCeilingLamp(XiaomiPhilipsBulb):
    def __init__(self, name: str, device: Any, entry: ConfigEntry, unique_id: str) -> None:
        super().__init__(name, device, entry, unique_id)
        self._state_attrs.update({ATTR_NIGHT_LIGHT_MODE: None, ATTR_AUTOMATIC_COLOR_TEMPERATURE: None})

    @property
    def _min_mireds(self) -> int:
        return 175

    @property
    def _max_mireds(self) -> int:
        return 370

    async def async_update(self) -> None:
        try:
            state = await self.hass.async_add_executor_job(self._device.status)
        except DeviceException as ex:
            if self._available:
                self._available = False
                _LOGGER.error('Got exception while fetching the state: %s', ex)
            return
        _LOGGER.debug('Got new state: %s', state)
        self._available = True
        self._state = state.is_on
        self._brightness = ceil(255 / 100.0 * state.brightness)
        self._color_temp = self.translate(state.color_temperature, CCT_MIN, CCT_MAX, self._max_mireds, self._min_mireds)
        delayed_turn_off = self.delayed_turn_off_timestamp(state.delay_off_countdown