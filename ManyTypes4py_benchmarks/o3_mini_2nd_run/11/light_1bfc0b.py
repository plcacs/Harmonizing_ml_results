from __future__ import annotations
import asyncio
import datetime
from datetime import timedelta, datetime as dt_datetime
from functools import partial
import logging
from math import ceil
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from miio import Ceil, Device as MiioDevice, DeviceException, PhilipsBulb, PhilipsEyecare, PhilipsMoonlight
from miio.gateway.gateway import (
    GATEWAY_MODEL_AC_V1,
    GATEWAY_MODEL_AC_V2,
    GATEWAY_MODEL_AC_V3,
    GatewayException,
)
import voluptuous as vol
from homeassistant.components.light import (
    ATTR_BRIGHTNESS,
    ATTR_COLOR_TEMP_KELVIN,
    ATTR_HS_COLOR,
    ColorMode,
    LightEntity,
)
from homeassistant.config_entries import ConfigEntry
from homeassistant.const import ATTR_ENTITY_ID, CONF_DEVICE, CONF_HOST, CONF_MODEL, CONF_TOKEN
from homeassistant.core import HomeAssistant, ServiceCall
from homeassistant.helpers import config_validation as cv
from homeassistant.helpers.device_registry import DeviceInfo
from homeassistant.helpers.entity_platform import AddConfigEntryEntitiesCallback
from homeassistant.util import color as color_util, dt as dt_util
from .const import (
    CONF_FLOW_TYPE,
    CONF_GATEWAY,
    DOMAIN,
    KEY_COORDINATOR,
    MODELS_LIGHT_BULB,
    MODELS_LIGHT_CEILING,
    MODELS_LIGHT_EYECARE,
    MODELS_LIGHT_MONO,
    MODELS_LIGHT_MOON,
    SERVICE_EYECARE_MODE_OFF,
    SERVICE_EYECARE_MODE_ON,
    SERVICE_NIGHT_LIGHT_MODE_OFF,
    SERVICE_NIGHT_LIGHT_MODE_ON,
    SERVICE_REMINDER_OFF,
    SERVICE_REMINDER_ON,
    SERVICE_SET_DELAYED_TURN_OFF,
    SERVICE_SET_SCENE,
)
from .entity import XiaomiGatewayDevice, XiaomiMiioEntity
from .typing import ServiceMethodDetails

_LOGGER: logging.Logger = logging.getLogger(__name__)
DEFAULT_NAME: str = 'Xiaomi Philips Light'
DATA_KEY: str = 'light.xiaomi_miio'
CCT_MIN: int = 1
CCT_MAX: int = 100
DELAYED_TURN_OFF_MAX_DEVIATION_SECONDS: int = 4
DELAYED_TURN_OFF_MAX_DEVIATION_MINUTES: int = 1
SUCCESS: List[str] = ['ok']
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
SERVICE_SCHEMA_SET_SCENE: vol.Schema = XIAOMI_MIIO_SERVICE_SCHEMA.extend(
    {vol.Required(ATTR_SCENE): vol.All(vol.Coerce(int), vol.Clamp(min=1, max=6))}
)
SERVICE_SCHEMA_SET_DELAYED_TURN_OFF: vol.Schema = XIAOMI_MIIO_SERVICE_SCHEMA.extend(
    {vol.Required(ATTR_TIME_PERIOD): cv.positive_time_period}
)
SERVICE_TO_METHOD: Dict[str, ServiceMethodDetails] = {
    SERVICE_SET_DELAYED_TURN_OFF: ServiceMethodDetails(method='async_set_delayed_turn_off', schema=SERVICE_SCHEMA_SET_DELAYED_TURN_OFF),
    SERVICE_SET_SCENE: ServiceMethodDetails(method='async_set_scene', schema=SERVICE_SCHEMA_SET_SCENE),
    SERVICE_REMINDER_ON: ServiceMethodDetails(method='async_reminder_on'),
    SERVICE_REMINDER_OFF: ServiceMethodDetails(method='async_reminder_off'),
    SERVICE_NIGHT_LIGHT_MODE_ON: ServiceMethodDetails(method='async_night_light_mode_on'),
    SERVICE_NIGHT_LIGHT_MODE_OFF: ServiceMethodDetails(method='async_night_light_mode_off'),
    SERVICE_EYECARE_MODE_ON: ServiceMethodDetails(method='async_eyecare_mode_on'),
    SERVICE_EYECARE_MODE_OFF: ServiceMethodDetails(method='async_eyecare_mode_off'),
}


async def async_setup_entry(
    hass: HomeAssistant,
    config_entry: ConfigEntry,
    async_add_entities: AddConfigEntryEntitiesCallback,
) -> None:
    """Set up the Xiaomi light from a config entry."""
    entities: List[LightEntity] = []
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
        unique_id: Optional[str] = config_entry.unique_id
        _LOGGER.debug('Initializing with host %s (token %s...)', host, token[:5])
        if model in MODELS_LIGHT_EYECARE:
            light: PhilipsEyecare = PhilipsEyecare(host, token)
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
            light: PhilipsMoonlight = PhilipsMoonlight(host, token)
            entity = XiaomiPhilipsMoonlightLamp(name, light, config_entry, unique_id)
            entities.append(entity)
            hass.data[DATA_KEY][host] = entity
        elif model in MODELS_LIGHT_BULB:
            light: PhilipsBulb = PhilipsBulb(host, token)
            entity = XiaomiPhilipsBulb(name, light, config_entry, unique_id)
            entities.append(entity)
            hass.data[DATA_KEY][host] = entity
        elif model in MODELS_LIGHT_MONO:
            light: PhilipsBulb = PhilipsBulb(host, token)
            entity = XiaomiPhilipsGenericLight(name, light, config_entry, unique_id)
            entities.append(entity)
            hass.data[DATA_KEY][host] = entity
        else:
            _LOGGER.error('Unsupported device found! Please create an issue at https://github.com/syssi/philipslight/issues and provide the following data: %s', model)
            return

        async def async_service_handler(service: ServiceCall) -> None:
            """Map services to methods on Xiaomi Philips Lights."""
            method: ServiceMethodDetails = SERVICE_TO_METHOD[service.service]
            params: Dict[str, Any] = {key: value for key, value in service.data.items() if key != ATTR_ENTITY_ID}
            if (entity_ids := service.data.get(ATTR_ENTITY_ID)) is not None:
                target_devices = [dev for dev in hass.data[DATA_KEY].values() if dev.entity_id in entity_ids]
            else:
                target_devices = list(hass.data[DATA_KEY].values())
            update_tasks: List[asyncio.Task[Any]] = []
            for target_device in target_devices:
                if not hasattr(target_device, method.method):
                    continue
                await getattr(target_device, method.method)(**params)
                update_tasks.append(asyncio.create_task(target_device.async_update_ha_state(True)))
            if update_tasks:
                await asyncio.wait(update_tasks)

        for xiaomi_miio_service, method_data in SERVICE_TO_METHOD.items():
            schema: vol.Schema = method_data.schema or XIAOMI_MIIO_SERVICE_SCHEMA
            hass.services.async_register(DOMAIN, xiaomi_miio_service, async_service_handler, schema=schema)
    async_add_entities(entities, update_before_add=True)


class XiaomiPhilipsAbstractLight(XiaomiMiioEntity, LightEntity):
    """Representation of an Abstract Xiaomi Philips Light."""
    _attr_color_mode: ColorMode = ColorMode.BRIGHTNESS
    _attr_supported_color_modes: set[ColorMode] = {ColorMode.BRIGHTNESS}

    def __init__(self, name: str, device: Any, entry: ConfigEntry, unique_id: Optional[str]) -> None:
        """Initialize the light device."""
        super().__init__(name, device, entry, unique_id)
        self._brightness: Optional[int] = None
        self._available: bool = False
        self._state: Optional[bool] = None
        self._state_attrs: Dict[str, Any] = {}

    @property
    def available(self) -> bool:
        """Return true when state is known."""
        return self._available

    @property
    def extra_state_attributes(self) -> Dict[str, Any]:
        """Return the state attributes of the device."""
        return self._state_attrs

    @property
    def is_on(self) -> Optional[bool]:
        """Return true if light is on."""
        return self._state

    @property
    def brightness(self) -> Optional[int]:
        """Return the brightness of this light between 0..255."""
        return self._brightness

    async def _try_command(self, mask_error: str, func: Callable[..., Any], *args: Any, **kwargs: Any) -> Union[bool, Any]:
        """Call a light command handling error messages."""
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
        """Turn the light on."""
        if ATTR_BRIGHTNESS in kwargs:
            brightness: int = kwargs[ATTR_BRIGHTNESS]
            percent_brightness: int = ceil(100 * brightness / 255.0)
            _LOGGER.debug('Setting brightness: %s %s%%', brightness, percent_brightness)
            result: Union[bool, Any] = await self._try_command('Turning on brightness failed: %s', self._device.set_brightness, percent_brightness)
            if result:
                self._brightness = brightness
        else:
            await self._try_command('Turning the light on failed.', self._device.on)

    async def async_turn_off(self, **kwargs: Any) -> None:
        """Turn the light off."""
        await self._try_command('Turning the light off failed.', self._device.off)

    async def async_update(self) -> None:
        """Fetch state from the device."""
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
    """Representation of a Generic Xiaomi Philips Light."""

    def __init__(self, name: str, device: Any, entry: ConfigEntry, unique_id: Optional[str]) -> None:
        """Initialize the light device."""
        super().__init__(name, device, entry, unique_id)
        self._state_attrs.update({ATTR_SCENE: None, ATTR_DELAYED_TURN_OFF: None})

    async def async_update(self) -> None:
        """Fetch state from the device."""
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

    async def async_set_scene(self, scene: int = 1) -> None:
        """Set the fixed scene."""
        await self._try_command('Setting a fixed scene failed.', self._device.set_scene, scene)

    async def async_set_delayed_turn_off(self, time_period: datetime.timedelta) -> None:
        """Set delayed turn off."""
        await self._try_command('Setting the turn off delay failed.', self._device.delay_off, time_period.total_seconds())

    @staticmethod
    def delayed_turn_off_timestamp(
        countdown: Optional[int],
        current: dt_datetime,
        previous: Optional[dt_datetime],
    ) -> Optional[dt_datetime]:
        """Update the turn off timestamp only if necessary."""
        if countdown is not None and countdown > 0:
            new: dt_datetime = current.replace(microsecond=0) + timedelta(seconds=countdown)
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
    """Representation of a Xiaomi Philips Bulb."""
    _attr_color_mode: ColorMode = ColorMode.COLOR_TEMP
    _attr_supported_color_modes: set[ColorMode] = {ColorMode.COLOR_TEMP}

    def __init__(self, name: str, device: Any, entry: ConfigEntry, unique_id: Optional[str]) -> None:
        """Initialize the light device."""
        super().__init__(name, device, entry, unique_id)
        self._color_temp: Optional[int] = None

    @property
    def _current_mireds(self) -> Optional[int]:
        """Return the color temperature."""
        return self._color_temp

    @property
    def _min_mireds(self) -> int:
        """Return the coldest color_temp that this light supports."""
        return 175

    @property
    def _max_mireds(self) -> int:
        """Return the warmest color_temp that this light supports."""
        return 333

    @property
    def color_temp_kelvin(self) -> Optional[float]:
        """Return the color temperature value in Kelvin."""
        return color_util.color_temperature_mired_to_kelvin(self._color_temp) if self._color_temp else None

    @property
    def min_color_temp_kelvin(self) -> Optional[float]:
        """Return the warmest color_temp_kelvin that this light supports."""
        return color_util.color_temperature_mired_to_kelvin(self._max_mireds)

    @property
    def max_color_temp_kelvin(self) -> Optional[float]:
        """Return the coldest color_temp_kelvin that this light supports."""
        return color_util.color_temperature_mired_to_kelvin(self._min_mireds)

    async def async_turn_on(self, **kwargs: Any) -> None:
        """Turn the light on."""
        if ATTR_COLOR_TEMP_KELVIN in kwargs:
            color_temp: int = color_util.color_temperature_kelvin_to_mired(kwargs[ATTR_COLOR_TEMP_KELVIN])
            percent_color_temp: int = self.translate(color_temp, self._max_mireds, self._min_mireds, CCT_MIN, CCT_MAX)
        if ATTR_BRIGHTNESS in kwargs:
            brightness: int = kwargs[ATTR_BRIGHTNESS]
            percent_brightness: int = ceil(100 * brightness / 255.0)
        if ATTR_BRIGHTNESS in kwargs and ATTR_COLOR_TEMP_KELVIN in kwargs:
            _LOGGER.debug('Setting brightness and color temperature: %s %s%%, %s mireds, %s%% cct', brightness, percent_brightness, color_temp, percent_color_temp)
            result: Union[bool, Any] = await self._try_command('Setting brightness and color temperature failed: %s bri, %s cct', self._device.set_brightness_and_color_temperature, percent_brightness, percent_color_temp)
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
        """Fetch state from the device."""
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
        """Map a value from left span to right span."""
        left_span: int = left_max - left_min
        right_span: int = right_max - right_min
        value_scaled: float = float(value - left_min) / float(left_span)
        return int(right_min + value_scaled * right_span)


class XiaomiPhilipsCeilingLamp(XiaomiPhilipsBulb):
    """Representation of a Xiaomi Philips Ceiling Lamp."""

    def __init__(self, name: str, device: Any, entry: ConfigEntry, unique_id: Optional[str]) -> None:
        """Initialize the light device."""
        super().__init__(name, device, entry, unique_id)
        self._state_attrs.update({ATTR_NIGHT_LIGHT_MODE: None, ATTR_AUTOMATIC_COLOR_TEMPERATURE: None})

    @property
    def _min_mireds(self) -> int:
        """Return the coldest color_temp that this light supports."""
        return 175

    @property
    def _max_mireds(self) -> int:
        """Return the warmest color_temp that this light supports."""
        return 370

    async def async_update(self) -> None:
        """Fetch state from the device."""
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
        self._state_attrs.update({
            ATTR_SCENE: state.scene,
            ATTR_DELAYED_TURN_OFF: delayed_turn_off,
            ATTR_NIGHT_LIGHT_MODE: state.smart_night_light,
            ATTR_AUTOMATIC_COLOR_TEMPERATURE: state.automatic_color_temperature,
        })


class XiaomiPhilipsEyecareLamp(XiaomiPhilipsGenericLight):
    """Representation of a Xiaomi Philips Eyecare Lamp 2."""

    def __init__(self, name: str, device: Any, entry: ConfigEntry, unique_id: Optional[str]) -> None:
        """Initialize the light device."""
        super().__init__(name, device, entry, unique_id)
        self._state_attrs.update({ATTR_REMINDER: None, ATTR_NIGHT_LIGHT_MODE: None, ATTR_EYECARE_MODE: None})

    async def async_update(self) -> None:
        """Fetch state from the device."""
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
        self._state_attrs.update({
            ATTR_SCENE: state.scene,
            ATTR_DELAYED_TURN_OFF: delayed_turn_off,
            ATTR_REMINDER: state.reminder,
            ATTR_NIGHT_LIGHT_MODE: state.smart_night_light,
            ATTR_EYECARE_MODE: state.eyecare,
        })

    async def async_set_delayed_turn_off(self, time_period: datetime.timedelta) -> None:
        """Set delayed turn off."""
        await self._try_command('Setting the turn off delay failed.', self._device.delay_off, round(time_period.total_seconds() / 60))

    async def async_reminder_on(self) -> None:
        """Enable the eye fatigue notification."""
        await self._try_command('Turning on the reminder failed.', self._device.reminder_on)

    async def async_reminder_off(self) -> None:
        """Disable the eye fatigue notification."""
        await self._try_command('Turning off the reminder failed.', self._device.reminder_off)

    async def async_night_light_mode_on(self) -> None:
        """Turn the smart night light mode on."""
        await self._try_command('Turning on the smart night light mode failed.', self._device.smart_night_light_on)

    async def async_night_light_mode_off(self) -> None:
        """Turn the smart night light mode off."""
        await self._try_command('Turning off the smart night light mode failed.', self._device.smart_night_light_off)

    async def async_eyecare_mode_on(self) -> None:
        """Turn the eyecare mode on."""
        await self._try_command('Turning on the eyecare mode failed.', self._device.eyecare_on)

    async def async_eyecare_mode_off(self) -> None:
        """Turn the eyecare mode off."""
        await self._try_command('Turning off the eyecare mode failed.', self._device.eyecare_off)

    @staticmethod
    def delayed_turn_off_timestamp(
        countdown: Optional[int],
        current: dt_datetime,
        previous: Optional[dt_datetime],
    ) -> Optional[dt_datetime]:
        """Update the turn off timestamp only if necessary."""
        if countdown is not None and countdown > 0:
            new: dt_datetime = current.replace(second=0, microsecond=0) + timedelta(minutes=countdown)
            if previous is None:
                return new
            lower = timedelta(minutes=-DELAYED_TURN_OFF_MAX_DEVIATION_MINUTES)
            upper = timedelta(minutes=DELAYED_TURN_OFF_MAX_DEVIATION_MINUTES)
            diff = previous - new
            if lower < diff < upper:
                return previous
            return new
        return None


class XiaomiPhilipsEyecareLampAmbientLight(XiaomiPhilipsAbstractLight):
    """Representation of a Xiaomi Philips Eyecare Lamp Ambient Light."""

    def __init__(self, name: str, device: Any, entry: ConfigEntry, unique_id: Optional[str]) -> None:
        """Initialize the light device."""
        name = f'{name} Ambient Light'
        if unique_id is not None:
            unique_id = f'{unique_id}-ambient'
        super().__init__(name, device, entry, unique_id)

    async def async_turn_on(self, **kwargs: Any) -> None:
        """Turn the light on."""
        if ATTR_BRIGHTNESS in kwargs:
            brightness: int = kwargs[ATTR_BRIGHTNESS]
            percent_brightness: int = ceil(100 * brightness / 255.0)
            _LOGGER.debug('Setting brightness of the ambient light: %s %s%%', brightness, percent_brightness)
            result = await self._try_command('Setting brightness of the ambient failed: %s', self._device.set_ambient_brightness, percent_brightness)
            if result:
                self._brightness = brightness
        else:
            await self._try_command('Turning the ambient light on failed.', self._device.ambient_on)

    async def async_turn_off(self, **kwargs: Any) -> None:
        """Turn the light off."""
        await self._try_command('Turning the ambient light off failed.', self._device.ambient_off)

    async def async_update(self) -> None:
        """Fetch state from the device."""
        try:
            state = await self.hass.async_add_executor_job(self._device.status)
        except DeviceException as ex:
            if self._available:
                self._available = False
                _LOGGER.error('Got exception while fetching the state: %s', ex)
            return
        _LOGGER.debug('Got new state: %s', state)
        self._available = True
        self._state = state.ambient
        self._brightness = ceil(255 / 100.0 * state.ambient_brightness)


class XiaomiPhilipsMoonlightLamp(XiaomiPhilipsBulb):
    """Representation of a Xiaomi Philips Zhirui Bedside Lamp."""
    _attr_supported_color_modes: set[ColorMode] = {ColorMode.COLOR_TEMP, ColorMode.HS}

    def __init__(self, name: str, device: Any, entry: ConfigEntry, unique_id: Optional[str]) -> None:
        """Initialize the light device."""
        super().__init__(name, device, entry, unique_id)
        self._hs_color: Optional[Tuple[float, float]] = None
        self._state_attrs.pop(ATTR_DELAYED_TURN_OFF)
        self._state_attrs.update({
            ATTR_SLEEP_ASSISTANT: None,
            ATTR_SLEEP_OFF_TIME: None,
            ATTR_TOTAL_ASSISTANT_SLEEP_TIME: None,
            ATTR_BAND_SLEEP: None,
            ATTR_BAND: None,
        })

    @property
    def _min_mireds(self) -> int:
        """Return the coldest color_temp that this light supports."""
        return 153

    @property
    def _max_mireds(self) -> int:
        """Return the warmest color_temp that this light supports."""
        return 588

    @property
    def hs_color(self) -> Optional[Tuple[float, float]]:
        """Return the hs color value."""
        return self._hs_color

    @property
    def color_mode(self) -> ColorMode:
        """Return the color mode of the light."""
        if self.hs_color:
            return ColorMode.HS
        return ColorMode.COLOR_TEMP

    async def async_turn_on(self, **kwargs: Any) -> None:
        """Turn the light on."""
        if ATTR_COLOR_TEMP_KELVIN in kwargs:
            color_temp: int = color_util.color_temperature_kelvin_to_mired(kwargs[ATTR_COLOR_TEMP_KELVIN])
            percent_color_temp: int = self.translate(color_temp, self._max_mireds, self._min_mireds, CCT_MIN, CCT_MAX)
        if ATTR_BRIGHTNESS in kwargs:
            brightness: int = kwargs[ATTR_BRIGHTNESS]
            percent_brightness: int = ceil(100 * brightness / 255.0)
        if ATTR_HS_COLOR in kwargs:
            hs_color: Tuple[float, float] = kwargs[ATTR_HS_COLOR]
            rgb: Tuple[int, int, int] = color_util.color_hs_to_RGB(*hs_color)
        if ATTR_BRIGHTNESS in kwargs and ATTR_HS_COLOR in kwargs:
            _LOGGER.debug('Setting brightness and color: %s %s%%, %s', brightness, percent_brightness, rgb)
            result = await self._try_command('Setting brightness and color failed: %s bri, %s color', self._device.set_brightness_and_rgb, percent_brightness, rgb)
            if result:
                self._hs_color = hs_color
                self._brightness = brightness
        elif ATTR_BRIGHTNESS in kwargs and ATTR_COLOR_TEMP_KELVIN in kwargs:
            _LOGGER.debug('Setting brightness and color temperature: %s %s%%, %s mireds, %s%% cct', brightness, percent_brightness, color_temp, percent_color_temp)
            result = await self._try_command('Setting brightness and color temperature failed: %s bri, %s cct', self._device.set_brightness_and_color_temperature, percent_brightness, percent_color_temp)
            if result:
                self._color_temp = color_temp
                self._brightness = brightness
        elif ATTR_HS_COLOR in kwargs:
            _LOGGER.debug('Setting color: %s', rgb)
            result = await self._try_command('Setting color failed: %s', self._device.set_rgb, rgb)
            if result:
                self._hs_color = hs_color
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
        """Fetch state from the device."""
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
        self._hs_color = color_util.color_RGB_to_hs(*state.rgb)
        self._state_attrs.update({
            ATTR_SCENE: state.scene,
            ATTR_SLEEP_ASSISTANT: state.sleep_assistant,
            ATTR_SLEEP_OFF_TIME: state.sleep_off_time,
            ATTR_TOTAL_ASSISTANT_SLEEP_TIME: state.total_assistant_sleep_time,
            ATTR_BAND_SLEEP: state.brand_sleep,
            ATTR_BAND: state.brand,
        })

    async def async_set_delayed_turn_off(self, time_period: datetime.timedelta) -> None:
        """Set delayed turn off. Unsupported."""
        return


class XiaomiGatewayLight(LightEntity):
    """Representation of a gateway device's light."""
    _attr_color_mode: ColorMode = ColorMode.HS
    _attr_supported_color_modes: set[ColorMode] = {ColorMode.HS}

    def __init__(self, gateway_device: Any, gateway_name: str, gateway_device_id: str) -> None:
        """Initialize the XiaomiGatewayLight."""
        self._gateway: Any = gateway_device
        self._name: str = f'{gateway_name} Light'
        self._gateway_device_id: str = gateway_device_id
        self._unique_id: str = gateway_device_id
        self._available: bool = False
        self._is_on: Optional[bool] = None
        self._brightness_pct: int = 100
        self._rgb: Tuple[int, int, int] = (255, 255, 255)
        self._hs: Tuple[float, float] = (0, 0)

    @property
    def unique_id(self) -> str:
        """Return an unique ID."""
        return self._unique_id

    @property
    def device_info(self) -> DeviceInfo:
        """Return the device info of the gateway."""
        return DeviceInfo(identifiers={(DOMAIN, self._gateway_device_id)})

    @property
    def name(self) -> str:
        """Return the name of this entity, if any."""
        return self._name

    @property
    def available(self) -> bool:
        """Return true when state is known."""
        return self._available

    @property
    def is_on(self) -> Optional[bool]:
        """Return true if it is on."""
        return self._is_on

    @property
    def brightness(self) -> int:
        """Return the brightness of this light between 0..255."""
        return int(255 * self._brightness_pct / 100)

    @property
    def hs_color(self) -> Tuple[float, float]:
        """Return the hs color value."""
        return self._hs

    def turn_on(self, **kwargs: Any) -> None:
        """Turn the light on."""
        if ATTR_HS_COLOR in kwargs:
            rgb: Tuple[int, int, int] = color_util.color_hs_to_RGB(*kwargs[ATTR_HS_COLOR])
        else:
            rgb = self._rgb
        if ATTR_BRIGHTNESS in kwargs:
            brightness_pct: int = int(100 * kwargs[ATTR_BRIGHTNESS] / 255)
        else:
            brightness_pct = self._brightness_pct
        self._gateway.light.set_rgb(brightness_pct, rgb)
        self.schedule_update_ha_state()

    def turn_off(self, **kwargs: Any) -> None:
        """Turn the light off."""
        self._gateway.light.set_rgb(0, self._rgb)
        self.schedule_update_ha_state()

    async def async_update(self) -> None:
        """Fetch state from the device."""
        try:
            state_dict: Dict[str, Any] = await self.hass.async_add_executor_job(self._gateway.light.rgb_status)
        except GatewayException as ex:
            if self._available:
                self._available = False
                _LOGGER.error('Got exception while fetching the gateway light state: %s', ex)
            return
        self._available = True
        self._is_on = state_dict['is_on']
        if self._is_on:
            self._brightness_pct = state_dict['brightness']
            self._rgb = state_dict['rgb']
            self._hs = color_util.color_RGB_to_hs(*self._rgb)


class XiaomiGatewayBulb(XiaomiGatewayDevice, LightEntity):
    """Representation of Xiaomi Gateway Bulb."""
    _attr_color_mode: ColorMode = ColorMode.COLOR_TEMP
    _attr_supported_color_modes: set[ColorMode] = {ColorMode.COLOR_TEMP}

    @property
    def brightness(self) -> int:
        """Return the brightness of the light."""
        return round(self._sub_device.status['brightness'] * 255 / 100)

    @property
    def _current_mireds(self) -> int:
        """Return current color temperature."""
        return self._sub_device.status['color_temp']

    @property
    def is_on(self) -> bool:
        """Return true if light is on."""
        return self._sub_device.status['status'] == 'on'

    @property
    def _min_mireds(self) -> int:
        """Return min cct."""
        return self._sub_device.status['cct_min']

    @property
    def _max_mireds(self) -> int:
        """Return max cct."""
        return self._sub_device.status['cct_max']

    async def async_turn_on(self, **kwargs: Any) -> None:
        """Instruct the light to turn on."""
        await self.hass.async_add_executor_job(self._sub_device.on)
        if ATTR_COLOR_TEMP_KELVIN in kwargs:
            color_temp: int = color_util.color_temperature_kelvin_to_mired(kwargs[ATTR_COLOR_TEMP_KELVIN])
            await self.hass.async_add_executor_job(self._sub_device.set_color_temp, color_temp)
        if ATTR_BRIGHTNESS in kwargs:
            brightness: int = round(kwargs[ATTR_BRIGHTNESS] * 100 / 255)
            await self.hass.async_add_executor_job(self._sub_device.set_brightness, brightness)

    async def async_turn_off(self, **kwargs: Any) -> None:
        """Instruct the light to turn off."""
        await self.hass.async_add_executor_job(self._sub_device.off)