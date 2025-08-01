from __future__ import annotations
from abc import abstractmethod
import asyncio
import logging
import math
from typing import Any, Callable, Dict, List, Optional, Type
from miio.fan_common import MoveDirection as FanMoveDirection, OperationMode as FanOperationMode
from miio.integrations.airpurifier.dmaker.airfresh_t2017 import OperationMode as AirfreshOperationModeT2017
from miio.integrations.airpurifier.zhimi.airfresh import OperationMode as AirfreshOperationMode
from miio.integrations.airpurifier.zhimi.airpurifier import OperationMode as AirpurifierOperationMode
from miio.integrations.airpurifier.zhimi.airpurifier_miot import OperationMode as AirpurifierMiotOperationMode
from miio.integrations.fan.zhimi.zhimi_miot import OperationModeFanZA5 as FanZA5OperationMode
import voluptuous as vol
from homeassistant.components.fan import FanEntity, FanEntityFeature
from homeassistant.config_entries import ConfigEntry
from homeassistant.const import ATTR_ENTITY_ID, CONF_DEVICE, CONF_MODEL
from homeassistant.core import HomeAssistant, ServiceCall, callback
from homeassistant.helpers import config_validation as cv
from homeassistant.helpers.entity_platform import AddConfigEntryEntitiesCallback
from homeassistant.util.percentage import percentage_to_ranged_value, ranged_value_to_percentage
from .const import (
    CONF_FLOW_TYPE,
    DOMAIN,
    FEATURE_FLAGS_AIRFRESH,
    FEATURE_FLAGS_AIRFRESH_A1,
    FEATURE_FLAGS_AIRFRESH_T2017,
    FEATURE_FLAGS_AIRPURIFIER_2S,
    FEATURE_FLAGS_AIRPURIFIER_3C,
    FEATURE_FLAGS_AIRPURIFIER_4,
    FEATURE_FLAGS_AIRPURIFIER_4_LITE,
    FEATURE_FLAGS_AIRPURIFIER_MIIO,
    FEATURE_FLAGS_AIRPURIFIER_MIOT,
    FEATURE_FLAGS_AIRPURIFIER_PRO,
    FEATURE_FLAGS_AIRPURIFIER_PRO_V7,
    FEATURE_FLAGS_AIRPURIFIER_V3,
    FEATURE_FLAGS_AIRPURIFIER_ZA1,
    FEATURE_FLAGS_FAN,
    FEATURE_FLAGS_FAN_1C,
    FEATURE_FLAGS_FAN_P5,
    FEATURE_FLAGS_FAN_P9,
    FEATURE_FLAGS_FAN_P10_P11_P18,
    FEATURE_FLAGS_FAN_ZA5,
    FEATURE_RESET_FILTER,
    FEATURE_SET_EXTRA_FEATURES,
    KEY_COORDINATOR,
    KEY_DEVICE,
    MODEL_AIRFRESH_A1,
    MODEL_AIRFRESH_T2017,
    MODEL_AIRPURIFIER_2H,
    MODEL_AIRPURIFIER_2S,
    MODEL_AIRPURIFIER_3C,
    MODEL_AIRPURIFIER_3C_REV_A,
    MODEL_AIRPURIFIER_4,
    MODEL_AIRPURIFIER_4_LITE_RMA1,
    MODEL_AIRPURIFIER_4_LITE_RMB1,
    MODEL_AIRPURIFIER_4_PRO,
    MODEL_AIRPURIFIER_PRO,
    MODEL_AIRPURIFIER_PRO_V7,
    MODEL_AIRPURIFIER_V3,
    MODEL_AIRPURIFIER_ZA1,
    MODEL_FAN_1C,
    MODEL_FAN_P5,
    MODEL_FAN_P9,
    MODEL_FAN_P10,
    MODEL_FAN_P11,
    MODEL_FAN_P18,
    MODEL_FAN_ZA5,
    MODELS_FAN_MIIO,
    MODELS_FAN_MIOT,
    MODELS_PURIFIER_MIOT,
    SERVICE_RESET_FILTER,
    SERVICE_SET_EXTRA_FEATURES,
)
from .entity import XiaomiCoordinatedMiioEntity
from .typing import ServiceMethodDetails

_LOGGER = logging.getLogger(__name__)
DATA_KEY: str = 'fan.xiaomi_miio'
ATTR_MODE_NATURE: str = 'nature'
ATTR_MODE_NORMAL: str = 'normal'
ATTR_BRIGHTNESS: str = 'brightness'
ATTR_FAN_LEVEL: str = 'fan_level'
ATTR_SLEEP_TIME: str = 'sleep_time'
ATTR_SLEEP_LEARN_COUNT: str = 'sleep_mode_learn_count'
ATTR_EXTRA_FEATURES: str = 'extra_features'
ATTR_FEATURES: str = 'features'
ATTR_TURBO_MODE_SUPPORTED: str = 'turbo_mode_supported'
ATTR_SLEEP_MODE: str = 'sleep_mode'
ATTR_USE_TIME: str = 'use_time'
ATTR_BUTTON_PRESSED: str = 'button_pressed'
ATTR_FAVORITE_SPEED: str = 'favorite_speed'
ATTR_FAVORITE_RPM: str = 'favorite_rpm'
ATTR_MOTOR_SPEED: str = 'motor_speed'
AVAILABLE_ATTRIBUTES_AIRPURIFIER_COMMON: Dict[str, str] = {
    ATTR_EXTRA_FEATURES: 'extra_features',
    ATTR_TURBO_MODE_SUPPORTED: 'turbo_mode_supported',
    ATTR_BUTTON_PRESSED: 'button_pressed',
}
AVAILABLE_ATTRIBUTES_AIRPURIFIER: Dict[str, str] = {
    **AVAILABLE_ATTRIBUTES_AIRPURIFIER_COMMON,
    ATTR_SLEEP_TIME: 'sleep_time',
    ATTR_SLEEP_LEARN_COUNT: 'sleep_mode_learn_count',
    ATTR_USE_TIME: 'use_time',
    ATTR_SLEEP_MODE: 'sleep_mode',
}
AVAILABLE_ATTRIBUTES_AIRPURIFIER_PRO: Dict[str, str] = {
    **AVAILABLE_ATTRIBUTES_AIRPURIFIER_COMMON,
    ATTR_USE_TIME: 'use_time',
    ATTR_SLEEP_TIME: 'sleep_time',
    ATTR_SLEEP_LEARN_COUNT: 'sleep_mode_learn_count',
}
AVAILABLE_ATTRIBUTES_AIRPURIFIER_MIOT: Dict[str, str] = {ATTR_USE_TIME: 'use_time'}
AVAILABLE_ATTRIBUTES_AIRPURIFIER_PRO_V7: Dict[str, str] = AVAILABLE_ATTRIBUTES_AIRPURIFIER_COMMON
AVAILABLE_ATTRIBUTES_AIRPURIFIER_V3: Dict[str, str] = {
    ATTR_SLEEP_TIME: 'sleep_time',
    ATTR_SLEEP_LEARN_COUNT: 'sleep_mode_learn_count',
    ATTR_EXTRA_FEATURES: 'extra_features',
    ATTR_USE_TIME: 'use_time',
    ATTR_BUTTON_PRESSED: 'button_pressed',
}
AVAILABLE_ATTRIBUTES_AIRFRESH: Dict[str, str] = {ATTR_USE_TIME: 'use_time', ATTR_EXTRA_FEATURES: 'extra_features'}
PRESET_MODES_AIRPURIFIER: List[str] = ['Auto', 'Silent', 'Favorite', 'Idle']
PRESET_MODES_AIRPURIFIER_4_LITE: List[str] = ['Auto', 'Silent', 'Favorite']
PRESET_MODES_AIRPURIFIER_MIOT: List[str] = ['Auto', 'Silent', 'Favorite', 'Fan']
PRESET_MODES_AIRPURIFIER_PRO: List[str] = ['Auto', 'Silent', 'Favorite']
PRESET_MODES_AIRPURIFIER_PRO_V7: List[str] = PRESET_MODES_AIRPURIFIER_PRO
PRESET_MODES_AIRPURIFIER_2S: List[str] = ['Auto', 'Silent', 'Favorite']
PRESET_MODES_AIRPURIFIER_3C: List[str] = ['Auto', 'Silent', 'Favorite']
PRESET_MODES_AIRPURIFIER_ZA1: List[str] = ['Auto', 'Silent', 'Favorite']
PRESET_MODES_AIRPURIFIER_V3: List[str] = ['Auto', 'Silent', 'Favorite', 'Idle', 'Medium', 'High', 'Strong']
PRESET_MODES_AIRFRESH: List[str] = ['Auto', 'Interval']
PRESET_MODES_AIRFRESH_A1: List[str] = ['Auto', 'Sleep', 'Favorite']
AIRPURIFIER_SERVICE_SCHEMA: vol.Schema = vol.Schema({vol.Optional(ATTR_ENTITY_ID): cv.entity_ids})
SERVICE_SCHEMA_EXTRA_FEATURES: vol.Schema = AIRPURIFIER_SERVICE_SCHEMA.extend({vol.Required(ATTR_FEATURES): cv.positive_int})
SERVICE_TO_METHOD: Dict[str, ServiceMethodDetails] = {
    SERVICE_RESET_FILTER: ServiceMethodDetails(method='async_reset_filter'),
    SERVICE_SET_EXTRA_FEATURES: ServiceMethodDetails(method='async_set_extra_features', schema=SERVICE_SCHEMA_EXTRA_FEATURES),
}
FAN_DIRECTIONS_MAP: Dict[str, str] = {'forward': 'right', 'reverse': 'left'}


async def async_setup_entry(
    hass: HomeAssistant,
    config_entry: ConfigEntry,
    async_add_entities: AddConfigEntryEntitiesCallback,
) -> None:
    """Set up the Fan from a config entry."""
    entities: List[XiaomiGenericDevice] = []
    if config_entry.data[CONF_FLOW_TYPE] != CONF_DEVICE:
        return
    hass.data.setdefault(DATA_KEY, {})
    model: str = config_entry.data[CONF_MODEL]
    unique_id: str = config_entry.unique_id  # type: ignore
    coordinator: Any = hass.data[DOMAIN][config_entry.entry_id][KEY_COORDINATOR]
    device: Any = hass.data[DOMAIN][config_entry.entry_id][KEY_DEVICE]
    if model in (MODEL_AIRPURIFIER_3C, MODEL_AIRPURIFIER_3C_REV_A):
        entity: XiaomiGenericDevice = XiaomiAirPurifierMB4(device, config_entry, unique_id, coordinator)
    elif model in MODELS_PURIFIER_MIOT:
        entity = XiaomiAirPurifierMiot(device, config_entry, unique_id, coordinator)
    elif model.startswith('zhimi.airpurifier.'):
        entity = XiaomiAirPurifier(device, config_entry, unique_id, coordinator)
    elif model.startswith('zhimi.airfresh.'):
        entity = XiaomiAirFresh(device, config_entry, unique_id, coordinator)
    elif model == MODEL_AIRFRESH_A1:
        entity = XiaomiAirFreshA1(device, config_entry, unique_id, coordinator)
    elif model == MODEL_AIRFRESH_T2017:
        entity = XiaomiAirFreshT2017(device, config_entry, unique_id, coordinator)
    elif model == MODEL_FAN_P5:
        entity = XiaomiFanP5(device, config_entry, unique_id, coordinator)
    elif model in MODELS_FAN_MIIO:
        entity = XiaomiFan(device, config_entry, unique_id, coordinator)
    elif model == MODEL_FAN_ZA5:
        entity = XiaomiFanZA5(device, config_entry, unique_id, coordinator)
    elif model == MODEL_FAN_1C:
        entity = XiaomiFan1C(device, config_entry, unique_id, coordinator)
    elif model in MODELS_FAN_MIOT:
        entity = XiaomiFanMiot(device, config_entry, unique_id, coordinator)
    else:
        return
    hass.data[DATA_KEY][unique_id] = entity
    entities.append(entity)

    async def async_service_handler(service: ServiceCall) -> None:
        """Map services to methods on XiaomiAirPurifier."""
        method_detail: ServiceMethodDetails = SERVICE_TO_METHOD[service.service]
        params: Dict[str, Any] = {key: value for key, value in service.data.items() if key != ATTR_ENTITY_ID}
        if (entity_ids := service.data.get(ATTR_ENTITY_ID)):
            filtered_entities: List[Any] = [
                entity for entity in hass.data[DATA_KEY].values() if entity.entity_id in entity_ids
            ]
        else:
            filtered_entities = list(hass.data[DATA_KEY].values())
        update_tasks: List[asyncio.Task] = []
        for entity in filtered_entities:
            entity_method: Optional[Callable[..., Any]] = getattr(entity, method_detail.method, None)
            if not entity_method:
                continue
            await entity_method(**params)
            update_tasks.append(asyncio.create_task(entity.async_update_ha_state(True)))
        if update_tasks:
            await asyncio.wait(update_tasks)

    for air_purifier_service, method in SERVICE_TO_METHOD.items():
        schema = method.schema or AIRPURIFIER_SERVICE_SCHEMA
        hass.services.async_register(DOMAIN, air_purifier_service, async_service_handler, schema=schema)
    async_add_entities(entities)


class XiaomiGenericDevice(XiaomiCoordinatedMiioEntity, FanEntity):
    """Representation of a generic Xiaomi device."""
    _attr_name: Optional[str] = None

    def __init__(self, device: Any, entry: ConfigEntry, unique_id: str, coordinator: Any) -> None:
        """Initialize the generic Xiaomi device."""
        super().__init__(device, entry, unique_id, coordinator)
        self._available_attributes: Dict[str, str] = {}
        self._state: Optional[bool] = None
        self._mode: Any = None
        self._fan_level: Any = None
        self._state_attrs: Dict[str, Any] = {}
        self._device_features: int = 0
        self._preset_modes: List[str] = []

    @property
    @abstractmethod
    def operation_mode_class(self) -> Any:
        """Hold operation mode class."""

    @property
    def preset_modes(self) -> List[str]:
        """Get the list of available preset modes."""
        return self._preset_modes

    @property
    def percentage(self) -> Optional[int]:
        """Return the percentage based speed of the fan."""
        return None

    @property
    def extra_state_attributes(self) -> Dict[str, Any]:
        """Return the state attributes of the device."""
        return self._state_attrs

    @property
    def is_on(self) -> Optional[bool]:
        """Return true if device is on."""
        return self._state

    async def async_turn_on(self, percentage: Optional[int] = None, preset_mode: Optional[str] = None, **kwargs: Any) -> None:
        """Turn the device on."""
        result = await self._try_command('Turning the miio device on failed.', self._device.on)
        if percentage:
            await self.async_set_percentage(percentage)
        if preset_mode:
            await self.async_set_preset_mode(preset_mode)
        if result:
            self._state = True
            self.async_write_ha_state()

    async def async_turn_off(self, **kwargs: Any) -> None:
        """Turn the device off."""
        result = await self._try_command('Turning the miio device off failed.', self._device.off)
        if result:
            self._state = False
            self.async_write_ha_state()


class XiaomiGenericAirPurifier(XiaomiGenericDevice):
    """Representation of a generic AirPurifier device."""

    def __init__(self, device: Any, entry: ConfigEntry, unique_id: str, coordinator: Any) -> None:
        """Initialize the generic AirPurifier device."""
        super().__init__(device, entry, unique_id, coordinator)
        self._speed_count: int = 100

    @property
    def speed_count(self) -> int:
        """Return the number of speeds of the fan supported."""
        return self._speed_count

    @property
    def preset_mode(self) -> Optional[str]:
        """Get the active preset mode."""
        if self._state:
            preset_mode: str = self.operation_mode_class(self._mode).name  # type: ignore
            return preset_mode if preset_mode in self._preset_modes else None
        return None

    @callback
    def _handle_coordinator_update(self) -> None:
        """Fetch state from the device."""
        self._state = self.coordinator.data.is_on
        self._state_attrs.update({key: self._extract_value_from_attribute(self.coordinator.data, value) for key, value in self._available_attributes.items()})
        self._mode = self.coordinator.data.mode.value
        self._fan_level = getattr(self.coordinator.data, ATTR_FAN_LEVEL, None)
        self.async_write_ha_state()


class XiaomiAirPurifier(XiaomiGenericAirPurifier):
    """Representation of a Xiaomi Air Purifier."""
    SPEED_MODE_MAPPING: Dict[int, Any] = {
        1: AirpurifierOperationMode.Silent,
        2: AirpurifierOperationMode.Medium,
        3: AirpurifierOperationMode.High,
        4: AirpurifierOperationMode.Strong,
    }
    REVERSE_SPEED_MODE_MAPPING: Dict[Any, int] = {v: k for k, v in SPEED_MODE_MAPPING.items()}

    def __init__(self, device: Any, entry: ConfigEntry, unique_id: str, coordinator: Any) -> None:
        """Initialize the plug switch."""
        super().__init__(device, entry, unique_id, coordinator)
        if self._model == MODEL_AIRPURIFIER_PRO:
            self._device_features = FEATURE_FLAGS_AIRPURIFIER_PRO
            self._available_attributes = AVAILABLE_ATTRIBUTES_AIRPURIFIER_PRO
            self._preset_modes = PRESET_MODES_AIRPURIFIER_PRO
            self._attr_supported_features = FanEntityFeature.PRESET_MODE
            self._speed_count = 1
        elif self._model in [MODEL_AIRPURIFIER_4, MODEL_AIRPURIFIER_4_PRO]:
            self._device_features = FEATURE_FLAGS_AIRPURIFIER_4
            self._available_attributes = AVAILABLE_ATTRIBUTES_AIRPURIFIER_MIOT
            self._preset_modes = PRESET_MODES_AIRPURIFIER_MIOT
            self._attr_supported_features = FanEntityFeature.SET_SPEED | FanEntityFeature.PRESET_MODE
            self._speed_count = 3
        elif self._model in [MODEL_AIRPURIFIER_4_LITE_RMA1, MODEL_AIRPURIFIER_4_LITE_RMB1]:
            self._device_features = FEATURE_FLAGS_AIRPURIFIER_4_LITE
            self._available_attributes = AVAILABLE_ATTRIBUTES_AIRPURIFIER_MIOT
            self._preset_modes = PRESET_MODES_AIRPURIFIER_4_LITE
            self._attr_supported_features = FanEntityFeature.PRESET_MODE
            self._speed_count = 1
        elif self._model == MODEL_AIRPURIFIER_PRO_V7:
            self._device_features = FEATURE_FLAGS_AIRPURIFIER_PRO_V7
            self._available_attributes = AVAILABLE_ATTRIBUTES_AIRPURIFIER_PRO_V7
            self._preset_modes = PRESET_MODES_AIRPURIFIER_PRO_V7
            self._attr_supported_features = FanEntityFeature.PRESET_MODE
            self._speed_count = 1
        elif self._model in [MODEL_AIRPURIFIER_2S, MODEL_AIRPURIFIER_2H]:
            self._device_features = FEATURE_FLAGS_AIRPURIFIER_2S
            self._available_attributes = AVAILABLE_ATTRIBUTES_AIRPURIFIER_COMMON
            self._preset_modes = PRESET_MODES_AIRPURIFIER_2S
            self._attr_supported_features = FanEntityFeature.PRESET_MODE
            self._speed_count = 1
        elif self._model == MODEL_AIRPURIFIER_ZA1:
            self._device_features = FEATURE_FLAGS_AIRPURIFIER_ZA1
            self._available_attributes = AVAILABLE_ATTRIBUTES_AIRPURIFIER_MIOT
            self._preset_modes = PRESET_MODES_AIRPURIFIER_ZA1
            self._attr_supported_features = FanEntityFeature.PRESET_MODE
            self._speed_count = 1
        elif self._model in MODELS_PURIFIER_MIOT:
            self._device_features = FEATURE_FLAGS_AIRPURIFIER_MIOT
            self._available_attributes = AVAILABLE_ATTRIBUTES_AIRPURIFIER_MIOT
            self._preset_modes = PRESET_MODES_AIRPURIFIER_MIOT
            self._attr_supported_features = FanEntityFeature.SET_SPEED | FanEntityFeature.PRESET_MODE
            self._speed_count = 3
        elif self._model == MODEL_AIRPURIFIER_V3:
            self._device_features = FEATURE_FLAGS_AIRPURIFIER_V3
            self._available_attributes = AVAILABLE_ATTRIBUTES_AIRPURIFIER_V3
            self._preset_modes = PRESET_MODES_AIRPURIFIER_V3
            self._attr_supported_features = FanEntityFeature.PRESET_MODE
            self._speed_count = 1
        else:
            self._device_features = FEATURE_FLAGS_AIRPURIFIER_MIIO
            self._available_attributes = AVAILABLE_ATTRIBUTES_AIRPURIFIER
            self._preset_modes = PRESET_MODES_AIRPURIFIER
            self._attr_supported_features = FanEntityFeature.PRESET_MODE
            self._speed_count = 1
        self._attr_supported_features |= FanEntityFeature.TURN_OFF | FanEntityFeature.TURN_ON
        self._state = self.coordinator.data.is_on
        self._state_attrs.update({key: self._extract_value_from_attribute(self.coordinator.data, value) for key, value in self._available_attributes.items()})
        self._mode = self.coordinator.data.mode.value
        self._fan_level = getattr(self.coordinator.data, ATTR_FAN_LEVEL, None)

    @property
    def operation_mode_class(self) -> Type[Any]:
        """Hold operation mode class."""
        return AirpurifierOperationMode

    @property
    def percentage(self) -> Optional[int]:
        """Return the current percentage based speed."""
        if self._state:
            mode = self.operation_mode_class(self._mode)
            if mode in self.REVERSE_SPEED_MODE_MAPPING:
                return ranged_value_to_percentage((1, self._speed_count), self.REVERSE_SPEED_MODE_MAPPING[mode])
        return None

    async def async_set_percentage(self, percentage: int) -> None:
        """Set the percentage of the fan.

        This method is a coroutine.
        """
        if percentage == 0:
            await self.async_turn_off()
            return
        speed_mode: int = math.ceil(percentage_to_ranged_value((1, self._speed_count), percentage))
        if speed_mode:
            await self._try_command('Setting operation mode of the miio device failed.', self._device.set_mode, self.operation_mode_class(self.SPEED_MODE_MAPPING[speed_mode]))

    async def async_set_preset_mode(self, preset_mode: str) -> None:
        """Set the preset mode of the fan.

        This method is a coroutine.
        """
        if await self._try_command('Setting operation mode of the miio device failed.', self._device.set_mode, self.operation_mode_class[preset_mode]):
            self._mode = self.operation_mode_class[preset_mode].value
            self.async_write_ha_state()

    async def async_set_extra_features(self, features: int = 1) -> None:
        """Set the extra features."""
        if self._device_features & FEATURE_SET_EXTRA_FEATURES == 0:
            return
        await self._try_command('Setting the extra features of the miio device failed.', self._device.set_extra_features, features)

    async def async_reset_filter(self) -> None:
        """Reset the filter lifetime and usage."""
        if self._device_features & FEATURE_RESET_FILTER == 0:
            return
        await self._try_command('Resetting the filter lifetime of the miio device failed.', self._device.reset_filter)


class XiaomiAirPurifierMiot(XiaomiAirPurifier):
    """Representation of a Xiaomi Air Purifier (MiOT protocol)."""

    @property
    def operation_mode_class(self) -> Type[Any]:
        """Hold operation mode class."""
        return AirpurifierMiotOperationMode

    @property
    def percentage(self) -> Optional[int]:
        """Return the current percentage based speed."""
        if self._fan_level is None:
            return None
        if self._state:
            return ranged_value_to_percentage((1, 3), self._fan_level)
        return None

    async def async_set_percentage(self, percentage: int) -> None:
        """Set the percentage of the fan.

        This method is a coroutine.
        """
        if percentage == 0:
            await self.async_turn_off()
            return
        fan_level: int = math.ceil(percentage_to_ranged_value((1, 3), percentage))
        if not fan_level:
            return
        if await self._try_command('Setting fan level of the miio device failed.', self._device.set_fan_level, fan_level):
            self._fan_level = fan_level
            self.async_write_ha_state()


class XiaomiAirPurifierMB4(XiaomiGenericAirPurifier):
    """Representation of a Xiaomi Air Purifier MB4."""

    def __init__(self, device: Any, entry: ConfigEntry, unique_id: str, coordinator: Any) -> None:
        """Initialize Air Purifier MB4."""
        super().__init__(device, entry, unique_id, coordinator)
        self._device_features: int = FEATURE_FLAGS_AIRPURIFIER_3C
        self._preset_modes = PRESET_MODES_AIRPURIFIER_3C
        self._attr_supported_features = FanEntityFeature.SET_SPEED | FanEntityFeature.PRESET_MODE | FanEntityFeature.TURN_OFF | FanEntityFeature.TURN_ON
        self._state = self.coordinator.data.is_on
        self._mode = self.coordinator.data.mode.value
        self._favorite_rpm: Optional[int] = None
        self._speed_range: tuple[int, int] = (300, 2200)
        self._motor_speed: int = 0

    @property
    def operation_mode_class(self) -> Type[Any]:
        """Hold operation mode class."""
        return AirpurifierMiotOperationMode

    @property
    def percentage(self) -> Optional[int]:
        """Return the current percentage based speed."""
        if self._mode != self.operation_mode_class['Favorite'].value:
            return ranged_value_to_percentage(self._speed_range, self._motor_speed)
        if self._favorite_rpm is None:
            return None
        if self._state:
            return ranged_value_to_percentage(self._speed_range, self._favorite_rpm)
        return None

    async def async_set_percentage(self, percentage: int) -> None:
        """Set the percentage of the fan. This method is a coroutine."""
        if percentage == 0:
            await self.async_turn_off()
            return
        favorite_rpm: int = int(round(percentage_to_ranged_value(self._speed_range, percentage), -1))
        if not favorite_rpm:
            return
        if await self._try_command('Setting fan level of the miio device failed.', self._device.set_favorite_rpm, favorite_rpm):
            self._favorite_rpm = favorite_rpm
            self._mode = self.operation_mode_class['Favorite'].value
            self.async_write_ha_state()

    async def async_set_preset_mode(self, preset_mode: str) -> None:
        """Set the preset mode of the fan."""
        if not self._state:
            await self.async_turn_on()
        if await self._try_command('Setting operation mode of the miio device failed.', self._device.set_mode, self.operation_mode_class[preset_mode]):
            self._mode = self.operation_mode_class[preset_mode].value
            self.async_write_ha_state()

    @callback
    def _handle_coordinator_update(self) -> None:
        """Fetch state from the device."""
        self._state = self.coordinator.data.is_on
        self._mode = self.coordinator.data.mode.value
        self._favorite_rpm = getattr(self.coordinator.data, ATTR_FAVORITE_RPM, None)
        self._motor_speed = min(self._speed_range[1], max(self._speed_range[0], getattr(self.coordinator.data, ATTR_MOTOR_SPEED, 0)))
        self.async_write_ha_state()


class XiaomiAirFresh(XiaomiGenericAirPurifier):
    """Representation of a Xiaomi Air Fresh."""
    SPEED_MODE_MAPPING: Dict[int, Any] = {
        1: AirfreshOperationMode.Silent,
        2: AirfreshOperationMode.Low,
        3: AirfreshOperationMode.Middle,
        4: AirfreshOperationMode.Strong,
    }
    REVERSE_SPEED_MODE_MAPPING: Dict[Any, int] = {v: k for k, v in SPEED_MODE_MAPPING.items()}
    PRESET_MODE_MAPPING: Dict[str, Any] = {'Auto': AirfreshOperationMode.Auto, 'Interval': AirfreshOperationMode.Interval}

    def __init__(self, device: Any, entry: ConfigEntry, unique_id: str, coordinator: Any) -> None:
        """Initialize the miio device."""
        super().__init__(device, entry, unique_id, coordinator)
        self._device_features = FEATURE_FLAGS_AIRFRESH
        self._available_attributes = AVAILABLE_ATTRIBUTES_AIRFRESH
        self._speed_count = 4
        self._preset_modes = PRESET_MODES_AIRFRESH
        self._attr_supported_features = FanEntityFeature.SET_SPEED | FanEntityFeature.PRESET_MODE | FanEntityFeature.TURN_OFF | FanEntityFeature.TURN_ON
        self._state = self.coordinator.data.is_on
        self._state_attrs.update({key: getattr(self.coordinator.data, value) for key, value in self._available_attributes.items()})
        self._mode = self.coordinator.data.mode.value

    @property
    def operation_mode_class(self) -> Type[Any]:
        """Hold operation mode class."""
        return AirfreshOperationMode

    @property
    def percentage(self) -> Optional[int]:
        """Return the current percentage based speed."""
        if self._state:
            mode = AirfreshOperationMode(self._mode)
            if mode in self.REVERSE_SPEED_MODE_MAPPING:
                return ranged_value_to_percentage((1, self._speed_count), self.REVERSE_SPEED_MODE_MAPPING[mode])
        return None

    async def async_set_percentage(self, percentage: int) -> None:
        """Set the percentage of the fan.

        This method is a coroutine.
        """
        speed_mode: int = math.ceil(percentage_to_ranged_value((1, self._speed_count), percentage))
        if speed_mode:
            if await self._try_command('Setting operation mode of the miio device failed.', self._device.set_mode, AirfreshOperationMode(self.SPEED_MODE_MAPPING[speed_mode])):
                self._mode = AirfreshOperationMode(self.SPEED_MODE_MAPPING[speed_mode]).value
                self.async_write_ha_state()

    async def async_set_preset_mode(self, preset_mode: str) -> None:
        """Set the preset mode of the fan.

        This method is a coroutine.
        """
        if await self._try_command('Setting operation mode of the miio device failed.', self._device.set_mode, self.operation_mode_class[preset_mode]):
            self._mode = self.operation_mode_class[preset_mode].value
            self.async_write_ha_state()

    async def async_set_extra_features(self, features: int = 1) -> None:
        """Set the extra features."""
        if self._device_features & FEATURE_SET_EXTRA_FEATURES == 0:
            return
        await self._try_command('Setting the extra features of the miio device failed.', self._device.set_extra_features, features)

    async def async_reset_filter(self) -> None:
        """Reset the filter lifetime and usage."""
        if self._device_features & FEATURE_RESET_FILTER == 0:
            return
        await self._try_command('Resetting the filter lifetime of the miio device failed.', self._device.reset_filter)


class XiaomiAirFreshA1(XiaomiGenericAirPurifier):
    """Representation of a Xiaomi Air Fresh A1."""

    def __init__(self, device: Any, entry: ConfigEntry, unique_id: str, coordinator: Any) -> None:
        """Initialize the miio device."""
        super().__init__(device, entry, unique_id, coordinator)
        self._favorite_speed: Optional[int] = None
        self._device_features = FEATURE_FLAGS_AIRFRESH_A1
        self._preset_modes = PRESET_MODES_AIRFRESH_A1
        self._attr_supported_features = FanEntityFeature.SET_SPEED | FanEntityFeature.PRESET_MODE | FanEntityFeature.TURN_OFF | FanEntityFeature.TURN_ON
        self._state = self.coordinator.data.is_on
        self._mode = self.coordinator.data.mode.value
        self._speed_range: tuple[int, int] = (60, 150)

    @property
    def operation_mode_class(self) -> Type[Any]:
        """Hold operation mode class."""
        return AirfreshOperationModeT2017

    @property
    def percentage(self) -> Optional[int]:
        """Return the current percentage based speed."""
        if self._favorite_speed is None:
            return None
        if self._state:
            return ranged_value_to_percentage(self._speed_range, self._favorite_speed)
        return None

    async def async_set_percentage(self, percentage: int) -> None:
        """Set the percentage of the fan. This method is a coroutine."""
        if percentage == 0:
            await self.async_turn_off()
            return
        await self.async_set_preset_mode('Favorite')
        favorite_speed: int = math.ceil(percentage_to_ranged_value(self._speed_range, percentage))
        if not favorite_speed:
            return
        if await self._try_command('Setting fan level of the miio device failed.', self._device.set_favorite_speed, favorite_speed):
            self._favorite_speed = favorite_speed
            self.async_write_ha_state()

    async def async_set_preset_mode(self, preset_mode: str) -> None:
        """Set the preset mode of the fan. This method is a coroutine."""
        if await self._try_command('Setting operation mode of the miio device failed.', self._device.set_mode, self.operation_mode_class[preset_mode]):
            self._mode = self.operation_mode_class[preset_mode].value
            self.async_write_ha_state()

    @callback
    def _handle_coordinator_update(self) -> None:
        """Fetch state from the device."""
        self._state = self.coordinator.data.is_on
        self._mode = self.coordinator.data.mode.value
        self._favorite_speed = getattr(self.coordinator.data, ATTR_FAVORITE_SPEED, None)
        self.async_write_ha_state()


class XiaomiAirFreshT2017(XiaomiAirFreshA1):
    """Representation of a Xiaomi Air Fresh T2017."""

    def __init__(self, device: Any, entry: ConfigEntry, unique_id: str, coordinator: Any) -> None:
        """Initialize the miio device."""
        super().__init__(device, entry, unique_id, coordinator)
        self._device_features = FEATURE_FLAGS_AIRFRESH_T2017
        self._speed_range = (60, 300)


class XiaomiGenericFan(XiaomiGenericDevice):
    """Representation of a generic Xiaomi Fan."""
    _attr_translation_key: str = 'generic_fan'

    def __init__(self, device: Any, entry: ConfigEntry, unique_id: str, coordinator: Any) -> None:
        """Initialize the fan."""
        super().__init__(device, entry, unique_id, coordinator)
        if self._model == MODEL_FAN_P5:
            self._device_features = FEATURE_FLAGS_FAN_P5
        elif self._model == MODEL_FAN_ZA5:
            self._device_features = FEATURE_FLAGS_FAN_ZA5
        elif self._model == MODEL_FAN_1C:
            self._device_features = FEATURE_FLAGS_FAN_1C
        elif self._model == MODEL_FAN_P9:
            self._device_features = FEATURE_FLAGS_FAN_P9
        elif self._model in (MODEL_FAN_P10, MODEL_FAN_P11, MODEL_FAN_P18):
            self._device_features = FEATURE_FLAGS_FAN_P10_P11_P18
        else:
            self._device_features = FEATURE_FLAGS_FAN
        self._attr_supported_features = FanEntityFeature.SET_SPEED | FanEntityFeature.OSCILLATE | FanEntityFeature.PRESET_MODE | FanEntityFeature.TURN_OFF | FanEntityFeature.TURN_ON
        if self._model != MODEL_FAN_1C:
            self._attr_supported_features |= FanEntityFeature.DIRECTION
        self._preset_mode: Optional[str] = None
        self._oscillating: Optional[bool] = None
        self._percentage: Optional[int] = None

    @property
    def preset_mode(self) -> Optional[str]:
        """Get the active preset mode."""
        return self._preset_mode

    @property
    def preset_modes(self) -> List[str]:
        """Get the list of available preset modes."""
        return [mode.name for mode in self.operation_mode_class]  # type: ignore

    @property
    def percentage(self) -> Optional[int]:
        """Return the current speed as a percentage."""
        if self._state:
            return self._percentage
        return None

    @property
    def oscillating(self) -> Optional[bool]:
        """Return whether or not the fan is currently oscillating."""
        return self._oscillating

    async def async_oscillate(self, oscillating: bool) -> None:
        """Set oscillation."""
        await self._try_command('Setting oscillate on/off of the miio device failed.', self._device.set_oscillate, oscillating)
        self._oscillating = oscillating
        self.async_write_ha_state()

    async def async_set_direction(self, direction: str) -> None:
        """Set the direction of the fan."""
        if self._oscillating:
            await self.async_oscillate(oscillating=False)
        await self._try_command('Setting move direction of the miio device failed.', self._device.set_rotate, FanMoveDirection(FAN_DIRECTIONS_MAP[direction]))


class XiaomiFan(XiaomiGenericFan):
    """Representation of a Xiaomi Fan."""

    def __init__(self, device: Any, entry: ConfigEntry, unique_id: str, coordinator: Any) -> None:
        """Initialize the fan."""
        super().__init__(device, entry, unique_id, coordinator)
        self._state = self.coordinator.data.is_on
        self._oscillating = self.coordinator.data.oscillate
        self._nature_mode: bool = self.coordinator.data.natural_speed != 0
        if self._nature_mode:
            self._percentage = self.coordinator.data.natural_speed
        else:
            self._percentage = self.coordinator.data.direct_speed

    @property
    def operation_mode_class(self) -> Any:
        """Hold operation mode class."""
        # This property is intentionally left without a concrete implementation.
        return None

    @property
    def preset_mode(self) -> str:
        """Get the active preset mode."""
        return ATTR_MODE_NATURE if self._nature_mode else ATTR_MODE_NORMAL

    @property
    def preset_modes(self) -> List[str]:
        """Get the list of available preset modes."""
        return [ATTR_MODE_NATURE, ATTR_MODE_NORMAL]

    @callback
    def _handle_coordinator_update(self) -> None:
        """Fetch state from the device."""
        self._state = self.coordinator.data.is_on
        self._oscillating = self.coordinator.data.oscillate
        self._nature_mode = self.coordinator.data.natural_speed != 0
        if self._nature_mode:
            self._percentage = self.coordinator.data.natural_speed
        else:
            self._percentage = self.coordinator.data.direct_speed
        self.async_write_ha_state()

    async def async_set_preset_mode(self, preset_mode: str) -> None:
        """Set the preset mode of the fan."""
        if preset_mode == ATTR_MODE_NATURE:
            await self._try_command('Setting natural fan speed percentage of the miio device failed.', self._device.set_natural_speed, self._percentage)
        else:
            await self._try_command('Setting direct fan speed percentage of the miio device failed.', self._device.set_direct_speed, self._percentage)
        self._preset_mode = preset_mode
        self.async_write_ha_state()

    async def async_set_percentage(self, percentage: int) -> None:
        """Set the percentage of the fan."""
        if percentage == 0:
            self._percentage = 0
            await self.async_turn_off()
            return
        if self._nature_mode:
            await self._try_command('Setting fan speed percentage of the miio device failed.', self._device.set_natural_speed, percentage)
        else:
            await self._try_command('Setting fan speed percentage of the miio device failed.', self._device.set_direct_speed, percentage)
        self._percentage = percentage
        if not self.is_on:
            await self.async_turn_on()
        else:
            self.async_write_ha_state()


class XiaomiFanP5(XiaomiGenericFan):
    """Representation of a Xiaomi Fan P5."""

    def __init__(self, device: Any, entry: ConfigEntry, unique_id: str, coordinator: Any) -> None:
        """Initialize the fan."""
        super().__init__(device, entry, unique_id, coordinator)
        self._state = self.coordinator.data.is_on
        self._preset_mode = self.coordinator.data.mode.name
        self._oscillating = self.coordinator.data.oscillate
        self._percentage = self.coordinator.data.speed

    @property
    def operation_mode_class(self) -> Type[Any]:
        """Hold operation mode class."""
        return FanOperationMode

    @callback
    def _handle_coordinator_update(self) -> None:
        """Fetch state from the device."""
        self._state = self.coordinator.data.is_on
        self._preset_mode = self.coordinator.data.mode.name
        self._oscillating = self.coordinator.data.oscillate
        self._percentage = self.coordinator.data.speed
        self.async_write_ha_state()

    async def async_set_preset_mode(self, preset_mode: str) -> None:
        """Set the preset mode of the fan."""
        await self._try_command('Setting operation mode of the miio device failed.', self._device.set_mode, self.operation_mode_class[preset_mode])
        self._preset_mode = preset_mode
        self.async_write_ha_state()

    async def async_set_percentage(self, percentage: int) -> None:
        """Set the percentage of the fan."""
        if percentage == 0:
            self._percentage = 0
            await self.async_turn_off()
            return
        await self._try_command('Setting fan speed percentage of the miio device failed.', self._device.set_speed, percentage)
        self._percentage = percentage
        if not self.is_on:
            await self.async_turn_on()
        else:
            self.async_write_ha_state()


class XiaomiFanMiot(XiaomiGenericFan):
    """Representation of a Xiaomi Fan Miot."""

    @property
    def operation_mode_class(self) -> Type[Any]:
        """Hold operation mode class."""
        return FanOperationMode

    @property
    def preset_mode(self) -> Optional[str]:
        """Get the active preset mode."""
        return self._preset_mode

    @callback
    def _handle_coordinator_update(self) -> None:
        """Fetch state from the device."""
        self._state = self.coordinator.data.is_on
        self._preset_mode = self.coordinator.data.mode.name
        self._oscillating = self.coordinator.data.oscillate
        if self.coordinator.data.is_on:
            self._percentage = self.coordinator.data.speed
        else:
            self._percentage = 0
        self.async_write_ha_state()

    async def async_set_preset_mode(self, preset_mode: str) -> None:
        """Set the preset mode of the fan."""
        await self._try_command('Setting operation mode of the miio device failed.', self._device.set_mode, self.operation_mode_class[preset_mode])
        self._preset_mode = preset_mode
        self.async_write_ha_state()

    async def async_set_percentage(self, percentage: int) -> None:
        """Set the percentage of the fan."""
        if percentage == 0:
            self._percentage = 0
            await self.async_turn_off()
            return
        result = await self._try_command('Setting fan speed percentage of the miio device failed.', self._device.set_speed, percentage)
        if result:
            self._percentage = percentage
        if not self.is_on:
            await self.async_turn_on()
        elif result:
            self.async_write_ha_state()


class XiaomiFanZA5(XiaomiFanMiot):
    """Representation of a Xiaomi Fan ZA5."""

    @property
    def operation_mode_class(self) -> Type[Any]:
        """Hold operation mode class."""
        return FanZA5OperationMode


class XiaomiFan1C(XiaomiFanMiot):
    """Representation of a Xiaomi Fan 1C (Standing Fan 2 Lite)."""

    def __init__(self, device: Any, entry: ConfigEntry, unique_id: str, coordinator: Any) -> None:
        """Initialize MIOT fan with speed count."""
        super().__init__(device, entry, unique_id, coordinator)
        self._speed_count: int = 3

    @callback
    def _handle_coordinator_update(self) -> None:
        """Fetch state from the device."""
        self._state = self.coordinator.data.is_on
        self._preset_mode = self.coordinator.data.mode.name
        self._oscillating = self.coordinator.data.oscillate
        if self.coordinator.data.is_on:
            self._percentage = ranged_value_to_percentage((1, self._speed_count), self.coordinator.data.speed)
        else:
            self._percentage = 0
        self.async_write_ha_state()

    async def async_set_percentage(self, percentage: int) -> None:
        """Set the percentage of the fan."""
        if percentage == 0:
            self._percentage = 0
            await self.async_turn_off()
            return
        speed: int = math.ceil(percentage_to_ranged_value((1, self._speed_count), percentage))
        if not self.is_on:
            await self.async_turn_on()
        result = await self._try_command('Setting fan speed percentage of the miio device failed.', self._device.set_speed, speed)
        if result:
            self._percentage = ranged_value_to_percentage((1, self._speed_count), speed)
            self.async_write_ha_state()