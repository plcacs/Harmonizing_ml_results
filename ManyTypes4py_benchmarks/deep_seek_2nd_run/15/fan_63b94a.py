"""Support for Xiaomi Mi Air Purifier and Xiaomi Mi Air Humidifier."""
from __future__ import annotations
from abc import abstractmethod
import asyncio
import logging
import math
from typing import Any, Dict, List, Optional, Tuple, Union, cast
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
from .const import CONF_FLOW_TYPE, DOMAIN, FEATURE_FLAGS_AIRFRESH, FEATURE_FLAGS_AIRFRESH_A1, FEATURE_FLAGS_AIRFRESH_T2017, FEATURE_FLAGS_AIRPURIFIER_2S, FEATURE_FLAGS_AIRPURIFIER_3C, FEATURE_FLAGS_AIRPURIFIER_4, FEATURE_FLAGS_AIRPURIFIER_4_LITE, FEATURE_FLAGS_AIRPURIFIER_MIIO, FEATURE_FLAGS_AIRPURIFIER_MIOT, FEATURE_FLAGS_AIRPURIFIER_PRO, FEATURE_FLAGS_AIRPURIFIER_PRO_V7, FEATURE_FLAGS_AIRPURIFIER_V3, FEATURE_FLAGS_AIRPURIFIER_ZA1, FEATURE_FLAGS_FAN, FEATURE_FLAGS_FAN_1C, FEATURE_FLAGS_FAN_P5, FEATURE_FLAGS_FAN_P9, FEATURE_FLAGS_FAN_P10_P11_P18, FEATURE_FLAGS_FAN_ZA5, FEATURE_RESET_FILTER, FEATURE_SET_EXTRA_FEATURES, KEY_COORDINATOR, KEY_DEVICE, MODEL_AIRFRESH_A1, MODEL_AIRFRESH_T2017, MODEL_AIRPURIFIER_2H, MODEL_AIRPURIFIER_2S, MODEL_AIRPURIFIER_3C, MODEL_AIRPURIFIER_3C_REV_A, MODEL_AIRPURIFIER_4, MODEL_AIRPURIFIER_4_LITE_RMA1, MODEL_AIRPURIFIER_4_LITE_RMB1, MODEL_AIRPURIFIER_4_PRO, MODEL_AIRPURIFIER_PRO, MODEL_AIRPURIFIER_PRO_V7, MODEL_AIRPURIFIER_V3, MODEL_AIRPURIFIER_ZA1, MODEL_FAN_1C, MODEL_FAN_P5, MODEL_FAN_P9, MODEL_FAN_P10, MODEL_FAN_P11, MODEL_FAN_P18, MODEL_FAN_ZA5, MODELS_FAN_MIIO, MODELS_FAN_MIOT, MODELS_PURIFIER_MIOT, SERVICE_RESET_FILTER, SERVICE_SET_EXTRA_FEATURES
from .entity import XiaomiCoordinatedMiioEntity
from .typing import ServiceMethodDetails
_LOGGER = logging.getLogger(__name__)
DATA_KEY = 'fan.xiaomi_miio'
ATTR_MODE_NATURE = 'nature'
ATTR_MODE_NORMAL = 'normal'
ATTR_BRIGHTNESS = 'brightness'
ATTR_FAN_LEVEL = 'fan_level'
ATTR_SLEEP_TIME = 'sleep_time'
ATTR_SLEEP_LEARN_COUNT = 'sleep_mode_learn_count'
ATTR_EXTRA_FEATURES = 'extra_features'
ATTR_FEATURES = 'features'
ATTR_TURBO_MODE_SUPPORTED = 'turbo_mode_supported'
ATTR_SLEEP_MODE = 'sleep_mode'
ATTR_USE_TIME = 'use_time'
ATTR_BUTTON_PRESSED = 'button_pressed'
ATTR_FAVORITE_SPEED = 'favorite_speed'
ATTR_FAVORITE_RPM = 'favorite_rpm'
ATTR_MOTOR_SPEED = 'motor_speed'
AVAILABLE_ATTRIBUTES_AIRPURIFIER_COMMON: Dict[str, str] = {ATTR_EXTRA_FEATURES: 'extra_features', ATTR_TURBO_MODE_SUPPORTED: 'turbo_mode_supported', ATTR_BUTTON_PRESSED: 'button_pressed'}
AVAILABLE_ATTRIBUTES_AIRPURIFIER: Dict[str, str] = {**AVAILABLE_ATTRIBUTES_AIRPURIFIER_COMMON, ATTR_SLEEP_TIME: 'sleep_time', ATTR_SLEEP_LEARN_COUNT: 'sleep_mode_learn_count', ATTR_USE_TIME: 'use_time', ATTR_SLEEP_MODE: 'sleep_mode'}
AVAILABLE_ATTRIBUTES_AIRPURIFIER_PRO: Dict[str, str] = {**AVAILABLE_ATTRIBUTES_AIRPURIFIER_COMMON, ATTR_USE_TIME: 'use_time', ATTR_SLEEP_TIME: 'sleep_time', ATTR_SLEEP_LEARN_COUNT: 'sleep_mode_learn_count'}
AVAILABLE_ATTRIBUTES_AIRPURIFIER_MIOT: Dict[str, str] = {ATTR_USE_TIME: 'use_time'}
AVAILABLE_ATTRIBUTES_AIRPURIFIER_PRO_V7: Dict[str, str] = AVAILABLE_ATTRIBUTES_AIRPURIFIER_COMMON
AVAILABLE_ATTRIBUTES_AIRPURIFIER_V3: Dict[str, str] = {ATTR_SLEEP_TIME: 'sleep_time', ATTR_SLEEP_LEARN_COUNT: 'sleep_mode_learn_count', ATTR_EXTRA_FEATURES: 'extra_features', ATTR_USE_TIME: 'use_time', ATTR_BUTTON_PRESSED: 'button_pressed'}
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
AIRPURIFIER_SERVICE_SCHEMA = vol.Schema({vol.Optional(ATTR_ENTITY_ID): cv.entity_ids})
SERVICE_SCHEMA_EXTRA_FEATURES = AIRPURIFIER_SERVICE_SCHEMA.extend({vol.Required(ATTR_FEATURES): cv.positive_int})
SERVICE_TO_METHOD: Dict[str, ServiceMethodDetails] = {SERVICE_RESET_FILTER: ServiceMethodDetails(method='async_reset_filter'), SERVICE_SET_EXTRA_FEATURES: ServiceMethodDetails(method='async_set_extra_features', schema=SERVICE_SCHEMA_EXTRA_FEATURES)}
FAN_DIRECTIONS_MAP: Dict[str, str] = {'forward': 'right', 'reverse': 'left'}

async def async_setup_entry(hass: HomeAssistant, config_entry: ConfigEntry, async_add_entities: AddConfigEntryEntitiesCallback) -> None:
    """Set up the Fan from a config entry."""
    entities: List[Union[XiaomiAirPurifier, XiaomiAirFresh, XiaomiFan]] = []
    if config_entry.data[CONF_FLOW_TYPE] != CONF_DEVICE:
        return
    hass.data.setdefault(DATA_KEY, {})
    model: str = config_entry.data[CONF_MODEL]
    unique_id: str = config_entry.unique_id
    coordinator = hass.data[DOMAIN][config_entry.entry_id][KEY_COORDINATOR]
    device = hass.data[DOMAIN][config_entry.entry_id][KEY_DEVICE]
    entity: Union[XiaomiAirPurifier, XiaomiAirFresh, XiaomiFan, None] = None
    if model in (MODEL_AIRPURIFIER_3C, MODEL_AIRPURIFIER_3C_REV_A):
        entity = XiaomiAirPurifierMB4(device, config_entry, unique_id, coordinator)
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
    if entity is None:
        return
    hass.data[DATA_KEY][unique_id] = entity
    entities.append(entity)

    async def async_service_handler(service: ServiceCall) -> None:
        """Map services to methods on XiaomiAirPurifier."""
        method = SERVICE_TO_METHOD[service.service]
        params = {key: value for key, value in service.data.items() if key != ATTR_ENTITY_ID}
        filtered_entities: List[Union[XiaomiAirPurifier, XiaomiAirFresh, XiaomiFan]]
        if (entity_ids := service.data.get(ATTR_ENTITY_ID)):
            filtered_entities = [entity for entity in hass.data[DATA_KEY].values() if entity.entity_id in entity_ids]
        else:
            filtered_entities = list(hass.data[DATA_KEY].values())
        update_tasks = []
        for entity in filtered_entities:
            entity_method = getattr(entity, method.method, None)
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
        self._mode: Optional[int] = None
        self._fan_level: Optional[int] = None
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
    def is_on(self) -> bool:
        """Return true if device is on."""
        return self._state is True

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
            preset_mode = self.operation_mode_class(self._mode).name
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
    SPEED_MODE_MAPPING: Dict[int, Any] = {1: AirpurifierOperationMode.Silent, 2: AirpurifierOperationMode.Medium, 3: AirpurifierOperationMode.High, 4: AirpurifierOperationMode.Strong}
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
            self._attr_s