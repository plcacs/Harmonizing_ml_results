"""Support for Xiaomi Mi Air Purifier and Xiaomi Mi Air Humidifier."""

from __future__ import annotations
from abc import abstractmethod
import asyncio
import logging
import math
from typing import Any, Callable, Dict, List, Optional, Union

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

_LOGGER: logging.Logger = logging.getLogger(__name__)
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

AVAILABLE_ATTRIBUTES_AIRPURIFIER_COMMON: dict[str, str] = {
    ATTR_EXTRA_FEATURES: 'extra_features',
    ATTR_TURBO_MODE_SUPPORTED: 'turbo_mode_supported',
    ATTR_BUTTON_PRESSED: 'button_pressed',
}

AVAILABLE_ATTRIBUTES_AIRPURIFIER: dict[str, str] = {
    **AVAILABLE_ATTRIBUTES_AIRPURIFIER_COMMON,
    ATTR_SLEEP_TIME: 'sleep_time',
    ATTR_SLEEP_LEARN_COUNT: 'sleep_mode_learn_count',
    ATTR_USE_TIME: 'use_time',
    ATTR_SLEEP_MODE: 'sleep_mode',
}

AVAILABLE_ATTRIBUTES_AIRPURIFIER_PRO: dict[str, str] = {
    **AVAILABLE_ATTRIBUTES_AIRPURIFIER_COMMON,
    ATTR_USE_TIME: 'use_time',
    ATTR_SLEEP_TIME: 'sleep_time',
    ATTR_SLEEP_LEARN_COUNT: 'sleep_mode_learn_count',
}

AVAILABLE_ATTRIBUTES_AIRPURIFIER_MIOT: dict[str, str] = {ATTR_USE_TIME: 'use_time'}

AVAILABLE_ATTRIBUTES_AIRPURIFIER_PRO_V7: dict[str, str] = AVAILABLE_ATTRIBUTES_AIRPURIFIER_COMMON

AVAILABLE_ATTRIBUTES_AIRPURIFIER_V3: dict[str, str] = {
    ATTR_SLEEP_TIME: 'sleep_time',
    ATTR_SLEEP_LEARN_COUNT: 'sleep_mode_learn_count',
    ATTR_EXTRA_FEATURES: 'extra_features',
    ATTR_USE_TIME: 'use_time',
    ATTR_BUTTON_PRESSED: 'button_pressed',
}

AVAILABLE_ATTRIBUTES_AIRFRESH: dict[str, str] = {ATTR_USE_TIME: 'use_time', ATTR_EXTRA_FEATURES: 'extra_features'}

PRESET_MODES_AIRPURIFIER: list[str] = ['Auto', 'Silent', 'Favorite', 'Idle']
PRESET_MODES_AIRPURIFIER_4_LITE: list[str] = ['Auto', 'Silent', 'Favorite']
PRESET_MODES_AIRPURIFIER_MIOT: list[str] = ['Auto', 'Silent', 'Favorite', 'Fan']
PRESET_MODES_AIRPURIFIER_PRO: list[str] = ['Auto', 'Silent', 'Favorite']
PRESET_MODES_AIRPURIFIER_PRO_V7: list[str] = PRESET_MODES_AIRPURIFIER_PRO
PRESET_MODES_AIRPURIFIER_2S: list[str] = ['Auto', 'Silent', 'Favorite']
PRESET_MODES_AIRPURIFIER_3C: list[str] = ['Auto', 'Silent', 'Favorite']
PRESET_MODES_AIRPURIFIER_ZA1: list[str] = ['Auto', 'Silent', 'Favorite']
PRESET_MODES_AIRPURIFIER_V3: list[str] = ['Auto', 'Silent', 'Favorite', 'Idle', 'Medium', 'High', 'Strong']
PRESET_MODES_AIRFRESH: list[str] = ['Auto', 'Interval']
PRESET_MODES_AIRFRESH_A1: list[str] = ['Auto', 'Sleep', 'Favorite']

AIRPURIFIER_SERVICE_SCHEMA: vol.Schema = vol.Schema({vol.Optional(ATTR_ENTITY_ID): cv.entity_ids})

SERVICE_SCHEMA_EXTRA_FEATURES: vol.Schema = AIRPURIFIER_SERVICE_SCHEMA.extend({vol.Required(ATTR_FEATURES): cv.positive_int})

SERVICE_TO_METHOD: dict[str, ServiceMethodDetails] = {
    SERVICE_RESET_FILTER: ServiceMethodDetails(method='async_reset_filter'),
    SERVICE_SET_EXTRA_FEATURES: ServiceMethodDetails(method='async_set_extra_features', schema=SERVICE_SCHEMA_EXTRA_FEATURES),
}

FAN_DIRECTIONS_MAP: dict[str, str] = {'forward': 'right', 'reverse': 'left'}


async def async_setup_entry(
    hass: HomeAssistant,
    config_entry: ConfigEntry,
    async_add_entities: AddConfigEntryEntitiesCallback,
) -> None:
    ...


class XiaomiGenericDevice(XiaomiCoordinatedMiioEntity, FanEntity):
    _attr_name: str = None

    def __init__(
        self,
        device: Any,
        entry: ConfigEntry,
        unique_id: str,
        coordinator: Any,
    ) -> None:
        ...

    @property
    @abstractmethod
    def operation_mode_class(self) -> type:
        ...

    @property
    def preset_modes(self) -> list[str]:
        ...

    @property
    def percentage(self) -> Optional[float]:
        ...

    @property
    def extra_state_attributes(self) -> dict[str, Any]:
        ...

    @property
    def is_on(self) -> bool:
        ...

    async def async_turn_on(self, percentage: Optional[float] = None, preset_mode: Optional[str] = None, **kwargs: Any) -> None:
        ...

    async def async_turn_off(self, **kwargs: Any) -> None:
        ...


class XiaomiGenericAirPurifier(XiaomiGenericDevice):
    def __init__(
        self,
        device: Any,
        entry: ConfigEntry,
        unique_id: str,
        coordinator: Any,
    ) -> None:
        ...

    @property
    def speed_count(self) -> int:
        ...

    @property
    def preset_mode(self) -> Optional[str]:
        ...

    @callback
    def _handle_coordinator_update(self) -> None:
        ...


class XiaomiAirPurifier(XiaomiGenericAirPurifier):
    SPEED_MODE_MAPPING: dict[int, Any] = ...
    REVERSE_SPEED_MODE_MAPPING: dict[Any, int] = ...

    def __init__(
        self,
        device: Any,
        entry: ConfigEntry,
        unique_id: str,
        coordinator: Any,
    ) -> None:
        ...

    @property
    def operation_mode_class(self) -> type:
        ...

    @property
    def percentage(self) -> Optional[float]:
        ...

    async def async_set_percentage(self, percentage: float) -> None:
        ...

    async def async_set_preset_mode(self, preset_mode: str) -> None:
        ...

    async def async_set_extra_features(self, features: int = 1) -> None:
        ...

    async def async_reset_filter(self) -> None:
        ...


class XiaomiAirPurifierMiot(XiaomiAirPurifier):
    @property
    def operation_mode_class(self) -> type:
        ...

    @property
    def percentage(self) -> Optional[float]:
        ...

    async def async_set_percentage(self, percentage: float) -> None:
        ...


class XiaomiAirPurifierMB4(XiaomiGenericAirPurifier):
    def __init__(
        self,
        device: Any,
        entry: ConfigEntry,
        unique_id: str,
        coordinator: Any,
    ) -> None:
        ...

    @property
    def operation_mode_class(self) -> type:
        ...

    @property
    def percentage(self) -> Optional[float]:
        ...

    async def async_set_percentage(self, percentage: float) -> None:
        ...

    async def async_set_preset_mode(self, preset_mode: str) -> None:
        ...

    @callback
    def _handle_coordinator_update(self) -> None:
        ...


class XiaomiAirFresh(XiaomiGenericAirPurifier):
    SPEED_MODE_MAPPING: dict[int, Any] = ...
    REVERSE_SPEED_MODE_MAPPING: dict[Any, int] = ...
    PRESET_MODE_MAPPING: dict[str, Any] = ...

    def __init__(
        self,
        device: Any,
        entry: ConfigEntry,
        unique_id: str,
        coordinator: Any,
    ) -> None:
        ...

    @property
    def operation_mode_class(self) -> type:
        ...

    @property
    def percentage(self) -> Optional[float]:
        ...

    async def async_set_percentage(self, percentage: float) -> None:
        ...

    async def async_set_preset_mode(self, preset_mode: str) -> None:
        ...

    async def async_set_extra_features(self, features: int = 1) -> None:
        ...

    async def async_reset_filter(self) -> None:
        ...


class XiaomiAirFreshA1(XiaomiGenericAirPurifier):
    def __init__(
        self,
        device: Any,
        entry: ConfigEntry,
        unique_id: str,
        coordinator: Any,
    ) -> None:
        ...

    @property
    def operation_mode_class(self) -> type:
        ...

    @property
    def percentage(self) -> Optional[float]:
        ...

    async def async_set_percentage(self, percentage: float) -> None:
        ...

    async def async_set_preset_mode(self, preset_mode: str) -> None:
        ...

    @callback
    def _handle_coordinator_update(self) -> None:
        ...


class XiaomiAirFreshT2017(XiaomiAirFreshA1):
    def __init__(
        self,
        device: Any,
        entry: ConfigEntry,
        unique_id: str,
        coordinator: Any,
    ) -> None:
        ...


class XiaomiGenericFan(XiaomiGenericDevice):
    _attr_translation_key: str = 'generic_fan'

    def __init__(
        self,
        device: Any,
        entry: ConfigEntry,
        unique_id: str,
        coordinator: Any,
    ) -> None:
        ...

    @property
    def operation_mode_class(self) -> type:
        ...

    @property
    def preset_mode(self) -> Optional[str]:
        ...

    @property
    def preset_modes(self) -> list[str]:
        ...

    @property
    def percentage(self) -> Optional[float]:
        ...

    @property
    def oscillating(self) -> Optional[bool]:
        ...

    async def async_oscillate(self, oscillating: bool) -> None:
        ...

    async def async_set_direction(self, direction: str) -> None:
        ...


class XiaomiFan(XiaomiGenericFan):
    def __init__(
        self,
        device: Any,
        entry: ConfigEntry,
        unique_id: str,
        coordinator: Any,
    ) -> None:
        ...

    @property
    def operation_mode_class(self) -> type:
        ...

    @property
    def preset_mode(self) -> Optional[str]:
        ...

    @property
    def preset_modes(self) -> list[str]:
        ...

    @callback
    def _handle_coordinator_update(self) -> None:
        ...

    async def async_set_preset_mode(self, preset_mode: str) -> None:
        ...

    async def async_set_percentage(self, percentage: float) -> None:
        ...


class XiaomiFanP5(XiaomiGenericFan):
    def __init__(
        self,
        device: Any,
        entry: ConfigEntry,
        unique_id: str,
        coordinator: Any,
    ) -> None:
        ...

    @property
    def operation_mode_class(self) -> type:
        ...

    @callback
    def _handle_coordinator_update(self) -> None:
        ...

    async def async_set_preset_mode(self, preset_mode: str) -> None:
        ...

    async def async_set_percentage(self, percentage: float) -> None:
        ...


class XiaomiFanMiot(XiaomiGenericFan):
    @property
    def operation_mode_class(self) -> type:
        ...

    @property
    def preset_mode(self) -> Optional[str]:
        ...

    @callback
    def _handle_coordinator_update(self) -> None:
        ...

    async def async_set_preset_mode(self, preset_mode: str) -> None:
        ...

    async def async_set_percentage(self, percentage: float) -> None:
        ...


class XiaomiFanZA5(XiaomiFanMiot):
    @property
    def operation_mode_class(self) -> type:
        ...


class XiaomiFan1C(XiaomiFanMiot):
    def __init__(
        self,
        device: Any,
        entry: ConfigEntry,
        unique_id: str,
        coordinator: Any,
    ) -> None:
        ...

    @callback
    def _handle_coordinator_update(self) -> None:
        ...

    async def async_set_percentage(self, percentage: float) -> None:
        ...