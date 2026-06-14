from __future__ import annotations

import logging
from abc import abstractmethod
from typing import Any

import voluptuous as vol

from homeassistant.components.fan import FanEntity, FanEntityFeature
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant, ServiceCall, callback
from homeassistant.helpers.entity_platform import AddConfigEntryEntitiesCallback

from miio.fan_common import MoveDirection as FanMoveDirection, OperationMode as FanOperationMode
from miio.integrations.airpurifier.dmaker.airfresh_t2017 import OperationMode as AirfreshOperationModeT2017
from miio.integrations.airpurifier.zhimi.airfresh import OperationMode as AirfreshOperationMode
from miio.integrations.airpurifier.zhimi.airpurifier import OperationMode as AirpurifierOperationMode
from miio.integrations.airpurifier.zhimi.airpurifier_miot import OperationMode as AirpurifierMiotOperationMode
from miio.integrations.fan.zhimi.zhimi_miot import OperationModeFanZA5 as FanZA5OperationMode

from .entity import XiaomiCoordinatedMiioEntity
from .typing import ServiceMethodDetails

_LOGGER: logging.Logger
DATA_KEY: str
ATTR_MODE_NATURE: str
ATTR_MODE_NORMAL: str
ATTR_BRIGHTNESS: str
ATTR_FAN_LEVEL: str
ATTR_SLEEP_TIME: str
ATTR_SLEEP_LEARN_COUNT: str
ATTR_EXTRA_FEATURES: str
ATTR_FEATURES: str
ATTR_TURBO_MODE_SUPPORTED: str
ATTR_SLEEP_MODE: str
ATTR_USE_TIME: str
ATTR_BUTTON_PRESSED: str
ATTR_FAVORITE_SPEED: str
ATTR_FAVORITE_RPM: str
ATTR_MOTOR_SPEED: str

AVAILABLE_ATTRIBUTES_AIRPURIFIER_COMMON: dict[str, str]
AVAILABLE_ATTRIBUTES_AIRPURIFIER: dict[str, str]
AVAILABLE_ATTRIBUTES_AIRPURIFIER_PRO: dict[str, str]
AVAILABLE_ATTRIBUTES_AIRPURIFIER_MIOT: dict[str, str]
AVAILABLE_ATTRIBUTES_AIRPURIFIER_PRO_V7: dict[str, str]
AVAILABLE_ATTRIBUTES_AIRPURIFIER_V3: dict[str, str]
AVAILABLE_ATTRIBUTES_AIRFRESH: dict[str, str]

PRESET_MODES_AIRPURIFIER: list[str]
PRESET_MODES_AIRPURIFIER_4_LITE: list[str]
PRESET_MODES_AIRPURIFIER_MIOT: list[str]
PRESET_MODES_AIRPURIFIER_PRO: list[str]
PRESET_MODES_AIRPURIFIER_PRO_V7: list[str]
PRESET_MODES_AIRPURIFIER_2S: list[str]
PRESET_MODES_AIRPURIFIER_3C: list[str]
PRESET_MODES_AIRPURIFIER_ZA1: list[str]
PRESET_MODES_AIRPURIFIER_V3: list[str]
PRESET_MODES_AIRFRESH: list[str]
PRESET_MODES_AIRFRESH_A1: list[str]

AIRPURIFIER_SERVICE_SCHEMA: vol.Schema
SERVICE_SCHEMA_EXTRA_FEATURES: vol.Schema
SERVICE_TO_METHOD: dict[str, ServiceMethodDetails]
FAN_DIRECTIONS_MAP: dict[str, str]

async def async_setup_entry(
    hass: HomeAssistant,
    config_entry: ConfigEntry,
    async_add_entities: AddConfigEntryEntitiesCallback,
) -> None: ...

class XiaomiGenericDevice(XiaomiCoordinatedMiioEntity, FanEntity):
    _attr_name: None
    _available_attributes: dict[str, str]
    _state: bool | None
    _mode: Any
    _fan_level: int | None
    _state_attrs: dict[str, Any]
    _device_features: int
    _preset_modes: list[str]

    def __init__(self, device: Any, entry: ConfigEntry, unique_id: str | None, coordinator: Any) -> None: ...

    @property
    @abstractmethod
    def operation_mode_class(self) -> Any: ...

    @property
    def preset_modes(self) -> list[str]: ...

    @property
    def percentage(self) -> int | None: ...

    @property
    def extra_state_attributes(self) -> dict[str, Any]: ...

    @property
    def is_on(self) -> bool | None: ...

    async def async_turn_on(
        self,
        percentage: int | None = ...,
        preset_mode: str | None = ...,
        **kwargs: Any,
    ) -> None: ...

    async def async_turn_off(self, **kwargs: Any) -> None: ...

class XiaomiGenericAirPurifier(XiaomiGenericDevice):
    _speed_count: int

    def __init__(self, device: Any, entry: ConfigEntry, unique_id: str | None, coordinator: Any) -> None: ...

    @property
    def speed_count(self) -> int: ...

    @property
    def preset_mode(self) -> str | None: ...

    @callback
    def _handle_coordinator_update(self) -> None: ...

class XiaomiAirPurifier(XiaomiGenericAirPurifier):
    SPEED_MODE_MAPPING: dict[int, AirpurifierOperationMode]
    REVERSE_SPEED_MODE_MAPPING: dict[AirpurifierOperationMode, int]

    def __init__(self, device: Any, entry: ConfigEntry, unique_id: str | None, coordinator: Any) -> None: ...

    @property
    def operation_mode_class(self) -> type[AirpurifierOperationMode]: ...

    @property
    def percentage(self) -> int | None: ...

    async def async_set_percentage(self, percentage: int) -> None: ...
    async def async_set_preset_mode(self, preset_mode: str) -> None: ...
    async def async_set_extra_features(self, features: int = ...) -> None: ...
    async def async_reset_filter(self) -> None: ...

class XiaomiAirPurifierMiot(XiaomiAirPurifier):
    @property
    def operation_mode_class(self) -> type[AirpurifierMiotOperationMode]: ...

    @property
    def percentage(self) -> int | None: ...

    async def async_set_percentage(self, percentage: int) -> None: ...

class XiaomiAirPurifierMB4(XiaomiGenericAirPurifier):
    _favorite_rpm: int | None
    _speed_range: tuple[int, int]
    _motor_speed: int

    def __init__(self, device: Any, entry: ConfigEntry, unique_id: str | None, coordinator: Any) -> None: ...

    @property
    def operation_mode_class(self) -> type[AirpurifierMiotOperationMode]: ...

    @property
    def percentage(self) -> int | None: ...

    async def async_set_percentage(self, percentage: int) -> None: ...
    async def async_set_preset_mode(self, preset_mode: str) -> None: ...

    @callback
    def _handle_coordinator_update(self) -> None: ...

class XiaomiAirFresh(XiaomiGenericAirPurifier):
    SPEED_MODE_MAPPING: dict[int, AirfreshOperationMode]
    REVERSE_SPEED_MODE_MAPPING: dict[AirfreshOperationMode, int]
    PRESET_MODE_MAPPING: dict[str, AirfreshOperationMode]

    def __init__(self, device: Any, entry: ConfigEntry, unique_id: str | None, coordinator: Any) -> None: ...

    @property
    def operation_mode_class(self) -> type[AirfreshOperationMode]: ...

    @property
    def percentage(self) -> int | None: ...

    async def async_set_percentage(self, percentage: int) -> None: ...
    async def async_set_preset_mode(self, preset_mode: str) -> None: ...
    async def async_set_extra_features(self, features: int = ...) -> None: ...
    async def async_reset_filter(self) -> None: ...

class XiaomiAirFreshA1(XiaomiGenericAirPurifier):
    _favorite_speed: int | None
    _speed_range: tuple[int, int]

    def __init__(self, device: Any, entry: ConfigEntry, unique_id: str | None, coordinator: Any) -> None: ...

    @property
    def operation_mode_class(self) -> type[AirfreshOperationModeT2017]: ...

    @property
    def percentage(self) -> int | None: ...

    async def async_set_percentage(self, percentage: int) -> None: ...
    async def async_set_preset_mode(self, preset_mode: str) -> None: ...

    @callback
    def _handle_coordinator_update(self) -> None: ...

class XiaomiAirFreshT2017(XiaomiAirFreshA1):
    def __init__(self, device: Any, entry: ConfigEntry, unique_id: str | None, coordinator: Any) -> None: ...

class XiaomiGenericFan(XiaomiGenericDevice):
    _attr_translation_key: str
    _preset_mode: str | None
    _oscillating: bool | None
    _percentage: int | None

    def __init__(self, device: Any, entry: ConfigEntry, unique_id: str | None, coordinator: Any) -> None: ...

    @property
    def preset_mode(self) -> str | None: ...

    @property
    def preset_modes(self) -> list[str]: ...

    @property
    def percentage(self) -> int | None: ...

    @property
    def oscillating(self) -> bool | None: ...

    async def async_oscillate(self, oscillating: bool) -> None: ...
    async def async_set_direction(self, direction: str) -> None: ...

class XiaomiFan(XiaomiGenericFan):
    _nature_mode: bool

    def __init__(self, device: Any, entry: ConfigEntry, unique_id: str | None, coordinator: Any) -> None: ...

    @property
    def operation_mode_class(self) -> None: ...

    @property
    def preset_mode(self) -> str: ...

    @property
    def preset_modes(self) -> list[str]: ...

    @callback
    def _handle_coordinator_update(self) -> None: ...

    async def async_set_preset_mode(self, preset_mode: str) -> None: ...
    async def async_set_percentage(self, percentage: int) -> None: ...

class XiaomiFanP5(XiaomiGenericFan):
    def __init__(self, device: Any, entry: ConfigEntry, unique_id: str | None, coordinator: Any) -> None: ...

    @property
    def operation_mode_class(self) -> type[FanOperationMode]: ...

    @callback
    def _handle_coordinator_update(self) -> None: ...

    async def async_set_preset_mode(self, preset_mode: str) -> None: ...
    async def async_set_percentage(self, percentage: int) -> None: ...

class XiaomiFanMiot(XiaomiGenericFan):
    @property
    def operation_mode_class(self) -> type[FanOperationMode]: ...

    @property
    def preset_mode(self) -> str | None: ...

    @callback
    def _handle_coordinator_update(self) -> None: ...

    async def async_set_preset_mode(self, preset_mode: str) -> None: ...
    async def async_set_percentage(self, percentage: int) -> None: ...

class XiaomiFanZA5(XiaomiFanMiot):
    @property
    def operation_mode_class(self) -> type[FanZA5OperationMode]: ...

class XiaomiFan1C(XiaomiFanMiot):
    _speed_count: int

    def __init__(self, device: Any, entry: ConfigEntry, unique_id: str | None, coordinator: Any) -> None: ...

    @callback
    def _handle_coordinator_update(self) -> None: ...

    async def async_set_percentage(self, percentage: int) -> None: ...