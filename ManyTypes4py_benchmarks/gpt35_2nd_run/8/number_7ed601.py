from __future__ import annotations
from collections.abc import Callable
from contextlib import suppress
from dataclasses import dataclass
import logging
from typing import Any, List, Optional
from PyViCare.PyViCareDevice import Device as PyViCareDevice
from PyViCare.PyViCareDeviceConfig import PyViCareDeviceConfig
from PyViCare.PyViCareHeatingDevice import HeatingDeviceWithComponent as PyViCareHeatingDeviceComponent
from PyViCare.PyViCareUtils import PyViCareInvalidDataError, PyViCareNotSupportedFeatureError, PyViCareRateLimitError
from requests.exceptions import ConnectionError as RequestConnectionError
from homeassistant.components.number import NumberDeviceClass, NumberEntity, NumberEntityDescription
from homeassistant.const import EntityCategory, UnitOfTemperature
from homeassistant.core import HomeAssistant
from homeassistant.helpers.entity_platform import AddConfigEntryEntitiesCallback
from .entity import ViCareEntity
from .types import HeatingProgram, ViCareConfigEntry, ViCareDevice, ViCareRequiredKeysMixin
from .utils import get_circuits, get_device_serial, is_supported

@dataclass(frozen=True)
class ViCareNumberEntityDescription(NumberEntityDescription, ViCareRequiredKeysMixin):
    value_setter: Optional[Callable]
    min_value_getter: Optional[Callable]
    max_value_getter: Optional[Callable]
    stepping_getter: Optional[Callable]

def _build_entities(device_list: List[ViCareDevice]) -> List[ViCareNumber]:
    ...

async def async_setup_entry(hass: HomeAssistant, config_entry, async_add_entities: AddConfigEntryEntitiesCallback):
    ...

class ViCareNumber(ViCareEntity, NumberEntity):
    def __init__(self, description: ViCareNumberEntityDescription, device_serial: str, device_config: PyViCareDeviceConfig, device: PyViCareDevice, component: Optional[PyViCareHeatingDeviceComponent] = None):
        ...

    @property
    def available(self) -> bool:
        ...

    def set_native_value(self, value: Any):
        ...

    def update(self):
        ...

def _get_value(fn: Optional[Callable], api) -> Any:
    ...
