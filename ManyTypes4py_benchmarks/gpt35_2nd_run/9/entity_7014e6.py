from __future__ import annotations
from collections.abc import Callable, Coroutine, Sequence
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from functools import partial
import logging
from operator import attrgetter
from typing import TYPE_CHECKING, Any, Generic, TypeVar
from uiprotect import make_enabled_getter, make_required_getter, make_value_getter
from uiprotect.data import NVR, Event, ModelType, ProtectAdoptableDeviceModel, SmartDetectObjectType, StateType
from homeassistant.core import callback
from homeassistant.helpers import device_registry as dr
from homeassistant.helpers.device_registry import DeviceInfo
from homeassistant.helpers.entity import Entity, EntityDescription
from .const import ATTR_EVENT_ID, ATTR_EVENT_SCORE, DEFAULT_ATTRIBUTION, DEFAULT_BRAND, DOMAIN
from .data import ProtectData, ProtectDeviceType

_LOGGER: logging.Logger
T: TypeVar('T', bound=ProtectAdoptableDeviceModel | NVR)

class PermRequired(int, Enum):
    """Type of permission level required for entity."""
    NO_WRITE: int
    WRITE: int
    DELETE: int

@callback
def _async_device_entities(data: ProtectData, klass: Any, model_type: ModelType, descs: Sequence[EntityDescription], unadopted_descs: Sequence[EntityDescription] = None, ufp_device: ProtectAdoptableDeviceModel = None) -> Sequence[Entity]:
    ...

@callback
def _combine_model_descs(model_type: ModelType, model_descriptions: dict[ModelType, Sequence[EntityDescription]], all_descs: Sequence[EntityDescription]) -> Sequence[EntityDescription]:
    ...

@callback
def async_all_device_entities(data: ProtectData, klass: Any, model_descriptions: dict[ModelType, Sequence[EntityDescription]] = None, all_descs: Sequence[EntityDescription] = None, unadopted_descs: Sequence[EntityDescription] = None, ufp_device: ProtectAdoptableDeviceModel = None) -> Sequence[Entity]:
    ...

class BaseProtectEntity(Entity):
    ...

class ProtectIsOnEntity(BaseProtectEntity):
    ...

class ProtectDeviceEntity(BaseProtectEntity):
    ...

class ProtectNVREntity(BaseProtectEntity):
    ...

class EventEntityMixin(ProtectDeviceEntity):
    ...

@dataclass(frozen=True, kw_only=True)
class ProtectEntityDescription(EntityDescription, Generic[T]):
    ...

@dataclass(frozen=True, kw_only=True)
class ProtectEventMixin(ProtectEntityDescription[T]):
    ...

@dataclass(frozen=True, kw_only=True)
class ProtectSetableKeysMixin(ProtectEntityDescription[T]):
    ...
