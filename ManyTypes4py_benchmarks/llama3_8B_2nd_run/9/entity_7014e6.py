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
_LOGGER = logging.getLogger(__name__)
T = TypeVar('T', bound=ProtectAdoptableDeviceModel | NVR)

class PermRequired(int, Enum):
    """Type of permission level required for entity."""
    NO_WRITE = 1
    WRITE = 2
    DELETE = 3

@callback
def _async_device_entities(data: ProtectData, klass: type[Entity], model_type: ModelType, descs: Sequence[EntityDescription], unadopted_descs: Sequence[EntityDescription] = None, ufp_device: ProtectAdoptableDeviceModel | NVR = None) -> Sequence[Entity]:
    ...

@callback
def _combine_model_descs(model_type: ModelType, model_descriptions: dict[ModelType, Sequence[EntityDescription]], all_descs: Sequence[EntityDescription]) -> Sequence[EntityDescription]:
    ...

@callback
def async_all_device_entities(data: ProtectData, klass: type[Entity], model_descriptions: dict[ModelType, Sequence[EntityDescription]] = None, all_descs: Sequence[EntityDescription] = None, unadopted_descs: Sequence[EntityDescription] = None, ufp_device: ProtectAdoptableDeviceModel | NVR = None) -> Sequence[Entity]:
    ...

class BaseProtectEntity(Entity):
    """Base class for UniFi protect entities."""
    _attr_should_poll: bool
    _attr_attribution: str
    _state_attrs: tuple[str]
    _attr_has_entity_name: bool
    _async_get_ufp_enabled: Callable[[ProtectAdoptableDeviceModel | NVR], bool]

    def __init__(self, data: ProtectData, device: ProtectAdoptableDeviceModel | NVR, description: EntityDescription | None = None) -> None:
        ...

    async def async_update(self) -> None:
        ...

    @callback
    def _async_set_device_info(self) -> None:
        ...

    @callback
    def _async_update_device_from_protect(self, device: ProtectAdoptableDeviceModel | NVR) -> None:
        ...

    @callback
    def _async_updated_event(self, device: ProtectAdoptableDeviceModel | NVR) -> None:
        ...

    async def async_added_to_hass(self) -> None:
        ...

class ProtectIsOnEntity(BaseProtectEntity):
    """Base class for entities with is_on property."""
    _state_attrs: tuple[str]

    def _async_update_device_from_protect(self, device: ProtectAdoptableDeviceModel | NVR) -> None:
        ...

class ProtectDeviceEntity(BaseProtectEntity):
    """Base class for UniFi protect entities."""

    @callback
    def _async_set_device_info(self) -> None:
        ...

class ProtectNVREntity(BaseProtectEntity):
    """Base class for unifi protect entities."""

    @callback
    def _async_set_device_info(self) -> None:
        ...

class EventEntityMixin(ProtectDeviceEntity):
    """Adds motion event attributes to sensor."""
    _unrecorded_attributes: frozenset[str]
    _event: Event | None
    _event_end: datetime | None

    @callback
    def _set_event_done(self) -> None:
        ...

    @callback
    def _set_event_attrs(self, event: Event) -> None:
        ...

    @callback
    def _async_event_with_immediate_end(self) -> None:
        ...

    @callback
    def _event_already_ended(self, prev_event: Event | None, prev_event_end: datetime | None) -> bool:
        ...

@dataclass(frozen=True, kw_only=True)
class ProtectEntityDescription(EntityDescription, Generic[T]):
    """Base class for protect entity descriptions."""
    ufp_required_field: str | None
    ufp_value: str | None
    ufp_value_fn: Callable[[T], str] | None
    ufp_enabled: bool | None
    ufp_perm: PermRequired | None
    has_required: Callable[[T], bool]
    get_ufp_enabled: Callable[[T], bool] | None

    def get_ufp_value(self, obj: T) -> str:
        ...

    def __post_init__(self) -> None:
        ...

@dataclass(frozen=True, kw_only=True)
class ProtectEventMixin(ProtectEntityDescription[T]):
    """Mixin for events."""
    ufp_event_obj: str | None
    ufp_obj_type: SmartDetectObjectType | None

    def get_event_obj(self, obj: T) -> Event | None:
        ...

    def has_matching_smart(self, event: Event) -> bool:
        ...

    def __post_init__(self) -> None:
        ...

@dataclass(frozen=True, kw_only=True)
class ProtectSetableKeysMixin(ProtectEntityDescription[T]):
    """Mixin for settable values."""
    ufp_set_method: str | None
    ufp_set_method_fn: Callable[[T, str], None] | None

    async def ufp_set(self, obj: T, value: str) -> None:
        ...
