"""Shared Entity definition for UniFi Protect Integration."""
from __future__ import annotations
from collections.abc import Callable, Coroutine, Sequence
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from functools import partial
import logging
from operator import attrgetter
from typing import TYPE_CHECKING, Any, Generic, TypeVar, cast, Optional, Union, List, Dict, Tuple, Callable
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
def _async_device_entities(data: ProtectData, 
                          klass: type, 
                          model_type: ModelType, 
                          descs: List[ProtectEntityDescription], 
                          unadopted_descs: Optional[List[ProtectEntityDescription]] = None, 
                          ufp_device: Optional[ProtectAdoptableDeviceModel | NVR] = None) -> List[BaseProtectEntity]:
    if not descs and (not unadopted_descs):
        return []
    entities: List[BaseProtectEntity] = []
    devices: List[ProtectAdoptableDeviceModel | NVR] = [ufp_device] if ufp_device is not None else data.get_by_types({model_type}, ignore_unadopted=False)
    auth_user = data.api.bootstrap.auth_user
    for device in devices:
        if TYPE_CHECKING:
            assert isinstance(device, ProtectAdoptableDeviceModel)
        if not device.is_adopted_by_us:
            if unadopted_descs:
                for description in unadopted_descs:
                    entities.append(klass(data, device=device, description=description))
                    _LOGGER.debug('Adding %s entity %s for %s', klass.__name__, description.name, device.display_name)
            continue
        can_write = device.can_write(auth_user)
        for description in descs:
            if (perms := description.ufp_perm) is not None:
                if perms is PermRequired.WRITE and (not can_write):
                    continue
                if perms is PermRequired.NO_WRITE and can_write:
                    continue
                if perms is PermRequired.DELETE and (not device.can_delete(auth_user)):
                    continue
            if not description.has_required(device):
                continue
            entities.append(klass(data, device=device, description=description))
            _LOGGER.debug('Adding %s entity %s for %s', klass.__name__, description.name, device.display_name)
    return entities

_ALL_MODEL_TYPES: Tuple[ModelType, ...] = (ModelType.AIPORT, ModelType.CAMERA, ModelType.LIGHT, ModelType.SENSOR, ModelType.VIEWPORT, ModelType.DOORLOCK, ModelType.CHIME)

@callback
def _combine_model_descs(model_type: ModelType, 
                         model_descriptions: Optional[Dict[ModelType, List[ProtectEntityDescription]]], 
                         all_descs: Optional[List[ProtectEntityDescription]]) -> List[ProtectEntityDescription]:
    """Combine all the descriptions with descriptions a model type."""
    descs: List[ProtectEntityDescription] = list(all_descs) if all_descs else []
    if model_descriptions and (model_descs := model_descriptions.get(model_type)):
        descs.extend(model_descs)
    return descs

@callback
def async_all_device_entities(data: ProtectData, 
                             klass: type, 
                             model_descriptions: Optional[Dict[ModelType, List[ProtectEntityDescription]]] = None, 
                             all_descs: Optional[List[ProtectEntityDescription]] = None, 
                             unadopted_descs: Optional[List[ProtectEntityDescription]] = None, 
                             ufp_device: Optional[ProtectAdoptableDeviceModel | NVR] = None) -> List[BaseProtectEntity]:
    """Generate a list of all the device entities."""
    if ufp_device is None:
        entities: List[BaseProtectEntity] = []
        for model_type in _ALL_MODEL_TYPES:
            descs = _combine_model_descs(model_type, model_descriptions, all_descs)
            entities.extend(_async_device_entities(data, klass, model_type, descs, unadopted_descs))
        return entities
    device_model_type = ufp_device.model
    assert device_model_type is not None
    descs = _combine_model_descs(device_model_type, model_descriptions, all_descs)
    return _async_device_entities(data, klass, device_model_type, descs, unadopted_descs, ufp_device)

class BaseProtectEntity(Entity):
    """Base class for UniFi protect entities."""
    _attr_should_poll: bool = False
    _attr_attribution: str = DEFAULT_ATTRIBUTION
    _state_attrs: Tuple[str, ...] = ('_attr_available',)
    _attr_has_entity_name: bool = True
    _async_get_ufp_enabled: Optional[Callable[[ProtectAdoptableDeviceModel | NVR], bool]] = None

    def __init__(self, data: ProtectData, device: ProtectAdoptableDeviceModel | NVR, description: Optional[ProtectEntityDescription] = None) -> None:
        """Initialize the entity."""
        super().__init__()
        self.data: ProtectData = data
        self.device: ProtectAdoptableDeviceModel | NVR = device
        if description is None:
            self._attr_unique_id: str = self.device.mac
            self._attr_name: Optional[str] = None
        else:
            self.entity_description: ProtectEntityDescription = description
            self._attr_unique_id = f'{self.device.mac}_{description.key}'
            if isinstance(description, ProtectEntityDescription):
                self._async_get_ufp_enabled = description.get_ufp_enabled
        self._async_set_device_info()
        self._state_getters: Tuple[Callable[[], Any], ...] = tuple((partial(attrgetter(attr), self) for attr in self._state_attrs))

    async def async_update(self) -> None:
        """Update the entity.

        Only used by the generic entity update service.
        """
        await self.data.async_refresh()

    @callback
    def _async_set_device_info(self) -> None:
        """Set device info."""

    @callback
    def _async_update_device_from_protect(self, device: ProtectAdoptableDeviceModel | NVR) -> None:
        """Update Entity object from Protect device."""
        was_available: bool = self._attr_available
        if (last_updated_success := self.data.last_update_success):
            self.device = device
        if device.model is ModelType.NVR:
            available = last_updated_success
        else:
            if TYPE_CHECKING:
                assert isinstance(device, ProtectAdoptableDeviceModel)
            connected = device.state is StateType.CONNECTED or (not device.is_adopted_by_us and device.can_adopt)
            async_get_ufp_enabled = self._async_get_ufp_enabled
            enabled = not async_get_ufp_enabled or async_get_ufp_enabled(device)
            available = last_updated_success and connected and enabled
        if available != was_available:
            self._attr_available = available

    @callback
    def _async_updated_event(self, device: ProtectAdoptableDeviceModel | NVR) -> None:
        """When device is updated from Protect."""
        previous_attrs: List[Any] = [getter() for getter in self._state_getters]
        self._async_update_device_from_protect(device)
        changed: bool = False
        for idx, getter in enumerate(self._state_getters):
            if previous_attrs[idx] != getter():
                changed = True
                break
        if changed:
            if _LOGGER.isEnabledFor(logging.DEBUG):
                device_name = device.name or ''
                if hasattr(self, 'entity_description') and self.entity_description.name:
                    device_name += f' {self.entity_description.name}'
                _LOGGER.debug('Updating state [%s (%s)] %s -> %s', device_name, device.mac, previous_attrs, tuple((getattr(self, attr) for attr in self._state_attrs)))
            self.async_write_ha_state()

    async def async_added_to_hass(self) -> None:
        """When entity is added to hass."""
        await super().async_added_to_hass()
        self.async_on_remove(self.data.async_subscribe(self.device.mac, self._async_updated_event))
        self._async_update_device_from_protect(self.device)

class ProtectIsOnEntity(BaseProtectEntity):
    """Base class for entities with is_on property."""
    _state_attrs: Tuple[str, ...] = ('_attr_available', '_attr_is_on')

    def _async_update_device_from_protect(self, device: ProtectAdoptableDeviceModel | NVR) -> None:
        super()._async_update_device_from_protect(device)
        was_on: bool = self._attr_is_on
        if was_on != (is_on := (self.entity_description.get_ufp_value(device) is True)):
            self._attr_is_on = is_on

class ProtectDeviceEntity(BaseProtectEntity):
    """Base class for UniFi protect entities."""

    @callback
    def _async_set_device_info(self) -> None:
        self._attr_device_info = DeviceInfo(name=self.device.display_name, manufacturer=DEFAULT_BRAND, model=self.device.market_name or self.device.type, model_id=self.device.type, via_device=(DOMAIN, self.data.api.bootstrap.nvr.mac), sw_version=self.device.firmware_version, connections={(dr.CONNECTION_NETWORK_MAC, self.device.mac)}, configuration_url=self.device.protect_url)

class ProtectNVREntity(BaseProtectEntity):
    """Base class for unifi protect entities."""

    @callback
    def _async_set_device_info(self) -> None:
        self._attr_device_info = DeviceInfo(connections={(dr.CONNECTION_NETWORK_MAC, self.device.mac)}, identifiers={(DOMAIN, self.device.mac)}, manufacturer=DEFAULT_BRAND, name=self.device.display_name, model=self.device.type, sw_version=str(self.device.version), configuration_url=self.device.api.base_url)

class EventEntityMixin(ProtectDeviceEntity):
    """Adds motion event attributes to sensor."""
    _unrecorded_attributes: frozenset[str] = frozenset({ATTR_EVENT_ID, ATTR_EVENT_SCORE})
    _event: Optional[Event] = None
    _event_end: Optional[datetime] = None

    @callback
    def _set_event_done(self) -> None:
        """Clear the event and state."""

    @callback
    def _set_event_attrs(self, event: Event) -> None:
        """Set event attrs."""
        self._attr_extra_state_attributes = {ATTR_EVENT_ID: event.id, ATTR_EVENT_SCORE: event.score}

    @callback
    def _async_event_with_immediate_end(self) -> None:
        self.async_write_ha_state()
        self._set_event_done()
        self.async_write_ha_state()

    @callback
    def _event_already_ended(self, prev_event: Optional[Event], prev_event_end: Optional[datetime]) -> bool:
        """Determine if the event has already ended.

        The event_end time is passed because the prev_event and event object
        may be the same object, and the uiprotect code will mutate the
        event object so we need to check the datetime object that was
        saved from the last time the entity was updated.
        """
        return bool((event := self._event) and event.end and prev_event and prev_event_end and (prev_event.id == event.id))

@dataclass(frozen=True, kw_only=True)
class ProtectEntityDescription(EntityDescription, Generic[T]):
    """Base class for protect entity descriptions."""
    ufp_required_field: Optional[str] = None
    ufp_value: Optional[str] = None
    ufp_value_fn: Optional[Callable[[T], Any]] = None
    ufp_enabled: Optional[str] = None
    ufp_perm: Optional[PermRequired] = None
    has_required: Callable[[T], bool] = bool
    get_ufp_enabled: Optional[Callable[[T], bool]] = None

    def get_ufp_value(self, obj: T) -> Any:
        """Return value from UniFi Protect device; overridden in __post_init__."""
        raise RuntimeError(f'`ufp_value` or `ufp_value_fn` is required for {self}')

    def __post_init__(self) -> None:
        """Override get_ufp_value, has_required, and get_ufp_enabled if required."""
        _setter = partial(object.__setattr__, self)
        if (ufp_value := self.ufp_value) is not None:
            _setter('get_ufp_value', make_value_getter(ufp_value))
        elif (ufp_value_fn := self.ufp_value_fn) is not None:
            _setter('get_ufp_value', ufp_value_fn)
        if (ufp_enabled := self.ufp_enabled) is not None:
            _setter('get_ufp_enabled', make_enabled_getter(ufp_enabled))
        if (ufp_required_field := self.ufp_required_field) is not None:
            _setter('has_required', make_required_getter(ufp_required_field))

@dataclass(frozen=True, kw_only=True)
class ProtectEventMixin(ProtectEntityDescription[T]):
    """Mixin for events."""
    ufp_event_obj: Optional[str] = None
    ufp_obj_type: Optional[SmartDetectObjectType] = None

    def get_event_obj(self, obj: T) -> Optional[Any]:
        """Return value from UniFi Protect device."""
        return None

    def has_matching_smart(self, event: Event) -> bool:
        """Determine if the detection type is a match."""
        return not (obj_type := self.ufp_obj_type) or obj_type in event.smart_detect_types

    def __post_init__(self) -> None:
        """Override get_event_obj if ufp_event_obj is set."""
        if (_ufp_event_obj := self.ufp_event_obj) is not None:
            object.__setattr__(self, 'get_event_obj', attrgetter(_ufp_event_obj))
        super().__post_init__()

@dataclass(frozen=True, kw_only=True)
class ProtectSetableKeysMixin(ProtectEntityDescription[T]):
    """Mixin for settable values."""
    ufp_set_method: Optional[str] = None
    ufp_set_method_fn: Optional[Callable[[T, Any], Coroutine[Any, Any, None]]] = None

    async def ufp_set(self, obj: T, value: Any) -> None:
        """Set value for UniFi Protect device."""
        _LOGGER.debug('Setting %s to %s for %s', self.name, value, obj.display_name)
        if self.ufp_set_method is not None:
            await getattr(obj, self.ufp_set_method)(value)
        elif self.ufp_set_method_fn is not None:
            await self.ufp_set_method_fn(obj, value)
