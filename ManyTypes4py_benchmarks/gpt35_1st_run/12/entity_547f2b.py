from __future__ import annotations
from collections.abc import Callable, Coroutine
from dataclasses import dataclass
from typing import Any, Generic, TypeVar
from deebot_client.capabilities import Capabilities
from deebot_client.device import Device
from deebot_client.events import AvailabilityEvent
from deebot_client.events.base import Event
from sucks import EventListener, VacBot
from homeassistant.helpers import device_registry as dr
from homeassistant.helpers.device_registry import DeviceInfo
from homeassistant.helpers.entity import Entity, EntityDescription
from .const import DOMAIN

CapabilityEntity = TypeVar('CapabilityEntity')
EventT = TypeVar('EventT', bound=Event)

class EcovacsEntity(Entity, Generic[CapabilityEntity]):
    _attr_should_poll: bool = False
    _attr_has_entity_name: bool = True
    _always_available: bool = False

    def __init__(self, device: Device, capability: Capabilities, **kwargs: Any):
        self._attr_unique_id: str = f'{device.device_info["did"]}_{self.entity_description.key}'
        self._device: Device = device
        self._capability: Capabilities = capability
        self._subscribed_events: set[Type[Event]] = set()

    @property
    def device_info(self) -> DeviceInfo:
        device_info: dict = self._device.device_info
        info: DeviceInfo = DeviceInfo(identifiers={(DOMAIN, device_info['did'])}, manufacturer='Ecovacs', sw_version=self._device.fw_version, serial_number=device_info['name'], model_id=device_info['class'])
        if (nick := device_info.get('nick')):
            info['name'] = nick
        if (model := device_info.get('deviceName')):
            info['model'] = model
        if (mac := self._device.mac):
            info['connections'] = {(dr.CONNECTION_NETWORK_MAC, mac)}
        return info

    async def async_added_to_hass(self) -> None:
        await super().async_added_to_hass()
        if not self._always_available:

            async def on_available(event: AvailabilityEvent) -> None:
                self._attr_available = event.available
                self.async_write_ha_state()
            self._subscribe(AvailabilityEvent, on_available)

    def _subscribe(self, event_type: Type[Event], callback: Callable[[Event], None]) -> None:
        self._subscribed_events.add(event_type)
        self.async_on_remove(self._device.events.subscribe(event_type, callback))

    async def async_update(self) -> None:
        for event_type in self._subscribed_events:
            self._device.events.request_refresh(event_type)

class EcovacsDescriptionEntity(EcovacsEntity[CapabilityEntity]):
    def __init__(self, device: Device, capability: Capabilities, entity_description: EntityDescription, **kwargs: Any):
        self.entity_description: EntityDescription = entity_description
        super().__init__(device, capability, **kwargs)

@dataclass(kw_only=True, frozen=True)
class EcovacsCapabilityEntityDescription(EntityDescription, Generic[CapabilityEntity]):
    pass

class EcovacsLegacyEntity(Entity):
    _attr_has_entity_name: bool = True
    _attr_should_poll: bool = False

    def __init__(self, device: Device):
        self.device: Device = device
        vacuum: dict = device.vacuum
        self.error: Any = None
        self._attr_unique_id: str = vacuum['did']
        if (name := vacuum.get('nick')) is None:
            name = vacuum['did']
        self._attr_device_info: DeviceInfo = DeviceInfo(identifiers={(DOMAIN, vacuum['did'])}, manufacturer='Ecovacs', model=vacuum.get('deviceName'), name=name, serial_number=vacuum['did'])
        self._event_listeners: list[EventListener] = []

    @property
    def available(self) -> bool:
        return super().available and self.state is not None

    async def async_will_remove_from_hass(self) -> None:
        for listener in self._event_listeners:
            listener.unsubscribe()
