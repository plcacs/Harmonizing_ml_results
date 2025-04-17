"""Generic entity for the HomematicIP Cloud component."""
from __future__ import annotations
import logging
from typing import Any, Optional, Union
from homematicip.aio.device import AsyncDevice
from homematicip.aio.group import AsyncGroup
from homematicip.base.functionalChannels import FunctionalChannel
from homeassistant.const import ATTR_ID
from homeassistant.core import callback
from homeassistant.helpers import device_registry as dr, entity_registry as er
from homeassistant.helpers.device_registry import DeviceInfo
from homeassistant.helpers.entity import Entity
from .const import DOMAIN
from .hap import AsyncHome, HomematicipHAP
_LOGGER = logging.getLogger(__name__)
ATTR_MODEL_TYPE = 'model_type'
ATTR_LOW_BATTERY = 'low_battery'
ATTR_CONFIG_PENDING = 'config_pending'
ATTR_CONNECTION_TYPE = 'connection_type'
ATTR_DUTY_CYCLE_REACHED = 'duty_cycle_reached'
ATTR_IS_GROUP = 'is_group'
ATTR_RSSI_DEVICE = 'rssi_device'
ATTR_RSSI_PEER = 'rssi_peer'
ATTR_SABOTAGE = 'sabotage'
ATTR_GROUP_MEMBER_UNREACHABLE = 'group_member_unreachable'
ATTR_DEVICE_OVERHEATED = 'device_overheated'
ATTR_DEVICE_OVERLOADED = 'device_overloaded'
ATTR_DEVICE_UNTERVOLTAGE = 'device_undervoltage'
ATTR_EVENT_DELAY = 'event_delay'
DEVICE_ATTRIBUTE_ICONS: dict[str, str] = {'lowBat': 'mdi:battery-outline',
    'sabotage': 'mdi:shield-alert', 'dutyCycle': 'mdi:alert',
    'deviceOverheated': 'mdi:alert', 'deviceOverloaded': 'mdi:alert',
    'deviceUndervoltage': 'mdi:alert', 'configPending': 'mdi:alert-circle'}
DEVICE_ATTRIBUTES: dict[str, str] = {'modelType': ATTR_MODEL_TYPE,
    'connectionType': ATTR_CONNECTION_TYPE, 'sabotage': ATTR_SABOTAGE,
    'dutyCycle': ATTR_DUTY_CYCLE_REACHED, 'rssiDeviceValue':
    ATTR_RSSI_DEVICE, 'rssiPeerValue': ATTR_RSSI_PEER, 'deviceOverheated':
    ATTR_DEVICE_OVERHEATED, 'deviceOverloaded': ATTR_DEVICE_OVERLOADED,
    'deviceUndervoltage': ATTR_DEVICE_UNTERVOLTAGE, 'configPending':
    ATTR_CONFIG_PENDING, 'eventDelay': ATTR_EVENT_DELAY, 'id': ATTR_ID}
GROUP_ATTRIBUTES: dict[str, str] = {'modelType': ATTR_MODEL_TYPE, 'lowBat':
    ATTR_LOW_BATTERY, 'sabotage': ATTR_SABOTAGE, 'dutyCycle':
    ATTR_DUTY_CYCLE_REACHED, 'configPending': ATTR_CONFIG_PENDING,
    'unreach': ATTR_GROUP_MEMBER_UNREACHABLE}


class HomematicipGenericEntity(Entity):
    """Representation of the HomematicIP generic entity."""
    _attr_should_poll: bool = False

    def __init__(self, hap, device, post=None, channel=None,
        is_multi_channel=False):
        """Initialize the generic entity."""
        self._hap: HomematicipHAP = hap
        self._home: AsyncHome = hap.home
        self._device: Union[AsyncDevice, AsyncGroup] = device
        self._post: Optional[str] = post
        self._channel: Optional[int] = channel
        self._is_multi_channel: bool = is_multi_channel
        self.functional_channel: Optional[FunctionalChannel
            ] = self.get_current_channel()
        self.hmip_device_removed: bool = False

    @property
    def device_info(self):
        """Return device specific attributes."""
        if isinstance(self._device, AsyncDevice):
            return DeviceInfo(identifiers={(DOMAIN, self._device.id)},
                manufacturer=self._device.oem, model=self._device.modelType,
                name=self._device.label, sw_version=self._device.
                firmwareVersion, via_device=(DOMAIN, self._device.homeId))
        return None

    async def async_added_to_hass(self) ->None:
        """Register callbacks."""
        self._hap.hmip_device_by_entity_id[self.entity_id] = self._device
        self._device.on_update(self._async_device_changed)
        self._device.on_remove(self._async_device_removed)

    @callback
    def _async_device_changed(self, *args: Any, **kwargs: Any):
        """Handle device state changes."""
        if self.enabled:
            _LOGGER.debug('Event %s (%s)', self.name, self._device.modelType)
            self.async_write_ha_state()
        else:
            _LOGGER.debug(
                'Device Changed Event for %s (%s) not fired. Entity is disabled'
                , self.name, self._device.modelType)

    async def async_will_remove_from_hass(self) ->None:
        """Run when hmip device will be removed from hass."""
        if self.hmip_device_removed:
            try:
                del self._hap.hmip_device_by_entity_id[self.entity_id]
                self.async_remove_from_registries()
            except KeyError as err:
                _LOGGER.debug('Error removing HMIP device from registry: %s',
                    err)

    @callback
    def async_remove_from_registries(self):
        """Remove entity/device from registry."""
        self._device.remove_callback(self._async_device_changed)
        self._device.remove_callback(self._async_device_removed)
        if not self.registry_entry:
            return
        device_id: Optional[str] = self.registry_entry.device_id
        if device_id:
            device_registry = dr.async_get(self.hass)
            if device_id in device_registry.devices:
                device_registry.async_remove_device(device_id)
        else:
            entity_id: Optional[str] = self.registry_entry.entity_id
            if entity_id:
                entity_registry = er.async_get(self.hass)
                if entity_id in entity_registry.entities:
                    entity_registry.async_remove(entity_id)

    @callback
    def _async_device_removed(self, *args: Any, **kwargs: Any):
        """Handle hmip device removal."""
        self.hmip_device_removed = True
        self.hass.async_create_task(self.async_remove(force_remove=True),
            eager_start=False)

    @property
    def name(self):
        """Return the name of the generic entity."""
        name: Optional[str] = None
        if hasattr(self._device, 'functionalChannels'):
            if self._is_multi_channel and self._channel is not None:
                name = self._device.functionalChannels[self._channel].label
            elif len(self._device.functionalChannels) > 1:
                name = self._device.functionalChannels[1].label
        if not name:
            name = self._device.label
            if self._post:
                name = f'{name} {self._post}'
            elif self._is_multi_channel and self._channel is not None:
                name = f'{name} Channel{self._channel}'
        if name and self._home.name:
            name = f'{self._home.name} {name}'
        return name

    @property
    def available(self):
        """Return if entity is available."""
        return not self._device.unreach

    @property
    def unique_id(self):
        """Return a unique ID."""
        unique_id: str = f'{self.__class__.__name__}_{self._device.id}'
        if self._is_multi_channel and self._channel is not None:
            unique_id = (
                f'{self.__class__.__name__}_Channel{self._channel}_{self._device.id}'
                )
        return unique_id

    @property
    def icon(self):
        """Return the icon."""
        for attr, icon in DEVICE_ATTRIBUTE_ICONS.items():
            if getattr(self._device, attr, None):
                return icon
        return None

    @property
    def extra_state_attributes(self):
        """Return the state attributes of the generic entity."""
        state_attr: dict[str, Any] = {}
        if isinstance(self._device, AsyncDevice):
            for attr, attr_key in DEVICE_ATTRIBUTES.items():
                if (attr_value := getattr(self._device, attr, None)
                    ) is not None:
                    state_attr[attr_key] = attr_value
            state_attr[ATTR_IS_GROUP] = False
        if isinstance(self._device, AsyncGroup):
            for attr, attr_key in GROUP_ATTRIBUTES.items():
                if (attr_value := getattr(self._device, attr, None)
                    ) is not None:
                    state_attr[attr_key] = attr_value
            state_attr[ATTR_IS_GROUP] = True
        return state_attr

    def get_current_channel(self):
        """Return the FunctionalChannel for device."""
        if hasattr(self._device, 'functionalChannels'):
            if self._is_multi_channel and self._channel is not None:
                return self._device.functionalChannels[self._channel]
            if len(self._device.functionalChannels) > 1:
                return self._device.functionalChannels[1]
        return None
