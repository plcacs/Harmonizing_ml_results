"""Support for AlarmDecoder zone states- represented as binary sensors."""
import logging
from typing import Any, Dict, List, Optional, Union

from homeassistant.components.binary_sensor import BinarySensorEntity
from homeassistant.core import HomeAssistant, callback
from homeassistant.helpers.dispatcher import async_dispatcher_connect
from homeassistant.helpers.entity_platform import AddConfigEntryEntitiesCallback
from homeassistant.config_entries import ConfigEntry

from . import AlarmDecoderConfigEntry
from .const import CONF_RELAY_ADDR, CONF_RELAY_CHAN, CONF_ZONE_LOOP, CONF_ZONE_NAME, CONF_ZONE_NUMBER, CONF_ZONE_RFID, CONF_ZONE_TYPE, DEFAULT_ZONE_OPTIONS, OPTIONS_ZONES, SIGNAL_REL_MESSAGE, SIGNAL_RFX_MESSAGE, SIGNAL_ZONE_FAULT, SIGNAL_ZONE_RESTORE
from .entity import AlarmDecoderEntity

_LOGGER = logging.getLogger(__name__)
ATTR_RF_BIT0 = 'rf_bit0'
ATTR_RF_LOW_BAT = 'rf_low_battery'
ATTR_RF_SUPERVISED = 'rf_supervised'
ATTR_RF_BIT3 = 'rf_bit3'
ATTR_RF_LOOP3 = 'rf_loop3'
ATTR_RF_LOOP2 = 'rf_loop2'
ATTR_RF_LOOP4 = 'rf_loop4'
ATTR_RF_LOOP1 = 'rf_loop1'

async def async_setup_entry(
    hass: HomeAssistant, 
    entry: ConfigEntry, 
    async_add_entities: AddConfigEntryEntitiesCallback
) -> None:
    """Set up for AlarmDecoder sensor."""
    client = entry.runtime_data.client
    zones: Dict[str, Dict[str, Any]] = entry.options.get(OPTIONS_ZONES, DEFAULT_ZONE_OPTIONS)
    entities: List[AlarmDecoderBinarySensor] = []
    for zone_num in zones:
        zone_info = zones[zone_num]
        zone_type = zone_info[CONF_ZONE_TYPE]
        zone_name = zone_info[CONF_ZONE_NAME]
        zone_rfid = zone_info.get(CONF_ZONE_RFID)
        zone_loop = zone_info.get(CONF_ZONE_LOOP)
        relay_addr = zone_info.get(CONF_RELAY_ADDR)
        relay_chan = zone_info.get(CONF_RELAY_CHAN)
        entity = AlarmDecoderBinarySensor(client, zone_num, zone_name, zone_type, zone_rfid, zone_loop, relay_addr, relay_chan)
        entities.append(entity)
    async_add_entities(entities)

class AlarmDecoderBinarySensor(AlarmDecoderEntity, BinarySensorEntity):
    """Representation of an AlarmDecoder binary sensor."""
    _attr_should_poll: bool = False

    def __init__(
        self, 
        client: Any, 
        zone_number: str, 
        zone_name: str, 
        zone_type: str, 
        zone_rfid: Optional[str], 
        zone_loop: Optional[int], 
        relay_addr: Optional[int], 
        relay_chan: Optional[int]
    ) -> None:
        """Initialize the binary_sensor."""
        super().__init__(client)
        self._attr_unique_id: str = f'{client.serial_number}-zone-{zone_number}'
        self._zone_number: int = int(zone_number)
        self._zone_type: str = zone_type
        self._attr_name: str = zone_name
        self._attr_is_on: bool = False
        self._rfid: Optional[str] = zone_rfid
        self._loop: Optional[int] = zone_loop
        self._relay_addr: Optional[int] = relay_addr
        self._relay_chan: Optional[int] = relay_chan
        self._attr_device_class: str = zone_type
        self._attr_extra_state_attributes: Dict[str, Any] = {CONF_ZONE_NUMBER: self._zone_number}

    async def async_added_to_hass(self) -> None:
        """Register callbacks."""
        self.async_on_remove(async_dispatcher_connect(self.hass, SIGNAL_ZONE_FAULT, self._fault_callback))
        self.async_on_remove(async_dispatcher_connect(self.hass, SIGNAL_ZONE_RESTORE, self._restore_callback))
        self.async_on_remove(async_dispatcher_connect(self.hass, SIGNAL_RFX_MESSAGE, self._rfx_message_callback))
        self.async_on_remove(async_dispatcher_connect(self.hass, SIGNAL_REL_MESSAGE, self._rel_message_callback))

    @callback
    def _fault_callback(self, zone: Optional[Union[str, int]]) -> None:
        """Update the zone's state, if needed."""
        if zone is None or int(zone) == self._zone_number:
            self._attr_is_on = True
            self.schedule_update_ha_state()

    @callback
    def _restore_callback(self, zone: Optional[Union[str, int]]) -> None:
        """Update the zone's state, if needed."""
        if zone is None or (int(zone) == self._zone_number and (not self._loop)):
            self._attr_is_on = False
            self.schedule_update_ha_state()

    @callback
    def _rfx_message_callback(self, message: Any) -> None:
        """Update RF state."""
        if self._rfid and message and (message.serial_number == self._rfid):
            rfstate = message.value
            if self._loop:
                self._attr_is_on = bool(message.loop[self._loop - 1])
            attr: Dict[str, Any] = {CONF_ZONE_NUMBER: self._zone_number}
            if self._rfid and rfstate is not None:
                attr[ATTR_RF_BIT0] = bool(rfstate & 1)
                attr[ATTR_RF_LOW_BAT] = bool(rfstate & 2)
                attr[ATTR_RF_SUPERVISED] = bool(rfstate & 4)
                attr[ATTR_RF_BIT3] = bool(rfstate & 8)
                attr[ATTR_RF_LOOP3] = bool(rfstate & 16)
                attr[ATTR_RF_LOOP2] = bool(rfstate & 32)
                attr[ATTR_RF_LOOP4] = bool(rfstate & 64)
                attr[ATTR_RF_LOOP1] = bool(rfstate & 128)
            self._attr_extra_state_attributes = attr
            self.schedule_update_ha_state()

    @callback
    def _rel_message_callback(self, message: Any) -> None:
        """Update relay / expander state."""
        if self._relay_addr == message.address and self._relay_chan == message.channel:
            _LOGGER.debug('%s %d:%d value:%d', 'Relay' if message.type == message.RELAY else 'ZoneExpander', message.address, message.channel, message.value)
            self._attr_is_on = bool(message.value)
            self.schedule_update_ha_state()
