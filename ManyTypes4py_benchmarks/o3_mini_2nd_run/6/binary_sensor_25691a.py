from typing import Any, Dict, Optional
import logging
from homeassistant.components.binary_sensor import BinarySensorEntity
from homeassistant.core import HomeAssistant
from homeassistant.helpers.dispatcher import async_dispatcher_connect
from homeassistant.helpers.entity_platform import AddConfigEntryEntitiesCallback
from . import AlarmDecoderConfigEntry
from .const import (
    CONF_RELAY_ADDR,
    CONF_RELAY_CHAN,
    CONF_ZONE_LOOP,
    CONF_ZONE_NAME,
    CONF_ZONE_NUMBER,
    CONF_ZONE_RFID,
    CONF_ZONE_TYPE,
    DEFAULT_ZONE_OPTIONS,
    OPTIONS_ZONES,
    SIGNAL_REL_MESSAGE,
    SIGNAL_RFX_MESSAGE,
    SIGNAL_ZONE_FAULT,
    SIGNAL_ZONE_RESTORE,
)
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
    entry: AlarmDecoderConfigEntry,
    async_add_entities: AddConfigEntryEntitiesCallback,
) -> None:
    client: Any = entry.runtime_data.client
    zones: Dict[Any, Any] = entry.options.get(OPTIONS_ZONES, DEFAULT_ZONE_OPTIONS)
    entities: list[AlarmDecoderBinarySensor] = []
    for zone_num in zones:
        zone_info: Dict[str, Any] = zones[zone_num]
        zone_type: Any = zone_info[CONF_ZONE_TYPE]
        zone_name: str = zone_info[CONF_ZONE_NAME]
        zone_rfid: Optional[Any] = zone_info.get(CONF_ZONE_RFID)
        zone_loop: Optional[int] = zone_info.get(CONF_ZONE_LOOP)
        relay_addr: Optional[Any] = zone_info.get(CONF_RELAY_ADDR)
        relay_chan: Optional[Any] = zone_info.get(CONF_RELAY_CHAN)
        entity = AlarmDecoderBinarySensor(
            client, zone_num, zone_name, zone_type, zone_rfid, zone_loop, relay_addr, relay_chan
        )
        entities.append(entity)
    async_add_entities(entities)


class AlarmDecoderBinarySensor(AlarmDecoderEntity, BinarySensorEntity):
    _attr_should_poll = False

    def __init__(
        self,
        client: Any,
        zone_number: Any,
        zone_name: str,
        zone_type: Any,
        zone_rfid: Optional[Any],
        zone_loop: Optional[int],
        relay_addr: Optional[Any],
        relay_chan: Optional[Any],
    ) -> None:
        super().__init__(client)
        self._attr_unique_id: str = f'{client.serial_number}-zone-{zone_number}'
        self._zone_number: int = int(zone_number)
        self._zone_type: Any = zone_type
        self._attr_name: str = zone_name
        self._attr_is_on: bool = False
        self._rfid: Optional[Any] = zone_rfid
        self._loop: Optional[int] = zone_loop
        self._relay_addr: Optional[Any] = relay_addr
        self._relay_chan: Optional[Any] = relay_chan
        self._attr_device_class: Any = zone_type
        self._attr_extra_state_attributes: Dict[str, Any] = {CONF_ZONE_NUMBER: self._zone_number}

    async def async_added_to_hass(self) -> None:
        self.async_on_remove(async_dispatcher_connect(self.hass, SIGNAL_ZONE_FAULT, self._fault_callback))
        self.async_on_remove(async_dispatcher_connect(self.hass, SIGNAL_ZONE_RESTORE, self._restore_callback))
        self.async_on_remove(async_dispatcher_connect(self.hass, SIGNAL_RFX_MESSAGE, self._rfx_message_callback))
        self.async_on_remove(async_dispatcher_connect(self.hass, SIGNAL_REL_MESSAGE, self._rel_message_callback))

    def _fault_callback(self, zone: Optional[Any]) -> None:
        if zone is None or int(zone) == self._zone_number:
            self._attr_is_on = True
            self.schedule_update_ha_state()

    def _restore_callback(self, zone: Optional[Any]) -> None:
        if zone is None or (int(zone) == self._zone_number and not self._loop):
            self._attr_is_on = False
            self.schedule_update_ha_state()

    def _rfx_message_callback(self, message: Any) -> None:
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

    def _rel_message_callback(self, message: Any) -> None:
        if self._relay_addr == message.address and self._relay_chan == message.channel:
            _LOGGER.debug(
                '%s %d:%d value:%d',
                'Relay' if message.type == message.RELAY else 'ZoneExpander',
                message.address,
                message.channel,
                message.value,
            )
            self._attr_is_on = bool(message.value)
            self.schedule_update_ha_state()