from __future__ import annotations
import logging
from homeassistant.components.sensor import SensorEntity
from homeassistant.core import HomeAssistant, callback
from homeassistant.helpers.dispatcher import async_dispatcher_connect
from homeassistant.helpers.entity_platform import AddEntitiesCallback
from homeassistant.helpers.typing import ConfigType, DiscoveryInfoType
from . import CONF_PARTITIONNAME, DATA_EVL, PARTITION_SCHEMA, SIGNAL_KEYPAD_UPDATE, SIGNAL_PARTITION_UPDATE
from .entity import EnvisalinkEntity
_LOGGER: logging.Logger = logging.getLogger(__name__)

async def async_setup_platform(hass: HomeAssistant, config: ConfigType, async_add_entities: AddEntitiesCallback, discovery_info: DiscoveryInfoType = None) -> None:
    if not discovery_info:
        return
    configured_partitions: dict = discovery_info['partitions']
    entities: list[EnvisalinkSensor] = []
    for part_num in configured_partitions:
        entity_config_data: dict = PARTITION_SCHEMA(configured_partitions[part_num])
        entity: EnvisalinkSensor = EnvisalinkSensor(hass, entity_config_data[CONF_PARTITIONNAME], part_num, hass.data[DATA_EVL].alarm_state['partition'][part_num], hass.data[DATA_EVL])
        entities.append(entity)
    async_add_entities(entities)

class EnvisalinkSensor(EnvisalinkEntity, SensorEntity):
    def __init__(self, hass: HomeAssistant, partition_name: str, partition_number: int, info: dict, controller: Any) -> None:
        self._icon: str = 'mdi:alarm'
        self._partition_number: int = partition_number
        _LOGGER.debug('Setting up sensor for partition: %s', partition_name)
        super().__init__(f'{partition_name} Keypad', info, controller)

    async def async_added_to_hass(self) -> None:
        self.async_on_remove(async_dispatcher_connect(self.hass, SIGNAL_KEYPAD_UPDATE, self.async_update_callback))
        self.async_on_remove(async_dispatcher_connect(self.hass, SIGNAL_PARTITION_UPDATE, self.async_update_callback))

    @property
    def icon(self) -> str:
        return self._icon

    @property
    def native_value(self) -> str:
        return self._info['status']['alpha']

    @property
    def extra_state_attributes(self) -> dict:
        return self._info['status']

    @callback
    def async_update_callback(self, partition: str) -> None:
        if partition is None or int(partition) == self._partition_number:
            self.async_write_ha_state()
