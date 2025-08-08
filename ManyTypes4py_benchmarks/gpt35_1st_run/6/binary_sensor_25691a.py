from typing import Any, Dict, List, Optional
from homeassistant.config_entries import ConfigEntry
from homeassistant.helpers.entity import DeviceInfo
from homeassistant.helpers.typing import ConfigType

async def async_setup_entry(hass: HomeAssistant, entry: ConfigEntry, async_add_entities: AddConfigEntryEntitiesCallback) -> None:
    ...

class AlarmDecoderBinarySensor(AlarmDecoderEntity, BinarySensorEntity):
    def __init__(self, client: Any, zone_number: int, zone_name: str, zone_type: str, zone_rfid: Optional[str], zone_loop: Optional[int], relay_addr: Optional[int], relay_chan: Optional[int]) -> None:
        ...

    async def async_added_to_hass(self) -> None:
        ...

    def _fault_callback(self, zone: Optional[int]) -> None:
        ...

    def _restore_callback(self, zone: Optional[int]) -> None:
        ...

    def _rfx_message_callback(self, message: Any) -> None:
        ...

    def _rel_message_callback(self, message: Any) -> None:
        ...
