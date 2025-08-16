from typing import Any, Dict, List, Optional
from homeassistant.helpers.entity import DeviceInfo

async def async_setup_entry(hass: HomeAssistant, entry: AlarmDecoderConfigEntry, async_add_entities: AddConfigEntryEntitiesCallback) -> None:
    ...

class AlarmDecoderBinarySensor(AlarmDecoderEntity, BinarySensorEntity):
    _attr_should_poll: bool = False

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
