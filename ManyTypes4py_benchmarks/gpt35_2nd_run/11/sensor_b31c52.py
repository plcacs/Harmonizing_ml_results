from typing import List, Dict, Optional, Callable

def thread_node_capability_to_str(char: int) -> str:
    ...

def thread_status_to_str(char: int) -> str:
    ...

class SimpleSensor(CharacteristicEntity, SensorEntity):
    def __init__(self, conn, info, char, description):
        ...

    def get_characteristic_types(self) -> List[str]:
        ...

    @property
    def name(self) -> str:
        ...

    @property
    def native_value(self) -> Optional[str]:
        ...

class RSSISensor(HomeKitEntity, SensorEntity):
    def __init__(self, accessory, devinfo):
        ...

    def get_characteristic_types(self) -> List[str]:
        ...

    @property
    def available(self) -> bool:
        ...

    @property
    def name(self) -> str:
        ...

    @property
    def old_unique_id(self) -> str:
        ...

    @property
    def native_value(self) -> Optional[int]:
        ...

async def async_setup_entry(hass, config_entry, async_add_entities):
    ...

def async_add_service(service) -> bool:
    ...

def async_add_characteristic(char) -> bool:
    ...

def async_add_accessory(accessory) -> bool:
    ...
