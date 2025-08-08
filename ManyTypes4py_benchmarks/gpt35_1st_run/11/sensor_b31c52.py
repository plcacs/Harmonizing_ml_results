from typing import List, Dict, Optional, Callable

def thread_node_capability_to_str(char: int) -> str:
    ...

def thread_status_to_str(char: int) -> str:
    ...

class HomeKitSensorEntityDescription(SensorEntityDescription):
    ...

SIMPLE_SENSOR: Dict[str, HomeKitSensorEntityDescription] = {...}

class HomeKitSensor(HomeKitEntity, SensorEntity):
    ...

class HomeKitHumiditySensor(HomeKitSensor):
    ...

class HomeKitTemperatureSensor(HomeKitSensor):
    ...

class HomeKitLightSensor(HomeKitSensor):
    ...

class HomeKitCarbonDioxideSensor(HomeKitSensor):
    ...

class HomeKitBatterySensor(HomeKitSensor):
    ...

class SimpleSensor(CharacteristicEntity, SensorEntity):
    ...

ENTITY_TYPES: Dict[str, Callable] = {...}

REQUIRED_CHAR_BY_TYPE: Dict[str, str] = {...}

class RSSISensor(HomeKitEntity, SensorEntity):
    ...

async def async_setup_entry(hass, config_entry, async_add_entities):
    ...
