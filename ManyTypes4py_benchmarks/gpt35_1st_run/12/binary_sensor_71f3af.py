from typing import List, Optional, Union

def is_valid_notification_binary_sensor(info: ZwaveDiscoveryInfo) -> bool:
    ...

async def async_setup_entry(hass: HomeAssistant, config_entry: ConfigEntry, async_add_entities: AddConfigEntryEntitiesCallback) -> None:
    ...

class ZWaveBooleanBinarySensor(ZWaveBaseEntity, BinarySensorEntity):
    def __init__(self, config_entry: ConfigEntry, driver: Driver, info: ZwaveDiscoveryInfo) -> None:
        ...

class ZWaveNotificationBinarySensor(ZWaveBaseEntity, BinarySensorEntity):
    def __init__(self, config_entry: ConfigEntry, driver: Driver, info: ZwaveDiscoveryInfo, state_key: str, description: Optional[NotificationZWaveJSEntityDescription] = None) -> None:
        ...

class ZWavePropertyBinarySensor(ZWaveBaseEntity, BinarySensorEntity):
    def __init__(self, config_entry: ConfigEntry, driver: Driver, info: ZwaveDiscoveryInfo, description: PropertyZWaveJSEntityDescription) -> None:
        ...

class ZWaveConfigParameterBinarySensor(ZWaveBooleanBinarySensor):
    def __init__(self, config_entry: ConfigEntry, driver: Driver, info: ZwaveDiscoveryInfo) -> None:
        ...
