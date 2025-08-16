def setup_platform(hass: HomeAssistant, config: ConfigType, add_entities: AddEntitiesCallback, discovery_info: DiscoveryInfoType = None) -> None:
    ...

class ArestBinarySensor(BinarySensorEntity):
    def __init__(self, arest: ArestData, resource: str, name: str, device_class: str, pin: str) -> None:
        ...

    def update(self) -> None:
        ...

class ArestData:
    def __init__(self, resource: str, pin: str) -> None:
        ...

    @Throttle(MIN_TIME_BETWEEN_UPDATES)
    def update(self) -> None:
        ...
