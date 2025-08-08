def setup_platform(hass: HomeAssistant, config: ConfigType, add_entities: AddEntitiesCallback, discovery_info: DiscoveryInfoType = None) -> None:
    name: str = config.get(CONF_NAME)
    device: str = config.get(CONF_DEVICE)
    value_template: cv.template = config.get(CONF_VALUE_TEMPLATE)
    unit: str = config.get(CONF_UNIT_OF_MEASUREMENT)

class DweetSensor(SensorEntity):
    def __init__(self, hass: HomeAssistant, dweet: DweetData, name: str, value_template: cv.template, unit_of_measurement: str) -> None:
    @property
    def name(self) -> str:
    @property
    def native_unit_of_measurement(self) -> str:
    @property
    def native_value(self) -> str:
    def update(self) -> None:

class DweetData:
    def __init__(self, device: str) -> None:
    def update(self) -> None:
