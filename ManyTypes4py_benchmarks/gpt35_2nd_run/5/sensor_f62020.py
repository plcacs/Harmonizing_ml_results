def setup_platform(hass: HomeAssistant, config: ConfigType, add_entities: AddEntitiesCallback, discovery_info: DiscoveryInfoType = None) -> None:
    ...

class EtherscanSensor(SensorEntity):
    _attr_attribution: str = 'Data provided by etherscan.io'

    def __init__(self, name: str, address: str, token: str, token_address: str) -> None:
        ...

    @property
    def name(self) -> str:
        ...

    @property
    def native_value(self) -> str:
        ...

    @property
    def native_unit_of_measurement(self) -> str:
        ...

    def update(self) -> None:
        ...
