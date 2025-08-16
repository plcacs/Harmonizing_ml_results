def setup_platform(hass: HomeAssistant, config: ConfigType, add_entities_callback: AddEntitiesCallback, discovery_info: DiscoveryInfoType = None) -> None:
    switches: dict[str, Any] = config.get('switches', {})
    devices: list[KankunSwitch] = []
    for dev_name, properties in switches.items():
        devices.append(KankunSwitch(hass, properties.get(CONF_NAME, dev_name), properties.get(CONF_HOST), properties.get(CONF_PORT, DEFAULT_PORT), properties.get(CONF_PATH, DEFAULT_PATH), properties.get(CONF_USERNAME), properties.get(CONF_PASSWORD))
    add_entities_callback(devices)

class KankunSwitch(SwitchEntity):
    def __init__(self, hass: HomeAssistant, name: str, host: str, port: int, path: str, user: str, passwd: str) -> None:
    def _switch(self, newstate: str) -> bool:
    def _query_state(self) -> bool:
    @property
    def name(self) -> str:
    @property
    def is_on(self) -> bool:
    def update(self) -> None:
    def turn_on(self, **kwargs) -> None:
    def turn_off(self, **kwargs) -> None:
