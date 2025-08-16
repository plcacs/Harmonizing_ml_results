def setup_platform(hass: HomeAssistant, config: ConfigType, add_entities: AddEntitiesCallback, discovery_info: DiscoveryInfoType = None) -> None:
def _setup_traditional_switches(logger: logging.Logger, config: ConfigType, scsgate: Any, add_entities_callback: AddEntitiesCallback) -> None:
def _setup_scenario_switches(logger: logging.Logger, config: ConfigType, scsgate: Any, hass: HomeAssistant) -> None:
class SCSGateSwitch(SwitchEntity):
    def __init__(self, scs_id: str, name: str, logger: logging.Logger, scsgate: Any) -> None:
    def scs_id(self) -> str:
    def name(self) -> str:
    def is_on(self) -> bool:
    def turn_on(self, **kwargs: Any) -> None:
    def turn_off(self, **kwargs: Any) -> None:
    def process_event(self, message: Any) -> None:
class SCSGateScenarioSwitch:
    def __init__(self, scs_id: str, name: str, logger: logging.Logger, hass: HomeAssistant) -> None:
    def scs_id(self) -> str:
    def name(self) -> str:
    def process_event(self, message: Any) -> None:
