def _above_greater_than_below(config: dict) -> dict:
def _no_overlapping(configs: list) -> list:
def update_probability(prior: float, prob_given_true: float, prob_given_false: float) -> float:
async def async_setup_platform(hass: HomeAssistant, config: ConfigType, async_add_entities: AddEntitiesCallback, discovery_info: DiscoveryInfoType = None) -> None:
class BayesianBinarySensor(BinarySensorEntity):
    def __init__(self, name: str, unique_id: str, prior: float, observations: list, probability_threshold: float, device_class: str) -> None:
    async def async_added_to_hass(self) -> None:
    def _recalculate_and_write_state(self) -> None:
    def _initialize_current_observations(self) -> OrderedDict:
    def _record_entity_observations(self, entity: str) -> OrderedDict:
    def _calculate_new_probability(self) -> float:
    def _build_observations_by_entity(self) -> dict:
    def _build_observations_by_template(self) -> dict:
    def _process_numeric_state(self, entity_observation: Observation, multi: bool = False) -> Union[bool, None]:
    def _process_state(self, entity_observation: Observation, multi: bool = False) -> Union[bool, None]:
    @property
    def extra_state_attributes(self) -> dict:
    async def async_update(self) -> None:
