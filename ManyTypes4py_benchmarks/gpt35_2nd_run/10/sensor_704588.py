async def async_setup_platform(hass: HomeAssistant, config: ConfigType, async_add_entities: AddEntitiesCallback, discovery_info: DiscoveryInfoType = None) -> None:
    ...

class SensorManager:
    def __init__(self, manager: EnergyManager, async_add_entities: AddEntitiesCallback) -> None:
        ...

    async def async_start(self) -> None:
        ...

    async def _process_manager_data(self) -> None:
        ...

    @callback
    def _process_sensor_data(self, adapter: SourceAdapter, config: Mapping[str, Any], to_add: list[EnergyCostSensor], to_remove: dict) -> None:
        ...

def _set_result_unless_done(future: asyncio.Future) -> None:
    ...

class EnergyCostSensor(SensorEntity):
    def __init__(self, adapter: SourceAdapter, config: Mapping[str, Any]) -> None:
        ...

    def _reset(self, energy_state: State) -> None:
        ...

    @callback
    def _update_cost(self) -> None:
        ...

    async def async_added_to_hass(self) -> None:
        ...

    @callback
    def _async_state_changed_listener(self, *_) -> None:
        ...

    @callback
    def add_to_platform_abort(self) -> None:
        ...

    async def async_will_remove_from_hass(self) -> None:
        ...

    @callback
    def update_config(self, config: Mapping[str, Any]) -> None:
        ...

    @property
    def native_unit_of_measurement(self) -> str:
        ...

    @property
    def unique_id(self) -> str:
        ...
