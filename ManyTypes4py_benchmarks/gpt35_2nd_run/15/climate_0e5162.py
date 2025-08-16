async def async_setup_entry(hass: HomeAssistant, config_entry: ConfigEntry, async_add_entities: AddEntitiesCallback) -> None:
    ...

@callback
def async_setup_climate_entities(async_add_entities: AddEntitiesCallback, coordinator: ShellyBlockCoordinator) -> None:
    ...

@callback
def async_restore_climate_entities(hass: HomeAssistant, config_entry: ConfigEntry, async_add_entities: AddEntitiesCallback, coordinator: ShellyBlockCoordinator) -> None:
    ...

@callback
def async_setup_rpc_entry(hass: HomeAssistant, config_entry: ConfigEntry, async_add_entities: AddEntitiesCallback) -> None:
    ...

@dataclass
class ShellyClimateExtraStoredData(ExtraStoredData):
    last_target_temp: Any = None

    def as_dict(self) -> Mapping[str, Any]:
        ...

class BlockSleepingClimate(CoordinatorEntity[ShellyBlockCoordinator], RestoreEntity, ClimateEntity):
    def __init__(self, coordinator: ShellyBlockCoordinator, sensor_block: Block, device_block: Block, entry: RegistryEntry = None) -> None:
        ...

    @property
    def extra_restore_state_data(self) -> ShellyClimateExtraStoredData:
        ...

    @property
    def unique_id(self) -> str:
        ...

    @property
    def target_temperature(self) -> float:
        ...

    @property
    def current_temperature(self) -> float:
        ...

    @property
    def available(self) -> bool:
        ...

    @property
    def hvac_mode(self) -> HVACMode:
        ...

    @property
    def preset_mode(self) -> str:
        ...

    @property
    def hvac_action(self) -> HVACAction:
        ...

    @property
    def preset_modes(self) -> List[str]:
        ...

    async def set_state_full_path(self, **kwargs) -> Any:
        ...

    async def async_set_temperature(self, **kwargs) -> None:
        ...

    async def async_set_hvac_mode(self, hvac_mode: HVACMode) -> None:
        ...

    async def async_set_preset_mode(self, preset_mode: str) -> None:
        ...

    async def async_added_to_hass(self) -> None:
        ...

    @callback
    def _handle_coordinator_update(self) -> None:
        ...

class RpcClimate(ShellyRpcEntity, ClimateEntity):
    def __init__(self, coordinator: ShellyRpcCoordinator, id_: str) -> None:
        ...

    @property
    def target_temperature(self) -> float:
        ...

    @property
    def current_temperature(self) -> float:
        ...

    @property
    def current_humidity(self) -> float:
        ...

    @property
    def hvac_mode(self) -> HVACMode:
        ...

    @property
    def hvac_action(self) -> HVACAction:
        ...

    async def async_set_temperature(self, **kwargs) -> None:
        ...

    async def async_set_hvac_mode(self, hvac_mode: HVACMode) -> None:
        ...

class RpcBluTrvClimate(ShellyRpcEntity, ClimateEntity):
    def __init__(self, coordinator: ShellyRpcCoordinator, id_: str) -> None:
        ...

    @property
    def target_temperature(self) -> float:
        ...

    @property
    def current_temperature(self) -> float:
        ...

    @property
    def hvac_action(self) -> HVACAction:
        ...

    async def async_set_temperature(self, **kwargs) -> None:
        ...
