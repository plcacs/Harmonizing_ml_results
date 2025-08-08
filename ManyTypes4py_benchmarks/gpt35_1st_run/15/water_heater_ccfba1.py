def _build_entities(device_list: list[ViCareDevice]) -> list[ViCareWater]:
    ...

async def async_setup_entry(hass: HomeAssistant, config_entry: ViCareConfigEntry, async_add_entities: AddConfigEntryEntitiesCallback) -> None:
    ...

class ViCareWater(ViCareEntity, WaterHeaterEntity):
    _attr_precision: float = PRECISION_TENTHS
    _attr_supported_features: int = WaterHeaterEntityFeature.TARGET_TEMPERATURE
    _attr_temperature_unit: UnitOfTemperature = UnitOfTemperature.CELSIUS
    _attr_min_temp: int = VICARE_TEMP_WATER_MIN
    _attr_max_temp: int = VICARE_TEMP_WATER_MAX
    _attr_operation_list: list[str] = list(HA_TO_VICARE_HVAC_DHW)
    _attr_translation_key: str = 'domestic_hot_water'
    _current_mode: str | None = None

    def __init__(self, device_serial: str, device_config: PyViCareDeviceConfig, device: PyViCareDevice, circuit: PyViCareHeatingCircuit) -> None:
        ...

    def update(self) -> None:
        ...

    def set_temperature(self, **kwargs: Any) -> None:
        ...

    @property
    def current_operation(self) -> str | None:
        ...
