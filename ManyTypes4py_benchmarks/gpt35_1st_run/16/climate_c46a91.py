def _build_entities(device_list: list[PyViCareDevice]) -> list[ViCareClimate]:
    ...

async def async_setup_entry(hass: HomeAssistant, config_entry: ViCareConfigEntry, async_add_entities: AddConfigEntryEntitiesCallback) -> None:
    ...

class ViCareClimate(ViCareEntity, ClimateEntity):
    def __init__(self, device_serial: str, device_config: PyViCareDeviceConfig, device: PyViCareDevice, circuit: PyViCareHeatingCircuit) -> None:
        ...

    def update(self) -> None:
        ...

    @property
    def hvac_mode(self) -> HVACMode | None:
        ...

    def set_hvac_mode(self, hvac_mode: HVACMode) -> None:
        ...

    def vicare_mode_from_hvac_mode(self, hvac_mode: HVACMode) -> str | None:
        ...

    @property
    def hvac_modes(self) -> list[HVACMode]:
        ...

    @property
    def hvac_action(self) -> HVACAction:
        ...

    def set_temperature(self, **kwargs: Any) -> None:
        ...

    @property
    def preset_mode(self) -> str | None:
        ...

    def set_preset_mode(self, preset_mode: str) -> None:
        ...

    @property
    def extra_state_attributes(self) -> dict[str, Any]:
        ...

    def set_vicare_mode(self, vicare_mode: str) -> None:
        ...
