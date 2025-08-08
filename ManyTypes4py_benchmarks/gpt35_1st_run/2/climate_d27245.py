async def _generate_entities(tado: TadoDataUpdateCoordinator) -> List[TadoClimate]:
    ...

async def create_climate_entity(tado: TadoDataUpdateCoordinator, name: str, zone_id: int, device_info: Dict[str, Any]) -> Optional[TadoClimate]:
    ...

class TadoClimate(TadoZoneEntity, ClimateEntity):
    def __init__(self, coordinator: TadoDataUpdateCoordinator, zone_name: str, zone_id: int, zone_type: str, supported_hvac_modes: List[str], support_flags: int, device_info: Dict[str, Any], capabilities: Dict[str, Any], auto_geofencing_supported: bool, heat_min_temp: Optional[float] = None, heat_max_temp: Optional[float] = None, heat_step: Optional[float] = None, cool_min_temp: Optional[float] = None, cool_max_temp: Optional[float] = None, cool_step: Optional[float] = None, supported_fan_modes: Optional[List[str]] = None, supported_swing_modes: Optional[List[str]] = None):
        ...

    @property
    def current_humidity(self) -> float:
        ...

    @property
    def current_temperature(self) -> float:
        ...

    @property
    def hvac_mode(self) -> str:
        ...

    @property
    def hvac_action(self) -> str:
        ...

    @property
    def fan_mode(self) -> str:
        ...

    async def async_set_fan_mode(self, fan_mode: str) -> None:
        ...

    @property
    def preset_mode(self) -> str:
        ...

    @property
    def preset_modes(self) -> List[str]:
        ...

    async def async_set_preset_mode(self, preset_mode: str) -> None:
        ...

    @property
    def target_temperature_step(self) -> float:
        ...

    @property
    def target_temperature(self) -> float:
        ...

    async def set_timer(self, temperature: float, time_period: Optional[timedelta] = None, requested_overlay: Optional[str] = None) -> None:
        ...

    async def set_temp_offset(self, offset: float) -> None:
        ...

    async def async_set_temperature(self, **kwargs: Any) -> None:
        ...

    async def async_set_hvac_mode(self, hvac_mode: str) -> None:
        ...

    @property
    def available(self) -> bool:
        ...

    @property
    def min_temp(self) -> float:
        ...

    @property
    def max_temp(self) -> float:
        ...

    @property
    def swing_mode(self) -> str:
        ...

    @property
    def extra_state_attributes(self) -> Dict[str, Any]:
        ...

    async def async_set_swing_mode(self, swing_mode: str) -> None:
        ...

    def _normalize_target_temp_for_hvac_mode(self) -> None:
        ...

    async def _control_hvac(self, hvac_mode: Optional[str] = None, target_temp: Optional[float] = None, fan_mode: Optional[str] = None, swing_mode: Optional[str] = None, duration: Optional[timedelta] = None, overlay_mode: Optional[str] = None, vertical_swing: Optional[str] = None, horizontal_swing: Optional[str] = None) -> None:
        ...

    def _is_valid_setting_for_hvac_mode(self, setting: str) -> bool:
        ...

    def _is_current_setting_supported_by_current_hvac_mode(self, setting: str, current_state: str) -> bool:
        ...
