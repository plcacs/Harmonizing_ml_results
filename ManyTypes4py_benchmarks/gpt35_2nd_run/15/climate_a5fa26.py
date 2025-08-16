async def async_setup_entry(hass: HomeAssistant, config_entry: EcobeeConfigEntry, async_add_entities: AddConfigEntryEntitiesCallback) -> None:

class Thermostat(ClimateEntity):
    def __init__(self, data: EcobeeData, thermostat_index: int, thermostat: dict, hass: HomeAssistant) -> None:

    async def async_update(self) -> None:

    @property
    def available(self) -> bool:

    @property
    def supported_features(self) -> int:

    @property
    def device_info(self) -> DeviceInfo:

    @property
    def current_temperature(self) -> float:

    @property
    def target_temperature_low(self) -> Optional[float]:

    @property
    def target_temperature_high(self) -> Optional[float]:

    @property
    def target_temperature_step(self) -> float:

    @property
    def settings(self) -> dict:

    @property
    def has_humidifier_control(self) -> bool:

    @property
    def target_humidity(self) -> Optional[int]:

    @property
    def target_temperature(self) -> Optional[float]:

    @property
    def fan(self) -> str:

    @property
    def fan_mode(self) -> str:

    @property
    def preset_mode(self) -> Optional[str]:

    @property
    def hvac_mode(self) -> HVACMode:

    @property
    def current_humidity(self) -> Optional[int]:

    @property
    def hvac_action(self) -> HVACAction:

    @property
    def extra_state_attributes(self) -> dict:

    @property
    def remote_sensors(self) -> List[str]:

    @property
    def remote_sensor_devices(self) -> List[str]:

    @property
    def remote_sensor_ids_names(self) -> List[dict]:

    @property
    def active_sensors_in_preset_mode(self) -> List[str]:

    @property
    def active_sensor_devices_in_preset_mode(self) -> List[str]:

    def set_preset_mode(self, preset_mode: str) -> None:

    @property
    def preset_modes(self) -> List[str]:

    @property
    def comfort_settings(self) -> dict:

    def set_auto_temp_hold(self, heat_temp: Optional[float], cool_temp: Optional[float]) -> None:

    def set_fan_mode(self, fan_mode: str) -> None:

    def set_temp_hold(self, temp: float) -> None:

    def set_temperature(self, **kwargs: Any) -> None:

    def set_humidity(self, humidity: int) -> None:

    def set_hvac_mode(self, hvac_mode: HVACMode) -> None:

    def set_fan_min_on_time(self, fan_min_on_time: int) -> None:

    def resume_program(self, resume_all: bool) -> None:

    def set_sensors_used_in_climate(self, device_ids: List[str], preset_mode: Optional[str] = None) -> None:

    def hold_preference(self) -> str:

    def hold_hours(self) -> Optional[int]:

    def create_vacation(self, service_data: dict) -> None:

    def delete_vacation(self, vacation_name: str) -> None:

    def turn_on(self) -> None:

    def set_dst_mode(self, dst_enabled: bool) -> None:

    def set_mic_mode(self, mic_enabled: bool) -> None:

    def set_occupancy_modes(self, auto_away: Optional[bool] = None, follow_me: Optional[bool] = None) -> None:
