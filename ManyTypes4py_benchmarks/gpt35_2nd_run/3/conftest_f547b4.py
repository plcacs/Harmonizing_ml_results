def assert_sensor_state(hass: HomeAssistant, entity_id: str, expected_state: Any, attributes: dict = None) -> None:
def assert_temperature_sensor_registered(hass: HomeAssistant, serial_number: str, number: int, name: str) -> None:
def assert_pulse_counter_registered(hass: HomeAssistant, serial_number: str, number: int, name: str, quantity: str, per_time: str) -> None:
def assert_power_sensor_registered(hass: HomeAssistant, serial_number: str, number: int, name: str) -> None:
def assert_voltage_sensor_registered(hass: HomeAssistant, serial_number: str, number: int, name: str) -> None:
def assert_sensor_registered(hass: HomeAssistant, serial_number: str, sensor_type: str, number: int, name: str) -> Any:
def monitors() -> Generator[AsyncMock, None, None]:
