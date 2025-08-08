from typing import List, Optional, Tuple

class DemoWeather(WeatherEntity):
    def __init__(self, name: str, condition: str, temperature: float, humidity: int, pressure: int, wind_speed: float, temperature_unit: UnitOfTemperature, pressure_unit: UnitOfPressure, wind_speed_unit: UnitOfSpeed, forecast_daily: Optional[List[List[str]]], forecast_hourly: Optional[List[List[str]]], forecast_twice_daily: Optional[List[List[Tuple[str, int, int, int, int, bool]]]):
        ...

    async def async_added_to_hass(self) -> None:
        ...

    @property
    def native_temperature(self) -> float:
        ...

    @property
    def native_temperature_unit(self) -> UnitOfTemperature:
        ...

    @property
    def humidity(self) -> int:
        ...

    @property
    def native_wind_speed(self) -> float:
        ...

    @property
    def native_wind_speed_unit(self) -> UnitOfSpeed:
        ...

    @property
    def native_pressure(self) -> int:
        ...

    @property
    def native_pressure_unit(self) -> UnitOfPressure:
        ...

    @property
    def condition(self) -> str:
        ...

    async def async_forecast_daily(self) -> List[Forecast]:
        ...

    async def async_forecast_hourly(self) -> List[Forecast]:
        ...

    async def async_forecast_twice_daily(self) -> List[Forecast]:
        ...
