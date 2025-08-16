from __future__ import annotations
from typing import Any

async def async_setup_entry(hass: HomeAssistant, config_entry: Any, async_add_entities: AddConfigEntryEntitiesCallback) -> None:

def _calculate_unique_id(config: Any, hourly: bool) -> str:

def format_condition(condition: str) -> str:

class MetWeather(SingleCoordinatorWeatherEntity[MetDataUpdateCoordinator]):

    def __init__(self, coordinator: MetDataUpdateCoordinator, config_entry: Any, name: str, is_metric: bool) -> None:

    @property
    def condition(self) -> str:

    @property
    def native_temperature(self) -> Any:

    @property
    def native_pressure(self) -> Any:

    @property
    def humidity(self) -> Any:

    @property
    def native_wind_speed(self) -> Any:

    @property
    def wind_bearing(self) -> Any:

    @property
    def native_wind_gust_speed(self) -> Any:

    @property
    def cloud_coverage(self) -> Any:

    @property
    def native_dew_point(self) -> Any:

    @property
    def uv_index(self) -> Any:

    def _forecast(self, hourly: bool) -> Any:

    @callback
    def _async_forecast_daily(self) -> Any:

    @callback
    def _async_forecast_hourly(self) -> Any:
