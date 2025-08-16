from __future__ import annotations
from datetime import datetime, timedelta
from typing import List, Tuple, Union

from homeassistant.components.weather import ATTR_CONDITION_CLOUDY, ATTR_CONDITION_EXCEPTIONAL, ATTR_CONDITION_FOG, ATTR_CONDITION_HAIL, ATTR_CONDITION_LIGHTNING, ATTR_CONDITION_LIGHTNING_RAINY, ATTR_CONDITION_PARTLYCLOUDY, ATTR_CONDITION_POURING, ATTR_CONDITION_RAINY, ATTR_CONDITION_SNOWY, ATTR_CONDITION_SNOWY_RAINY, ATTR_CONDITION_SUNNY, ATTR_CONDITION_WINDY, ATTR_CONDITION_WINDY_VARIANT, Forecast, WeatherEntity, WeatherEntityFeature
from homeassistant.config_entries import ConfigEntry
from homeassistant.const import UnitOfPressure, UnitOfSpeed, UnitOfTemperature
from homeassistant.core import HomeAssistant
from homeassistant.helpers.entity_platform import AddConfigEntryEntitiesCallback
from homeassistant.helpers.event import async_track_time_interval
from homeassistant.util import dt as dt_util

CONDITION_CLASSES: dict[str, List[str]] = {ATTR_CONDITION_CLOUDY: [], ATTR_CONDITION_FOG: [], ATTR_CONDITION_HAIL: [], ATTR_CONDITION_LIGHTNING: [], ATTR_CONDITION_LIGHTNING_RAINY: [], ATTR_CONDITION_PARTLYCLOUDY: [], ATTR_CONDITION_POURING: [], ATTR_CONDITION_RAINY: ['shower rain'], ATTR_CONDITION_SNOWY: [], ATTR_CONDITION_SNOWY_RAINY: [], ATTR_CONDITION_SUNNY: ['sunshine'], ATTR_CONDITION_WINDY: [], ATTR_CONDITION_WINDY_VARIANT: [], ATTR_CONDITION_EXCEPTIONAL: []}
CONDITION_MAP: dict[str, str] = {cond_code: cond_ha for cond_ha, cond_codes in CONDITION_CLASSES.items() for cond_code in cond_codes}

WEATHER_UPDATE_INTERVAL: timedelta = timedelta(minutes=30)

async def async_setup_entry(hass: HomeAssistant, config_entry: ConfigEntry, async_add_entities: AddConfigEntryEntitiesCallback) -> None:
    async_add_entities([DemoWeather('South', 'Sunshine', 21.6414, 92, 1099, 0.5, UnitOfTemperature.CELSIUS, UnitOfPressure.HPA, UnitOfSpeed.METERS_PER_SECOND, [[ATTR_CONDITION_RAINY, 1, 22, 15, 60], [ATTR_CONDITION_RAINY, 5, 19, 8, 30], [ATTR_CONDITION_CLOUDY, 0, 15, 9, 10], [ATTR_CONDITION_SUNNY, 0, 12, 6, 0], [ATTR_CONDITION_PARTLYCLOUDY, 2, 14, 7, 20], [ATTR_CONDITION_RAINY, 15, 18, 7, 0], [ATTR_CONDITION_FOG, 0.2, 21, 12, 100]], None, None), DemoWeather('North', 'Shower rain', -12, 54, 987, 4.8, UnitOfTemperature.FAHRENHEIT, UnitOfPressure.INHG, UnitOfSpeed.MILES_PER_HOUR, [[ATTR_CONDITION_SNOWY, 2, -10, -15, 60], [ATTR_CONDITION_PARTLYCLOUDY, 1, -13, -14, 25], [ATTR_CONDITION_SUNNY, 0, -18, -22, 70], [ATTR_CONDITION_SUNNY, 0.1, -23, -23, 90], [ATTR_CONDITION_SNOWY, 4, -19, -20, 40], [ATTR_CONDITION_SUNNY, 0.3, -14, -19, 0], [ATTR_CONDITION_SUNNY, 0, -9, -12, 0]], [[ATTR_CONDITION_SUNNY, 2, -10, -15, 60], [ATTR_CONDITION_SUNNY, 1, -13, -14, 25], [ATTR_CONDITION_SUNNY, 0, -18, -22, 70], [ATTR_CONDITION_SUNNY, 0.1, -23, -23, 90], [ATTR_CONDITION_SUNNY, 4, -19, -20, 40], [ATTR_CONDITION_SUNNY, 0.3, -14, -19, 0], [ATTR_CONDITION_SUNNY, 0, -9, -12, 0]], [[ATTR_CONDITION_SNOWY, 2, -10, -15, 60, True], [ATTR_CONDITION_PARTLYCLOUDY, 1, -13, -14, 25, False], [ATTR_CONDITION_SUNNY, 0, -18, -22, 70, True], [ATTR_CONDITION_SUNNY, 0.1, -23, -23, 90, False], [ATTR_CONDITION_SNOWY, 4, -19, -20, 40, True], [ATTR_CONDITION_SUNNY, 0.3, -14, -19, 0, False], [ATTR_CONDITION_SUNNY, 0, -9, -12, 0, True]])])

class DemoWeather(WeatherEntity):
    """Representation of a weather condition."""
    _attr_attribution: str = 'Powered by Home Assistant'
    _attr_should_poll: bool = False

    def __init__(self, name: str, condition: str, temperature: float, humidity: int, pressure: int, wind_speed: float, temperature_unit: UnitOfTemperature, pressure_unit: UnitOfPressure, wind_speed_unit: UnitOfSpeed, forecast_daily: List[List[Union[str, int]]], forecast_hourly: List[List[Union[str, int]]], forecast_twice_daily: List[List[Union[str, int, bool]]]) -> None:
        """Initialize the Demo weather."""
        self._attr_name: str = f'Demo Weather {name}'
        self._attr_unique_id: str = f'demo-weather-{name.lower()}'
        self._condition: str = condition
        self._native_temperature: float = temperature
        self._native_temperature_unit: UnitOfTemperature = temperature_unit
        self._humidity: int = humidity
        self._native_pressure: int = pressure
        self._native_pressure_unit: UnitOfPressure = pressure_unit
        self._native_wind_speed: float = wind_speed
        self._native_wind_speed_unit: UnitOfSpeed = wind_speed_unit
        self._forecast_daily: List[List[Union[str, int]]] = forecast_daily
        self._forecast_hourly: List[List[Union[str, int]]] = forecast_hourly
        self._forecast_twice_daily: List[List[Union[str, int, bool]]] = forecast_twice_daily
        self._attr_supported_features: int = 0
        if self._forecast_daily:
            self._attr_supported_features |= WeatherEntityFeature.FORECAST_DAILY
        if self._forecast_hourly:
            self._attr_supported_features |= WeatherEntityFeature.FORECAST_HOURLY
        if self._forecast_twice_daily:
            self._attr_supported_features |= WeatherEntityFeature.FORECAST_TWICE_DAILY

    async def async_added_to_hass(self) -> None:
        """Set up a timer updating the forecasts."""

        async def update_forecasts(_) -> None:
            if self._forecast_daily:
                self._forecast_daily = self._forecast_daily[1:] + self._forecast_daily[:1]
            if self._forecast_hourly:
                self._forecast_hourly = self._forecast_hourly[1:] + self._forecast_hourly[:1]
            if self._forecast_twice_daily:
                self._forecast_twice_daily = self._forecast_twice_daily[1:] + self._forecast_twice_daily[:1]
            await self.async_update_listeners(None)
        self.async_on_remove(async_track_time_interval(self.hass, update_forecasts, WEATHER_UPDATE_INTERVAL))

    @property
    def native_temperature(self) -> float:
        """Return the temperature."""
        return self._native_temperature

    @property
    def native_temperature_unit(self) -> UnitOfTemperature:
        """Return the unit of measurement."""
        return self._native_temperature_unit

    @property
    def humidity(self) -> int:
        """Return the humidity."""
        return self._humidity

    @property
    def native_wind_speed(self) -> float:
        """Return the wind speed."""
        return self._native_wind_speed

    @property
    def native_wind_speed_unit(self) -> UnitOfSpeed:
        """Return the wind speed."""
        return self._native_wind_speed_unit

    @property
    def native_pressure(self) -> int:
        """Return the pressure."""
        return self._native_pressure

    @property
    def native_pressure_unit(self) -> UnitOfPressure:
        """Return the pressure."""
        return self._native_pressure_unit

    @property
    def condition(self) -> str:
        """Return the weather condition."""
        return CONDITION_MAP[self._condition.lower()]

    async def async_forecast_daily(self) -> List[Forecast]:
        """Return the daily forecast."""
        reftime = dt_util.now().replace(hour=16, minute=0)
        forecast_data = []
        assert self._forecast_daily is not None
        for entry in self._forecast_daily:
            data_dict = Forecast(datetime=reftime.isoformat(), condition=entry[0], precipitation=entry[1], temperature=entry[2], templow=entry[3], precipitation_probability=entry[4])
            reftime = reftime + timedelta(hours=24)
            forecast_data.append(data_dict)
        return forecast_data

    async def async_forecast_hourly(self) -> List[Forecast]:
        """Return the hourly forecast."""
        reftime = dt_util.now().replace(hour=16, minute=0)
        forecast_data = []
        assert self._forecast_hourly is not None
        for entry in self._forecast_hourly:
            data_dict = Forecast(datetime=reftime.isoformat(), condition=entry[0], precipitation=entry[1], temperature=entry[2], templow=entry[3], precipitation_probability=entry[4])
            reftime = reftime + timedelta(hours=1)
            forecast_data.append(data_dict)
        return forecast_data

    async def async_forecast_twice_daily(self) -> List[Forecast]:
        """Return the twice daily forecast."""
        reftime = dt_util.now().replace(hour=11, minute=0)
        forecast_data = []
        assert self._forecast_twice_daily is not None
        for entry in self._forecast_twice_daily:
            data_dict = Forecast(datetime=reftime.isoformat(), condition=entry[0], precipitation=entry[1], temperature=entry[2], templow=entry[3], precipitation_probability=entry[4], is_daytime=entry[5])
            reftime = reftime + timedelta(hours=12)
            forecast_data.append(data_dict)
        return forecast_data
