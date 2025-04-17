"""Demo platform that offers fake meteorological data."""
from __future__ import annotations
from datetime import datetime, timedelta
from typing import Callable, List, Optional, Union
from homeassistant.components.weather import ATTR_CONDITION_CLOUDY, ATTR_CONDITION_EXCEPTIONAL, ATTR_CONDITION_FOG, ATTR_CONDITION_HAIL, ATTR_CONDITION_LIGHTNING, ATTR_CONDITION_LIGHTNING_RAINY, ATTR_CONDITION_PARTLYCLOUDY, ATTR_CONDITION_POURING, ATTR_CONDITION_RAINY, ATTR_CONDITION_SNOWY, ATTR_CONDITION_SNOWY_RAINY, ATTR_CONDITION_SUNNY, ATTR_CONDITION_WINDY, ATTR_CONDITION_WINDY_VARIANT, Forecast, WeatherEntity, WeatherEntityFeature
from homeassistant.config_entries import ConfigEntry
from homeassistant.const import UnitOfPressure, UnitOfSpeed, UnitOfTemperature
from homeassistant.core import HomeAssistant
from homeassistant.helpers.entity_platform import AddConfigEntryEntitiesCallback
from homeassistant.helpers.event import CALLBACK_TYPE, Event, async_track_time_interval
from homeassistant.util import dt as dt_util
CONDITION_CLASSES: dict[str, List[str]] = {ATTR_CONDITION_CLOUDY: [],
    ATTR_CONDITION_FOG: [], ATTR_CONDITION_HAIL: [],
    ATTR_CONDITION_LIGHTNING: [], ATTR_CONDITION_LIGHTNING_RAINY: [],
    ATTR_CONDITION_PARTLYCLOUDY: [], ATTR_CONDITION_POURING: [],
    ATTR_CONDITION_RAINY: ['shower rain'], ATTR_CONDITION_SNOWY: [],
    ATTR_CONDITION_SNOWY_RAINY: [], ATTR_CONDITION_SUNNY: ['sunshine'],
    ATTR_CONDITION_WINDY: [], ATTR_CONDITION_WINDY_VARIANT: [],
    ATTR_CONDITION_EXCEPTIONAL: []}
CONDITION_MAP: dict[str, str] = {cond_code.lower(): cond_ha for cond_ha,
    cond_codes in CONDITION_CLASSES.items() for cond_code in cond_codes}
WEATHER_UPDATE_INTERVAL: timedelta = timedelta(minutes=30)


async def async_setup_entry(hass: HomeAssistant, config_entry: ConfigEntry,
    async_add_entities: AddConfigEntryEntitiesCallback) ->None:
    """Set up the Demo config entry."""
    async_add_entities([DemoWeather('South', 'Sunshine', 21.6414, 92.0, 
        1099.0, 0.5, UnitOfTemperature.CELSIUS, UnitOfPressure.HPA,
        UnitOfSpeed.METERS_PER_SECOND, [[ATTR_CONDITION_RAINY, 1.0, 22.0, 
        15.0, 60.0], [ATTR_CONDITION_RAINY, 5.0, 19.0, 8.0, 30.0], [
        ATTR_CONDITION_CLOUDY, 0.0, 15.0, 9.0, 10.0], [ATTR_CONDITION_SUNNY,
        0.0, 12.0, 6.0, 0.0], [ATTR_CONDITION_PARTLYCLOUDY, 2.0, 14.0, 7.0,
        20.0], [ATTR_CONDITION_RAINY, 15.0, 18.0, 7.0, 0.0], [
        ATTR_CONDITION_FOG, 0.2, 21.0, 12.0, 100.0]], None, None),
        DemoWeather('North', 'Shower rain', -12.0, 54.0, 987.0, 4.8,
        UnitOfTemperature.FAHRENHEIT, UnitOfPressure.INHG, UnitOfSpeed.
        MILES_PER_HOUR, [[ATTR_CONDITION_SNOWY, 2.0, -10.0, -15.0, 60.0], [
        ATTR_CONDITION_PARTLYCLOUDY, 1.0, -13.0, -14.0, 25.0], [
        ATTR_CONDITION_SUNNY, 0.0, -18.0, -22.0, 70.0], [
        ATTR_CONDITION_SUNNY, 0.1, -23.0, -23.0, 90.0], [
        ATTR_CONDITION_SNOWY, 4.0, -19.0, -20.0, 40.0], [
        ATTR_CONDITION_SUNNY, 0.3, -14.0, -19.0, 0.0], [
        ATTR_CONDITION_SUNNY, 0.0, -9.0, -12.0, 0.0]], [[
        ATTR_CONDITION_SUNNY, 2.0, -10.0, -15.0, 60.0], [
        ATTR_CONDITION_SUNNY, 1.0, -13.0, -14.0, 25.0], [
        ATTR_CONDITION_SUNNY, 0.0, -18.0, -22.0, 70.0], [
        ATTR_CONDITION_SUNNY, 0.1, -23.0, -23.0, 90.0], [
        ATTR_CONDITION_SUNNY, 4.0, -19.0, -20.0, 40.0], [
        ATTR_CONDITION_SUNNY, 0.3, -14.0, -19.0, 0.0], [
        ATTR_CONDITION_SUNNY, 0.0, -9.0, -12.0, 0.0]], [[
        ATTR_CONDITION_SNOWY, 2.0, -10.0, -15.0, 60.0, True], [
        ATTR_CONDITION_PARTLYCLOUDY, 1.0, -13.0, -14.0, 25.0, False], [
        ATTR_CONDITION_SUNNY, 0.0, -18.0, -22.0, 70.0, True], [
        ATTR_CONDITION_SUNNY, 0.1, -23.0, -23.0, 90.0, False], [
        ATTR_CONDITION_SNOWY, 4.0, -19.0, -20.0, 40.0, True], [
        ATTR_CONDITION_SUNNY, 0.3, -14.0, -19.0, 0.0, False], [
        ATTR_CONDITION_SUNNY, 0.0, -9.0, -12.0, 0.0, True]])])


class DemoWeather(WeatherEntity):
    """Representation of a weather condition."""
    _attr_attribution: str = 'Powered by Home Assistant'
    _attr_should_poll: bool = False

    def __init__(self, name, condition, temperature, humidity, pressure,
        wind_speed, temperature_unit, pressure_unit, wind_speed_unit,
        forecast_daily, forecast_hourly, forecast_twice_daily):
        """Initialize the Demo weather."""
        self._attr_name: str = f'Demo Weather {name}'
        self._attr_unique_id: str = f'demo-weather-{name.lower()}'
        self._condition: str = condition
        self._native_temperature: float = temperature
        self._native_temperature_unit: str = temperature_unit
        self._humidity: float = humidity
        self._native_pressure: float = pressure
        self._native_pressure_unit: str = pressure_unit
        self._native_wind_speed: float = wind_speed
        self._native_wind_speed_unit: str = wind_speed_unit
        self._forecast_daily: Optional[List[List[Union[str, float]]]
            ] = forecast_daily
        self._forecast_hourly: Optional[List[List[Union[str, float]]]
            ] = forecast_hourly
        self._forecast_twice_daily: Optional[List[List[Union[str, float,
            bool]]]] = forecast_twice_daily
        self._attr_supported_features: WeatherEntityFeature = 0
        if self._forecast_daily:
            self._attr_supported_features |= (WeatherEntityFeature.
                FORECAST_DAILY)
        if self._forecast_hourly:
            self._attr_supported_features |= (WeatherEntityFeature.
                FORECAST_HOURLY)
        if self._forecast_twice_daily:
            self._attr_supported_features |= (WeatherEntityFeature.
                FORECAST_TWICE_DAILY)

    async def async_added_to_hass(self) ->None:
        """Set up a timer updating the forecasts."""

        async def update_forecasts(_: datetime) ->None:
            if self._forecast_daily:
                self._forecast_daily = self._forecast_daily[1:
                    ] + self._forecast_daily[:1]
            if self._forecast_hourly:
                self._forecast_hourly = self._forecast_hourly[1:
                    ] + self._forecast_hourly[:1]
            if self._forecast_twice_daily:
                self._forecast_twice_daily = self._forecast_twice_daily[1:
                    ] + self._forecast_twice_daily[:1]
            await self.async_update_listeners(None)
        self.async_on_remove(async_track_time_interval(self.hass,
            update_forecasts, WEATHER_UPDATE_INTERVAL))

    @property
    def native_temperature(self):
        """Return the temperature."""
        return self._native_temperature

    @property
    def native_temperature_unit(self):
        """Return the unit of measurement."""
        return self._native_temperature_unit

    @property
    def humidity(self):
        """Return the humidity."""
        return self._humidity

    @property
    def native_wind_speed(self):
        """Return the wind speed."""
        return self._native_wind_speed

    @property
    def native_wind_speed_unit(self):
        """Return the wind speed unit."""
        return self._native_wind_speed_unit

    @property
    def native_pressure(self):
        """Return the pressure."""
        return self._native_pressure

    @property
    def native_pressure_unit(self):
        """Return the pressure unit."""
        return self._native_pressure_unit

    @property
    def condition(self):
        """Return the weather condition."""
        return CONDITION_MAP.get(self._condition.lower(),
            ATTR_CONDITION_EXCEPTIONAL)

    async def async_forecast_daily(self) ->List[Forecast]:
        """Return the daily forecast."""
        reftime: datetime = dt_util.now().replace(hour=16, minute=0, second
            =0, microsecond=0)
        forecast_data: List[Forecast] = []
        assert self._forecast_daily is not None
        for entry in self._forecast_daily:
            condition_entry: str = entry[0]
            precipitation: float = entry[1]
            temperature: float = entry[2]
            templow: float = entry[3]
            precipitation_probability: float = entry[4]
            data_dict: Forecast = Forecast(datetime=reftime.isoformat(),
                condition=condition_entry, precipitation=precipitation,
                temperature=temperature, templow=templow,
                precipitation_probability=precipitation_probability)
            reftime += timedelta(hours=24)
            forecast_data.append(data_dict)
        return forecast_data

    async def async_forecast_hourly(self) ->List[Forecast]:
        """Return the hourly forecast."""
        reftime: datetime = dt_util.now().replace(hour=16, minute=0, second
            =0, microsecond=0)
        forecast_data: List[Forecast] = []
        assert self._forecast_hourly is not None
        for entry in self._forecast_hourly:
            condition_entry: str = entry[0]
            precipitation: float = entry[1]
            temperature: float = entry[2]
            templow: float = entry[3]
            precipitation_probability: float = entry[4]
            data_dict: Forecast = Forecast(datetime=reftime.isoformat(),
                condition=condition_entry, precipitation=precipitation,
                temperature=temperature, templow=templow,
                precipitation_probability=precipitation_probability)
            reftime += timedelta(hours=1)
            forecast_data.append(data_dict)
        return forecast_data

    async def async_forecast_twice_daily(self) ->List[Forecast]:
        """Return the twice daily forecast."""
        reftime: datetime = dt_util.now().replace(hour=11, minute=0, second
            =0, microsecond=0)
        forecast_data: List[Forecast] = []
        assert self._forecast_twice_daily is not None
        for entry in self._forecast_twice_daily:
            condition_entry: str = entry[0]
            precipitation: float = entry[1]
            temperature: float = entry[2]
            templow: float = entry[3]
            precipitation_probability: float = entry[4]
            is_daytime: bool = entry[5]
            data_dict: Forecast = Forecast(datetime=reftime.isoformat(),
                condition=condition_entry, precipitation=precipitation,
                temperature=temperature, templow=templow,
                precipitation_probability=precipitation_probability,
                is_daytime=is_daytime)
            reftime += timedelta(hours=12)
            forecast_data.append(data_dict)
        return forecast_data
