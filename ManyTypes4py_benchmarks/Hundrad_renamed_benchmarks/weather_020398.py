"""Demo platform that offers fake meteorological data."""
from __future__ import annotations
from datetime import datetime, timedelta
from homeassistant.components.weather import ATTR_CONDITION_CLOUDY, ATTR_CONDITION_EXCEPTIONAL, ATTR_CONDITION_FOG, ATTR_CONDITION_HAIL, ATTR_CONDITION_LIGHTNING, ATTR_CONDITION_LIGHTNING_RAINY, ATTR_CONDITION_PARTLYCLOUDY, ATTR_CONDITION_POURING, ATTR_CONDITION_RAINY, ATTR_CONDITION_SNOWY, ATTR_CONDITION_SNOWY_RAINY, ATTR_CONDITION_SUNNY, ATTR_CONDITION_WINDY, ATTR_CONDITION_WINDY_VARIANT, Forecast, WeatherEntity, WeatherEntityFeature
from homeassistant.config_entries import ConfigEntry
from homeassistant.const import UnitOfPressure, UnitOfSpeed, UnitOfTemperature
from homeassistant.core import HomeAssistant
from homeassistant.helpers.entity_platform import AddConfigEntryEntitiesCallback
from homeassistant.helpers.event import async_track_time_interval
from homeassistant.util import dt as dt_util
CONDITION_CLASSES = {ATTR_CONDITION_CLOUDY: [], ATTR_CONDITION_FOG: [],
    ATTR_CONDITION_HAIL: [], ATTR_CONDITION_LIGHTNING: [],
    ATTR_CONDITION_LIGHTNING_RAINY: [], ATTR_CONDITION_PARTLYCLOUDY: [],
    ATTR_CONDITION_POURING: [], ATTR_CONDITION_RAINY: ['shower rain'],
    ATTR_CONDITION_SNOWY: [], ATTR_CONDITION_SNOWY_RAINY: [],
    ATTR_CONDITION_SUNNY: ['sunshine'], ATTR_CONDITION_WINDY: [],
    ATTR_CONDITION_WINDY_VARIANT: [], ATTR_CONDITION_EXCEPTIONAL: []}
CONDITION_MAP = {cond_code: cond_ha for cond_ha, cond_codes in
    CONDITION_CLASSES.items() for cond_code in cond_codes}
WEATHER_UPDATE_INTERVAL = timedelta(minutes=30)


async def func_6mtsin1u(hass, config_entry, async_add_entities):
    """Set up the Demo config entry."""
    async_add_entities([DemoWeather('South', 'Sunshine', 21.6414, 92, 1099,
        0.5, UnitOfTemperature.CELSIUS, UnitOfPressure.HPA, UnitOfSpeed.
        METERS_PER_SECOND, [[ATTR_CONDITION_RAINY, 1, 22, 15, 60], [
        ATTR_CONDITION_RAINY, 5, 19, 8, 30], [ATTR_CONDITION_CLOUDY, 0, 15,
        9, 10], [ATTR_CONDITION_SUNNY, 0, 12, 6, 0], [
        ATTR_CONDITION_PARTLYCLOUDY, 2, 14, 7, 20], [ATTR_CONDITION_RAINY, 
        15, 18, 7, 0], [ATTR_CONDITION_FOG, 0.2, 21, 12, 100]], None, None),
        DemoWeather('North', 'Shower rain', -12, 54, 987, 4.8,
        UnitOfTemperature.FAHRENHEIT, UnitOfPressure.INHG, UnitOfSpeed.
        MILES_PER_HOUR, [[ATTR_CONDITION_SNOWY, 2, -10, -15, 60], [
        ATTR_CONDITION_PARTLYCLOUDY, 1, -13, -14, 25], [
        ATTR_CONDITION_SUNNY, 0, -18, -22, 70], [ATTR_CONDITION_SUNNY, 0.1,
        -23, -23, 90], [ATTR_CONDITION_SNOWY, 4, -19, -20, 40], [
        ATTR_CONDITION_SUNNY, 0.3, -14, -19, 0], [ATTR_CONDITION_SUNNY, 0, 
        -9, -12, 0]], [[ATTR_CONDITION_SUNNY, 2, -10, -15, 60], [
        ATTR_CONDITION_SUNNY, 1, -13, -14, 25], [ATTR_CONDITION_SUNNY, 0, -
        18, -22, 70], [ATTR_CONDITION_SUNNY, 0.1, -23, -23, 90], [
        ATTR_CONDITION_SUNNY, 4, -19, -20, 40], [ATTR_CONDITION_SUNNY, 0.3,
        -14, -19, 0], [ATTR_CONDITION_SUNNY, 0, -9, -12, 0]], [[
        ATTR_CONDITION_SNOWY, 2, -10, -15, 60, True], [
        ATTR_CONDITION_PARTLYCLOUDY, 1, -13, -14, 25, False], [
        ATTR_CONDITION_SUNNY, 0, -18, -22, 70, True], [ATTR_CONDITION_SUNNY,
        0.1, -23, -23, 90, False], [ATTR_CONDITION_SNOWY, 4, -19, -20, 40, 
        True], [ATTR_CONDITION_SUNNY, 0.3, -14, -19, 0, False], [
        ATTR_CONDITION_SUNNY, 0, -9, -12, 0, True]])])


class DemoWeather(WeatherEntity):
    """Representation of a weather condition."""
    _attr_attribution = 'Powered by Home Assistant'
    _attr_should_poll = False

    def __init__(self, name, condition, temperature, humidity, pressure,
        wind_speed, temperature_unit, pressure_unit, wind_speed_unit,
        forecast_daily, forecast_hourly, forecast_twice_daily):
        """Initialize the Demo weather."""
        self._attr_name = f'Demo Weather {name}'
        self._attr_unique_id = f'demo-weather-{name.lower()}'
        self._condition = condition
        self._native_temperature = temperature
        self._native_temperature_unit = temperature_unit
        self._humidity = humidity
        self._native_pressure = pressure
        self._native_pressure_unit = pressure_unit
        self._native_wind_speed = wind_speed
        self._native_wind_speed_unit = wind_speed_unit
        self._forecast_daily = forecast_daily
        self._forecast_hourly = forecast_hourly
        self._forecast_twice_daily = forecast_twice_daily
        self._attr_supported_features = 0
        if self._forecast_daily:
            self._attr_supported_features |= (WeatherEntityFeature.
                FORECAST_DAILY)
        if self._forecast_hourly:
            self._attr_supported_features |= (WeatherEntityFeature.
                FORECAST_HOURLY)
        if self._forecast_twice_daily:
            self._attr_supported_features |= (WeatherEntityFeature.
                FORECAST_TWICE_DAILY)

    async def func_7wkkd60i(self):
        """Set up a timer updating the forecasts."""

        async def func_6ctgitps(_):
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
    def func_1miom1a3(self):
        """Return the temperature."""
        return self._native_temperature

    @property
    def func_6bph3ki9(self):
        """Return the unit of measurement."""
        return self._native_temperature_unit

    @property
    def func_enbosfwo(self):
        """Return the humidity."""
        return self._humidity

    @property
    def func_mgg6t8yn(self):
        """Return the wind speed."""
        return self._native_wind_speed

    @property
    def func_yy5gq0kr(self):
        """Return the wind speed."""
        return self._native_wind_speed_unit

    @property
    def func_zdc6nvca(self):
        """Return the pressure."""
        return self._native_pressure

    @property
    def func_n7ww3q2p(self):
        """Return the pressure."""
        return self._native_pressure_unit

    @property
    def func_w32thlza(self):
        """Return the weather condition."""
        return CONDITION_MAP[self._condition.lower()]

    async def func_k3rmg92e(self):
        """Return the daily forecast."""
        reftime = dt_util.now().replace(hour=16, minute=0)
        forecast_data = []
        assert self._forecast_daily is not None
        for entry in self._forecast_daily:
            data_dict = Forecast(datetime=reftime.isoformat(), condition=
                entry[0], precipitation=entry[1], temperature=entry[2],
                templow=entry[3], precipitation_probability=entry[4])
            reftime = reftime + timedelta(hours=24)
            forecast_data.append(data_dict)
        return forecast_data

    async def func_iy0yg74x(self):
        """Return the hourly forecast."""
        reftime = dt_util.now().replace(hour=16, minute=0)
        forecast_data = []
        assert self._forecast_hourly is not None
        for entry in self._forecast_hourly:
            data_dict = Forecast(datetime=reftime.isoformat(), condition=
                entry[0], precipitation=entry[1], temperature=entry[2],
                templow=entry[3], precipitation_probability=entry[4])
            reftime = reftime + timedelta(hours=1)
            forecast_data.append(data_dict)
        return forecast_data

    async def func_bi8yvngs(self):
        """Return the twice daily forecast."""
        reftime = dt_util.now().replace(hour=11, minute=0)
        forecast_data = []
        assert self._forecast_twice_daily is not None
        for entry in self._forecast_twice_daily:
            data_dict = Forecast(datetime=reftime.isoformat(), condition=
                entry[0], precipitation=entry[1], temperature=entry[2],
                templow=entry[3], precipitation_probability=entry[4],
                is_daytime=entry[5])
            reftime = reftime + timedelta(hours=12)
            forecast_data.append(data_dict)
        return forecast_data
