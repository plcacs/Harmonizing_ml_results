"""Weather component that handles meteorological data for your location."""
from __future__ import annotations
from datetime import datetime
from typing import Any, Dict, List, Optional, Union
from pytomorrowio.const import DAILY, FORECASTS, HOURLY, NOWCAST, WeatherCode
from homeassistant.components.weather import (
    ATTR_FORECAST_CONDITION,
    ATTR_FORECAST_HUMIDITY,
    ATTR_FORECAST_NATIVE_DEW_POINT,
    ATTR_FORECAST_NATIVE_PRECIPITATION,
    ATTR_FORECAST_NATIVE_TEMP,
    ATTR_FORECAST_NATIVE_TEMP_LOW,
    ATTR_FORECAST_NATIVE_WIND_SPEED,
    ATTR_FORECAST_PRECIPITATION_PROBABILITY,
    ATTR_FORECAST_TIME,
    ATTR_FORECAST_WIND_BEARING,
    DOMAIN as WEATHER_DOMAIN,
    Forecast,
    SingleCoordinatorWeatherEntity,
    WeatherEntityFeature,
)
from homeassistant.config_entries import ConfigEntry
from homeassistant.const import (
    CONF_API_KEY,
    UnitOfLength,
    UnitOfPrecipitationDepth,
    UnitOfPressure,
    UnitOfSpeed,
    UnitOfTemperature,
)
from homeassistant.core import HomeAssistant, callback
from homeassistant.helpers import entity_registry as er
from homeassistant.helpers.entity_platform import AddConfigEntryEntitiesCallback
from homeassistant.helpers.sun import is_up
from homeassistant.util import dt as dt_util
from .const import (
    CLEAR_CONDITIONS,
    CONDITIONS,
    CONF_TIMESTEP,
    DEFAULT_FORECAST_TYPE,
    DOMAIN,
    MAX_FORECASTS,
    TMRW_ATTR_CONDITION,
    TMRW_ATTR_DEW_POINT,
    TMRW_ATTR_HUMIDITY,
    TMRW_ATTR_OZONE,
    TMRW_ATTR_PRECIPITATION,
    TMRW_ATTR_PRECIPITATION_PROBABILITY,
    TMRW_ATTR_PRESSURE,
    TMRW_ATTR_TEMPERATURE,
    TMRW_ATTR_TEMPERATURE_HIGH,
    TMRW_ATTR_TEMPERATURE_LOW,
    TMRW_ATTR_TIMESTAMP,
    TMRW_ATTR_VISIBILITY,
    TMRW_ATTR_WIND_DIRECTION,
    TMRW_ATTR_WIND_SPEED,
)
from .coordinator import TomorrowioDataUpdateCoordinator
from .entity import TomorrowioEntity


async def async_setup_entry(
    hass: HomeAssistant,
    config_entry: ConfigEntry,
    async_add_entities: AddConfigEntryEntitiesCallback,
) -> None:
    """Set up a config entry."""
    coordinator: TomorrowioDataUpdateCoordinator = hass.data[DOMAIN][config_entry.data[CONF_API_KEY]]
    entity_registry = er.async_get(hass)
    entities: List[TomorrowioWeatherEntity] = [TomorrowioWeatherEntity(config_entry, coordinator, 4, DAILY)]
    for forecast_type in (HOURLY, NOWCAST):
        unique_id = _calculate_unique_id(config_entry.unique_id, forecast_type)
        if not entity_registry.async_get_entity_id(WEATHER_DOMAIN, DOMAIN, unique_id):
            continue
        entities.append(TomorrowioWeatherEntity(config_entry, coordinator, 4, forecast_type))
    async_add_entities(entities)


def _calculate_unique_id(config_entry_unique_id: str, forecast_type: str) -> str:
    """Calculate unique ID."""
    return f"{config_entry_unique_id}_{forecast_type}"


class TomorrowioWeatherEntity(TomorrowioEntity, SingleCoordinatorWeatherEntity):
    """Entity that talks to Tomorrow.io v4 API to retrieve weather data."""

    _attr_native_precipitation_unit: UnitOfPrecipitationDepth = UnitOfPrecipitationDepth.MILLIMETERS
    _attr_native_pressure_unit: UnitOfPressure = UnitOfPressure.HPA
    _attr_native_temperature_unit: UnitOfTemperature = UnitOfTemperature.CELSIUS
    _attr_native_visibility_unit: UnitOfLength = UnitOfLength.KILOMETERS
    _attr_native_wind_speed_unit: UnitOfSpeed = UnitOfSpeed.METERS_PER_SECOND
    _attr_supported_features: WeatherEntityFeature = (
        WeatherEntityFeature.FORECAST_DAILY | WeatherEntityFeature.FORECAST_HOURLY
    )

    def __init__(
        self,
        config_entry: ConfigEntry,
        coordinator: TomorrowioDataUpdateCoordinator,
        api_version: int,
        forecast_type: str,
    ) -> None:
        """Initialize Tomorrow.io Weather Entity."""
        super().__init__(config_entry, coordinator, api_version)
        self.forecast_type: str = forecast_type
        self._attr_entity_registry_enabled_default: bool = forecast_type == DEFAULT_FORECAST_TYPE
        self._attr_name: str = forecast_type.title()
        self._attr_unique_id: str = _calculate_unique_id(config_entry.unique_id, forecast_type)

    def _forecast_dict(
        self,
        forecast_dt: datetime,
        use_datetime: bool,
        condition: Optional[int],
        precipitation: Optional[float],
        precipitation_probability: Optional[float],
        temp: Optional[float],
        temp_low: Optional[float],
        humidity: Optional[float],
        dew_point: Optional[float],
        wind_direction: Optional[float],
        wind_speed: Optional[float],
    ) -> Dict[str, Any]:
        """Return formatted Forecast dict from Tomorrow.io forecast data."""
        if use_datetime:
            translated_condition: Optional[str] = self._translate_condition(
                condition, is_up(self.hass, forecast_dt)
            )
        else:
            translated_condition = self._translate_condition(condition, True)
        return {
            ATTR_FORECAST_TIME: forecast_dt.isoformat(),
            ATTR_FORECAST_CONDITION: translated_condition,
            ATTR_FORECAST_NATIVE_PRECIPITATION: precipitation,
            ATTR_FORECAST_PRECIPITATION_PROBABILITY: precipitation_probability,
            ATTR_FORECAST_NATIVE_TEMP: temp,
            ATTR_FORECAST_NATIVE_TEMP_LOW: temp_low,
            ATTR_FORECAST_HUMIDITY: humidity,
            ATTR_FORECAST_NATIVE_DEW_POINT: dew_point,
            ATTR_FORECAST_WIND_BEARING: wind_direction,
            ATTR_FORECAST_NATIVE_WIND_SPEED: wind_speed,
        }

    @staticmethod
    def _translate_condition(condition: Optional[int], sun_is_up: bool = True) -> Optional[str]:
        """Translate Tomorrow.io condition into an HA condition."""
        if condition is None:
            return None
        condition_enum = WeatherCode(condition)
        if condition_enum in (WeatherCode.CLEAR, WeatherCode.MOSTLY_CLEAR):
            if sun_is_up:
                return CLEAR_CONDITIONS["day"]
            return CLEAR_CONDITIONS["night"]
        return CONDITIONS.get(condition_enum)

    @property
    def native_temperature(self) -> Optional[float]:
        """Return the platform temperature."""
        return self._get_current_property(TMRW_ATTR_TEMPERATURE)

    @property
    def native_pressure(self) -> Optional[float]:
        """Return the raw pressure."""
        return self._get_current_property(TMRW_ATTR_PRESSURE)

    @property
    def humidity(self) -> Optional[float]:
        """Return the humidity."""
        return self._get_current_property(TMRW_ATTR_HUMIDITY)

    @property
    def native_wind_speed(self) -> Optional[float]:
        """Return the raw wind speed."""
        return self._get_current_property(TMRW_ATTR_WIND_SPEED)

    @property
    def wind_bearing(self) -> Optional[float]:
        """Return the wind bearing."""
        return self._get_current_property(TMRW_ATTR_WIND_DIRECTION)

    @property
    def ozone(self) -> Optional[float]:
        """Return the O3 (ozone) level."""
        return self._get_current_property(TMRW_ATTR_OZONE)

    @property
    def condition(self) -> Optional[str]:
        """Return the condition."""
        return self._translate_condition(
            self._get_current_property(TMRW_ATTR_CONDITION), is_up(self.hass)
        )

    @property
    def native_visibility(self) -> Optional[float]:
        """Return the raw visibility."""
        return self._get_current_property(TMRW_ATTR_VISIBILITY)

    def _forecast(self, forecast_type: str) -> Optional[List[Forecast]]:
        """Return the forecast."""
        raw_forecasts: Optional[List[Dict[str, Any]]] = self.coordinator.data.get(
            self._config_entry.entry_id, {}
        ).get(FORECASTS, {}).get(forecast_type)
        if not raw_forecasts:
            return None
        forecasts: List[Forecast] = []
        max_forecasts: int = MAX_FORECASTS[forecast_type]
        forecast_count: int = 0
        today: datetime.date = dt_util.as_local(dt_util.utcnow()).date()
        for forecast in raw_forecasts:
            forecast_dt_str: str = forecast.get(TMRW_ATTR_TIMESTAMP)
            forecast_dt: Optional[datetime] = dt_util.parse_datetime(forecast_dt_str)
            if forecast_dt is None or dt_util.as_local(forecast_dt).date() < today:
                continue
            values: Dict[str, Any] = forecast.get("values", {})
            use_datetime: bool = True
            condition: Optional[int] = values.get(TMRW_ATTR_CONDITION)
            precipitation: Optional[float] = values.get(TMRW_ATTR_PRECIPITATION)
            precipitation_probability: Optional[float] = values.get(TMRW_ATTR_PRECIPITATION_PROBABILITY)
            try:
                precipitation_probability = round(precipitation_probability)
            except (TypeError, ValueError):
                precipitation_probability = None
            temp: Optional[float] = values.get(TMRW_ATTR_TEMPERATURE_HIGH)
            temp_low: Optional[float] = None
            dew_point: Optional[float] = values.get(TMRW_ATTR_DEW_POINT)
            humidity: Optional[float] = values.get(TMRW_ATTR_HUMIDITY)
            wind_direction: Optional[float] = values.get(TMRW_ATTR_WIND_DIRECTION)
            wind_speed: Optional[float] = values.get(TMRW_ATTR_WIND_SPEED)
            if forecast_type == DAILY:
                use_datetime = False
                temp_low = values.get(TMRW_ATTR_TEMPERATURE_LOW)
                if precipitation is not None:
                    precipitation = precipitation * 24
            elif forecast_type == NOWCAST:
                if precipitation is not None:
                    timestep: int = self._config_entry.options.get(CONF_TIMESTEP, 60)
                    precipitation = precipitation / 60 * timestep
            forecast_entry: Dict[str, Any] = self._forecast_dict(
                forecast_dt,
                use_datetime,
                condition,
                precipitation,
                precipitation_probability,
                temp,
                temp_low,
                humidity,
                dew_point,
                wind_direction,
                wind_speed,
            )
            forecasts.append(forecast_entry)
            forecast_count += 1
            if forecast_count == max_forecasts:
                break
        return forecasts

    @callback
    def _async_forecast_daily(self) -> Optional[List[Forecast]]:
        """Return the daily forecast in native units."""
        return self._forecast(DAILY)

    @callback
    def _async_forecast_hourly(self) -> Optional[List[Forecast]]:
        """Return the hourly forecast in native units."""
        return self._forecast(HOURLY)
