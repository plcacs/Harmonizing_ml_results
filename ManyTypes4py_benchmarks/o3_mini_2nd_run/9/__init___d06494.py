"""Weather component that handles meteorological data for your location."""
from __future__ import annotations
import abc
from collections.abc import Callable, Iterable
from contextlib import suppress
from datetime import timedelta
from functools import partial
import logging
from typing import Any, Final, Generic, Literal, Optional, Set, TypeVar, cast, final, Dict, List
from homeassistant.config_entries import ConfigEntry
from homeassistant.const import PRECISION_HALVES, PRECISION_TENTHS, PRECISION_WHOLE, UnitOfPressure, UnitOfSpeed, UnitOfTemperature
from homeassistant.core import CALLBACK_TYPE, HomeAssistant, ServiceCall, ServiceResponse, SupportsResponse, callback
from homeassistant.exceptions import HomeAssistantError
from homeassistant.helpers import config_validation as cv
from homeassistant.helpers.entity import ABCCachedProperties, Entity, EntityDescription
from homeassistant.helpers.entity_component import EntityComponent
from homeassistant.helpers.typing import ConfigType
from homeassistant.helpers.update_coordinator import CoordinatorEntity, DataUpdateCoordinator, TimestampDataUpdateCoordinator
from homeassistant.util.dt import utcnow
from homeassistant.util.json import JsonValueType
from homeassistant.util.unit_system import US_CUSTOMARY_SYSTEM
from propcache.api import cached_property
import voluptuous as vol
from .const import (
    ATTR_WEATHER_APPARENT_TEMPERATURE,
    ATTR_WEATHER_CLOUD_COVERAGE,
    ATTR_WEATHER_DEW_POINT,
    ATTR_WEATHER_HUMIDITY,
    ATTR_WEATHER_OZONE,
    ATTR_WEATHER_PRECIPITATION_UNIT,
    ATTR_WEATHER_PRESSURE,
    ATTR_WEATHER_PRESSURE_UNIT,
    ATTR_WEATHER_TEMPERATURE,
    ATTR_WEATHER_TEMPERATURE_UNIT,
    ATTR_WEATHER_UV_INDEX,
    ATTR_WEATHER_VISIBILITY,
    ATTR_WEATHER_VISIBILITY_UNIT,
    ATTR_WEATHER_WIND_BEARING,
    ATTR_WEATHER_WIND_GUST_SPEED,
    ATTR_WEATHER_WIND_SPEED,
    ATTR_WEATHER_WIND_SPEED_UNIT,
    DATA_COMPONENT,
    DOMAIN,
    INTENT_GET_WEATHER,
    UNIT_CONVERSIONS,
    VALID_UNITS,
    WeatherEntityFeature,
)
from .websocket_api import async_setup as async_setup_ws_api

_LOGGER = logging.getLogger(__name__)
ENTITY_ID_FORMAT: Final = DOMAIN + '.{}'
PLATFORM_SCHEMA = cv.PLATFORM_SCHEMA
PLATFORM_SCHEMA_BASE = cv.PLATFORM_SCHEMA_BASE
SCAN_INTERVAL: Final = timedelta(seconds=30)
ATTR_CONDITION_CLASS: Final[str] = 'condition_class'
ATTR_CONDITION_CLEAR_NIGHT: Final[str] = 'clear-night'
ATTR_CONDITION_CLOUDY: Final[str] = 'cloudy'
ATTR_CONDITION_EXCEPTIONAL: Final[str] = 'exceptional'
ATTR_CONDITION_FOG: Final[str] = 'fog'
ATTR_CONDITION_HAIL: Final[str] = 'hail'
ATTR_CONDITION_LIGHTNING: Final[str] = 'lightning'
ATTR_CONDITION_LIGHTNING_RAINY: Final[str] = 'lightning-rainy'
ATTR_CONDITION_PARTLYCLOUDY: Final[str] = 'partlycloudy'
ATTR_CONDITION_POURING: Final[str] = 'pouring'
ATTR_CONDITION_RAINY: Final[str] = 'rainy'
ATTR_CONDITION_SNOWY: Final[str] = 'snowy'
ATTR_CONDITION_SNOWY_RAINY: Final[str] = 'snowy-rainy'
ATTR_CONDITION_SUNNY: Final[str] = 'sunny'
ATTR_CONDITION_WINDY: Final[str] = 'windy'
ATTR_CONDITION_WINDY_VARIANT: Final[str] = 'windy-variant'
ATTR_FORECAST_IS_DAYTIME: Final[str] = 'is_daytime'
ATTR_FORECAST_CONDITION: Final[str] = 'condition'
ATTR_FORECAST_HUMIDITY: Final[str] = 'humidity'
ATTR_FORECAST_NATIVE_PRECIPITATION: Final[str] = 'native_precipitation'
ATTR_FORECAST_PRECIPITATION: Final[str] = 'precipitation'
ATTR_FORECAST_PRECIPITATION_PROBABILITY: Final[str] = 'precipitation_probability'
ATTR_FORECAST_NATIVE_PRESSURE: Final[str] = 'native_pressure'
ATTR_FORECAST_PRESSURE: Final[str] = 'pressure'
ATTR_FORECAST_NATIVE_APPARENT_TEMP: Final[str] = 'native_apparent_temperature'
ATTR_FORECAST_APPARENT_TEMP: Final[str] = 'apparent_temperature'
ATTR_FORECAST_NATIVE_TEMP: Final[str] = 'native_temperature'
ATTR_FORECAST_TEMP: Final[str] = 'temperature'
ATTR_FORECAST_NATIVE_TEMP_LOW: Final[str] = 'native_templow'
ATTR_FORECAST_TEMP_LOW: Final[str] = 'templow'
ATTR_FORECAST_TIME: Final[str] = 'datetime'
ATTR_FORECAST_WIND_BEARING: Final[str] = 'wind_bearing'
ATTR_FORECAST_NATIVE_WIND_GUST_SPEED: Final[str] = 'native_wind_gust_speed'
ATTR_FORECAST_WIND_GUST_SPEED: Final[str] = 'wind_gust_speed'
ATTR_FORECAST_NATIVE_WIND_SPEED: Final[str] = 'native_wind_speed'
ATTR_FORECAST_WIND_SPEED: Final[str] = 'wind_speed'
ATTR_FORECAST_NATIVE_DEW_POINT: Final[str] = 'native_dew_point'
ATTR_FORECAST_DEW_POINT: Final[str] = 'dew_point'
ATTR_FORECAST_CLOUD_COVERAGE: Final[str] = 'cloud_coverage'
ATTR_FORECAST_UV_INDEX: Final[str] = 'uv_index'
ROUNDING_PRECISION: Final[int] = 2
SERVICE_GET_FORECASTS: Final[str] = 'get_forecasts'

_ObservationUpdateCoordinatorT = TypeVar(
    "_ObservationUpdateCoordinatorT", bound=DataUpdateCoordinator[Any], default=DataUpdateCoordinator[dict[str, Any]]
)
_DailyForecastUpdateCoordinatorT = TypeVar(
    "_DailyForecastUpdateCoordinatorT", bound=TimestampDataUpdateCoordinator[Any], default=TimestampDataUpdateCoordinator[None]
)
_HourlyForecastUpdateCoordinatorT = TypeVar(
    "_HourlyForecastUpdateCoordinatorT", bound=TimestampDataUpdateCoordinator[Any], default=_DailyForecastUpdateCoordinatorT
)
_TwiceDailyForecastUpdateCoordinatorT = TypeVar(
    "_TwiceDailyForecastUpdateCoordinatorT", bound=TimestampDataUpdateCoordinator[Any], default=_DailyForecastUpdateCoordinatorT
)

def round_temperature(temperature: Optional[float], precision: str) -> Optional[float]:
    """Convert temperature into preferred precision for display."""
    if temperature is None:
        return None
    if precision == PRECISION_HALVES:
        temperature = round(temperature * 2) / 2.0
    elif precision == PRECISION_TENTHS:
        temperature = round(temperature, 1)
    else:
        temperature = round(temperature)
    return temperature

class Forecast(TypedDict, total=False):
    """Typed weather forecast dict.

    All attributes are in native units and old attributes kept
    for backwards compatibility.
    """
    # Keys are dynamically added based on forecast conversion

async def async_setup(hass: HomeAssistant, config: ConfigType) -> bool:
    """Set up the weather component."""
    component: EntityComponent[WeatherEntity] = hass.data[DATA_COMPONENT] = EntityComponent[WeatherEntity](
        _LOGGER, DOMAIN, hass, SCAN_INTERVAL
    )
    component.async_register_entity_service(
        SERVICE_GET_FORECASTS,
        {vol.Required("type"): vol.In(("daily", "hourly", "twice_daily"))},
        async_get_forecasts_service,
        required_features=[
            WeatherEntityFeature.FORECAST_DAILY,
            WeatherEntityFeature.FORECAST_HOURLY,
            WeatherEntityFeature.FORECAST_TWICE_DAILY,
        ],
        supports_response=SupportsResponse.ONLY,
    )
    async_setup_ws_api(hass)
    await component.async_setup(config)
    return True

async def async_setup_entry(hass: HomeAssistant, entry: ConfigEntry) -> Any:
    """Set up a config entry."""
    return await hass.data[DATA_COMPONENT].async_setup_entry(entry)

async def async_unload_entry(hass: HomeAssistant, entry: ConfigEntry) -> Any:
    """Unload a config entry."""
    return await hass.data[DATA_COMPONENT].async_unload_entry(entry)

class WeatherEntityDescription(EntityDescription, frozen_or_thawed=True):
    """A class that describes weather entities."""
    # No additional attributes required

class PostInitMeta(ABCCachedProperties):
    """Meta class which calls __post_init__ after __new__ and __init__."""

    def __call__(cls, *args: Any, **kwargs: Any) -> Any:
        """Create an instance."""
        instance = super().__call__(*args, **kwargs)
        instance.__post_init__(*args, **kwargs)
        return instance

class PostInit(metaclass=PostInitMeta):
    """Class which calls __post_init__ after __new__ and __init__."""

    @abc.abstractmethod
    def __post_init__(self, *args: Any, **kwargs: Any) -> None:
        """Finish initializing."""
        
CACHED_PROPERTIES_WITH_ATTR_: Final[Set[str]] = {
    "native_apparent_temperature",
    "native_temperature",
    "native_temperature_unit",
    "native_dew_point",
    "native_pressure",
    "native_pressure_unit",
    "humidity",
    "native_wind_gust_speed",
    "native_wind_speed",
    "native_wind_speed_unit",
    "wind_bearing",
    "ozone",
    "cloud_coverage",
    "uv_index",
    "native_visibility",
    "native_visibility_unit",
    "native_precipitation_unit",
    "condition",
}

class WeatherEntity(Entity, PostInit, cached_properties=CACHED_PROPERTIES_WITH_ATTR_):
    """ABC for weather data."""
    _attr_condition: Optional[str] = None
    _attr_humidity: Optional[float] = None
    _attr_ozone: Optional[float] = None
    _attr_cloud_coverage: Optional[float] = None
    _attr_uv_index: Optional[float] = None
    _attr_state: Any = None
    _attr_wind_bearing: Optional[float] = None
    _attr_native_pressure: Optional[float] = None
    _attr_native_pressure_unit: Optional[str] = None
    _attr_native_apparent_temperature: Optional[float] = None
    _attr_native_temperature: Optional[float] = None
    _attr_native_temperature_unit: Optional[str] = None
    _attr_native_visibility: Optional[float] = None
    _attr_native_visibility_unit: Optional[str] = None
    _attr_native_precipitation_unit: Optional[str] = None
    _attr_native_wind_gust_speed: Optional[float] = None
    _attr_native_wind_speed: Optional[float] = None
    _attr_native_wind_speed_unit: Optional[str] = None
    _attr_native_dew_point: Optional[float] = None
    _weather_option_temperature_unit: Optional[str] = None
    _weather_option_pressure_unit: Optional[str] = None
    _weather_option_visibility_unit: Optional[str] = None
    _weather_option_precipitation_unit: Optional[str] = None
    _weather_option_wind_speed_unit: Optional[str] = None

    def __post_init__(self, *args: Any, **kwargs: Any) -> None:
        """Finish initializing."""
        self._forecast_listeners: Dict[str, List[Callable[[Optional[List[Forecast]]], None]]] = {
            "daily": [],
            "hourly": [],
            "twice_daily": [],
        }

    async def async_internal_added_to_hass(self) -> None:
        """Call when the weather entity is added to hass."""
        await super().async_internal_added_to_hass()  # type: ignore
        if not self.registry_entry:
            return
        self.async_registry_entry_updated()

    @cached_property
    def native_apparent_temperature(self) -> Optional[float]:
        """Return the apparent temperature in native units."""
        return self._attr_native_apparent_temperature

    @cached_property
    def native_temperature(self) -> Optional[float]:
        """Return the temperature in native units."""
        return self._attr_native_temperature

    @cached_property
    def native_temperature_unit(self) -> Optional[str]:
        """Return the native unit of measurement for temperature."""
        return self._attr_native_temperature_unit

    @cached_property
    def native_dew_point(self) -> Optional[float]:
        """Return the dew point temperature in native units."""
        return self._attr_native_dew_point

    @final
    @property
    def _default_temperature_unit(self) -> str:
        """Return the default unit of measurement for temperature.

        Should not be set by integrations.
        """
        return self.hass.config.units.temperature_unit  # type: ignore

    @final
    @property
    def _temperature_unit(self) -> str:
        """Return the converted unit of measurement for temperature.

        Should not be set by integrations.
        """
        if (weather_option_temperature_unit := self._weather_option_temperature_unit) is not None:
            return weather_option_temperature_unit
        return self._default_temperature_unit

    @cached_property
    def native_pressure(self) -> Optional[float]:
        """Return the pressure in native units."""
        return self._attr_native_pressure

    @cached_property
    def native_pressure_unit(self) -> Optional[str]:
        """Return the native unit of measurement for pressure."""
        return self._attr_native_pressure_unit

    @final
    @property
    def _default_pressure_unit(self) -> str:
        """Return the default unit of measurement for pressure.

        Should not be set by integrations.
        """
        if self.hass.config.units is US_CUSTOMARY_SYSTEM:  # type: ignore
            return UnitOfPressure.INHG
        return UnitOfPressure.HPA

    @final
    @property
    def _pressure_unit(self) -> str:
        """Return the converted unit of measurement for pressure.

        Should not be set by integrations.
        """
        if (weather_option_pressure_unit := self._weather_option_pressure_unit) is not None:
            return weather_option_pressure_unit
        return self._default_pressure_unit

    @cached_property
    def humidity(self) -> Optional[float]:
        """Return the humidity in native units."""
        return self._attr_humidity

    @cached_property
    def native_wind_gust_speed(self) -> Optional[float]:
        """Return the wind gust speed in native units."""
        return self._attr_native_wind_gust_speed

    @cached_property
    def native_wind_speed(self) -> Optional[float]:
        """Return the wind speed in native units."""
        return self._attr_native_wind_speed

    @cached_property
    def native_wind_speed_unit(self) -> Optional[str]:
        """Return the native unit of measurement for wind speed."""
        return self._attr_native_wind_speed_unit

    @final
    @property
    def _default_wind_speed_unit(self) -> str:
        """Return the default unit of measurement for wind speed.

        Should not be set by integrations.
        """
        if self.hass.config.units is US_CUSTOMARY_SYSTEM:  # type: ignore
            return UnitOfSpeed.MILES_PER_HOUR
        return UnitOfSpeed.KILOMETERS_PER_HOUR

    @final
    @property
    def _wind_speed_unit(self) -> str:
        """Return the converted unit of measurement for wind speed.

        Should not be set by integrations.
        """
        if (weather_option_wind_speed_unit := self._weather_option_wind_speed_unit) is not None:
            return weather_option_wind_speed_unit
        return self._default_wind_speed_unit

    @cached_property
    def wind_bearing(self) -> Optional[float]:
        """Return the wind bearing."""
        return self._attr_wind_bearing

    @cached_property
    def ozone(self) -> Optional[float]:
        """Return the ozone level."""
        return self._attr_ozone

    @cached_property
    def cloud_coverage(self) -> Optional[float]:
        """Return the Cloud coverage in %."""
        return self._attr_cloud_coverage

    @cached_property
    def uv_index(self) -> Optional[float]:
        """Return the UV index."""
        return self._attr_uv_index

    @cached_property
    def native_visibility(self) -> Optional[float]:
        """Return the visibility in native units."""
        return self._attr_native_visibility

    @cached_property
    def native_visibility_unit(self) -> Optional[str]:
        """Return the native unit of measurement for visibility."""
        return self._attr_native_visibility_unit

    @final
    @property
    def _default_visibility_unit(self) -> str:
        """Return the default unit of measurement for visibility.

        Should not be set by integrations.
        """
        return self.hass.config.units.length_unit  # type: ignore

    @final
    @property
    def _visibility_unit(self) -> str:
        """Return the converted unit of measurement for visibility.

        Should not be set by integrations.
        """
        if (weather_option_visibility_unit := self._weather_option_visibility_unit) is not None:
            return weather_option_visibility_unit
        return self._default_visibility_unit

    async def async_forecast_daily(self) -> Optional[List[Forecast]]:
        """Return the daily forecast in native units."""
        raise NotImplementedError

    async def async_forecast_twice_daily(self) -> Optional[List[Forecast]]:
        """Return the daily forecast in native units."""
        raise NotImplementedError

    async def async_forecast_hourly(self) -> Optional[List[Forecast]]:
        """Return the hourly forecast in native units."""
        raise NotImplementedError

    @cached_property
    def native_precipitation_unit(self) -> Optional[str]:
        """Return the native unit of measurement for accumulated precipitation."""
        return self._attr_native_precipitation_unit

    @final
    @property
    def _default_precipitation_unit(self) -> str:
        """Return the default unit of measurement for precipitation.

        Should not be set by integrations.
        """
        return self.hass.config.units.accumulated_precipitation_unit  # type: ignore

    @final
    @property
    def _precipitation_unit(self) -> str:
        """Return the converted unit of measurement for precipitation.

        Should not be set by integrations.
        """
        if (weather_option_precipitation_unit := self._weather_option_precipitation_unit) is not None:
            return weather_option_precipitation_unit
        return self._default_precipitation_unit

    @property
    def precision(self) -> str:
        """Return the precision of the temperature value, after unit conversion."""
        if hasattr(self, "_attr_precision"):
            return self._attr_precision  # type: ignore
        return PRECISION_TENTHS if self._temperature_unit == UnitOfTemperature.CELSIUS else PRECISION_WHOLE

    @final
    @property
    def state_attributes(self) -> Dict[str, Any]:
        """Return the state attributes, converted.

        Attributes are configured from native units to user-configured units.
        """
        data: Dict[str, Any] = {}
        precision = self.precision
        if (temperature := self.native_temperature) is not None:
            from_unit = self.native_temperature_unit or self._default_temperature_unit
            to_unit = self._temperature_unit
            try:
                temperature_f = float(temperature)
                value_temp = UNIT_CONVERSIONS[ATTR_WEATHER_TEMPERATURE_UNIT](temperature_f, from_unit, to_unit)
                data[ATTR_WEATHER_TEMPERATURE] = round_temperature(value_temp, precision)
            except (TypeError, ValueError):
                data[ATTR_WEATHER_TEMPERATURE] = temperature
        if (apparent_temperature := self.native_apparent_temperature) is not None:
            from_unit = self.native_temperature_unit or self._default_temperature_unit
            to_unit = self._temperature_unit
            try:
                apparent_temperature_f = float(apparent_temperature)
                value_apparent_temp = UNIT_CONVERSIONS[ATTR_WEATHER_TEMPERATURE_UNIT](apparent_temperature_f, from_unit, to_unit)
                data[ATTR_WEATHER_APPARENT_TEMPERATURE] = round_temperature(value_apparent_temp, precision)
            except (TypeError, ValueError):
                data[ATTR_WEATHER_APPARENT_TEMPERATURE] = apparent_temperature
        if (dew_point := self.native_dew_point) is not None:
            from_unit = self.native_temperature_unit or self._default_temperature_unit
            to_unit = self._temperature_unit
            try:
                dew_point_f = float(dew_point)
                value_dew_point = UNIT_CONVERSIONS[ATTR_WEATHER_TEMPERATURE_UNIT](dew_point_f, from_unit, to_unit)
                data[ATTR_WEATHER_DEW_POINT] = round_temperature(value_dew_point, precision)
            except (TypeError, ValueError):
                data[ATTR_WEATHER_DEW_POINT] = dew_point
        data[ATTR_WEATHER_TEMPERATURE_UNIT] = self._temperature_unit
        if (humidity := self.humidity) is not None:
            data[ATTR_WEATHER_HUMIDITY] = round(humidity)
        if (ozone := self.ozone) is not None:
            data[ATTR_WEATHER_OZONE] = ozone
        if (cloud_coverage := self.cloud_coverage) is not None:
            data[ATTR_WEATHER_CLOUD_COVERAGE] = cloud_coverage
        if (uv_index := self.uv_index) is not None:
            data[ATTR_WEATHER_UV_INDEX] = uv_index
        if (pressure := self.native_pressure) is not None:
            from_unit = self.native_pressure_unit or self._default_pressure_unit
            to_unit = self._pressure_unit
            try:
                pressure_f = float(pressure)
                value_pressure = UNIT_CONVERSIONS[ATTR_WEATHER_PRESSURE_UNIT](pressure_f, from_unit, to_unit)
                data[ATTR_WEATHER_PRESSURE] = round(value_pressure, ROUNDING_PRECISION)
            except (TypeError, ValueError):
                data[ATTR_WEATHER_PRESSURE] = pressure
        data[ATTR_WEATHER_PRESSURE_UNIT] = self._pressure_unit
        if (wind_bearing := self.wind_bearing) is not None:
            data[ATTR_WEATHER_WIND_BEARING] = wind_bearing
        if (wind_gust_speed := self.native_wind_gust_speed) is not None:
            from_unit = self.native_wind_speed_unit or self._default_wind_speed_unit
            to_unit = self._wind_speed_unit
            try:
                wind_gust_speed_f = float(wind_gust_speed)
                value_wind_gust_speed = UNIT_CONVERSIONS[ATTR_WEATHER_WIND_SPEED_UNIT](wind_gust_speed_f, from_unit, to_unit)
                data[ATTR_WEATHER_WIND_GUST_SPEED] = round(value_wind_gust_speed, ROUNDING_PRECISION)
            except (TypeError, ValueError):
                data[ATTR_WEATHER_WIND_GUST_SPEED] = wind_gust_speed
        if (wind_speed := self.native_wind_speed) is not None:
            from_unit = self.native_wind_speed_unit or self._default_wind_speed_unit
            to_unit = self._wind_speed_unit
            try:
                wind_speed_f = float(wind_speed)
                value_wind_speed = UNIT_CONVERSIONS[ATTR_WEATHER_WIND_SPEED_UNIT](wind_speed_f, from_unit, to_unit)
                data[ATTR_WEATHER_WIND_SPEED] = round(value_wind_speed, ROUNDING_PRECISION)
            except (TypeError, ValueError):
                data[ATTR_WEATHER_WIND_SPEED] = wind_speed
        data[ATTR_WEATHER_WIND_SPEED_UNIT] = self._wind_speed_unit
        if (visibility := self.native_visibility) is not None:
            from_unit = self.native_visibility_unit or self._default_visibility_unit
            to_unit = self._visibility_unit
            try:
                visibility_f = float(visibility)
                value_visibility = UNIT_CONVERSIONS[ATTR_WEATHER_VISIBILITY_UNIT](visibility_f, from_unit, to_unit)
                data[ATTR_WEATHER_VISIBILITY] = round(value_visibility, ROUNDING_PRECISION)
            except (TypeError, ValueError):
                data[ATTR_WEATHER_VISIBILITY] = visibility
        data[ATTR_WEATHER_VISIBILITY_UNIT] = self._visibility_unit
        data[ATTR_WEATHER_PRECIPITATION_UNIT] = self._precipitation_unit
        return data

    @final
    def _convert_forecast(self, native_forecast_list: List[Dict[str, Any]]) -> List[Forecast]:
        """Convert a forecast in native units to the unit configured by the user."""
        converted_forecast_list: List[Forecast] = []
        precision = self.precision
        from_temp_unit = self.native_temperature_unit or self._default_temperature_unit
        to_temp_unit = self._temperature_unit
        for _forecast_entry in native_forecast_list:
            forecast_entry: Dict[str, Any] = dict(_forecast_entry)
            temperature = forecast_entry.pop(ATTR_FORECAST_NATIVE_TEMP, forecast_entry.get(ATTR_FORECAST_TEMP))
            if temperature is None:
                forecast_entry[ATTR_FORECAST_TEMP] = None
            else:
                with suppress(TypeError, ValueError):
                    temperature_f = float(temperature)
                    value_temp = UNIT_CONVERSIONS[ATTR_WEATHER_TEMPERATURE_UNIT](temperature_f, from_temp_unit, to_temp_unit)
                    forecast_entry[ATTR_FORECAST_TEMP] = round_temperature(value_temp, precision)
            if (forecast_apparent_temp := forecast_entry.pop(ATTR_FORECAST_NATIVE_APPARENT_TEMP, forecast_entry.get(ATTR_FORECAST_NATIVE_APPARENT_TEMP))) is not None:
                with suppress(TypeError, ValueError):
                    forecast_apparent_temp = float(forecast_apparent_temp)
                    value_apparent_temp = UNIT_CONVERSIONS[ATTR_WEATHER_TEMPERATURE_UNIT](forecast_apparent_temp, from_temp_unit, to_temp_unit)
                    forecast_entry[ATTR_FORECAST_APPARENT_TEMP] = round_temperature(value_apparent_temp, precision)
            if (forecast_temp_low := forecast_entry.pop(ATTR_FORECAST_NATIVE_TEMP_LOW, forecast_entry.get(ATTR_FORECAST_TEMP_LOW))) is not None:
                with suppress(TypeError, ValueError):
                    forecast_temp_low_f = float(forecast_temp_low)
                    value_temp_low = UNIT_CONVERSIONS[ATTR_WEATHER_TEMPERATURE_UNIT](forecast_temp_low_f, from_temp_unit, to_temp_unit)
                    forecast_entry[ATTR_FORECAST_TEMP_LOW] = round_temperature(value_temp_low, precision)
            if (forecast_dew_point := forecast_entry.pop(ATTR_FORECAST_NATIVE_DEW_POINT, None)) is not None:
                with suppress(TypeError, ValueError):
                    forecast_dew_point_f = float(forecast_dew_point)
                    value_dew_point = UNIT_CONVERSIONS[ATTR_WEATHER_TEMPERATURE_UNIT](forecast_dew_point_f, from_temp_unit, to_temp_unit)
                    forecast_entry[ATTR_FORECAST_DEW_POINT] = round_temperature(value_dew_point, precision)
            if (forecast_pressure := forecast_entry.pop(ATTR_FORECAST_NATIVE_PRESSURE, forecast_entry.get(ATTR_FORECAST_PRESSURE))) is not None:
                from_pressure_unit = self.native_pressure_unit or self._default_pressure_unit
                to_pressure_unit = self._pressure_unit
                with suppress(TypeError, ValueError):
                    forecast_pressure_f = float(forecast_pressure)
                    forecast_entry[ATTR_FORECAST_PRESSURE] = round(
                        UNIT_CONVERSIONS[ATTR_WEATHER_PRESSURE_UNIT](forecast_pressure_f, from_pressure_unit, to_pressure_unit),
                        ROUNDING_PRECISION,
                    )
            if (forecast_wind_gust_speed := forecast_entry.pop(ATTR_FORECAST_NATIVE_WIND_GUST_SPEED, None)) is not None:
                from_wind_speed_unit = self.native_wind_speed_unit or self._default_wind_speed_unit
                to_wind_speed_unit = self._wind_speed_unit
                with suppress(TypeError, ValueError):
                    forecast_wind_gust_speed_f = float(forecast_wind_gust_speed)
                    forecast_entry[ATTR_FORECAST_WIND_GUST_SPEED] = round(
                        UNIT_CONVERSIONS[ATTR_WEATHER_WIND_SPEED_UNIT](forecast_wind_gust_speed_f, from_wind_speed_unit, to_wind_speed_unit),
                        ROUNDING_PRECISION,
                    )
            if (forecast_wind_speed := forecast_entry.pop(ATTR_FORECAST_NATIVE_WIND_SPEED, forecast_entry.get(ATTR_FORECAST_WIND_SPEED))) is not None:
                from_wind_speed_unit = self.native_wind_speed_unit or self._default_wind_speed_unit
                to_wind_speed_unit = self._wind_speed_unit
                with suppress(TypeError, ValueError):
                    forecast_wind_speed_f = float(forecast_wind_speed)
                    forecast_entry[ATTR_FORECAST_WIND_SPEED] = round(
                        UNIT_CONVERSIONS[ATTR_WEATHER_WIND_SPEED_UNIT](forecast_wind_speed_f, from_wind_speed_unit, to_wind_speed_unit),
                        ROUNDING_PRECISION,
                    )
            if (forecast_precipitation := forecast_entry.pop(ATTR_FORECAST_NATIVE_PRECIPITATION, forecast_entry.get(ATTR_FORECAST_PRECIPITATION))) is not None:
                from_precipitation_unit = self.native_precipitation_unit or self._default_precipitation_unit
                to_precipitation_unit = self._precipitation_unit
                with suppress(TypeError, ValueError):
                    forecast_precipitation_f = float(forecast_precipitation)
                    forecast_entry[ATTR_FORECAST_PRECIPITATION] = round(
                        UNIT_CONVERSIONS[ATTR_WEATHER_PRECIPITATION_UNIT](forecast_precipitation_f, from_precipitation_unit, to_precipitation_unit),
                        ROUNDING_PRECISION,
                    )
            if (forecast_humidity := forecast_entry.pop(ATTR_FORECAST_HUMIDITY, None)) is not None:
                with suppress(TypeError, ValueError):
                    forecast_humidity_f = float(forecast_humidity)
                    forecast_entry[ATTR_FORECAST_HUMIDITY] = round(forecast_humidity_f)
            converted_forecast_list.append(forecast_entry)  # type: ignore
        return converted_forecast_list

    @property
    @final
    def state(self) -> Optional[str]:
        """Return the current state."""
        return self.condition

    @cached_property
    def condition(self) -> Optional[str]:
        """Return the current condition."""
        return self._attr_condition

    @callback
    def async_registry_entry_updated(self) -> None:
        """Run when the entity registry entry has been updated."""
        assert self.registry_entry  # type: ignore
        self._weather_option_temperature_unit = None
        self._weather_option_pressure_unit = None
        self._weather_option_precipitation_unit = None
        self._weather_option_wind_speed_unit = None
        self._weather_option_visibility_unit = None
        if (weather_options := self.registry_entry.options.get(DOMAIN)):  # type: ignore
            if (custom_unit_temperature := weather_options.get(ATTR_WEATHER_TEMPERATURE_UNIT)) and custom_unit_temperature in VALID_UNITS[ATTR_WEATHER_TEMPERATURE_UNIT]:
                self._weather_option_temperature_unit = custom_unit_temperature
            if (custom_unit_pressure := weather_options.get(ATTR_WEATHER_PRESSURE_UNIT)) and custom_unit_pressure in VALID_UNITS[ATTR_WEATHER_PRESSURE_UNIT]:
                self._weather_option_pressure_unit = custom_unit_pressure
            if (custom_unit_precipitation := weather_options.get(ATTR_WEATHER_PRECIPITATION_UNIT)) and custom_unit_precipitation in VALID_UNITS[ATTR_WEATHER_PRECIPITATION_UNIT]:
                self._weather_option_precipitation_unit = custom_unit_precipitation
            if (custom_unit_wind_speed := weather_options.get(ATTR_WEATHER_WIND_SPEED_UNIT)) and custom_unit_wind_speed in VALID_UNITS[ATTR_WEATHER_WIND_SPEED_UNIT]:
                self._weather_option_wind_speed_unit = custom_unit_wind_speed
            if (custom_unit_visibility := weather_options.get(ATTR_WEATHER_VISIBILITY_UNIT)) and custom_unit_visibility in VALID_UNITS[ATTR_WEATHER_VISIBILITY_UNIT]:
                self._weather_option_visibility_unit = custom_unit_visibility

    @callback
    def _async_subscription_started(self, forecast_type: str) -> None:
        """Start subscription to forecast_type."""
        # Actual implementation to be provided by subclass or override if needed.

    @callback
    def _async_subscription_ended(self, forecast_type: str) -> None:
        """End subscription to forecast_type."""
        # Actual implementation to be provided by subclass or override if needed.

    @final
    @callback
    def async_subscribe_forecast(
        self, forecast_type: str, forecast_listener: Callable[[Optional[List[Forecast]]], None]
    ) -> Callable[[], None]:
        """Subscribe to forecast updates.

        Called by websocket API.
        """
        subscription_started: bool = not self._forecast_listeners[forecast_type]
        self._forecast_listeners[forecast_type].append(forecast_listener)
        if subscription_started:
            self._async_subscription_started(forecast_type)

        @callback
        def unsubscribe() -> None:
            self._forecast_listeners[forecast_type].remove(forecast_listener)
            if not self._forecast_listeners[forecast_type]:
                self._async_subscription_ended(forecast_type)

        return unsubscribe

    @final
    async def async_update_listeners(self, forecast_types: Optional[Set[str]] = None) -> None:
        """Push updated forecast to all listeners."""
        if forecast_types is None:
            forecast_types = {"daily", "hourly", "twice_daily"}
        for forecast_type in forecast_types:
            if not self._forecast_listeners[forecast_type]:
                continue
            native_forecast_list = await getattr(self, f"async_forecast_{forecast_type}")()
            if native_forecast_list is None:
                for listener in self._forecast_listeners[forecast_type]:
                    listener(None)
                continue
            if forecast_type == "twice_daily":
                for fc_twice_daily in native_forecast_list:
                    if fc_twice_daily.get(ATTR_FORECAST_IS_DAYTIME) is None:
                        raise ValueError("is_daytime mandatory attribute for forecast_twice_daily is missing")
            converted_forecast_list = self._convert_forecast(native_forecast_list)
            for listener in self._forecast_listeners[forecast_type]:
                listener(converted_forecast_list)

def raise_unsupported_forecast(entity_id: str, forecast_type: str) -> None:
    """Raise error on attempt to get an unsupported forecast."""
    raise HomeAssistantError(f"Weather entity '{entity_id}' does not support '{forecast_type}' forecast")

async def async_get_forecasts_service(weather: WeatherEntity, service_call: ServiceCall) -> Dict[str, Any]:
    """Get weather forecast."""
    forecast_type: str = service_call.data["type"]
    supported_features: int = weather.supported_features or 0
    if forecast_type == "daily":
        if supported_features & WeatherEntityFeature.FORECAST_DAILY == 0:
            raise_unsupported_forecast(weather.entity_id, forecast_type)
        native_forecast_list = await weather.async_forecast_daily()
    elif forecast_type == "hourly":
        if supported_features & WeatherEntityFeature.FORECAST_HOURLY == 0:
            raise_unsupported_forecast(weather.entity_id, forecast_type)
        native_forecast_list = await weather.async_forecast_hourly()
    else:
        if supported_features & WeatherEntityFeature.FORECAST_TWICE_DAILY == 0:
            raise_unsupported_forecast(weather.entity_id, forecast_type)
        native_forecast_list = await weather.async_forecast_twice_daily()
    if native_forecast_list is None:
        converted_forecast_list: List[Forecast] = []
    else:
        converted_forecast_list = weather._convert_forecast(native_forecast_list)
    return {"forecast": converted_forecast_list}

class CoordinatorWeatherEntity(
    CoordinatorEntity[_ObservationUpdateCoordinatorT],
    WeatherEntity,
    Generic[
        _ObservationUpdateCoordinatorT,
        _DailyForecastUpdateCoordinatorT,
        _HourlyForecastUpdateCoordinatorT,
        _TwiceDailyForecastUpdateCoordinatorT,
    ],
):
    """A class for weather entities using DataUpdateCoordinators."""

    def __init__(
        self,
        observation_coordinator: _ObservationUpdateCoordinatorT,
        *,
        context: Optional[Any] = None,
        daily_coordinator: Optional[_DailyForecastUpdateCoordinatorT] = None,
        hourly_coordinator: Optional[_HourlyForecastUpdateCoordinatorT] = None,
        twice_daily_coordinator: Optional[_TwiceDailyForecastUpdateCoordinatorT] = None,
        daily_forecast_valid: Optional[timedelta] = None,
        hourly_forecast_valid: Optional[timedelta] = None,
        twice_daily_forecast_valid: Optional[timedelta] = None,
    ) -> None:
        """Initialize."""
        super().__init__(observation_coordinator, context)
        self.forecast_coordinators: Dict[str, Optional[DataUpdateCoordinator[Any]]] = {
            "daily": daily_coordinator,
            "hourly": hourly_coordinator,
            "twice_daily": twice_daily_coordinator,
        }
        self.forecast_valid: Dict[str, Optional[timedelta]] = {
            "daily": daily_forecast_valid,
            "hourly": hourly_forecast_valid,
            "twice_daily": twice_daily_forecast_valid,
        }
        self.unsub_forecast: Dict[str, Optional[CALLBACK_TYPE]] = {"daily": None, "hourly": None, "twice_daily": None}

    async def async_added_to_hass(self) -> None:
        """When entity is added to hass."""
        await super().async_added_to_hass()
        self.async_on_remove(partial(self._remove_forecast_listener, "daily"))
        self.async_on_remove(partial(self._remove_forecast_listener, "hourly"))
        self.async_on_remove(partial(self._remove_forecast_listener, "twice_daily"))

    def _remove_forecast_listener(self, forecast_type: str) -> None:
        """Remove weather forecast listener."""
        if (unsub_fn := self.unsub_forecast[forecast_type]):
            unsub_fn()
            self.unsub_forecast[forecast_type] = None

    @callback
    def _async_subscription_started(self, forecast_type: str) -> None:
        """Start subscription to forecast_type."""
        if not (coordinator := self.forecast_coordinators[forecast_type]):
            return
        self.unsub_forecast[forecast_type] = coordinator.async_add_listener(partial(self._handle_forecast_update, forecast_type))

    @callback
    def _handle_daily_forecast_coordinator_update(self) -> None:
        """Handle updated data from the daily forecast coordinator."""
        # Implementation specific to daily forecast.

    @callback
    def _handle_hourly_forecast_coordinator_update(self) -> None:
        """Handle updated data from the hourly forecast coordinator."""
        # Implementation specific to hourly forecast.

    @callback
    def _handle_twice_daily_forecast_coordinator_update(self) -> None:
        """Handle updated data from the twice daily forecast coordinator."""
        # Implementation specific to twice daily forecast.

    @final
    @callback
    def _handle_forecast_update(self, forecast_type: str) -> None:
        """Update forecast data."""
        coordinator = self.forecast_coordinators[forecast_type]
        assert coordinator
        assert coordinator.config_entry is not None
        getattr(self, f"_handle_{forecast_type}_forecast_coordinator_update")()
        coordinator.config_entry.async_create_task(self.hass, self.async_update_listeners({forecast_type}))

    @callback
    def _async_subscription_ended(self, forecast_type: str) -> None:
        """End subscription to forecast_type."""
        self._remove_forecast_listener(forecast_type)

    @final
    async def _async_refresh_forecast(
        self, coordinator: DataUpdateCoordinator[Any], forecast_valid_time: Optional[timedelta]
    ) -> bool:
        """Refresh stale forecast if needed."""
        if coordinator.update_interval is None:
            return True
        if forecast_valid_time is None:
            forecast_valid_time = coordinator.update_interval
        if not (last_success_time := coordinator.last_update_success_time) or utcnow() - last_success_time >= coordinator.update_interval:
            await coordinator.async_refresh()
        if not (last_success_time := coordinator.last_update_success_time) or utcnow() - last_success_time >= forecast_valid_time:
            return False
        return True

    @callback
    def _async_forecast_daily(self) -> Any:
        """Return the daily forecast in native units."""
        raise NotImplementedError

    @callback
    def _async_forecast_hourly(self) -> Any:
        """Return the hourly forecast in native units."""
        raise NotImplementedError

    @callback
    def _async_forecast_twice_daily(self) -> Any:
        """Return the twice daily forecast in native units."""
        raise NotImplementedError

    @final
    async def _async_forecast(self, forecast_type: str) -> Optional[List[Forecast]]:
        """Return the forecast in native units."""
        coordinator = self.forecast_coordinators[forecast_type]
        if coordinator and (not await self._async_refresh_forecast(coordinator, self.forecast_valid[forecast_type])):
            return None
        return cast(Optional[List[Forecast]], getattr(self, f"_async_forecast_{forecast_type}")())

    @final
    async def async_forecast_daily(self) -> Optional[List[Forecast]]:
        """Return the daily forecast in native units."""
        return await self._async_forecast("daily")

    @final
    async def async_forecast_hourly(self) -> Optional[List[Forecast]]:
        """Return the hourly forecast in native units."""
        return await self._async_forecast("hourly")

    @final
    async def async_forecast_twice_daily(self) -> Optional[List[Forecast]]:
        """Return the twice daily forecast in native units."""
        return await self._async_forecast("twice_daily")

class SingleCoordinatorWeatherEntity(CoordinatorWeatherEntity[_ObservationUpdateCoordinatorT, TimestampDataUpdateCoordinator[None]]):
    """A class for weather entities using a single DataUpdateCoordinators.

    This class is added as a convenience.
    """

    def __init__(self, coordinator: _ObservationUpdateCoordinatorT, context: Optional[Any] = None) -> None:
        """Initialize."""
        super().__init__(coordinator, context=context)

    @callback
    def _handle_coordinator_update(self) -> None:
        """Handle updated data from the coordinator."""
        super()._handle_coordinator_update()  # type: ignore
        assert self.coordinator.config_entry  # type: ignore
        self.coordinator.config_entry.async_create_task(self.hass, self.async_update_listeners(None))
