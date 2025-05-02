"""Weather component that handles meteorological data for your location."""
from __future__ import annotations
import abc
from collections.abc import Callable, Iterable
from contextlib import suppress
from datetime import timedelta
from functools import partial
import logging
from typing import Any, Final, Generic, Literal, Optional, Required, TypedDict, TypeVar, Union, cast, final
from propcache.api import cached_property
import voluptuous as vol
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
from .const import ATTR_WEATHER_APPARENT_TEMPERATURE, ATTR_WEATHER_CLOUD_COVERAGE, ATTR_WEATHER_DEW_POINT, ATTR_WEATHER_HUMIDITY, ATTR_WEATHER_OZONE, ATTR_WEATHER_PRECIPITATION_UNIT, ATTR_WEATHER_PRESSURE, ATTR_WEATHER_PRESSURE_UNIT, ATTR_WEATHER_TEMPERATURE, ATTR_WEATHER_TEMPERATURE_UNIT, ATTR_WEATHER_UV_INDEX, ATTR_WEATHER_VISIBILITY, ATTR_WEATHER_VISIBILITY_UNIT, ATTR_WEATHER_WIND_BEARING, ATTR_WEATHER_WIND_GUST_SPEED, ATTR_WEATHER_WIND_SPEED, ATTR_WEATHER_WIND_SPEED_UNIT, DATA_COMPONENT, DOMAIN, INTENT_GET_WEATHER, UNIT_CONVERSIONS, VALID_UNITS, WeatherEntityFeature
from .websocket_api import async_setup as async_setup_ws_api

_LOGGER: Final = logging.getLogger(__name__)
ENTITY_ID_FORMAT: Final = DOMAIN + '.{}'
PLATFORM_SCHEMA: Final = cv.PLATFORM_SCHEMA
PLATFORM_SCHEMA_BASE: Final = cv.PLATFORM_SCHEMA_BASE
SCAN_INTERVAL: Final = timedelta(seconds=30)
ATTR_CONDITION_CLASS: Final = 'condition_class'
ATTR_CONDITION_CLEAR_NIGHT: Final = 'clear-night'
ATTR_CONDITION_CLOUDY: Final = 'cloudy'
ATTR_CONDITION_EXCEPTIONAL: Final = 'exceptional'
ATTR_CONDITION_FOG: Final = 'fog'
ATTR_CONDITION_HAIL: Final = 'hail'
ATTR_CONDITION_LIGHTNING: Final = 'lightning'
ATTR_CONDITION_LIGHTNING_RAINY: Final = 'lightning-rainy'
ATTR_CONDITION_PARTLYCLOUDY: Final = 'partlycloudy'
ATTR_CONDITION_POURING: Final = 'pouring'
ATTR_CONDITION_RAINY: Final = 'rainy'
ATTR_CONDITION_SNOWY: Final = 'snowy'
ATTR_CONDITION_SNOWY_RAINY: Final = 'snowy-rainy'
ATTR_CONDITION_SUNNY: Final = 'sunny'
ATTR_CONDITION_WINDY: Final = 'windy'
ATTR_CONDITION_WINDY_VARIANT: Final = 'windy-variant'
ATTR_FORECAST_IS_DAYTIME: Final = 'is_daytime'
ATTR_FORECAST_CONDITION: Final = 'condition'
ATTR_FORECAST_HUMIDITY: Final = 'humidity'
ATTR_FORECAST_NATIVE_PRECIPITATION: Final = 'native_precipitation'
ATTR_FORECAST_PRECIPITATION: Final = 'precipitation'
ATTR_FORECAST_PRECIPITATION_PROBABILITY: Final = 'precipitation_probability'
ATTR_FORECAST_NATIVE_PRESSURE: Final = 'native_pressure'
ATTR_FORECAST_PRESSURE: Final = 'pressure'
ATTR_FORECAST_NATIVE_APPARENT_TEMP: Final = 'native_apparent_temperature'
ATTR_FORECAST_APPARENT_TEMP: Final = 'apparent_temperature'
ATTR_FORECAST_NATIVE_TEMP: Final = 'native_temperature'
ATTR_FORECAST_TEMP: Final = 'temperature'
ATTR_FORECAST_NATIVE_TEMP_LOW: Final = 'native_templow'
ATTR_FORECAST_TEMP_LOW: Final = 'templow'
ATTR_FORECAST_TIME: Final = 'datetime'
ATTR_FORECAST_WIND_BEARING: Final = 'wind_bearing'
ATTR_FORECAST_NATIVE_WIND_GUST_SPEED: Final = 'native_wind_gust_speed'
ATTR_FORECAST_WIND_GUST_SPEED: Final = 'wind_gust_speed'
ATTR_FORECAST_NATIVE_WIND_SPEED: Final = 'native_wind_speed'
ATTR_FORECAST_WIND_SPEED: Final = 'wind_speed'
ATTR_FORECAST_NATIVE_DEW_POINT: Final = 'native_dew_point'
ATTR_FORECAST_DEW_POINT: Final = 'dew_point'
ATTR_FORECAST_CLOUD_COVERAGE: Final = 'cloud_coverage'
ATTR_FORECAST_UV_INDEX: Final = 'uv_index'
ROUNDING_PRECISION: Final = 2
SERVICE_GET_FORECASTS: Final = 'get_forecasts'

_ObservationUpdateCoordinatorT = TypeVar('_ObservationUpdateCoordinatorT', bound=DataUpdateCoordinator[Any], default=DataUpdateCoordinator[dict[str, Any]])
_DailyForecastUpdateCoordinatorT = TypeVar('_DailyForecastUpdateCoordinatorT', bound=TimestampDataUpdateCoordinator[Any], default=TimestampDataUpdateCoordinator[None])
_HourlyForecastUpdateCoordinatorT = TypeVar('_HourlyForecastUpdateCoordinatorT', bound=TimestampDataUpdateCoordinator[Any], default=_DailyForecastUpdateCoordinatorT)
_TwiceDailyForecastUpdateCoordinatorT = TypeVar('_TwiceDailyForecastUpdateCoordinatorT', bound=TimestampDataUpdateCoordinator[Any], default=_DailyForecastUpdateCoordinatorT)

def round_temperature(temperature: Optional[float], precision: int) -> Optional[float]:
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
    datetime: str
    condition: Optional[str]
    precipitation: Optional[float]
    precipitation_probability: Optional[float]
    temperature: Optional[float]
    templow: Optional[float]
    wind_speed: Optional[float]
    wind_bearing: Optional[float]
    humidity: Optional[float]
    pressure: Optional[float]
    is_daytime: Optional[bool]
    native_precipitation: Optional[float]
    native_temperature: Optional[float]
    native_templow: Optional[float]
    native_wind_speed: Optional[float]
    native_pressure: Optional[float]
    apparent_temperature: Optional[float]
    native_apparent_temperature: Optional[float]
    wind_gust_speed: Optional[float]
    native_wind_gust_speed: Optional[float]
    cloud_coverage: Optional[float]
    uv_index: Optional[float]
    dew_point: Optional[float]
    native_dew_point: Optional[float]

async def async_setup(hass: HomeAssistant, config: ConfigType) -> bool:
    """Set up the weather component."""
    component = hass.data[DATA_COMPONENT] = EntityComponent[WeatherEntity](_LOGGER, DOMAIN, hass, SCAN_INTERVAL)
    component.async_register_entity_service(SERVICE_GET_FORECASTS, {vol.Required('type'): vol.In(('daily', 'hourly', 'twice_daily'))}, async_get_forecasts_service, required_features=[WeatherEntityFeature.FORECAST_DAILY, WeatherEntityFeature.FORECAST_HOURLY, WeatherEntityFeature.FORECAST_TWICE_DAILY], supports_response=SupportsResponse.ONLY)
    async_setup_ws_api(hass)
    await component.async_setup(config)
    return True

async def async_setup_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    """Set up a config entry."""
    return await hass.data[DATA_COMPONENT].async_setup_entry(entry)

async def async_unload_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    """Unload a config entry."""
    return await hass.data[DATA_COMPONENT].async_unload_entry(entry)

class WeatherEntityDescription(EntityDescription, frozen_or_thawed=True):
    """A class that describes weather entities."""

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

CACHED_PROPERTIES_WITH_ATTR_: Final = {'native_apparent_temperature', 'native_temperature', 'native_temperature_unit', 'native_dew_point', 'native_pressure', 'native_pressure_unit', 'humidity', 'native_wind_gust_speed', 'native_wind_speed', 'native_wind_speed_unit', 'wind_bearing', 'ozone', 'cloud_coverage', 'uv_index', 'native_visibility', 'native_visibility_unit', 'native_precipitation_unit', 'condition'}

class WeatherEntity(Entity, PostInit, cached_properties=CACHED_PROPERTIES_WITH_ATTR_):
    """ABC for weather data."""
    _attr_condition: Optional[str] = None
    _attr_humidity: Optional[float] = None
    _attr_ozone: Optional[float] = None
    _attr_cloud_coverage: Optional[float] = None
    _attr_uv_index: Optional[float] = None
    _attr_state: Optional[str] = None
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
        self._forecast_listeners: dict[str, list[Callable[[Optional[list[Forecast]]], None]]] = {'daily': [], 'hourly': [], 'twice_daily': []}

    async def async_internal_added_to_hass(self) -> None:
        """Call when the weather entity is added to hass."""
        await super().async_internal_added_to_hass()
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
        return self.hass.config.units.temperature_unit

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
        if self.hass.config.units is US_CUSTOMARY_SYSTEM:
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
        if self.hass.config.units is US_CUSTOMARY_SYSTEM:
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
        """Return the