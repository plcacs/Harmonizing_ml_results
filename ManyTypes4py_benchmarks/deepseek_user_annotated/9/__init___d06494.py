"""Weather component that handles meteorological data for your location."""

from __future__ import annotations

import abc
from collections.abc import Callable, Iterable
from contextlib import suppress
from datetime import timedelta
from functools import partial
import logging
from typing import (
    Any,
    Final,
    Generic,
    Literal,
    Required,
    TypedDict,
    TypeVar,
    cast,
    final,
    Optional,
    Union,
    Dict,
    List,
    Set,
    Tuple,
)

from propcache.api import cached_property
import voluptuous as vol

from homeassistant.config_entries import ConfigEntry
from homeassistant.const import (
    PRECISION_HALVES,
    PRECISION_TENTHS,
    PRECISION_WHOLE,
    UnitOfPressure,
    UnitOfSpeed,
    UnitOfTemperature,
)
from homeassistant.core import (
    CALLBACK_TYPE,
    HomeAssistant,
    ServiceCall,
    ServiceResponse,
    SupportsResponse,
    callback,
)
from homeassistant.exceptions import HomeAssistantError
from homeassistant.helpers import config_validation as cv
from homeassistant.helpers.entity import ABCCachedProperties, Entity, EntityDescription
from homeassistant.helpers.entity_component import EntityComponent
from homeassistant.helpers.typing import ConfigType
from homeassistant.helpers.update_coordinator import (
    CoordinatorEntity,
    DataUpdateCoordinator,
    TimestampDataUpdateCoordinator,
)
from homeassistant.util.dt import utcnow
from homeassistant.util.json import JsonValueType
from homeassistant.util.unit_system import US_CUSTOMARY_SYSTEM

from .const import (  # noqa: F401
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

_LOGGER: Final = logging.getLogger(__name__)

ENTITY_ID_FORMAT: Final = DOMAIN + ".{}"
PLATFORM_SCHEMA: Final = cv.PLATFORM_SCHEMA
PLATFORM_SCHEMA_BASE: Final = cv.PLATFORM_SCHEMA_BASE
SCAN_INTERVAL: Final = timedelta(seconds=30)

ATTR_CONDITION_CLASS: Final = "condition_class"
ATTR_CONDITION_CLEAR_NIGHT: Final = "clear-night"
ATTR_CONDITION_CLOUDY: Final = "cloudy"
ATTR_CONDITION_EXCEPTIONAL: Final = "exceptional"
ATTR_CONDITION_FOG: Final = "fog"
ATTR_CONDITION_HAIL: Final = "hail"
ATTR_CONDITION_LIGHTNING: Final = "lightning"
ATTR_CONDITION_LIGHTNING_RAINY: Final = "lightning-rainy"
ATTR_CONDITION_PARTLYCLOUDY: Final = "partlycloudy"
ATTR_CONDITION_POURING: Final = "pouring"
ATTR_CONDITION_RAINY: Final = "rainy"
ATTR_CONDITION_SNOWY: Final = "snowy"
ATTR_CONDITION_SNOWY_RAINY: Final = "snowy-rainy"
ATTR_CONDITION_SUNNY: Final = "sunny"
ATTR_CONDITION_WINDY: Final = "windy"
ATTR_CONDITION_WINDY_VARIANT: Final = "windy-variant"
ATTR_FORECAST_IS_DAYTIME: Final = "is_daytime"
ATTR_FORECAST_CONDITION: Final = "condition"
ATTR_FORECAST_HUMIDITY: Final = "humidity"
ATTR_FORECAST_NATIVE_PRECIPITATION: Final = "native_precipitation"
ATTR_FORECAST_PRECIPITATION: Final = "precipitation"
ATTR_FORECAST_PRECIPITATION_PROBABILITY: Final = "precipitation_probability"
ATTR_FORECAST_NATIVE_PRESSURE: Final = "native_pressure"
ATTR_FORECAST_PRESSURE: Final = "pressure"
ATTR_FORECAST_NATIVE_APPARENT_TEMP: Final = "native_apparent_temperature"
ATTR_FORECAST_APPARENT_TEMP: Final = "apparent_temperature"
ATTR_FORECAST_NATIVE_TEMP: Final = "native_temperature"
ATTR_FORECAST_TEMP: Final = "temperature"
ATTR_FORECAST_NATIVE_TEMP_LOW: Final = "native_templow"
ATTR_FORECAST_TEMP_LOW: Final = "templow"
ATTR_FORECAST_TIME: Final = "datetime"
ATTR_FORECAST_WIND_BEARING: Final = "wind_bearing"
ATTR_FORECAST_NATIVE_WIND_GUST_SPEED: Final = "native_wind_gust_speed"
ATTR_FORECAST_WIND_GUST_SPEED: Final = "wind_gust_speed"
ATTR_FORECAST_NATIVE_WIND_SPEED: Final = "native_wind_speed"
ATTR_FORECAST_WIND_SPEED: Final = "wind_speed"
ATTR_FORECAST_NATIVE_DEW_POINT: Final = "native_dew_point"
ATTR_FORECAST_DEW_POINT: Final = "dew_point"
ATTR_FORECAST_CLOUD_COVERAGE: Final = "cloud_coverage"
ATTR_FORECAST_UV_INDEX: Final = "uv_index"

ROUNDING_PRECISION: Final = 2

SERVICE_GET_FORECASTS: Final = "get_forecasts"

_ObservationUpdateCoordinatorT = TypeVar(
    "_ObservationUpdateCoordinatorT",
    bound=DataUpdateCoordinator[Any],
    default=DataUpdateCoordinator[dict[str, Any]],
)

_DailyForecastUpdateCoordinatorT = TypeVar(
    "_DailyForecastUpdateCoordinatorT",
    bound=TimestampDataUpdateCoordinator[Any],
    default=TimestampDataUpdateCoordinator[None],
)
_HourlyForecastUpdateCoordinatorT = TypeVar(
    "_HourlyForecastUpdateCoordinatorT",
    bound=TimestampDataUpdateCoordinator[Any],
    default=_DailyForecastUpdateCoordinatorT,
)
_TwiceDailyForecastUpdateCoordinatorT = TypeVar(
    "_TwiceDailyForecastUpdateCoordinatorT",
    bound=TimestampDataUpdateCoordinator[Any],
    default=_DailyForecastUpdateCoordinatorT,
)

def round_temperature(temperature: float | None, precision: float) -> float | None:
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
    """Typed weather forecast dict."""
    condition: str | None
    datetime: Required[str]
    humidity: float | None
    precipitation_probability: int | None
    cloud_coverage: int | None
    native_precipitation: float | None
    precipitation: None
    native_pressure: float | None
    pressure: None
    native_temperature: float | None
    temperature: None
    native_templow: float | None
    templow: None
    native_apparent_temperature: float | None
    wind_bearing: float | str | None
    native_wind_gust_speed: float | None
    native_wind_speed: float | None
    wind_speed: None
    native_dew_point: float | None
    uv_index: float | None
    is_daytime: bool | None

async def async_setup(hass: HomeAssistant, config: ConfigType) -> bool:
    """Set up the weather component."""
    component = hass.data[DATA_COMPONENT] = EntityComponent[WeatherEntity](
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
        instance: PostInit = super().__call__(*args, **kwargs)
        instance.__post_init__(*args, **kwargs)
        return instance

class PostInit(metaclass=PostInitMeta):
    """Class which calls __post_init__ after __new__ and __init__."""

    @abc.abstractmethod
    def __post_init__(self, *args: Any, **kwargs: Any) -> None:
        """Finish initializing."""

CACHED_PROPERTIES_WITH_ATTR_: Final = {
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

    entity_description: WeatherEntityDescription
    _attr_condition: str | None = None
    _attr_humidity: float | None = None
    _attr_ozone: float | None = None
    _attr_cloud_coverage: int | None = None
    _attr_uv_index: float | None = None
    _attr_precision: float
    _attr_state: None = None
    _attr_wind_bearing: float | str | None = None

    _attr_native_pressure: float | None = None
    _attr_native_pressure_unit: str | None = None
    _attr_native_apparent_temperature: float | None = None
    _attr_native_temperature: float | None = None
    _attr_native_temperature_unit: str | None = None
    _attr_native_visibility: float | None = None
    _attr_native_visibility_unit: str | None = None
    _attr_native_precipitation_unit: str | None = None
    _attr_native_wind_gust_speed: float | None = None
    _attr_native_wind_speed: float | None = None
    _attr_native_wind_speed_unit: str | None = None
    _attr_native_dew_point: float | None = None

    _forecast_listeners: Dict[
        Literal["daily", "hourly", "twice_daily"],
        List[Callable[[List[JsonValueType] | None], None]],
    ]

    _weather_option_temperature_unit: str | None = None
    _weather_option_pressure_unit: str | None = None
    _weather_option_visibility_unit: str | None = None
    _weather_option_precipitation_unit: str | None = None
    _weather_option_wind_speed_unit: str | None = None

    def __post_init__(self, *args: Any, **kwargs: Any) -> None:
        """Finish initializing."""
        self._forecast_listeners = {"daily": [], "hourly": [], "twice_daily": []}

    async def async_internal_added_to_hass(self) -> None:
        """Call when the weather entity is added to hass."""
        await super().async_internal_added_to_hass()
        if not self.registry_entry:
            return
        self.async_registry_entry_updated()

    @cached_property
    def native_apparent_temperature(self) -> float | None:
        """Return the apparent temperature in native units."""
        return self._attr_native_apparent_temperature

    @cached_property
    def native_temperature(self) -> float | None:
        """Return the temperature in native units."""
        return self._attr_native_temperature

    @cached_property
    def native_temperature_unit(self) -> str | None:
        """Return the native unit of measurement for temperature."""
        return self._attr_native_temperature_unit

    @cached_property
    def native_dew_point(self) -> float | None:
        """Return the dew point temperature in native units."""
        return self._attr_native_dew_point

    @final
    @property
    def _default_temperature_unit(self) -> str:
        """Return the default unit of measurement for temperature."""
        return self.hass.config.units.temperature_unit

    @final
    @property
    def _temperature_unit(self) -> str:
        """Return the converted unit of measurement for temperature."""
        if (weather_option_temperature_unit := self._weather_option_temperature_unit) is not None:
            return weather_option_temperature_unit
        return self._default_temperature_unit

    @cached_property
    def native_pressure(self) -> float | None:
        """Return the pressure in native units."""
        return self._attr_native_pressure

    @cached_property
    def native_pressure_unit(self) -> str | None:
        """Return the native unit of measurement for pressure."""
        return self._attr_native_pressure_unit

    @final
    @property
    def _default_pressure_unit(self) -> str:
        """Return the default unit of measurement for pressure."""
        if self.hass.config.units is US_CUSTOMARY_SYSTEM:
            return UnitOfPressure.INHG
        return UnitOfPressure.HPA

    @final
    @property
    def _pressure_unit(self) -> str:
        """Return the converted unit of measurement for pressure."""
        if (weather_option_pressure_unit := self._weather_option_pressure_unit) is not None:
            return weather_option_pressure_unit
        return self._default_pressure_unit

    @cached_property
    def humidity(self) -> float | None:
        """Return the humidity in native units."""
        return self._attr_humidity

    @cached_property
    def native_wind_gust_speed(self) -> float | None:
        """Return the wind gust speed in native units."""
        return self._attr_native_wind_gust_speed

    @cached_property
    def native_wind_speed(self) -> float | None:
        """Return the wind speed in native units."""
        return self._attr_native_wind_speed

    @cached_property
    def native_wind_speed_unit(self) -> str | None:
        """Return the native unit of measurement for wind speed."""
        return self._attr_native_wind_speed_unit

    @final
    @property
    def _default_wind_speed_unit(self) -> str:
        """Return the default unit of measurement for wind speed."""
        if self.hass.config.units is US_CUSTOMARY_SYSTEM:
            return UnitOfSpeed.MILES_PER_HOUR
        return UnitOfSpeed.KILOMETERS_PER_HOUR

    @final
    @property
    def _wind_speed_unit(self) -> str:
        """Return the converted unit of measurement for wind speed."""
        if (weather_option_wind_speed_unit := self._weather_option_wind_speed_unit) is not None:
            return weather_option_wind_speed_unit
        return self._default_wind_speed_unit

    @cached_property
    def wind_bearing(self) -> float | str | None:
        """Return the wind bearing."""
        return self._attr_wind_bearing

    @cached_property
    def ozone(self) -> float | None:
        """Return the ozone level."""
        return self._attr_ozone

    @cached_property
    def cloud_coverage(self) -> float | None:
        """Return the Cloud coverage in %."""
        return self._attr_cloud_coverage

    @cached_property
    def uv_index(self) -> float | None:
        """Return the UV index."""
        return self._attr_uv_index

    @cached_property
   