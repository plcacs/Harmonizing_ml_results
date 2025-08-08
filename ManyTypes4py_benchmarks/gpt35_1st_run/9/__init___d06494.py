from __future__ import annotations
from typing import Any, Final, Generic, Literal, TypedDict, TypeVar, cast

from homeassistant.config_entries import ConfigEntry
from homeassistant.const import PRECISION_HALVES, PRECISION_TENTHS, PRECISION_WHOLE, UnitOfPressure, UnitOfSpeed, UnitOfTemperature
from homeassistant.core import CALLBACK_TYPE, HomeAssistant, ServiceCall, SupportsResponse, callback
from homeassistant.exceptions import HomeAssistantError
from homeassistant.helpers.entity import ABCCachedProperties, Entity, EntityDescription
from homeassistant.helpers.entity_component import EntityComponent
from homeassistant.helpers.typing import ConfigType
from homeassistant.helpers.update_coordinator import CoordinatorEntity, DataUpdateCoordinator, TimestampDataUpdateCoordinator
from homeassistant.util.dt import utcnow
from homeassistant.util.unit_system import US_CUSTOMARY_SYSTEM

from .const import ATTR_WEATHER_APPARENT_TEMPERATURE, ATTR_WEATHER_CLOUD_COVERAGE, ATTR_WEATHER_DEW_POINT, ATTR_WEATHER_HUMIDITY, ATTR_WEATHER_OZONE, ATTR_WEATHER_PRECIPITATION_UNIT, ATTR_WEATHER_PRESSURE, ATTR_WEATHER_PRESSURE_UNIT, ATTR_WEATHER_TEMPERATURE, ATTR_WEATHER_TEMPERATURE_UNIT, ATTR_WEATHER_UV_INDEX, ATTR_WEATHER_VISIBILITY, ATTR_WEATHER_VISIBILITY_UNIT, ATTR_WEATHER_WIND_BEARING, ATTR_WEATHER_WIND_GUST_SPEED, ATTR_WEATHER_WIND_SPEED, ATTR_WEATHER_WIND_SPEED_UNIT, DATA_COMPONENT, DOMAIN, INTENT_GET_WEATHER, UNIT_CONVERSIONS, VALID_UNITS, WeatherEntityFeature

_ObservationUpdateCoordinatorT = TypeVar('_ObservationUpdateCoordinatorT', bound=DataUpdateCoordinator[Any], default=DataUpdateCoordinator[dict[str, Any]])
_DailyForecastUpdateCoordinatorT = TypeVar('_DailyForecastUpdateCoordinatorT', bound=TimestampDataUpdateCoordinator[Any], default=TimestampDataUpdateCoordinator[None])
_HourlyForecastUpdateCoordinatorT = TypeVar('_HourlyForecastUpdateCoordinatorT', bound=TimestampDataUpdateCoordinator[Any], default=_DailyForecastUpdateCoordinatorT)
_TwiceDailyForecastUpdateCoordinatorT = TypeVar('_TwiceDailyForecastUpdateCoordinatorT', bound=TimestampDataUpdateCoordinator[Any], default=_DailyForecastUpdateCoordinatorT)

class WeatherEntityDescription(EntityDescription, frozen_or_thawed=True):
    """A class that describes weather entities."""

class PostInitMeta(ABCCachedProperties):
    """Meta class which calls __post_init__ after __new__ and __init__."""

    def __call__(cls, *args, **kwargs):
        """Create an instance."""
        instance = super().__call__(*args, **kwargs)
        instance.__post_init__(*args, **kwargs)
        return instance

class PostInit(metaclass=PostInitMeta):
    """Class which calls __post_init__ after __new__ and __init__."""

    @abc.abstractmethod
    def __post_init__(self, *args, **kwargs):
        """Finish initializing."""

CACHED_PROPERTIES_WITH_ATTR_: Final[set[str]] = {'native_apparent_temperature', 'native_temperature', 'native_temperature_unit', 'native_dew_point', 'native_pressure', 'native_pressure_unit', 'humidity', 'native_wind_gust_speed', 'native_wind_speed', 'native_wind_speed_unit', 'wind_bearing', 'ozone', 'cloud_coverage', 'uv_index', 'native_visibility', 'native_visibility_unit', 'native_precipitation_unit', 'condition'}

class WeatherEntity(Entity, PostInit, cached_properties=CACHED_PROPERTIES_WITH_ATTR_):
    """ABC for weather data."""
    _attr_condition: Any = None
    _attr_humidity: Any = None
    _attr_ozone: Any = None
    _attr_cloud_coverage: Any = None
    _attr_uv_index: Any = None
    _attr_state: Any = None
    _attr_wind_bearing: Any = None
    _attr_native_pressure: Any = None
    _attr_native_pressure_unit: Any = None
    _attr_native_apparent_temperature: Any = None
    _attr_native_temperature: Any = None
    _attr_native_temperature_unit: Any = None
    _attr_native_visibility: Any = None
    _attr_native_visibility_unit: Any = None
    _attr_native_precipitation_unit: Any = None
    _attr_native_wind_gust_speed: Any = None
    _attr_native_wind_speed: Any = None
    _attr_native_wind_speed_unit: Any = None
    _attr_native_dew_point: Any = None
    _weather_option_temperature_unit: Any = None
    _weather_option_pressure_unit: Any = None
    _weather_option_visibility_unit: Any = None
    _weather_option_precipitation_unit: Any = None
    _weather_option_wind_speed_unit: Any = None
