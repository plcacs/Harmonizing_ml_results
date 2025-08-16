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
